"""Trainer implementations for depth learning models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb

from utils.config import TrainingConfig
from data.data import create_visual_transform, AugmentationConfig
from utils.logging import get_logger

logger = get_logger(__name__)

from .base import BaseTrainer
class LeWMPlusTrainer(BaseTrainer):
    """Trainer for LeWM++ (Multi-modal JEPA with SigReg regularization)."""

    def build_model(self) -> nn.Module:
        """Build LeWM++ model."""
        from models.acoustic import TransformerEncoder
        from models.lewm_plus import LeWMPlus

        task = getattr(self.config, 'task', 'presence')

        ac_encoder = TransformerEncoder(embed_dim=self.config.embed_dim)

        return LeWMPlus(
            ac_encoder=ac_encoder,
            embed_dim=self.config.embed_dim,
            use_focal_loss=self.config.use_focal_loss,
            task=task,
            use_sigreg=True,
            sigreg_weight=self.config.sigreg_weight,
            use_decoder=False,
            n_classes=4,
        ).to(self.device)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_loss_jepa = 0
        total_loss_cls = 0
        total_loss_sigreg = 0

        # Task-specific metrics
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_mae = 0
        total_rmse = 0

        # Per-class F1 (for presence task)
        class_tp = torch.zeros(4)
        class_fp = torch.zeros(4)
        class_fn = torch.zeros(4)

        total_samples = 0

        pbar = tqdm(loader, desc="Training")
        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            predicted_target, target_latent, species_logits, recon_img, sigreg_loss = \
                self.model(vis, ac, labels)

            # Compute loss
            loss, loss_jepa, loss_cls, loss_sigreg, loss_recon = \
                self.model.compute_loss(
                    predicted_target, target_latent, species_logits, labels,
                    sigreg_loss=sigreg_loss,
                )

            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_loss_jepa += loss_jepa.item()
            total_loss_cls += loss_cls.item()
            total_loss_sigreg += loss_sigreg.item() if sigreg_loss is not None else 0

            # Task-specific metrics
            if self.model.task == "counting":
                # Counting metrics: MAE, RMSE
                pred_counts = species_logits.clamp(min=0)
                true_counts = labels.clamp(min=0)

                mae = F.l1_loss(pred_counts, true_counts, reduction='sum')
                mse = F.mse_loss(pred_counts, true_counts, reduction='sum')

                total_mae += mae.item()
                total_rmse += mse.item()
                total_samples += labels.shape[0]

                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "mae": f"{mae.item()/labels.shape[0]:.3f}",
                })
            else:
                # Multi-label metrics
                probs = torch.sigmoid(species_logits)
                preds = (probs > 0.5).float()

                for i in range(labels.shape[0]):
                    tp = preds[i] * labels[i]
                    fp = preds[i] * (1 - labels[i])
                    fn = (1 - preds[i]) * labels[i]

                    for c in range(4):
                        class_tp[c] += tp[c].item()
                        class_fp[c] += fp[c].item()
                        class_fn[c] += fn[c].item()

                    tp_sum = tp.sum().item()
                    fp_sum = fp.sum().item()
                    fn_sum = fn.sum().item()

                    precision = tp_sum / (tp_sum + fp_sum + 1e-8)
                    recall = tp_sum / (tp_sum + fn_sum + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1

                total_samples += labels.shape[0]

                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "f1": f"{100*total_f1/total_samples:.1f}%",
                })

        # Compute final metrics
        if self.model.task == "counting":
            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "loss_sigreg": total_loss_sigreg / len(loader),
                "mae": total_mae / total_samples if total_samples > 0 else 0,
                "rmse": torch.sqrt(torch.tensor(total_rmse / total_samples)).item() if total_samples > 0 else 0,
            }
        else:
            # Per-class F1
            class_precision = class_tp / (class_tp + class_fp + 1e-8)
            class_recall = class_tp / (class_tp + class_fn + 1e-8)
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)

            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "loss_sigreg": total_loss_sigreg / len(loader),
                "precision": total_precision / total_samples,
                "recall": total_recall / total_samples,
                "f1": total_f1 / total_samples,
                "f1_kingfish": class_f1[0].item(),
                "f1_snapper": class_f1[1].item(),
                "f1_cod": class_f1[2].item(),
                "f1_empty": class_f1[3].item(),
            }

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_loss_jepa = 0
        total_loss_cls = 0
        total_loss_sigreg = 0

        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_sim = 0
        total_mae = 0
        total_rmse = 0

        class_tp = torch.zeros(4)
        class_fp = torch.zeros(4)
        class_fn = torch.zeros(4)
        total_samples = 0

        with torch.no_grad():
            for vis, ac, labels in loader:
                vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

                # Forward pass
                predicted_target, target_latent, species_logits, recon_img, sigreg_loss = \
                    self.model(vis, ac, labels)

                # Compute loss
                loss, loss_jepa, loss_cls, loss_sigreg, loss_recon = \
                    self.model.compute_loss(
                        predicted_target, target_latent, species_logits, labels,
                        sigreg_loss=sigreg_loss,
                    )

                total_loss += loss.item()
                total_loss_jepa += loss_jepa.item()
                total_loss_cls += loss_cls.item()
                total_loss_sigreg += loss_sigreg.item() if sigreg_loss is not None else 0

                # Cosine similarity
                sim = F.cosine_similarity(predicted_target, target_latent, dim=-1).mean()
                total_sim += sim.item()

                # Task-specific metrics
                if self.model.task == "counting":
                    # Counting metrics: MAE, RMSE
                    pred_counts = species_logits.clamp(min=0)
                    true_counts = labels.clamp(min=0)

                    mae = F.l1_loss(pred_counts, true_counts, reduction='sum')
                    mse = F.mse_loss(pred_counts, true_counts, reduction='sum')

                    total_mae += mae.item()
                    total_rmse += mse.item()
                    total_samples += labels.shape[0]
                else:
                    # Multi-label metrics
                    probs = torch.sigmoid(species_logits)
                    preds = (probs > 0.5).float()

                    for i in range(labels.shape[0]):
                        tp = preds[i] * labels[i]
                        fp = preds[i] * (1 - labels[i])
                        fn = (1 - preds[i]) * labels[i]

                        for c in range(4):
                            class_tp[c] += tp[c].item()
                            class_fp[c] += fp[c].item()
                            class_fn[c] += fn[c].item()

                        tp_sum = tp.sum().item()
                        fp_sum = fp.sum().item()
                        fn_sum = fn.sum().item()

                        precision = tp_sum / (tp_sum + fp_sum + 1e-8)
                        recall = tp_sum / (tp_sum + fn_sum + 1e-8)
                        f1 = 2 * precision * recall / (precision + recall + 1e-8)

                        total_precision += precision
                        total_recall += recall
                        total_f1 += f1

                    total_samples += labels.shape[0]

        # Compute final metrics
        if self.model.task == "counting":
            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "loss_sigreg": total_loss_sigreg / len(loader),
                "sim": total_sim / len(loader),
                "mae": total_mae / total_samples if total_samples > 0 else 0,
                "rmse": torch.sqrt(torch.tensor(total_rmse / total_samples)).item() if total_samples > 0 else 0,
            }
        else:
            # Per-class F1
            class_precision = class_tp / (class_tp + class_fp + 1e-8)
            class_recall = class_tp / (class_tp + class_fn + 1e-8)
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)

            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "loss_sigreg": total_loss_sigreg / len(loader),
                "sim": total_sim / len(loader),
                "precision": total_precision / total_samples,
                "recall": total_recall / total_samples,
                "f1": total_f1 / total_samples,
                "f1_kingfish": class_f1[0].item(),
                "f1_snapper": class_f1[1].item(),
                "f1_cod": class_f1[2].item(),
                "f1_empty": class_f1[3].item(),
            }

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Use task-appropriate metric for model selection."""
        if self.model.task == "counting":
            return -val_metrics.get("mae", 0)  # Lower MAE is better
        return val_metrics.get("f1", 0)  # Higher F1 is better


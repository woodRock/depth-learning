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
class LeWMTrainer(BaseTrainer):
    """Trainer for LeWorldModel with multi-label presence/absence or counting."""

    def build_model(self) -> nn.Module:
        """Build LeWorldModel with task-specific head."""
        from models.lewm_multilabel import LeWorldModelMultiLabel

        task = getattr(self.config, 'task', 'presence')

        return LeWorldModelMultiLabel(
            embed_dim=self.config.embed_dim,
            num_layers=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop=0.1,
            n_classes=4,
            use_classifier=True,
            use_decoder=True,  # Enable world reconstruction decoder
            task=task,
        ).to(self.device)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train LeWM for one epoch with task-specific loss."""
        self.model.train()
        total_loss = 0
        total_loss_pred = 0
        total_loss_cls = 0
        total_loss_sigreg = 0
        total_loss_recon = 0

        # Multi-label metrics
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_mae = 0
        total_rmse = 0
        total_samples = 0

        # Per-class F1
        class_tp = torch.zeros(4)
        class_fp = torch.zeros(4)
        class_fn = torch.zeros(4)

        # Get task once at the beginning
        task = getattr(self.model, 'task', 'presence')

        pbar = tqdm(loader, desc="Training")
        for vis, ac, labels in pbar:
            # labels is now (B, 4) multi-hot vector
            vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            pred_emb, goal_emb, species_logits, recon_img = self.model(ac)

            # Compute reconstruction target (denormalize visual image)
            if self.model.use_decoder:
                # Denormalize target image from [-1, 1] (roughly) to [0, 1]
                # Using ImageNet normalization constants as defined in data.py
                inv_normalize = transforms.Normalize(
                    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                    std=[1/0.229, 1/0.224, 1/0.225],
                )
                target_img = torch.stack([inv_normalize(v) for v in vis])
            else:
                target_img = None

            loss, pred_loss, sigreg_loss, loss_cls, recon_loss = self.model.compute_loss(
                pred_emb, goal_emb, species_logits, labels,
                recon_img=recon_img,
                target_img=target_img,
                sigreg_weight=self.config.sigreg_weight,
                recon_weight=0.01,
            )
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_loss_pred += pred_loss.item()
            total_loss_cls += loss_cls.item()
            total_loss_sigreg += sigreg_loss.item()
            total_loss_recon += recon_loss.item()

            # Compute task-specific metrics
            if task == "counting":
                # Counting metrics: MAE, RMSE on original scale
                # Apply same scaling as loss: tanh(x/5) * 30
                pred_counts = torch.tanh(species_logits / 5.0) * 30.0
                pred_counts = pred_counts.clamp(min=0)  # Ensure non-negative
                labels_counts = labels.clamp(min=0)
                
                mae = F.l1_loss(pred_counts, labels_counts, reduction='sum')
                mse = F.mse_loss(pred_counts, labels_counts, reduction='sum')
                
                total_mae += mae.item()
                total_rmse += mse.item()
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "mae": f"{mae.item()/labels.shape[0]:.3f}",
                })
            else:
                # Presence/absence metrics: precision, recall, F1
                probs = torch.sigmoid(species_logits)
                preds = (probs > 0.5).float()

                # Per-sample F1, then average
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

                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "f1": f"{100*total_f1/(total_samples + labels.shape[0]):.1f}%"
                })

            total_samples += labels.shape[0]

        # Per-class F1
        class_precision = class_tp / (class_tp + class_fp + 1e-8)
        class_recall = class_tp / (class_tp + class_fn + 1e-8)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)

        return {
            "loss": total_loss / len(loader),
            "loss_pred": total_loss_pred / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "loss_sigreg": total_loss_sigreg / len(loader),
            "loss_recon": total_loss_recon / len(loader),
            "precision": total_precision / total_samples,
            "recall": total_recall / total_samples,
            "f1": total_f1 / total_samples,
            "f1_kingfish": class_f1[0].item(),
            "f1_snapper": class_f1[1].item(),
            "f1_cod": class_f1[2].item(),
            "f1_empty": class_f1[3].item(),
            "mae": total_mae / total_samples if total_samples > 0 else 0,
            "rmse": torch.sqrt(torch.tensor(total_rmse / total_samples)).item() if total_samples > 0 else 0,
        }

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate LeWM model with task-specific metrics."""
        self.model.eval()
        total_loss = 0
        total_loss_pred = 0
        total_loss_cls = 0
        total_loss_sigreg = 0
        total_loss_recon = 0

        # Task-specific metrics
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_mae = 0
        total_rmse = 0
        total_samples = 0

        # Per-class metrics (for presence task)
        class_tp = torch.zeros(4)
        class_fp = torch.zeros(4)
        class_fn = torch.zeros(4)

        last_recon = None
        last_target = None

        # Get task once at the beginning
        task = getattr(self.model, 'task', 'presence')

        with torch.no_grad():
            for vis, ac, labels in loader:
                vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

                pred_emb, goal_emb, species_logits, recon_img = self.model(ac)

                # Compute reconstruction target (denormalize visual image)
                if self.model.use_decoder:
                    inv_normalize = transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225],
                    )
                    target_img = torch.stack([inv_normalize(v) for v in vis])
                else:
                    target_img = None

                loss, pred_loss, sigreg_loss, cls_loss, recon_loss = self.model.compute_loss(
                    pred_emb, goal_emb, species_logits, labels,
                    recon_img=recon_img,
                    target_img=target_img,
                    sigreg_weight=self.config.sigreg_weight,
                    recon_weight=0.01,
                )

                total_loss += loss.item()
                total_loss_pred += pred_loss.item()
                total_loss_cls += cls_loss.item()
                total_loss_sigreg += sigreg_loss.item()
                total_loss_recon += recon_loss.item()

                # Compute task-specific metrics
                if task == "counting":
                    # Counting: MAE, RMSE on original scale
                    pred_counts = torch.tanh(species_logits / 5.0) * 30.0
                    pred_counts = pred_counts.clamp(min=0)
                    labels_counts = labels.clamp(min=0)
                    
                    mae = F.l1_loss(pred_counts, labels_counts, reduction='sum')
                    mse = F.mse_loss(pred_counts, labels_counts, reduction='sum')
                    total_mae += mae.item()
                    total_rmse += mse.item()
                else:
                    # Presence/absence: precision, recall, F1
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

                # Save last reconstruction for logging
                if recon_img is not None:
                    last_recon = recon_img[0].cpu().clamp(0, 1)
                    last_target = target_img[0].cpu().clamp(0, 1) if target_img is not None else None

        # Per-class F1 (only for presence task)
        class_precision = class_tp / (class_tp + class_fp + 1e-8)
        class_recall = class_tp / (class_tp + class_fn + 1e-8)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)

        return {
            "loss": total_loss / len(loader),
            "loss_pred": total_loss_pred / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "loss_sigreg": total_loss_sigreg / len(loader),
            "loss_recon": total_loss_recon / len(loader),
            "precision": total_precision / total_samples,
            "recall": total_recall / total_samples,
            "f1": total_f1 / total_samples,
            "mae": total_mae / total_samples if total_samples > 0 else 0,
            "rmse": torch.sqrt(torch.tensor(total_rmse / total_samples)).item() if total_samples > 0 else 0,
            "f1_kingfish": class_f1[0].item(),
            "f1_snapper": class_f1[1].item(),
            "f1_cod": class_f1[2].item(),
            "f1_empty": class_f1[3].item(),
            "last_recon": last_recon,
            "last_target": last_target,
        }

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Use task-appropriate metric for model selection."""
        # For counting task, use negative MAE (higher is better)
        if "mae" in val_metrics:
            return -val_metrics["mae"]
        # For presence task, use F1 (higher is better)
        if val_metrics.get("f1", 0) > 0:
            return val_metrics["f1"]
        return val_metrics.get("acc", 0)


def get_trainer(config: TrainingConfig, device: torch.device) -> BaseTrainer:
    """Factory function to get appropriate trainer for model type."""
    if config.model_type == "lewm":
        return LeWMTrainer(config, device)
    elif config.model_type == "lewm_plus":
        return LeWMPlusTrainer(config, device)
    else:
        return JEPATrainer(config, device)


# =============================================================================
# LeWM++ Trainer (Multi-modal JEPA + SigReg)
# =============================================================================


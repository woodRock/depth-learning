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
class JEPATrainer(BaseTrainer):
    """Trainer for JEPA (Joint Embedding Predictive Architecture) models."""
    
    def build_model(self, task: str = "presence") -> nn.Module:
        """Build JEPA model with specified acoustic encoder."""
        from models.acoustic import ConvEncoder, TransformerEncoder
        from models.lstm import AcousticLSTM
        from models.ast import AcousticAST
        from models.jepa import CrossModalJEPA

        if self.config.model_type == "conv":
            ac_encoder = ConvEncoder(embed_dim=self.config.embed_dim)
        elif self.config.model_type == "lstm":
            ac_encoder = AcousticLSTM(
                input_dim=768,
                hidden_dim=256,
                num_classes=self.config.embed_dim
            )
        elif self.config.model_type == "ast":
            ac_encoder = AcousticAST(
                d_model=self.config.embed_dim,
                num_classes=self.config.embed_dim
            )
        else:  # transformer (default)
            ac_encoder = TransformerEncoder(embed_dim=self.config.embed_dim)

        return CrossModalJEPA(
            ac_encoder=ac_encoder,
            embed_dim=self.config.embed_dim,
            use_focal_loss=self.config.use_focal_loss,
            task=task,
        ).to(self.device)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train JEPA for one epoch with task-specific metrics."""
        self.model.train()
        total_loss = 0
        total_loss_jepa = 0
        total_loss_cls = 0

        # Classification metrics
        correct = 0
        total = 0

        # Presence task metrics
        total_precision = 0
        total_recall = 0
        total_f1 = 0

        # Counting task metrics
        total_mae = 0
        total_rmse = 0
        total_samples = 0

        # Per-class F1 (for presence task)
        class_tp = torch.zeros(4)
        class_fp = torch.zeros(4)
        class_fn = torch.zeros(4)

        pbar = tqdm(loader, desc="Training")
        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            predicted_target, target_latent, species_logits = self.model(vis, ac)
            loss, loss_jepa, loss_cls = self.model.compute_loss(
                predicted_target, target_latent, species_logits, labels
            )
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_loss_jepa += loss_jepa.item()
            total_loss_cls += loss_cls.item()

        # Task-specific metrics
        task = getattr(self.model, 'task', 'presence')
        
        if task == "presence":
            # Multi-label: precision, recall, F1
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

            total += labels.shape[0]

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "f1": f"{100*total_f1/total:.1f}%"
            })
        elif task == "counting":
            # Counting: MAE, RMSE
            pred_counts = species_logits.clamp(min=0)
            true_counts = labels.clamp(min=0)
            
            mae = F.l1_loss(pred_counts, true_counts)
            rmse = torch.sqrt(F.mse_loss(pred_counts, true_counts))
            
            total_mae += mae.item() * len(labels)
            total_rmse += rmse.item() * len(labels)
            total_samples += len(labels)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "mae": f"{mae.item():.3f}",
            })
        else:
            # Single-label: accuracy
            preds = torch.argmax(species_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*correct/total:.1f}%"})

        # Return metrics based on task
        if task == "presence":
            # Per-class F1
            class_precision = class_tp / (class_tp + class_fp + 1e-8)
            class_recall = class_tp / (class_tp + class_fn + 1e-8)
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)

            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "precision": total_precision / total,
                "recall": total_recall / total,
                "f1": total_f1 / total,
                "f1_kingfish": class_f1[0].item(),
                "f1_snapper": class_f1[1].item(),
                "f1_cod": class_f1[2].item(),
                "f1_empty": class_f1[3].item(),
            }
        elif task == "counting":
            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "mae": total_mae / total_samples,
                "rmse": total_rmse / total_samples,
            }
        else:
            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "acc": correct / total,
            }
    
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate JEPA model with task-specific metrics."""
        self.model.eval()
        total_loss = 0
        total_loss_jepa = 0
        total_loss_cls = 0

        # Classification metrics
        correct = 0
        total = 0
        total_sim = 0

        # Presence task metrics
        total_precision = 0
        total_recall = 0
        total_f1 = 0

        # Counting task metrics
        total_mae = 0
        total_rmse = 0
        total_samples = 0

        # Per-class F1 (for presence task)
        class_tp = torch.zeros(4)
        class_fp = torch.zeros(4)
        class_fn = torch.zeros(4)

        with torch.no_grad():
            for vis, ac, labels in loader:
                vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

                predicted_target, target_latent, species_logits = self.model(vis, ac)
                loss, loss_jepa, loss_cls = self.model.compute_loss(
                    predicted_target, target_latent, species_logits, labels
                )

                total_loss += loss.item()
                total_loss_jepa += loss_jepa.item()
                total_loss_cls += loss_cls.item()

                sim = F.cosine_similarity(predicted_target, target_latent, dim=-1).mean()
                total_sim += sim.item()

                # Task-specific metrics
                task = getattr(self.model, 'task', 'presence')
                
                if task == "presence":
                    # Multi-label: precision, recall, F1
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

                    total += labels.shape[0]
                elif task == "counting":
                    # Counting: MAE, RMSE
                    pred_counts = species_logits.clamp(min=0)
                    true_counts = labels.clamp(min=0)
                    
                    mae = F.l1_loss(pred_counts, true_counts)
                    rmse = torch.sqrt(F.mse_loss(pred_counts, true_counts))
                    
                    total_mae += mae.item() * len(labels)
                    total_rmse += rmse.item() * len(labels)
                    total_samples += len(labels)
                else:
                    # Single-label: accuracy
                    preds = torch.argmax(species_logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        # Return metrics based on task
        if task == "presence":
            # Per-class F1
            class_precision = class_tp / (class_tp + class_fp + 1e-8)
            class_recall = class_tp / (class_tp + class_fn + 1e-8)
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)

            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "sim": total_sim / len(loader),
                "precision": total_precision / total,
                "recall": total_recall / total,
                "f1": total_f1 / total,
                "f1_kingfish": class_f1[0].item(),
                "f1_snapper": class_f1[1].item(),
                "f1_cod": class_f1[2].item(),
                "f1_empty": class_f1[3].item(),
            }
        elif task == "counting":
            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "sim": total_sim / len(loader),
                "mae": total_mae / total_samples,
                "rmse": total_rmse / total_samples,
            }
        else:
            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "sim": total_sim / len(loader),
                "acc": correct / total,
            }

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Use task-appropriate metric for model selection."""
        # For counting task, use negative MAE (higher is better)
        if "mae" in val_metrics:
            return -val_metrics["mae"]
        # For presence task, use F1 (higher is better)
        if val_metrics.get("f1", 0) > 0:
            return val_metrics["f1"]
        # For single-label, use combined score
        return val_metrics.get("acc", 0) * 0.7 + val_metrics.get("sim", 0) * 0.3



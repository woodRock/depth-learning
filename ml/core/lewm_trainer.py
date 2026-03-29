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
from utils.metrics import get_task_metrics

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
        """Train LeWM for one epoch with task-specific metrics."""
        self.model.train()
        total_loss = 0
        total_loss_pred = 0
        total_loss_cls = 0
        total_loss_sigreg = 0
        total_loss_recon = 0
        
        # Accumulators for batch-wise metric calculation
        total_samples = 0
        batch_metrics_sum = {}

        # Get task once at the beginning
        task = self.task

        pbar = tqdm(loader, desc="Training")
        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            pred_emb, goal_emb, species_logits, recon_img = self.model(ac)

            # Compute reconstruction target
            if self.model.use_decoder:
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

            # Calculate task-specific metrics using unified utility
            batch_metrics = get_task_metrics(task, species_logits, labels)
            
            # Accumulate metrics (weighted by batch size)
            batch_size = len(labels)
            for key, value in batch_metrics.items():
                if key not in batch_metrics_sum:
                    batch_metrics_sum[key] = 0.0
                batch_metrics_sum[key] += value * batch_size
            
            total_samples += batch_size

            # Display metrics
            if task == "counting":
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "mae": f"{batch_metrics['mae']:.3f}"})
            else:
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "f1": f"{batch_metrics['f1'] * 100:.1f}%"})

        # Average metrics over all samples
        avg_metrics = {
            key: value / total_samples if total_samples > 0 else 0.0
            for key, value in batch_metrics_sum.items()
        }

        return {
            "loss": total_loss / len(loader),
            "loss_pred": total_loss_pred / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "loss_sigreg": total_loss_sigreg / len(loader),
            "loss_recon": total_loss_recon / len(loader),
            **avg_metrics,
        }

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate LeWM model with task-specific metrics."""
        self.model.eval()
        total_loss = 0
        total_loss_pred = 0
        total_loss_cls = 0
        total_loss_sigreg = 0
        total_loss_recon = 0
        
        # Accumulators for batch-wise metric calculation
        total_samples = 0
        batch_metrics_sum = {}
        
        last_recon = None
        last_target = None
        
        task = self.task

        with torch.no_grad():
            for vis, ac, labels in loader:
                vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

                pred_emb, goal_emb, species_logits, recon_img = self.model(ac)

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

                # Calculate task-specific metrics using unified utility
                batch_metrics = get_task_metrics(task, species_logits, labels)
                
                # Accumulate metrics (weighted by batch size)
                batch_size = len(labels)
                for key, value in batch_metrics.items():
                    if key not in batch_metrics_sum:
                        batch_metrics_sum[key] = 0.0
                    batch_metrics_sum[key] += value * batch_size
                
                total_samples += batch_size

                # Save last reconstruction for logging
                if recon_img is not None:
                    last_recon = recon_img[0].cpu().clamp(0, 1)
                    if target_img is not None:
                        last_target = target_img[0].cpu().clamp(0, 1)

        # Average metrics
        avg_metrics = {
            key: value / total_samples if total_samples > 0 else 0.0
            for key, value in batch_metrics_sum.items()
        }

        result = {
            "loss": total_loss / len(loader),
            "loss_pred": total_loss_pred / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "loss_sigreg": total_loss_sigreg / len(loader),
            "loss_recon": total_loss_recon / len(loader),
            **avg_metrics,
        }
        
        # Add reconstructions for logging
        if last_recon is not None:
            result["last_recon"] = last_recon
        if last_target is not None:
            result["last_target"] = last_target
        
        return result

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Use task-appropriate metric for model selection."""
        if self.task == "counting":
            return -val_metrics.get("mae", 0)
        return val_metrics.get("f1", 0)


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


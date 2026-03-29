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
from utils.metrics import get_task_metrics, SPECIES_NAMES

logger = get_logger(__name__)

from .base import BaseTrainer


class JEPATrainer(BaseTrainer):
    """Trainer for JEPA (Joint Embedding Predictive Architecture) models."""

    def build_model(self) -> nn.Module:
        """Build JEPA model with specified acoustic encoder."""
        from models.acoustic import ConvEncoder, TransformerEncoder
        from models.lstm import AcousticLSTM
        from models.ast import AcousticAST
        from models.jepa import CrossModalJEPA

        task = getattr(self.config, 'task', 'presence')

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

        # Accumulators for batch-wise metric calculation
        total_samples = 0
        batch_metrics_sum = {}  # Sum of metrics across batches

        # Get task once at the beginning
        task = getattr(self.model, 'task', 'presence')

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
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "mae": f"{batch_metrics['mae']:.3f}",
                })
            elif task == "presence":
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "f1": f"{batch_metrics['f1'] * 100:.1f}%",
                })
            else:
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "acc": f"{batch_metrics.get('acc', 0) * 100:.1f}%",
                })

        # Average metrics over all samples
        avg_metrics = {
            key: value / total_samples if total_samples > 0 else 0.0
            for key, value in batch_metrics_sum.items()
        }

        # Build return dictionary
        result = {
            "loss": total_loss / len(loader),
            "loss_jepa": total_loss_jepa / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            **avg_metrics,
        }

        return result

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate JEPA model with task-specific metrics."""
        self.model.eval()
        total_loss = 0
        total_loss_jepa = 0
        total_loss_cls = 0
        total_sim = 0

        # Accumulators for batch-wise metric calculation
        total_samples = 0
        batch_metrics_sum = {}

        # Get task once at the beginning
        task = getattr(self.model, 'task', 'presence')

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

                # Calculate task-specific metrics using unified utility
                batch_metrics = get_task_metrics(task, species_logits, labels)
                
                # Accumulate metrics (weighted by batch size)
                batch_size = len(labels)
                for key, value in batch_metrics.items():
                    if key not in batch_metrics_sum:
                        batch_metrics_sum[key] = 0.0
                    batch_metrics_sum[key] += value * batch_size
                
                total_samples += batch_size

        # Average metrics over all samples
        avg_metrics = {
            key: value / total_samples if total_samples > 0 else 0.0
            for key, value in batch_metrics_sum.items()
        }

        # Build return dictionary
        result = {
            "loss": total_loss / len(loader),
            "loss_jepa": total_loss_jepa / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "sim": total_sim / len(loader),
            **avg_metrics,
        }

        return result

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



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
        """Train for one epoch with task-specific metrics."""
        self.model.train()
        total_loss = 0
        total_loss_jepa = 0
        total_loss_cls = 0
        total_loss_sigreg = 0
        
        # Accumulators for batch-wise metric calculation
        total_samples = 0
        batch_metrics_sum = {}
        
        # Get task once at the beginning
        task = self.task

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

            # Calculate task-specific metrics using unified utility
            # LeWM++ uses tanh scaling (part of model architecture)
            if self.task == "counting":
                scaled_logits = torch.tanh(species_logits / 5.0) * 30.0
                batch_metrics = get_task_metrics(self.task, scaled_logits, labels)
            else:
                batch_metrics = get_task_metrics(self.task, species_logits, labels)
            
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
            "loss_jepa": total_loss_jepa / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "loss_sigreg": total_loss_sigreg / len(loader),
            **avg_metrics,
        }

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation with task-specific metrics."""
        self.model.eval()
        total_loss = 0
        total_loss_jepa = 0
        total_loss_cls = 0
        total_loss_sigreg = 0
        total_sim = 0
        
        # Accumulators for batch-wise metric calculation
        total_samples = 0
        batch_metrics_sum = {}

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

                # Calculate task-specific metrics using unified utility
                # LeWM++ uses tanh scaling (part of model architecture)
                if self.task == "counting":
                    scaled_logits = torch.tanh(species_logits / 5.0) * 30.0
                    batch_metrics = get_task_metrics(self.task, scaled_logits, labels)
                else:
                    batch_metrics = get_task_metrics(self.task, species_logits, labels)
                
                # Accumulate metrics (weighted by batch size)
                batch_size = len(labels)
                for key, value in batch_metrics.items():
                    if key not in batch_metrics_sum:
                        batch_metrics_sum[key] = 0.0
                    batch_metrics_sum[key] += value * batch_size
                
                total_samples += batch_size

        # Average metrics
        avg_metrics = {
            key: value / total_samples if total_samples > 0 else 0.0
            for key, value in batch_metrics_sum.items()
        }

        return {
            "loss": total_loss / len(loader),
            "loss_jepa": total_loss_jepa / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "loss_sigreg": total_loss_sigreg / len(loader),
            "sim": total_sim / len(loader),
            **avg_metrics,
        }

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Use task-appropriate metric for model selection."""
        if self.task == "counting":
            return -val_metrics.get("mae", 0)
        return val_metrics.get("f1", 0)


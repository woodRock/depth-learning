"""Training strategies for different model architectures."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from config import TrainingConfig
from data import create_data_loaders, create_visual_transform, AugmentationConfig


class BaseTrainer(ABC):
    """Abstract base class for all training strategies."""
    
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        pass
    
    @abstractmethod
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        pass
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Full training loop."""
        best_score = 0.0
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Logging
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Model saving
            current_score = self._get_save_score(val_metrics)
            if current_score >= best_score:
                best_score = current_score
                self._save_model(epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log metrics to wandb."""
        log_dict = {
            "epoch": epoch + 1,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        
        # Log reconstruction images if available
        if "last_recon" in val_metrics and val_metrics["last_recon"] is not None:
            import wandb
            recon_img = val_metrics["last_recon"].permute(1, 2, 0).numpy()
            log_dict["reconstruction"] = wandb.Image(recon_img, caption="Reconstructed from acoustic")
            
            if "last_target" in val_metrics and val_metrics["last_target"] is not None:
                target_img = val_metrics["last_target"].permute(1, 2, 0).numpy()
                log_dict["ground_truth"] = wandb.Image(target_img, caption="Ground truth visual")
        
        wandb.log(log_dict)
    
    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Get score used for model selection."""
        return val_metrics.get("acc", 0.0)
    
    def _save_model(self, epoch: int) -> None:
        """Save model weights."""
        os.makedirs(self.config.weights_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(), 
            os.path.join(self.config.weights_dir, "fish_clip_model.pth")
        )
        with open(os.path.join(self.config.weights_dir, "model_config.json"), "w") as f:
            json.dump({
                "model_type": self.config.model_type, 
                "config": vars(self.config)
            }, f)


class JEPATrainer(BaseTrainer):
    """Trainer for JEPA (Joint Embedding Predictive Architecture) models."""
    
    def build_model(self) -> nn.Module:
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
        ).to(self.device)
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train JEPA for one epoch."""
        self.model.train()
        total_loss = 0
        total_loss_jepa = 0
        total_loss_cls = 0
        correct = 0
        total = 0
        
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
            
            preds = torch.argmax(species_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*correct/total:.1f}%"})
        
        return {
            "loss": total_loss / len(loader),
            "loss_jepa": total_loss_jepa / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "acc": correct / total,
        }
    
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate JEPA model."""
        self.model.eval()
        total_loss = 0
        total_loss_jepa = 0
        total_loss_cls = 0
        correct = 0
        total = 0
        total_sim = 0
        
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
                
                preds = torch.argmax(species_logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return {
            "loss": total_loss / len(loader),
            "loss_jepa": total_loss_jepa / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "acc": correct / total,
            "sim": total_sim / len(loader),
        }
    
    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Combined score for model selection."""
        return val_metrics["acc"] * 0.7 + val_metrics["sim"] * 0.3


class LeWMTrainer(BaseTrainer):
    """Trainer for LeWorldModel (LeWM) architecture."""
    
    def build_model(self) -> nn.Module:
        """Build LeWorldModel."""
        from models.lewm import LeWorldModel
        
        return LeWorldModel(
            embed_dim=self.config.embed_dim,
            num_layers=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop=0.1,
            n_classes=4,
            use_classifier=True,
            use_decoder=True,  # Enable world reconstruction decoder
        ).to(self.device)
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train LeWM for one epoch."""
        self.model.train()
        total_loss = 0
        total_loss_pred = 0
        total_loss_cls = 0
        total_loss_sigreg = 0
        total_loss_recon = 0
        correct = 0
        total = 0

        pbar = tqdm(loader, desc="Training")
        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            pred_emb, goal_emb, species_logits, recon_img = self.model(ac)

            # Compute reconstruction target (denormalize visual image)
            target_img = vis if self.model.use_decoder else None

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

            preds = torch.argmax(species_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*correct/total:.1f}%"})

        return {
            "loss": total_loss / len(loader),
            "loss_pred": total_loss_pred / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "loss_sigreg": total_loss_sigreg / len(loader),
            "loss_recon": total_loss_recon / len(loader),
            "acc": correct / total,
        }

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Validate LeWM model."""
        self.model.eval()
        total_loss = 0
        total_loss_pred = 0
        total_loss_cls = 0
        total_loss_sigreg = 0
        total_loss_recon = 0
        correct = 0
        total = 0
        last_recon = None
        last_target = None

        with torch.no_grad():
            for vis, ac, labels in loader:
                vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)

                pred_emb, goal_emb, species_logits, recon_img = self.model(ac)
                target_img = vis if self.model.use_decoder else None
                
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

                preds = torch.argmax(species_logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Save last reconstruction for logging
                if recon_img is not None:
                    last_recon = recon_img[0].cpu().clamp(0, 1)
                    last_target = target_img[0].cpu().clamp(0, 1) if target_img is not None else None

        return {
            "loss": total_loss / len(loader),
            "loss_pred": total_loss_pred / len(loader),
            "loss_cls": total_loss_cls / len(loader),
            "loss_sigreg": total_loss_sigreg / len(loader),
            "loss_recon": total_loss_recon / len(loader),
            "acc": correct / total,
            "last_recon": last_recon,
            "last_target": last_target,
        }
    
    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Use accuracy for model selection."""
        return val_metrics["acc"]


def get_trainer(config: TrainingConfig, device: torch.device) -> BaseTrainer:
    """Factory function to get appropriate trainer for model type."""
    if config.model_type == "lewm":
        return LeWMTrainer(config, device)
    else:
        return JEPATrainer(config, device)

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
        
        # Save model config with task information
        model_config = {
            "model_type": self.config.model_type,
            "config": vars(self.config)
        }
        
        # Add task information for LeWM
        if hasattr(self.model, 'task'):
            model_config["task"] = self.model.task
        
        with open(os.path.join(self.config.weights_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)


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
            if hasattr(self.model, 'task') and self.model.task == "presence":
                # Multi-label: precision, recall, F1
                probs = torch.sigmoid(species_logits)
                preds = (probs > 0.5).float()

                for i in range(labels.shape[0]):
                    tp = (preds[i] * labels[i]).sum().item()
                    fp = (preds[i] * (1 - labels[i])).sum().item()
                    fn = ((1 - preds[i]) * labels[i]).sum().item()

                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1

                total += labels.shape[0]  # Accumulate total, not reset!

                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "f1": f"{100*total_f1/total:.1f}%"
                })
            else:
                # Single-label: accuracy
                preds = torch.argmax(species_logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*correct/total:.1f}%"})

        if hasattr(self.model, 'task') and self.model.task == "presence":
            return {
                "loss": total_loss / len(loader),
                "loss_jepa": total_loss_jepa / len(loader),
                "loss_cls": total_loss_cls / len(loader),
                "precision": total_precision / total,
                "recall": total_recall / total,
                "f1": total_f1 / total,
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
                if hasattr(self.model, 'task') and self.model.task == "presence":
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

                    total += labels.shape[0]  # Accumulate total, not reset!
                else:
                    # Single-label: accuracy
                    preds = torch.argmax(species_logits, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        # Per-class F1
        class_precision = class_tp / (class_tp + class_fp + 1e-8)
        class_recall = class_tp / (class_tp + class_fn + 1e-8)
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)

        if hasattr(self.model, 'task') and self.model.task == "presence":
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
        # For presence task, use F1 (higher is better)
        if val_metrics.get("f1", 0) > 0:
            return val_metrics["f1"]
        # For single-label, use combined score
        return val_metrics.get("acc", 0) * 0.7 + val_metrics.get("sim", 0) * 0.3


class LeWMTrainer(BaseTrainer):
    """Trainer for LeWorldModel with multi-label presence/absence or counting."""

    def build_model(self, task: str = "presence") -> nn.Module:
        """Build LeWorldModel with task-specific head."""
        from models.lewm_multilabel import LeWorldModelMultiLabel

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

        pbar = tqdm(loader, desc="Training")
        for vis, ac, labels in pbar:
            # labels is now (B, 4) multi-hot vector
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

            # Compute task-specific metrics
            if hasattr(self.model, 'task') and self.model.task == "counting":
                # Counting metrics: MAE, RMSE on original scale
                # Apply same scaling as loss: tanh(x/5) * 30
                pred_counts = torch.tanh(species_logits / 5.0) * 30.0
                pred_counts = pred_counts.clamp(min=0)  # Ensure non-negative
                labels_counts = labels.clamp(min=0)
                
                mae = F.l1_loss(pred_counts, labels_counts)
                rmse = torch.sqrt(F.mse_loss(pred_counts, labels_counts))
                
                total_mae += mae.item()
                total_rmse += rmse.item()
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "mae": f"{mae.item():.3f}",
                })
            else:
                # Presence/absence metrics: precision, recall, F1
                probs = torch.sigmoid(species_logits)
                preds = (probs > 0.5).float()

                # Per-sample F1, then average
                for i in range(labels.shape[0]):
                    tp = (preds[i] * labels[i]).sum().item()
                    fp = (preds[i] * (1 - labels[i])).sum().item()
                    fn = ((1 - preds[i]) * labels[i]).sum().item()

                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)

                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1

                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "f1": f"{100*total_f1/(total_samples + labels.shape[0]):.1f}%"
                })

            total_samples += labels.shape[0]

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
            "rmse": total_rmse / total_samples if total_samples > 0 else 0,
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

                # Compute task-specific metrics
                if hasattr(self.model, 'task') and self.model.task == "counting":
                    # Counting: MAE, RMSE on original scale
                    pred_counts = torch.tanh(species_logits / 5.0) * 30.0
                    pred_counts = pred_counts.clamp(min=0)
                    labels_counts = labels.clamp(min=0)
                    
                    mae = F.l1_loss(pred_counts, labels_counts)
                    rmse = torch.sqrt(F.mse_loss(pred_counts, labels_counts))
                    total_mae += mae.item()
                    total_rmse += rmse.item()
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
            "rmse": total_rmse / total_samples if total_samples > 0 else 0,
            "f1_kingfish": class_f1[0].item(),
            "f1_snapper": class_f1[1].item(),
            "f1_cod": class_f1[2].item(),
            "f1_empty": class_f1[3].item(),
            "last_recon": last_recon,
            "last_target": last_target,
        }

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Use task-appropriate metric for model selection."""
        # For counting task, use negative MAE (lower is better)
        if val_metrics.get("mae", 0) > 0 and val_metrics.get("f1", 0) == 0:
            return -val_metrics["mae"]  # Negative because lower MAE is better
        # For presence task, use F1 (higher is better)
        return val_metrics["f1"]


def get_trainer(config: TrainingConfig, device: torch.device) -> BaseTrainer:
    """Factory function to get appropriate trainer for model type."""
    if config.model_type == "lewm":
        return LeWMTrainer(config, device)
    else:
        return JEPATrainer(config, device)

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
from data import create_visual_transform, AugmentationConfig
from utils.logging import get_logger

logger = get_logger(__name__)

class BaseTrainer(ABC):
    """Abstract base class for all training strategies."""

    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        # Determine task from config
        self.task = getattr(config, 'task', 'presence')

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
        """Full training loop with early stopping."""
        best_score = 0.0
        best_metrics = None
        best_multi_modal = None
        best_epoch = 0
        patience = getattr(self.config, 'early_stop_patience', 15)  # Stop if no improvement for N epochs
        min_delta = getattr(self.config, 'early_stop_min_delta', 0.001)  # Minimum improvement to count as progress
        epochs_without_improvement = 0

        print(f"\nTraining for up to {self.config.epochs} epochs (early stopping patience={patience}, min_delta={min_delta})")

        for epoch in range(self.config.epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader)

            # Validation phase - get multi-modal metrics (if model supports it)
            val_metrics = self.validate(val_loader)
            
            # Evaluate acoustic-only performance (PRIMARY METRICS)
            acoustic_metrics = None
            if hasattr(self, '_evaluate_acoustic_only'):
                acoustic_metrics = self._evaluate_acoustic_only(val_loader)
            
            # Determine primary metrics (acoustic-only if available, otherwise multi-modal)
            if acoustic_metrics:
                # Model supports both - acoustic is primary, multi-modal is secondary
                primary_metrics = acoustic_metrics
                multi_modal_metrics = val_metrics
            else:
                # Acoustic-only model - use multi-modal as primary
                primary_metrics = val_metrics
                multi_modal_metrics = None
            
            # Log to wandb with unified naming
            self._log_metrics(epoch, train_metrics, primary_metrics, multi_modal_metrics)
            
            # Model selection uses primary metrics (acoustic-only when available)
            current_score = self._get_save_score(primary_metrics)
            
            # Check for perfect score FIRST (stop immediately)
            if current_score >= 0.9999:  # Effectively 1.0 with floating point tolerance
                best_score = current_score
                best_metrics = {"train": train_metrics, "val": primary_metrics}
                best_multi_modal = {"train": train_metrics, "val": multi_modal_metrics} if multi_modal_metrics else None
                best_epoch = epoch
                self._save_model(epoch)
                print(f"  Epoch {epoch+1}: PERFECT SCORE! Score={best_score:.4f}")
                print(f"\n⏹ Perfect validation accuracy achieved! Stopping immediately.")
                break

            # Check for improvement
            improvement = current_score - best_score

            if improvement > min_delta:
                # Significant improvement
                best_score = current_score
                best_metrics = {"train": train_metrics, "val": primary_metrics}
                best_multi_modal = {"train": train_metrics, "val": multi_modal_metrics} if multi_modal_metrics else None
                best_epoch = epoch
                epochs_without_improvement = 0  # Reset counter
                self._save_model(epoch)
                print(f"  Epoch {epoch+1}: New best! Score={best_score:.4f} (improved by {improvement:.4f})")
            elif improvement > 0:
                # Very small improvement (within min_delta)
                best_metrics = {"train": train_metrics, "val": primary_metrics}
                best_multi_modal = {"train": train_metrics, "val": multi_modal_metrics} if multi_modal_metrics else None
                epochs_without_improvement = 0  # Reset counter
                self._save_model(epoch)
                print(f"  Epoch {epoch+1}: Small improvement. Score={current_score:.4f}")
            else:
                # No improvement
                epochs_without_improvement += 1
                print(f"  Epoch {epoch+1}: No improvement ({epochs_without_improvement}/{patience})")

                if epochs_without_improvement >= patience:
                    print(f"\n⏹ Early stopping at epoch {epoch+1}")
                    print(f"  Best epoch: {best_epoch + 1} (score={best_score:.4f})")
                    break

            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step()

        # Save final results to results.json
        if best_metrics:
            print(f"\n✓ Training complete. Best validation score: {best_score:.4f} at epoch {best_epoch + 1}")
            # Save results with both primary and multi-modal metrics
            self._record_final_results(best_metrics, best_multi_modal)

    def _record_final_results(self, metrics: Dict[str, Any], multi_modal_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Append the best metrics to results.json with unified naming.
        
        Args:
            metrics: Dictionary with 'train' and 'val' keys containing PRIMARY metrics (acoustic-only)
            multi_modal_metrics: Multi-modal metrics (if applicable)
        """
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "results.json")

        # Determine task type from metrics keys
        is_counting = "mae" in metrics["train"]

        # Prepare new entry
        architecture = getattr(self.config, 'architecture', 'jepa')
        if architecture == "lewm_plus":
            display_arch = "LeWM++"
        elif architecture == "lewm":
            display_arch = "LeWM"
        else:
            display_arch = architecture.upper()

        # Determine mode
        if architecture in ["jepa", "lewm_plus", "fusion"]:
            mode = "multi-modal"
        elif architecture in ["lewm", "mae"]:
            mode = "acoustic-only"
        elif architecture == "translator":
            mode = "cross-modal"
        else:
            mode = "visual"

        entry = {
            "architecture": display_arch,
            "model_type": getattr(self.config, 'model_type', 'default'),
            "dataset": self.config.dataset,
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": mode,
            "task": "counting" if is_counting else "presence",
            "train": {},  # PRIMARY (acoustic-only)
            "val": {},    # PRIMARY (acoustic-only)
            "multi_train": None,  # Multi-modal (if applicable)
            "multi_val": None,    # Multi-modal (if applicable)
            "test": None  # To be filled by simulation evaluation
        }

        if is_counting:
            # Counting task: save MAE/RMSE (unified format)
            entry["train"] = {
                "loss": metrics["train"].get("loss", 0),
                "mae": metrics["train"].get("mae", 0),
                "rmse": metrics["train"].get("rmse", 0),
                "kingfish_mae": metrics["train"].get("kingfish_mae", 0),
                "snapper_mae": metrics["train"].get("snapper_mae", 0),
                "cod_mae": metrics["train"].get("cod_mae", 0),
                "empty_mae": metrics["train"].get("empty_mae", 0),
            }
            entry["val"] = {
                "loss": metrics["val"].get("loss", 0),
                "mae": metrics["val"].get("mae", 0),
                "rmse": metrics["val"].get("rmse", 0),
                "kingfish_mae": metrics["val"].get("kingfish_mae", 0),
                "snapper_mae": metrics["val"].get("snapper_mae", 0),
                "cod_mae": metrics["val"].get("cod_mae", 0),
                "empty_mae": metrics["val"].get("empty_mae", 0),
            }
        else:
            # Presence task: save F1 scores (unified format)
            entry["train"] = {
                "loss": metrics["train"].get("loss", 0),
                "f1": metrics["train"].get("f1", 0),
                "precision": metrics["train"].get("precision", 0),
                "recall": metrics["train"].get("recall", 0),
                "kingfish_f1": metrics["train"].get("kingfish_f1", 0),
                "snapper_f1": metrics["train"].get("snapper_f1", 0),
                "cod_f1": metrics["train"].get("cod_f1", 0),
                "empty_f1": metrics["train"].get("empty_f1", 0),
            }
            entry["val"] = {
                "loss": metrics["val"].get("loss", 0),
                "f1": metrics["val"].get("f1", 0),
                "precision": metrics["val"].get("precision", 0),
                "recall": metrics["val"].get("recall", 0),
                "kingfish_f1": metrics["val"].get("kingfish_f1", 0),
                "snapper_f1": metrics["val"].get("snapper_f1", 0),
                "cod_f1": metrics["val"].get("cod_f1", 0),
                "empty_f1": metrics["val"].get("empty_f1", 0),
            }
        
        # Add multi-modal metrics if available (for models that support both)
        if multi_modal_metrics:
            if is_counting:
                entry["multi_train"] = {
                    "loss": multi_modal_metrics["train"].get("loss", 0),
                    "mae": multi_modal_metrics["train"].get("mae", 0),
                    "rmse": multi_modal_metrics["train"].get("rmse", 0),
                    "kingfish_mae": multi_modal_metrics["train"].get("kingfish_mae", 0),
                    "snapper_mae": multi_modal_metrics["train"].get("snapper_mae", 0),
                    "cod_mae": multi_modal_metrics["train"].get("cod_mae", 0),
                    "empty_mae": multi_modal_metrics["train"].get("empty_mae", 0),
                }
                entry["multi_val"] = {
                    "loss": multi_modal_metrics["val"].get("loss", 0),
                    "mae": multi_modal_metrics["val"].get("mae", 0),
                    "rmse": multi_modal_metrics["val"].get("rmse", 0),
                    "kingfish_mae": multi_modal_metrics["val"].get("kingfish_mae", 0),
                    "snapper_mae": multi_modal_metrics["val"].get("snapper_mae", 0),
                    "cod_mae": multi_modal_metrics["val"].get("cod_mae", 0),
                    "empty_mae": multi_modal_metrics["val"].get("empty_mae", 0),
                }
            else:
                entry["multi_train"] = {
                    "loss": multi_modal_metrics["train"].get("loss", 0),
                    "f1": multi_modal_metrics["train"].get("f1", 0),
                    "precision": multi_modal_metrics["train"].get("precision", 0),
                    "recall": multi_modal_metrics["train"].get("recall", 0),
                    "kingfish_f1": multi_modal_metrics["train"].get("kingfish_f1", 0),
                    "snapper_f1": multi_modal_metrics["train"].get("snapper_f1", 0),
                    "cod_f1": multi_modal_metrics["train"].get("cod_f1", 0),
                    "empty_f1": multi_modal_metrics["train"].get("empty_f1", 0),
                }
                entry["multi_val"] = {
                    "loss": multi_modal_metrics["val"].get("loss", 0),
                    "f1": multi_modal_metrics["val"].get("f1", 0),
                    "precision": multi_modal_metrics["val"].get("precision", 0),
                    "recall": multi_modal_metrics["val"].get("recall", 0),
                    "kingfish_f1": multi_modal_metrics["val"].get("kingfish_f1", 0),
                    "snapper_f1": multi_modal_metrics["val"].get("snapper_f1", 0),
                    "cod_f1": multi_modal_metrics["val"].get("cod_f1", 0),
                    "empty_f1": multi_modal_metrics["val"].get("empty_f1", 0),
                }

        # Load existing results
        results = []
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    results = json.load(f)
            except:
                pass

        results.append(entry)

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        multi_modal_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log metrics to wandb with unified naming.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics (PRIMARY - acoustic-only)
            val_metrics: Validation metrics (PRIMARY - acoustic-only)
            multi_modal_metrics: Multi-modal metrics (if applicable)
        """
        log_dict = {
            "epoch": epoch + 1,
            # Primary metrics use standard naming (train_*, val_*)
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        
        # Multi-modal metrics use multi_* prefix (for models that support both)
        if multi_modal_metrics:
            log_dict.update({
                f"multi_train_{k}": v for k, v in multi_modal_metrics.items()
            })
            log_dict.update({
                f"multi_val_{k}": v for k, v in multi_modal_metrics.items()
            })

        # Log reconstruction images if available
        if "last_recon" in val_metrics and val_metrics["last_recon"] is not None:
            recon_img = val_metrics["last_recon"].permute(1, 2, 0).numpy()
            log_dict["reconstruction"] = wandb.Image(recon_img, caption="Reconstructed from acoustic")

            if "last_target" in val_metrics and val_metrics["last_target"] is not None:
                target_img = val_metrics["last_target"].permute(1, 2, 0).numpy()
                log_dict["ground_truth"] = wandb.Image(target_img, caption="Ground truth visual")

        wandb.log(log_dict)

    def _evaluate_acoustic_only(self, loader: DataLoader) -> Optional[Dict[str, float]]:
        """
        Evaluate model in acoustic-only mode (no visual input).
        Override in subclasses that support acoustic-only evaluation.
        Returns None if model doesn't support acoustic-only evaluation.
        """
        return None

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        """Get score used for model selection.

        For counting tasks: returns negative MAE (lower MAE = higher score = better)
        For presence tasks: returns F1 score (higher = better)
        """
        # Use task attribute to determine which metric to use
        if self.task == "counting":
            return -val_metrics.get("mae", 0)
        else:  # presence task
            return val_metrics.get("f1", 0)
    
    def _save_model(self, epoch: int) -> None:
        """Save model weights."""
        os.makedirs(self.config.weights_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.weights_dir, "fish_clip_model.pth")
        )
        
        # Save model config with task information
        model_config = {
            "model_type": getattr(self.config, 'model_type', 'default'),
            "config": vars(self.config)
        }

        # Add task information for models that have it
        if hasattr(self.model, 'task'):
            model_config["task"] = self.model.task
        
        with open(os.path.join(self.config.weights_dir, "model_config.json"), "w") as f:
            json.dump(model_config, f, indent=2)



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
        best_epoch = 0
        patience = getattr(self.config, 'early_stop_patience', 15)  # Stop if no improvement for N epochs
        min_delta = getattr(self.config, 'early_stop_min_delta', 0.001)  # Minimum improvement to count as progress
        epochs_without_improvement = 0

        print(f"\nTraining for up to {self.config.epochs} epochs (early stopping patience={patience}, min_delta={min_delta})")

        for epoch in range(self.config.epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader)

            # Validation phase
            val_metrics = self.validate(val_loader)

            # Logging
            self._log_metrics(epoch, train_metrics, val_metrics)

            # Model saving and early stopping check
            current_score = self._get_save_score(val_metrics)
            
            # Check for perfect score FIRST (stop immediately)
            if current_score >= 0.9999:  # Effectively 1.0 with floating point tolerance
                best_score = current_score
                best_metrics = {"train": train_metrics, "val": val_metrics}
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
                best_metrics = {"train": train_metrics, "val": val_metrics}
                best_epoch = epoch
                epochs_without_improvement = 0  # Reset counter
                self._save_model(epoch)
                print(f"  Epoch {epoch+1}: New best! Score={best_score:.4f} (improved by {improvement:.4f})")
            elif improvement > 0:
                # Very small improvement (within min_delta)
                best_metrics = {"train": train_metrics, "val": val_metrics}
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
            self._record_final_results(best_metrics)
            # For JEPA, also record acoustic-only evaluation
            if self.config.model_type != "lewm":
                self._record_acoustic_only_results(best_metrics)

    def _record_final_results(self, metrics: Dict[str, Any]) -> None:
        """Append the best metrics to results.json (multi-modal for JEPA)."""
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
            "train": {},
            "val": {},
            "test": None  # To be filled by simulation evaluation
        }
        
        if is_counting:
            # Counting task: save MAE/RMSE
            entry["train"] = {
                "loss": metrics["train"].get("loss", 0),
                "mae": metrics["train"].get("mae", 0),
                "rmse": metrics["train"].get("rmse", 0),
            }
            entry["val"] = {
                "loss": metrics["val"].get("loss", 0),
                "mae": metrics["val"].get("mae", 0),
                "rmse": metrics["val"].get("rmse", 0),
                "sim": metrics["val"].get("sim", 0),
            }
        else:
            # Presence/single_label task: save F1 scores
            entry["train"] = {
                "kingfish_f1": metrics["train"].get("f1_kingfish", 0),
                "snapper_f1": metrics["train"].get("f1_snapper", 0),
                "cod_f1": metrics["train"].get("f1_cod", 0),
                "empty_f1": metrics["train"].get("f1_empty", 0),
                "avg_f1": metrics["train"].get("f1", 0),
            }
            entry["val"] = {
                "kingfish_f1": metrics["val"].get("f1_kingfish", 0),
                "snapper_f1": metrics["val"].get("f1_snapper", 0),
                "cod_f1": metrics["val"].get("f1_cod", 0),
                "empty_f1": metrics["val"].get("f1_empty", 0),
                "avg_f1": metrics["val"].get("f1", 0),
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

    def _record_acoustic_only_results(self, metrics: Dict[str, Any]) -> None:
        """Append acoustic-only evaluation entry for JEPA to results.json."""
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "results.json")

        print("\nEvaluating acoustic-only performance...")

        # Import here to avoid circular imports
        from data import FishDataset, create_stratified_split, create_visual_transform, AugmentationConfig
        from torch.utils.data import Subset
        import torch
        import torch.nn.functional as F

        device = self.device  # Use the same device as training
        
        # Determine task type
        task = getattr(self.model, 'task', 'presence')
        is_counting = (task == "counting")

        # Create evaluation transform (no augmentation)
        eval_transform = create_visual_transform(AugmentationConfig(enabled=False))
        
        # Correctly resolve project root dataset path
        ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_root = os.path.dirname(ml_dir)
        dataset_path = os.path.join(project_root, "dataset", self.config.dataset)

        # Create dataset and split (EXACTLY as training)
        # Use mode="train" to ensure balancing logic is applied if it was for training
        seed = getattr(self.config, 'seed', 42)
        full_dataset = FishDataset(dataset_path, transform=eval_transform, mode="train", multi_label=True, task=task, seed=seed)
        
        if len(full_dataset) == 0:
            print(f"  Warning: No samples found in {dataset_path}. Skipping acoustic-only evaluation.")
            return

        # Use the same stratified split logic with the same seed
        train_indices, val_indices = create_stratified_split(full_dataset)

        # Evaluate on train split (acoustic-only)
        train_ds = Subset(full_dataset, train_indices)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=False)

        # Evaluate on val split (acoustic-only)
        val_ds = Subset(full_dataset, val_indices)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        # Evaluate acoustic-only
        # Load the BEST model weights saved during training
        best_weights_path = os.path.join(self.config.weights_dir, "fish_clip_model.pth")
        if os.path.exists(best_weights_path):
            self.model.load_state_dict(torch.load(best_weights_path, map_location=device, weights_only=True))
        
        self.model.eval()

        if is_counting:
            # Counting task: evaluate MAE/RMSE
            def evaluate_acoustic(loader):
                total_mae = 0
                total_rmse = 0
                total_samples = 0

                with torch.no_grad():
                    for _, ac, labels in loader:
                        ac, labels = ac.to(device), labels.to(device)
                        _, species_logits = self.model.forward_ac_to_vis_latent(ac)

                        pred_counts = species_logits.clamp(min=0)
                        true_counts = labels.clamp(min=0)

                        mae = F.l1_loss(pred_counts, true_counts)
                        rmse = torch.sqrt(F.mse_loss(pred_counts, true_counts))

                        total_mae += mae.item() * len(labels)
                        total_rmse += rmse.item() * len(labels)
                        total_samples += len(labels)

                return {
                    "loss": 0,
                    "mae": total_mae / total_samples if total_samples > 0 else 0,
                    "rmse": total_rmse / total_samples if total_samples > 0 else 0,
                }

            acoustic_train_metrics = evaluate_acoustic(train_loader)
            acoustic_val_metrics = evaluate_acoustic(val_loader)

            print(f"  Acoustic-only Train MAE: {acoustic_train_metrics['mae']:.3f}")
            print(f"  Acoustic-only Val MAE: {acoustic_val_metrics['mae']:.3f}")
        else:
            # Presence task: evaluate F1 scores
            def evaluate_acoustic(loader):
                class_tp = torch.zeros(4)
                class_fp = torch.zeros(4)
                class_fn = torch.zeros(4)

                with torch.no_grad():
                    for _, ac, labels in loader:
                        ac, labels = ac.to(device), labels.to(device)
                        _, species_logits = self.model.forward_ac_to_vis_latent(ac)

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

                class_precision = class_tp / (class_tp + class_fp + 1e-8)
                class_recall = class_tp / (class_tp + class_fn + 1e-8)
                class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)

                return {
                    "kingfish_f1": class_f1[0].item(),
                    "snapper_f1": class_f1[1].item(),
                    "cod_f1": class_f1[2].item(),
                    "empty_f1": class_f1[3].item(),
                    "avg_f1": class_f1.mean().item(),
                }

            acoustic_train_metrics = evaluate_acoustic(train_loader)
            acoustic_val_metrics = evaluate_acoustic(val_loader)

            print(f"  Acoustic-only Train Macro F1: {acoustic_train_metrics['avg_f1']*100:.1f}%")
            print(f"  Acoustic-only Val Macro F1: {acoustic_val_metrics['avg_f1']*100:.1f}%")

        # Prepare acoustic-only entry
        entry = {
            "architecture": "JEPA",
            "model_type": self.config.model_type,
            "dataset": self.config.dataset,
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": "acoustic_only",
            "task": "counting" if is_counting else "presence",
            "train": acoustic_train_metrics,
            "val": acoustic_val_metrics,
            "test": None  # To be filled by simulation evaluation
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
        print(f"Acoustic-only results saved to {results_path}")

        # Log acoustic-only metrics to wandb
        wandb.log({
            "acoustic_only/train_mae": acoustic_train_metrics.get("mae", 0) if is_counting else acoustic_train_metrics.get("avg_f1", 0),
            "acoustic_only/val_mae": acoustic_val_metrics.get("mae", 0) if is_counting else acoustic_val_metrics.get("avg_f1", 0),
            "acoustic_only/train_rmse": acoustic_train_metrics.get("rmse", 0) if is_counting else 0,
            "acoustic_only/val_rmse": acoustic_val_metrics.get("rmse", 0) if is_counting else 0,
            "acoustic_only/train_f1": 0 if is_counting else acoustic_train_metrics.get("avg_f1", 0),
            "acoustic_only/val_f1": 0 if is_counting else acoustic_val_metrics.get("avg_f1", 0),
        })

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
        """Get score used for model selection.
        
        For counting tasks: returns negative MAE (lower MAE = higher score = better)
        For presence tasks: returns F1 score (higher = better)
        For single-label: returns accuracy (higher = better)
        """
        # Counting task: minimize MAE, so negate it (higher is better)
        if "mae" in val_metrics:
            return -val_metrics["mae"]
        # Presence task: maximize F1
        if "f1" in val_metrics and val_metrics["f1"] > 0:
            return val_metrics["f1"]
        # Single-label classification: maximize accuracy
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



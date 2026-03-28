"""Metric computation utilities for depth learning."""

import torch
import torch.nn.functional as F
from typing import Tuple


def compute_f1(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    Compute per-class F1 scores for multi-label classification.
    
    Args:
        predictions: Model predictions (probabilities or logits)
        targets: Ground truth labels (multi-hot encoded)
        threshold: Threshold for converting probabilities to binary predictions
    
    Returns:
        Tuple of (kingfish_f1, snapper_f1, cod_f1, empty_f1)
    """
    if predictions.dim() == 2 and targets.dim() == 2:
        # Multi-label case
        probs = torch.sigmoid(predictions) if predictions.min() < 0 else predictions
        preds = (probs > threshold).float()
        
        class_f1 = []
        for c in range(4):
            tp = (preds[:, c] * targets[:, c]).sum().item()
            fp = (preds[:, c] * (1 - targets[:, c])).sum().item()
            fn = ((1 - preds[:, c]) * targets[:, c]).sum().item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            class_f1.append(f1)
        
        return tuple(class_f1)
    
    raise ValueError(f"Invalid input shapes: {predictions.shape}, {targets.shape}")


def compute_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float, float, float]:
    """
    Compute per-class MAE for counting task.
    
    Args:
        predictions: Predicted counts
        targets: Ground truth counts
    
    Returns:
        Tuple of (kingfish_mae, snapper_mae, cod_mae, empty_mae)
    """
    predictions = predictions.clamp(min=0)
    targets = targets.clamp(min=0)
    
    class_mae = []
    for c in range(4):
        mae = F.l1_loss(predictions[:, c], targets[:, c]).item()
        class_mae.append(mae)
    
    return tuple(class_mae)


def compute_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float, float, float]:
    """
    Compute per-class RMSE for counting task.
    
    Args:
        predictions: Predicted counts
        targets: Ground truth counts
    
    Returns:
        Tuple of (kingfish_rmse, snapper_rmse, cod_rmse, empty_rmse)
    """
    predictions = predictions.clamp(min=0)
    targets = targets.clamp(min=0)
    
    class_rmse = []
    for c in range(4):
        mse = F.mse_loss(predictions[:, c], targets[:, c]).item()
        rmse = mse ** 0.5
        class_rmse.append(rmse)
    
    return tuple(class_rmse)


def compute_macro_average(metrics: Tuple[float, float, float, float]) -> float:
    """
    Compute macro average of per-class metrics.
    
    Args:
        metrics: Tuple of 4 per-class metric values
    
    Returns:
        Macro average
    """
    return sum(metrics) / len(metrics)

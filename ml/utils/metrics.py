"""Unified metric calculation for depth learning tasks.

This module provides consistent metric calculation for:
- Counting task: MAE, RMSE (overall and per-species)
- Presence task: Precision, Recall, F1 (overall and per-species)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


SPECIES_NAMES = ["kingfish", "snapper", "cod", "empty"]
NUM_CLASSES = 4


def calculate_counting_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    clamp_range: Tuple[float, float] = (0, 30),
    scale_method: str = "tanh"  # "tanh" or "clamp"
) -> Dict[str, float]:
    """
    Calculate counting task metrics: MAE and RMSE.

    Args:
        predictions: Model output logits (B, 4)
        targets: Ground truth counts (B, 4)
        clamp_range: Range to clamp predictions (min, max)
        scale_method: How to scale logits to counts ("tanh" or "clamp")

    Returns:
        Dictionary with overall and per-species metrics
    """
    # Scale predictions to valid range
    if scale_method == "tanh":
        # Tanh scaling: maps logits to [0, max_count]
        pred_counts = torch.tanh(predictions / 5.0) * clamp_range[1]
    else:
        # Just clamp to valid range
        pred_counts = predictions.clamp(min=clamp_range[0], max=clamp_range[1])
    
    true_counts = targets.clamp(min=clamp_range[0], max=clamp_range[1])

    # Overall metrics
    mae = F.l1_loss(pred_counts, true_counts).item()
    mse = F.mse_loss(pred_counts, true_counts).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    # Per-species metrics
    per_species_mae = {}
    per_species_rmse = {}
    
    for i, name in enumerate(SPECIES_NAMES):
        species_mae = F.l1_loss(pred_counts[:, i], true_counts[:, i]).item()
        species_mse = F.mse_loss(pred_counts[:, i], true_counts[:, i]).item()
        per_species_mae[f"{name}_mae"] = species_mae
        per_species_rmse[f"{name}_rmse"] = torch.sqrt(torch.tensor(species_mse)).item()
    
    return {
        "mae": mae,
        "rmse": rmse,
        **per_species_mae,
        **per_species_rmse,
    }


def calculate_presence_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate presence/absence task metrics: Precision, Recall, F1.
    
    Args:
        logits: Model output logits (B, 4)
        targets: Ground truth multi-hot (B, 4)
        threshold: Sigmoid threshold for positive prediction
    
    Returns:
        Dictionary with overall and per-species metrics
    """
    # Convert logits to predictions
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    # Initialize confusion matrices
    class_tp = torch.zeros(NUM_CLASSES)
    class_fp = torch.zeros(NUM_CLASSES)
    class_fn = torch.zeros(NUM_CLASSES)
    
    # Calculate per-sample metrics and accumulate confusion matrix
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_samples = len(logits)
    
    for i in range(num_samples):
        tp = preds[i] * targets[i]
        fp = preds[i] * (1 - targets[i])
        fn = (1 - preds[i]) * targets[i]
        
        for c in range(NUM_CLASSES):
            class_tp[c] += tp[c].item()
            class_fp[c] += fp[c].item()
            class_fn[c] += fn[c].item()
        
        # Per-sample macro metrics
        tp_sum = tp.sum().item()
        fp_sum = fp.sum().item()
        fn_sum = fn.sum().item()
        
        precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        recall = tp_sum / (tp_sum + fn_sum + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    # Calculate per-class metrics
    class_precision = class_tp / (class_tp + class_fp + 1e-8)
    class_recall = class_tp / (class_tp + class_fn + 1e-8)
    class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)
    
    # Build result dictionary
    metrics = {
        "precision": total_precision / num_samples,
        "recall": total_recall / num_samples,
        "f1": total_f1 / num_samples,
    }
    
    for i, name in enumerate(SPECIES_NAMES):
        metrics[f"{name}_precision"] = class_precision[i].item()
        metrics[f"{name}_recall"] = class_recall[i].item()
        metrics[f"{name}_f1"] = class_f1[i].item()
    
    return metrics


def calculate_single_label_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate single-label classification metrics: Accuracy.
    
    Args:
        logits: Model output logits (B, 4)
        targets: Ground truth class indices (B,)
    
    Returns:
        Dictionary with accuracy
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = len(targets)
    
    return {
        "acc": correct / total if total > 0 else 0.0,
    }


def apply_counting_transform(
    logits: torch.Tensor,
    method: str = "clamp",
    scale_factor: float = 5.0,
    max_count: float = 30.0
) -> torch.Tensor:
    """
    Transform logits to count predictions.
    
    Args:
        logits: Raw model output
        method: Transformation method ("clamp", "tanh", "relu")
        scale_factor: Scaling factor for tanh transformation
        max_count: Maximum count value
    
    Returns:
        Transformed count predictions
    """
    if method == "tanh":
        # Tanh scaling: maps to [-max_count, max_count]
        return torch.tanh(logits / scale_factor) * max_count
    elif method == "relu":
        # ReLU: maps to [0, inf)
        return torch.relu(logits)
    else:  # clamp (default)
        # Simple clamping to [0, max_count]
        return logits.clamp(min=0, max=max_count)


def get_task_metrics(
    task: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    **kwargs
) -> Dict[str, float]:
    """
    Get appropriate metrics for the given task.

    Args:
        task: Task type ("counting", "presence", "single_label")
        logits: Model output
        targets: Ground truth
        **kwargs: Additional arguments for specific metrics

    Returns:
        Dictionary of metrics (only relevant for the specified task)
    """
    if task == "counting":
        clamp_range = kwargs.get("clamp_range", (0, 30))
        scale_method = kwargs.get("scale_method", "tanh")  # Default to tanh scaling
        scale_factor = kwargs.get("scale_factor", 5.0)
        max_count = kwargs.get("max_count", 30.0)

        # Apply tanh scaling to convert logits to counts [0, max_count]
        # This matches the original LeWM implementation
        if scale_method == "tanh":
            # Tanh maps to [-1, 1], scale to [-max, max], then clamp to [0, max]
            scaled_logits = torch.tanh(logits / scale_factor) * max_count
            scaled_logits = scaled_logits.clamp(min=0, max=max_count)
            return calculate_counting_metrics(scaled_logits, targets, clamp_range, scale_method="clamp")
        else:
            return calculate_counting_metrics(logits, targets, clamp_range, scale_method="clamp")

    elif task == "presence":
        threshold = kwargs.get("threshold", 0.5)
        return calculate_presence_metrics(logits, targets, threshold)

    elif task == "single_label":
        return calculate_single_label_metrics(logits, targets)

    else:
        raise ValueError(f"Unknown task: {task}")


def format_metrics_for_display(
    metrics: Dict[str, float],
    task: str
) -> Dict[str, str]:
    """
    Format metrics for console display with appropriate precision.
    
    Args:
        metrics: Raw metrics dictionary
        task: Task type
    
    Returns:
        Formatted metrics for display
    """
    formatted = {}
    
    if task == "counting":
        # MAE/RMSE: 3 decimal places
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted[key] = f"{value:.3f}"
    
    elif task == "presence":
        # F1/Precision/Recall: percentage with 1 decimal
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ["precision", "recall", "f1"]:
                    formatted[key] = f"{value * 100:.1f}%"
                else:
                    formatted[key] = f"{value:.3f}"
    
    elif task == "single_label":
        # Accuracy: percentage with 1 decimal
        for key, value in metrics.items():
            if isinstance(value, float):
                if key == "acc":
                    formatted[key] = f"{value * 100:.1f}%"
                else:
                    formatted[key] = f"{value:.3f}"
    
    return formatted

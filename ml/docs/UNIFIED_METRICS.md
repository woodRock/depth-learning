# Unified Metrics System for Depth Learning

This document describes the unified metrics reporting system implemented across all models in the Depth Learning project.

## Overview

All models now report **consistent metrics** for each task type, ensuring fair comparison and standardized evaluation across different architectures.

## Task Types and Metrics

### 1. Counting Task

**Goal:** Predict the number of fish of each species in the echogram.

**Metrics Reported:**

| Metric | Description | Format |
|--------|-------------|--------|
| `mae` | Mean Absolute Error (overall) | float (lower is better) |
| `rmse` | Root Mean Square Error (overall) | float (lower is better) |
| `kingfish_mae` | MAE for Kingfish species | float |
| `snapper_mae` | MAE for Snapper species | float |
| `cod_mae` | MAE for Cod species | float |
| `empty_mae` | MAE for Empty class | float |

**Console Display:**
```
Training: loss=0.123, mae=0.375
Validation: loss=0.145, mae=0.412
```

**WandB Logging:**
- `train_mae`, `train_rmse`
- `val_mae`, `val_rmse`
- `val_kingfish_mae`, `val_snapper_mae`, `val_cod_mae`, `val_empty_mae`

**Results.json Entry:**
```json
{
  "task": "counting",
  "train": {
    "mae": 0.375,
    "rmse": 0.612,
    "kingfish_mae": 0.500,
    "snapper_mae": 0.500,
    "cod_mae": 0.250,
    "empty_mae": 0.100
  },
  "val": { ... }
}
```

### 2. Presence Task

**Goal:** Detect which species are present in the echogram (multi-label classification).

**Metrics Reported:**

| Metric | Description | Format |
|--------|-------------|--------|
| `f1` | Macro F1 Score (overall) | float 0-1 (higher is better) |
| `precision` | Macro Precision | float 0-1 |
| `recall` | Macro Recall | float 0-1 |
| `kingfish_f1` | F1 for Kingfish species | float 0-1 |
| `snapper_f1` | F1 for Snapper species | float 0-1 |
| `cod_f1` | F1 for Cod species | float 0-1 |
| `empty_f1` | F1 for Empty class | float 0-1 |

**Console Display:**
```
Training: loss=0.456, f1=85.2%
Validation: loss=0.512, f1=82.1%
```

**WandB Logging:**
- `train_f1`, `train_precision`, `train_recall`
- `val_f1`, `val_precision`, `val_recall`
- `val_kingfish_f1`, `val_snapper_f1`, `val_cod_mae`, `val_empty_f1`

**Results.json Entry:**
```json
{
  "task": "presence",
  "train": {
    "f1": 0.852,
    "precision": 0.870,
    "recall": 0.835,
    "kingfish_f1": 0.880,
    "snapper_f1": 0.850,
    "cod_f1": 0.820,
    "empty_f1": 0.910
  },
  "val": { ... }
}
```

## Models Using Unified Metrics

All models now use the unified metrics system:

| Model | Architecture | Counting | Presence |
|-------|-------------|----------|----------|
| **JEPA** | Cross-Modal Joint Embedding | ✅ | ✅ |
| **LeWM** | LeWorldModel | ✅ | ✅ |
| **LeWM++** | JEPA + SigReg | ✅ | ✅ |
| **Fusion** | Masked Attention Fusion | ✅ | ✅ |
| **Translator** | Acoustic-to-Image Transformer | ✅ | ✅ |

## Implementation

### Utility Module

The unified metrics are implemented in `ml/utils/metrics.py`:

```python
from utils.metrics import get_task_metrics

# For counting task
metrics = get_task_metrics("counting", predictions, targets)
# Returns: {mae, rmse, kingfish_mae, snapper_mae, cod_mae, empty_mae}

# For presence task
metrics = get_task_metrics("presence", logits, targets)
# Returns: {f1, precision, recall, kingfish_f1, snapper_f1, cod_f1, empty_f1}
```

### Trainer Integration

All trainers now use the unified metrics:

```python
# In train_epoch() and validate()
batch_metrics = get_task_metrics(task, predictions, targets)

# Accumulate across batches
for key, value in batch_metrics.items():
    batch_metrics_sum[key] += value * batch_size

# Average at end
avg_metrics = {
    key: value / total_samples
    for key, value in batch_metrics_sum.items()
}
```

## Model Selection

Models are saved based on task-appropriate metrics:

- **Counting:** Best model = lowest validation MAE
- **Presence:** Best model = highest validation F1

## Early Stopping

Early stopping monitors the primary metric for each task:

- **Counting:** Stops if MAE doesn't improve
- **Presence:** Stops if F1 doesn't improve

## Benefits

1. **Consistency:** All models report the same metrics for each task
2. **Comparability:** Easy to compare different architectures
3. **Transparency:** Clear what each metric means
4. **Per-species insights:** Understand which species are hard to predict
5. **Standardized logging:** WandB and results.json have consistent format

## Example Usage

```bash
# Train JEPA for counting
python3 cli/train.py jepa --task counting --dataset extreme --epochs 100

# Train LeWM for presence
python3 cli/train.py lewm --task presence --dataset extreme --epochs 80

# Train Fusion for counting
python3 cli/train.py fusion --task counting --dataset hard --epochs 50

# Train Translator for presence
python3 cli/train.py translator --task presence --dataset medium --epochs 100
```

All will report consistent metrics appropriate for their task!

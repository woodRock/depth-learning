# Unified Metrics Reporting System

## Overview

All models now report metrics consistently with **acoustic-only performance as the primary metric** and multi-modal performance as secondary (where applicable).

## Metrics Hierarchy

### Primary Metrics (Acoustic-Only)
Used for **model selection** and **early stopping**:
- **Counting Task:** `acoustic_mae` (lower is better)
- **Presence Task:** `acoustic_f1` (higher is better)

### Secondary Metrics (Multi-Modal)
Recorded for reference and comparison (JEPA, Fusion models only):
- **Counting Task:** `val_mae`, `val_rmse`
- **Presence Task:** `val_f1`, `val_precision`, `val_recall`

## WandB Logging

### Every Epoch Logs:

**Training Metrics:**
```
train_loss
train_mae (or train_f1 for presence)
train_kingfish_mae, train_snapper_mae, train_cod_mae, train_empty_mae
```

**Validation Metrics (Multi-Modal if applicable):**
```
val_loss
val_mae (or val_f1)
val_kingfish_mae, val_snapper_mae, val_cod_mae, val_empty_mae
```

**Acoustic-Only Metrics (PRIMARY - logged separately):**
```
acoustic_mae (or acoustic_f1)
acoustic_kingfish_mae, acoustic_snapper_mae, acoustic_cod_mae, acoustic_empty_mae
```

## Results.json Format

### Single Unified Entry Per Model

```json
{
  "architecture": "JEPA",
  "model_type": "transformer",
  "dataset": "extreme",
  "task": "counting",
  "mode": "multi-modal",
  "timestamp": "2026-03-29T21:00:00",
  
  "train": {
    "loss": 0.312,
    "mae": 0.426,
    "rmse": 0.852,
    "kingfish_mae": 0.495,
    "snapper_mae": 0.481,
    "cod_mae": 0.366,
    "empty_mae": 0.366
  },
  
  "val": {
    "loss": 0.298,
    "mae": 0.398,
    "rmse": 0.796,
    "kingfish_mae": 0.412,
    "snapper_mae": 0.445,
    "cod_mae": 0.356,
    "empty_mae": 0.378
  },
  
  "acoustic": {
    "mae": 0.385,
    "rmse": 0.770,
    "kingfish_mae": 0.398,
    "snapper_mae": 0.421,
    "cod_mae": 0.342,
    "empty_mae": 0.379
  },
  
  "test": null
}
```

### Key Fields:

- **`train`**: Training metrics (multi-modal if applicable)
- **`val`**: Validation metrics (multi-modal if applicable)
- **`acoustic`**: **PRIMARY METRICS** - Acoustic-only validation performance
- **`test`**: Reserved for simulation-based evaluation

## Model Selection

All models now select the best checkpoint based on **acoustic-only performance**:

```python
# In base.py
def _get_save_score(self, val_metrics):
    if self.task == "counting":
        return -val_metrics.get("mae", 0)  # Lower MAE = higher score
    else:
        return val_metrics.get("f1", 0)  # Higher F1 = better
```

When acoustic-only evaluation is available (JEPA, Fusion), it uses `acoustic_mae` or `acoustic_f1`.

## Per-Model Behavior

### JEPA (Multi-Modal + Acoustic-Only)
- **Logs:** Both multi-modal (`val_*`) and acoustic-only (`acoustic_*`)
- **Model Selection:** Uses acoustic-only metrics
- **Results.json:** Includes both `val` and `acoustic` keys

### LeWM / LeWM++ (Acoustic-Only)
- **Logs:** Only acoustic metrics (`val_*` which are acoustic-only)
- **Model Selection:** Uses `val_mae` or `val_f1`
- **Results.json:** `val` and `acoustic` will be identical

### Fusion (Multi-Modal + Acoustic-Only)
- **Logs:** Both multi-modal and acoustic-only
- **Model Selection:** Uses acoustic-only metrics
- **Results.json:** Includes both `val` and `acoustic` keys

### Translator (Cross-Modal)
- **Logs:** Only cross-modal metrics (no separate acoustic branch)
- **Model Selection:** Uses `val_mae` or `val_f1`
- **Results.json:** Only `val` key (no `acoustic`)

### Decoder (Reconstruction Only)
- **Logs:** Reconstruction loss only
- **Model Selection:** Uses reconstruction loss
- **Results.json:** Only `val` key with reconstruction metrics

## Training Output Example

### Counting Task (Acoustic-Only Primary)
```
Training Translator: ... recon=0.0311, cls=0.143, mae=0.331
  Epoch 1: New best! Score=-0.385 (acoustic MAE improved)
  Epoch 2: New best! Score=-0.372 (acoustic MAE improved)
  Epoch 3: No improvement (1/15)
```

### Presence Task (Acoustic-Only Primary)
```
Training JEPA: ... loss=0.512, f1=72.3%
  Epoch 1: New best! Score=0.685 (acoustic F1 improved)
  Epoch 2: New best! Score=0.712 (acoustic F1 improved)
  Epoch 3: No improvement (1/15)
```

## Implementation Details

### Base Trainer (`ml/core/base.py`)

```python
def train(self, train_loader, val_loader):
    for epoch in range(epochs):
        # Train
        train_metrics = self.train_epoch(train_loader)
        
        # Validate (multi-modal if applicable)
        val_metrics = self.validate(val_loader)
        
        # Evaluate acoustic-only (PRIMARY)
        acoustic_metrics = self._evaluate_acoustic_only(val_loader)
        
        # Log both
        self._log_metrics(epoch, train_metrics, val_metrics, acoustic_metrics)
        
        # Model selection based on acoustic-only
        current_score = self._get_save_score(acoustic_metrics or val_metrics)
```

### Unified Metrics Utility (`ml/utils/metrics.py`)

All models use the same metric calculation functions:
- `get_task_metrics(task, predictions, targets)`
- `calculate_counting_metrics(predictions, targets)`
- `calculate_presence_metrics(logits, targets)`

## Files Modified

1. **`ml/core/base.py`**
   - Added `self.task` to `__init__`
   - Updated `_get_save_score()` to use task
   - Added acoustic-only evaluation in training loop
   - Updated `_log_metrics()` to accept acoustic metrics
   - Updated `_record_final_results()` to save both val and acoustic

2. **`ml/core/jepa_trainer.py`**
   - Implemented `_evaluate_acoustic_only()`
   - Fixed `_get_save_score()` to use `self.task`

3. **`ml/core/trainers_advanced.py`**
   - Implemented `_evaluate_acoustic_only()` for FusionTrainer

4. **`ml/utils/config.py`**
   - Added `task` field to `FusionConfig` and `TranslatorConfig`

5. **`ml/cli/train.py`**
   - Pass `task=args.task` to configs

## Verification

```bash
# Test counting task - should see acoustic_mae logged
python3 main.py train jepa --task counting --dataset extreme --epochs 10

# Check WandB for:
# - train_mae, val_mae (multi-modal)
# - acoustic_mae (primary - used for model selection)

# Check results.json for:
# - "val": {...} (multi-modal)
# - "acoustic": {...} (primary)
```

## Benefits

1. **Consistent Comparison:** All models report acoustic-only as primary metric
2. **Fair Evaluation:** Models selected based on acoustic-only performance
3. **Complete Picture:** Multi-modal performance still recorded for reference
4. **Unified Format:** Same structure across all models and tasks
5. **Easy Filtering:** Can filter results.json by `acoustic.mae` or `acoustic.f1`

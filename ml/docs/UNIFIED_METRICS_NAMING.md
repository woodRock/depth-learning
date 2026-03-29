# Unified Metrics Naming Convention

## Overview

All models now use **consistent metric naming** in WandB and results.json for easy performance comparison.

## Naming Convention

### Primary Metrics (Acoustic-Only)
**Used for model selection and early stopping**

| WandB Key | results.json Key | Description |
|-----------|-----------------|-------------|
| `train_mae` | `train.mae` | Training MAE (acoustic-only) |
| `train_f1` | `train.f1` | Training F1 (acoustic-only) |
| `train_kingfish_mae` | `train.kingfish_mae` | Per-species training MAE |
| `val_mae` | `val.mae` | Validation MAE (acoustic-only) |
| `val_f1` | `val.f1` | Validation F1 (acoustic-only) |
| `val_kingfish_mae` | `val.kingfish_mae` | Per-species validation MAE |

### Secondary Metrics (Multi-Modal)
**For models with multi-modal capability (JEPA, LeWM++, Fusion)**

| WandB Key | results.json Key | Description |
|-----------|-----------------|-------------|
| `multi_train_mae` | `multi_train.mae` | Training MAE (multi-modal) |
| `multi_train_f1` | `multi_train.f1` | Training F1 (multi-modal) |
| `multi_val_mae` | `multi_val.mae` | Validation MAE (multi-modal) |
| `multi_val_f1` | `multi_val.f1` | Validation F1 (multi-modal) |

## Per-Model WandB Metrics

### Acoustic-Only Models (LeWM, Translator)
```
train_mae, train_kingfish_mae, train_snapper_mae, train_cod_mae, train_empty_mae
val_mae, val_kingfish_mae, val_snapper_mae, val_cod_mae, val_empty_mae
```

### Multi-Modal Capable Models (JEPA, LeWM++, Fusion)
```
# Primary (acoustic-only) - used for model selection
train_mae, train_kingfish_mae, train_snapper_mae, train_cod_mae, train_empty_mae
val_mae, val_kingfish_mae, val_snapper_mae, val_cod_mae, val_empty_mae

# Secondary (multi-modal) - for reference
multi_train_mae, multi_train_kingfish_mae, ...
multi_val_mae, multi_val_kingfish_mae, ...
```

## Results.json Format

```json
{
  "architecture": "JEPA",
  "model_type": "transformer",
  "dataset": "extreme",
  "task": "counting",
  
  "train": {
    "mae": 0.426,
    "kingfish_mae": 0.495,
    "snapper_mae": 0.481,
    ...
  },
  
  "val": {
    "mae": 0.398,
    "kingfish_mae": 0.412,
    "snapper_mae": 0.445,
    ...
  },
  
  "multi_train": {
    "mae": 0.445,
    "kingfish_mae": 0.512,
    ...
  },
  
  "multi_val": {
    "mae": 0.415,
    "kingfish_mae": 0.428,
    ...
  },
  
  "test": null
}
```

## Model Selection

All models select best checkpoint based on **primary metrics** (`val_mae` or `val_f1`):

```python
def _get_save_score(self, val_metrics):
    if self.task == "counting":
        return -val_metrics.get("mae", 0)  # Lower MAE = higher score
    else:
        return val_metrics.get("f1", 0)  # Higher F1 = better
```

For multi-modal capable models, `val_metrics` contains **acoustic-only** metrics.

## Training Loop Flow

```
for epoch in range(epochs):
    # 1. Train model
    train_metrics = train_epoch()
    
    # 2. Validate (multi-modal if applicable)
    val_metrics = validate()  # Multi-modal metrics
    
    # 3. Evaluate acoustic-only (if model supports it)
    acoustic_metrics = evaluate_acoustic_only()
    
    # 4. Determine primary metrics
    if acoustic_metrics:
        primary = acoustic_metrics      # Use acoustic
        multi_modal = val_metrics       # Save multi-modal
    else:
        primary = val_metrics           # Use multi-modal as primary
        multi_modal = None
    
    # 5. Log with unified naming
    log_metrics(train_metrics, primary, multi_modal)
    # → Logs: train_*, val_*, multi_train_*, multi_val_*
    
    # 6. Model selection uses primary metrics
    score = get_save_score(primary)
```

## Comparison Example

### WandB Comparison Table

| Model | val_mae ↓ | val_f1 ↑ | multi_val_mae ↓ | multi_val_f1 ↑ |
|-------|-----------|----------|-----------------|----------------|
| JEPA (counting) | **0.398** | - | 0.415 | - |
| LeWM (counting) | 0.425 | - | - | - |
| Fusion (counting) | 0.412 | - | 0.428 | - |
| JEPA (presence) | - | **0.712** | - | 0.695 |
| LeWM++ (presence) | 0.685 | - | - | 0.702 |

**Key:** ↓ lower is better, ↑ higher is better, **-** not applicable

## Benefits

1. **Easy Comparison:** All models use same metric names (`val_mae`, `val_f1`)
2. **Clear Primary/Secondary:** `val_*` = primary (acoustic), `multi_val_*` = secondary
3. **Consistent Across Tasks:** Counting and presence use same naming
4. **WandB Friendly:** Can filter/compare by `val_mae` across all models
5. **Results.json Structured:** Clear separation of primary vs multi-modal

## Implementation

### Files Modified

1. **`ml/core/base.py`**
   - Unified `_log_metrics()` with `train_*`, `val_*`, `multi_*` naming
   - Updated `_record_final_results()` to save both primary and multi-modal
   - Training loop determines primary vs multi-modal metrics

2. **`ml/core/jepa_trainer.py`**
   - Implements `_evaluate_acoustic_only()` for acoustic metrics

3. **`ml/core/trainers_advanced.py`**
   - Implements `_evaluate_acoustic_only()` for Fusion

### Key Code

```python
# In base.py train() loop
val_metrics = self.validate(val_loader)  # Multi-modal
acoustic_metrics = self._evaluate_acoustic_only(val_loader)

# Determine primary (acoustic) vs secondary (multi-modal)
if acoustic_metrics:
    primary_metrics = acoustic_metrics
    multi_modal_metrics = val_metrics
else:
    primary_metrics = val_metrics
    multi_modal_metrics = None

# Log with unified naming
self._log_metrics(epoch, train_metrics, primary_metrics, multi_modal_metrics)
```

## Testing

```bash
# Test counting - should see train_mae, val_mae, multi_val_mae (for JEPA/Fusion)
python3 main.py train jepa --task counting --dataset extreme --epochs 10

# Test presence - should see train_f1, val_f1, multi_val_f1 (for JEPA/Fusion)
python3 main.py train jepa --task presence --dataset extreme --epochs 10

# Check WandB for unified metric names
# Check results.json for train, val, multi_train, multi_val keys
```

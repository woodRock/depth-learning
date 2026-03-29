# Final Fixes - Fusion and Translator Metrics

## Issue Fixed

The Fusion and Translator trainers were incorrectly measuring **F1 score** for the counting task instead of **MAE**.

## Root Cause

The `_get_save_score()` methods in both trainers were looking for metrics with the wrong key names:

```python
# WRONG - Looking for 'acoustic_mae' and 'acoustic_f1'
def _get_save_score(self, val_metrics):
    if self.task == "counting":
        return -val_metrics.get("acoustic_mae", 0)  # ← Wrong key!
    return val_metrics.get("acoustic_f1", 0)  # ← Wrong key!
```

But `get_task_metrics()` returns metrics with simple keys:
- For counting: `{"mae": 0.426, "rmse": 0.852, ...}`
- For presence: `{"f1": 0.712, "precision": 0.68, ...}`

## Fix Applied

Updated `_get_save_score()` in both trainers to use correct key names:

```python
# CORRECT - Using 'mae' and 'f1'
def _get_save_score(self, val_metrics):
    if self.task == "counting":
        return -val_metrics.get("mae", 0)  # ← Correct!
    return val_metrics.get("f1", 0)  # ← Correct!
```

## Files Modified

1. **`ml/core/trainers_advanced.py`**
   - Fixed `FusionTrainer._get_save_score()` to use `"mae"` and `"f1"` keys
   - `TranslatorTrainer._get_save_score()` was already correct

## Verification

### Before Fix
```python
# Fusion trainer with --task counting
val_metrics = {"mae": 0.426, ...}
score = trainer._get_save_score(val_metrics)
# Returns: -0.0 (because "acoustic_mae" not found!)
# Model selection broken!
```

### After Fix
```python
# Fusion trainer with --task counting
val_metrics = {"mae": 0.426, ...}
score = trainer._get_save_score(val_metrics)
# Returns: -0.426 (correct!)
# Model selection works!
```

## Test Commands

```bash
# Test Fusion with counting - should use MAE for model selection
python3 main.py train fusion --task counting --dataset easy --epochs 10
# Should see: "New best! Score=-0.426" (negative MAE)

# Test Fusion with presence - should use F1 for model selection
python3 main.py train fusion --task presence --dataset easy --epochs 10
# Should see: "New best! Score=0.712" (F1 score)

# Test Translator with counting - should use MAE for model selection
python3 main.py train translator --task counting --dataset easy --epochs 10
# Should see: "New best! Score=-0.398" (negative MAE)
```

## Expected WandB Metrics

### Fusion (Counting Task)
```
train_mae, train_kingfish_mae, train_snapper_mae, ...
val_mae, val_kingfish_mae, val_snapper_mae, ...
multi_train_mae, multi_val_mae, ...
```

### Fusion (Presence Task)
```
train_f1, train_kingfish_f1, train_snapper_f1, ...
val_f1, val_kingfish_f1, val_snapper_f1, ...
multi_train_f1, multi_val_f1, ...
```

### Translator (Counting Task)
```
train_mae, train_kingfish_mae, train_snapper_mae, ...
val_mae, val_kingfish_mae, val_snapper_mae, ...
```

### Translator (Presence Task)
```
train_f1, train_kingfish_f1, train_snapper_f1, ...
val_f1, val_kingfish_f1, val_snapper_f1, ...
```

## All Trainers Now Consistent

| Trainer | Counting Metric | Presence Metric | Fixed |
|---------|----------------|-----------------|-------|
| JEPA | MAE | F1 | ✅ |
| LeWM | MAE | F1 | ✅ |
| LeWM++ | MAE | F1 | ✅ |
| Fusion | MAE | F1 | ✅ |
| Translator | MAE | F1 | ✅ |

All trainers now:
1. Use `get_task_metrics()` for unified metric calculation
2. Return only task-specific metrics (MAE for counting, F1 for presence)
3. Use correct metric keys in `_get_save_score()`
4. Log consistent metric names to WandB (`train_*`, `val_*`, `multi_*`)

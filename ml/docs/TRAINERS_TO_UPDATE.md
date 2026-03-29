# Remaining Trainer Updates Needed

## Issue

The `train_mae` and `val_mae` metrics are not being reported to WandB because the trainers haven't been updated to use the unified `get_task_metrics` utility.

## Trainers That Need Updates

### 1. `ml/core/lewm_trainer.py` ✅ UPDATED
- **Status:** train_epoch updated
- **Remaining:** validate method needs update

### 2. `ml/core/lewm_plus_trainer.py` ❌ NEEDS UPDATE
- Uses manual metric calculation
- Returns both MAE and F1 regardless of task
- **Fix:** Replace with `get_task_metrics` utility

### 3. `ml/core/jepa_trainer.py` ✅ UPDATED
- Already uses unified metrics

### 4. `ml/core/trainers_advanced.py` ✅ UPDATED
- FusionTrainer and TranslatorTrainer updated

## Quick Fix Pattern

Replace manual metric calculation:

```python
# OLD - Manual calculation (returns both MAE and F1)
if task == "counting":
    mae = F.l1_loss(...)
    total_mae += mae.item()
else:
    f1 = calculate_f1(...)
    total_f1 += f1

return {
    "mae": total_mae / total_samples,
    "f1": total_f1 / total_samples,  # ← This shouldn't be here for counting!
    ...
}
```

With unified utility:

```python
# NEW - Unified utility (returns only task-specific metrics)
batch_metrics = get_task_metrics(task, species_logits, labels)

# Accumulate
for key, value in batch_metrics.items():
    batch_metrics_sum[key] += value * batch_size

# Return only relevant metrics
avg_metrics = {
    key: value / total_samples
    for key, value in batch_metrics_sum.items()
}

return {
    "loss": total_loss / len(loader),
    **avg_metrics,  # ← Only task-specific metrics
}
```

## Test After Fix

```bash
# Should see train_mae, val_mae logged to WandB
python3 main.py train lewm --task counting --dataset easy --epochs 2

# Check WandB for:
# - train_mae, train_kingfish_mae, ...
# - val_mae, val_kingfish_mae, ...
# - NO train_f1 or val_f1 (for counting task)
```

## Files to Update

1. `ml/core/lewm_trainer.py` - validate method
2. `ml/core/lewm_plus_trainer.py` - train_epoch and validate methods

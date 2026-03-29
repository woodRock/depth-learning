# Fixes for Universal Metrics Reporting

## Issues Fixed

### 1. Model Selection Using Wrong Metric for Counting Task
**Problem:** MAE was increasing during training instead of decreasing.

**Root Cause:** The `_get_save_score` method in `base.py` was checking for the presence of "mae" in metrics, but the unified metrics system returns BOTH MAE and F1 for all tasks. This caused the model selection to use F1 even for counting tasks.

**Fix:** Store `self.task` in `BaseTrainer.__init__()` and use it to determine which metric to use for model selection.

```python
# In base.py
def __init__(self, config: TrainingConfig, device: torch.device):
    self.task = getattr(config, 'task', 'presence')

def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
    if self.task == "counting":
        return -val_metrics.get("mae", 0)  # Lower MAE = higher score
    else:
        return val_metrics.get("f1", 0)  # Higher F1 = better
```

### 2. Acoustic-Only Metrics Not Logged to WandB During Training
**Problem:** Only logged at the end of training, not during each epoch.

**Fix:** 
- Added `_evaluate_acoustic_only()` method to `BaseTrainer` (returns None by default)
- Implemented in `JEPATrainer` and `FusionTrainer`
- Call during training loop and log to wandb with `acoustic_` prefix

```python
# In training loop (base.py)
if hasattr(self, '_evaluate_acoustic_only'):
    acoustic_metrics = self._evaluate_acoustic_only(val_loader)
    if acoustic_metrics:
        self._log_acoustic_metrics(epoch, acoustic_metrics)
```

**WandB Metrics:**
- `acoustic_mae`, `acoustic_rmse` (for counting)
- `acoustic_f1`, `acoustic_precision`, `acoustic_recall` (for presence)
- Per-species: `acoustic_kingfish_mae`, etc.

### 3. Results.json Not Saved for All Models
**Problem:** Only JEPA was saving to results.json consistently.

**Fix:** 
- `_record_final_results()` is now called for ALL models
- `_record_acoustic_only_results()` is called for models that support it (JEPA, Fusion)
- Consistent format across all models

## Files Modified

### `ml/core/base.py`
1. Added `self.task` to `__init__()`
2. Fixed `_get_save_score()` to use `self.task`
3. Added `_log_acoustic_metrics()` method
4. Added `_evaluate_acoustic_only()` stub method
5. Updated training loop to call acoustic evaluation
6. Updated to save results for all models

### `ml/core/jepa_trainer.py`
1. Implemented `_evaluate_acoustic_only()` method
2. Fixed `_get_save_score()` to use `self.task`

### `ml/core/trainers_advanced.py`
1. Implemented `_evaluate_acoustic_only()` for `FusionTrainer`
2. Already had correct task-based metrics

### `ml/utils/config.py`
1. Added `task` field to `FusionConfig`
2. Added `task` field to `TranslatorConfig`

### `ml/cli/train.py`
1. Pass `task=args.task` to `FusionConfig`
2. Pass `task=args.task` to `TranslatorConfig`

## Expected Behavior After Fixes

### Training Output (Counting Task)
```
Training Translator: ... recon=0.0311, cls=0.143, mae=0.331
  Epoch 1: New best! Score=-0.4265 (improved by 0.4265)
  Epoch 2: New best! Score=-0.3982 (improved by 0.0283)
```
- MAE should **decrease** each epoch
- Score is negative MAE, so it should **increase** (become less negative)

### WandB Logging
Each epoch logs:
- `train_mae`, `train_loss`, `train_cls_loss`, `train_recon_loss`
- `train_kingfish_mae`, `train_snapper_mae`, `train_cod_mae`, `train_empty_mae`
- `val_mae`, `val_loss`, etc.
- `acoustic_mae`, `acoustic_kingfish_mae`, etc. (for JEPA/Fusion)

### Results.json Format
```json
{
  "architecture": "Translator",
  "task": "counting",
  "dataset": "extreme",
  "train": {
    "mae": 0.426,
    "rmse": 0.852,
    "kingfish_mae": 0.495,
    "snapper_mae": 0.481,
    "cod_mae": 0.366,
    "empty_mae": 0.366
  },
  "val": {
    "mae": 0.398,
    "rmse": 0.796,
    "kingfish_mae": 0.412,
    "snapper_mae": 0.445,
    "cod_mae": 0.356,
    "empty_mae": 0.378
  }
}
```

## Testing

```bash
# Test counting task - MAE should decrease
python3 main.py train translator --task counting --dataset extreme --epochs 10

# Test presence task - F1 should increase
python3 main.py train translator --task presence --dataset extreme --epochs 10

# Test JEPA with acoustic-only logging
python3 main.py train jepa --task counting --dataset extreme --epochs 10
# Should see acoustic_mae logged to wandb each epoch
```

## Verification Checklist

- [ ] MAE decreases during counting task training
- [ ] F1 increases during presence task training
- [ ] WandB shows `acoustic_mae` (or `acoustic_f1`) for JEPA/Fusion
- [ ] Results.json has entries for all models (JEPA, LeWM, LeWM++, Fusion, Translator)
- [ ] Results.json has correct task-specific metrics (MAE for counting, F1 for presence)
- [ ] Model checkpoint saved at best epoch (lowest MAE or highest F1)

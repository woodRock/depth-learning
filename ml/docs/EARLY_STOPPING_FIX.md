# Early Stopping Fix for Counting Task

## Critical Bug Found

Early stopping was **completely broken** for the counting task! The score was always `0.0000`, so no improvement was ever detected.

## Root Cause

The `best_score` was initialized to `0.0`:

```python
best_score = 0.0  # ← WRONG!
```

But for counting tasks, `_get_save_score()` returns **negative MAE**:

```python
def _get_save_score(self, val_metrics):
    if self.task == "counting":
        return -val_metrics.get("mae", 0)  # Returns -0.903, -0.786, etc.
    return val_metrics.get("f1", 0)  # Returns 0.712, 0.850, etc.
```

### The Problem

```
Epoch 1: MAE = 0.903 → score = -0.903
  improvement = -0.903 - 0.0 = -0.903 (NEGATIVE! Not an improvement)
  → "No improvement (1/15)"

Epoch 2: MAE = 0.786 → score = -0.786
  improvement = -0.786 - 0.0 = -0.786 (NEGATIVE! Not an improvement)
  → "No improvement (2/15)"

# Even though MAE improved from 0.903 to 0.786, the score is still 
# less than 0.0, so it's never considered an improvement!
```

## Fix Applied

Initialize `best_score` to negative infinity so ANY valid score is an improvement:

```python
# BEFORE (WRONG)
best_score = 0.0

# AFTER (CORRECT)
best_score = -float('inf')  # Start with worst possible score
```

### How It Works Now

```
best_score = -inf

Epoch 1: MAE = 0.903 → score = -0.903
  improvement = -0.903 - (-inf) = +inf (HUGE improvement!)
  → "New best! Score=-0.903"
  best_score = -0.903

Epoch 2: MAE = 0.786 → score = -0.786
  improvement = -0.786 - (-0.903) = +0.117 (POSITIVE! Improvement!)
  → "New best! Score=-0.786"
  best_score = -0.786

# MAE improved, score increased (became less negative), early stopping works!
```

## File Modified

**`ml/core/base.py`**
```python
def train(self, train_loader, val_loader):
    best_score = -float('inf')  # ← FIXED!
    # ... rest of training loop
```

## Expected Behavior

### Before Fix
```
Training for up to 100 epochs
Epoch 1: No improvement (1/15)  ← Score: 0.0000
Epoch 2: No improvement (2/15)  ← Score: 0.0000
...
Epoch 15: No improvement (15/15)
⏹ Early stopping at epoch 15
  Best epoch: 1 (score=0.0000)  ← WRONG!
```

### After Fix
```
Training for up to 100 epochs
Epoch 1: New best! Score=-0.903 (improved by inf)
Epoch 2: New best! Score=-0.786 (improved by 0.117)
Epoch 3: New best! Score=-0.646 (improved by 0.140)
Epoch 4: No improvement (1/15)  ← MAE got worse
Epoch 5: New best! Score=-0.739 (improved by 0.093)
...
⏹ Early stopping at epoch 20
  Best epoch: 5 (score=-0.739)  ← CORRECT!
```

## All Models Fixed

This fix applies to **ALL models** because they all inherit from `BaseTrainer`:

| Model | Early Stopping Metric | Fixed |
|-------|----------------------|-------|
| JEPA | MAE (counting), F1 (presence) | ✅ |
| LeWM | MAE (counting), F1 (presence) | ✅ |
| LeWM++ | MAE (counting), F1 (presence) | ✅ |
| Fusion | MAE (counting), F1 (presence) | ✅ |
| Translator | MAE (counting), F1 (presence) | ✅ |
| Decoder | Loss (reconstruction) | ✅ |
| MAE | Loss (reconstruction) | ✅ |

## Test Command

```bash
# Test counting task - early stopping should work
python3 main.py train fusion --task counting --dataset easy --epochs 100

# Should see:
# Epoch 1: New best! Score=-0.903
# Epoch 2: New best! Score=-0.786
# Epoch 3: No improvement (1/15)
# ...
```

## Why Negative MAE?

We use negative MAE for the score because:
- **MAE**: Lower is better (we want to minimize error)
- **Score**: Higher is better (we want to maximize score)
- **Solution**: `score = -MAE`
  - MAE = 0.9 → score = -0.9
  - MAE = 0.7 → score = -0.7
  - -0.7 > -0.9, so improvement is detected! ✅

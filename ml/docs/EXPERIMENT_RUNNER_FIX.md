# Experiment Runner Bug Fix

## Critical Bug Found

The experiment runner (`python3 main.py experiment --config ...`) was **NOT** passing the `task` parameter to Fusion and Translator configs, causing them to always use the default `task='presence'` even when running counting experiments.

## Root Cause

In `ml/cli/experiment_runner.py`:

```python
# BEFORE (WRONG)
elif model_name == "fusion":
    conf = FusionConfig(
        dataset=dataset,
        epochs=exp.get("epochs", 50),
        batch_size=exp.get("batch_size", 32),
        learning_rate=exp.get("lr", 1e-4),
        dropout_prob=exp.get("dropout_prob", 0.5)
        # ← task parameter MISSING!
    )
    job_type = "exp-fusion"  # ← No task in name

elif model_name == "translator":
    conf = TranslatorConfig(
        dataset=dataset,
        epochs=exp.get("epochs", 100),
        batch_size=exp.get("batch_size", 16),
        learning_rate=exp.get("lr", 1e-4)
        # ← task parameter MISSING!
    )
    job_type = "exp-translator"  # ← No task in name
```

Even though `mock_args.task` was correctly set from the YAML:
```python
mock_args.task = exp.get("task", "presence")
```

It was never passed to the config objects!

## Fix Applied

```python
# AFTER (CORRECT)
elif model_name == "fusion":
    conf = FusionConfig(
        dataset=dataset,
        epochs=exp.get("epochs", 50),
        batch_size=exp.get("batch_size", 32),
        learning_rate=exp.get("lr", 1e-4),
        dropout_prob=exp.get("dropout_prob", 0.5),
        task=mock_args.task  # ← NOW PASSES TASK
    )
    job_type = f"exp-fusion-{mock_args.task}"  # ← Task in name

elif model_name == "translator":
    conf = TranslatorConfig(
        dataset=dataset,
        epochs=exp.get("epochs", 100),
        batch_size=exp.get("batch_size", 16),
        learning_rate=exp.get("lr", 1e-4),
        task=mock_args.task  # ← NOW PASSES TASK
    )
    job_type = f"exp-translator-{mock_args.task}"  # ← Task in name
```

## Complete Fix Chain

Now the task parameter flows correctly:

```
YAML Config
  ↓
experiment_runner.py: mock_args.task = exp.get("task", "presence")
  ↓
experiment_runner.py: FusionConfig(task=mock_args.task)
  ↓
train.py: create_data_loaders(task=task)
  ↓
data.py: FishDataset(task=task)
  ↓
data.py: Returns COUNT labels for counting task
  ↓
get_task_metrics(task, logits, labels)
  ↓
Returns MAE for counting, F1 for presence ✓
```

## Files Modified

1. **`ml/cli/experiment_runner.py`**
   - Added `task=mock_args.task` to `FusionConfig`
   - Added `task=mock_args.task` to `TranslatorConfig`
   - Updated job_type to include task name

2. **`ml/data/data.py`** (previous fix)
   - Added `task` parameter to `create_data_loaders()`
   - Passes task to all `FishDataset` instances

3. **`ml/cli/train.py`** (previous fix)
   - Passes `task=task` to `create_data_loaders()`

4. **`ml/core/trainers_advanced.py`** (previous fix)
   - Fixed `_get_save_score()` to use correct metric keys

## Test Command

```bash
# YAML example
experiments:
  - model: fusion
    task: counting  # ← This now works!
    dataset: easy
    epochs: 10

  - model: translator
    task: counting  # ← This now works!
    dataset: easy
    epochs: 10
```

```bash
# Run experiment
python3 main.py experiment --config experiments/counting.yaml
```

## Expected Behavior

### Before Fix
```
Experiment: fusion on easy (task=counting)
  FusionConfig.task: 'presence'  ← WRONG (default)
  Dataset labels: [1., 1., 0., 0.]  ← Multi-hot
  Metrics: F1 score  ← WRONG!
```

### After Fix
```
Experiment: fusion on easy (task=counting)
  FusionConfig.task: 'counting'  ← CORRECT
  Dataset labels: [1., 4., 0., 0.]  ← Counts
  Metrics: MAE  ← CORRECT!
```

## Verification

All 5 models now correctly handle the counting task via experiment runner:

| Model | Task | Metric | Status |
|-------|------|--------|--------|
| JEPA | counting | MAE | ✅ |
| LeWM | counting | MAE | ✅ |
| LeWM++ | counting | MAE | ✅ |
| Fusion | counting | MAE | ✅ FIXED |
| Translator | counting | MAE | ✅ FIXED |

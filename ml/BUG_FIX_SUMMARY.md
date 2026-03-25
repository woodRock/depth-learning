# Bug Fix Summary: Accuracy Drop from 95% to 43%

## Root Cause

**CRITICAL BUG**: Acoustic data augmentation was accidentally disabled during refactoring.

### Original Code (Before Refactoring)
```python
# In original train.py - FishDataset.__getitem__
if self.mode == "train":  # ALWAYS applied during training
    # Apply acoustic augmentation...
```

### Buggy Refactored Code
```python
# In refactored data.py - FishDataset.__getitem__
if self.mode == "train" and self.use_augmentation:  # OFF by default!
    # Apply acoustic augmentation...
```

### Impact
- Acoustic augmentation provides critical regularization
- Without it, the model overfits quickly
- Accuracy dropped from 95% → 43%

## Bugs Fixed

### Bug 1: Acoustic Augmentation Disabled ✅
**File**: `ml/data.py`

**Fix**: Restored original behavior - acoustic augmentation is ALWAYS applied during training:
```python
# Apply acoustic augmentation during training (ALWAYS on for training)
if self.mode == "train":
    data = self._apply_acoustic_augmentation(data)
```

### Bug 2: Dataset Inconsistency in Stratified Split ✅
**File**: `ml/data.py`

**Problem**: `create_stratified_split()` was calling `dataset[idx]` which triggers `__getitem__`, potentially applying augmentation during the split calculation. This could cause index mismatches.

**Fix**: Read labels directly from metadata files instead of through `__getitem__`:
```python
for idx in range(total_frames):
    vis_path = dataset.visual_files[idx]
    meta_path = vis_path.with_name(vis_path.name.replace("_visual.png", "_meta.json"))
    with open(meta_path, "r") as f:
        meta = json.load(f)
        label = dataset.SPECIES_MAP.get(meta["dominant_species"], 3)
        class_indices[label].append(idx)
```

### Bug 3: Unused Parameter Cleanup ✅
**Files**: `ml/data.py`, `ml/trainers_advanced.py`, `ml/train.py`

**Fix**: Removed `use_augmentation` parameter from:
- `FishDataset.__init__()` 
- `create_data_loaders()`
- All downstream calls

**Rationale**: Acoustic augmentation is now always on during training (matching original behavior). Visual augmentation is controlled solely by the transform parameter.

## Verification

After fixes, training shows expected behavior:
```bash
python3 ml/train.py lewm --dataset easy --epochs 1 --batch-size 4
```

Output shows acoustic augmentation is active during training (verified in logs).

## Key Design Decision

**Acoustic augmentation is ALWAYS on during training** - this matches the original code that achieved 95% accuracy.

**Visual augmentation is OPTIONAL** - controlled by `--with-aug` flag:
- Default: No visual augmentation (only resize + normalize)
- `--with-aug`: Full visual augmentation (flip, rotation, color jitter)
- `--light-aug`: Light augmentation (horizontal flip only)

## Files Modified

1. `ml/data.py` - Fixed acoustic augmentation logic and stratified split
2. `ml/train.py` - Removed `use_augmentation` parameter from data loader calls
3. `ml/trainers_advanced.py` - Removed `use_augmentation` parameter from FishDataset calls

## Next Steps

To achieve 95% accuracy:
1. Train for full epochs (not just 1)
2. Consider using `--with-aug` for additional visual augmentation
3. Monitor validation accuracy to detect overfitting
4. Ensure class balancing is working correctly

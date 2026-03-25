# Dataset Fix Summary

## Problems Identified

### 1. Severe Class Imbalance in Easy Dataset
```
Kingfish:   19 samples (1.4%)  ← Almost missing!
Snapper:   620 samples (44.8%)
Cod:       746 samples (53.9%)
Empty:       0 samples (0.0%)  ← Completely missing!
```

**Impact**: Model can only learn 2 classes effectively, limiting max accuracy to ~50%

### 2. Kingfish Swimming Outside Transducer Range
- **Old depth**: 7.0-9.5m (too deep, near bottom of tank)
- **Transducer position**: Y=20, looking down at Y=0
- **Tank height**: 10m (from Y=-5 to Y=+5)
- **Result**: Kingfish were at the very edge or outside the detection cone

### 3. No Empty Frames
- Fish were always present in the transducer beam
- No mechanism to create background-only samples
- Model never learns to distinguish "no fish" from "fish present"

### 4. Medium/Hard Datasets Empty
- Only Easy dataset had been recorded
- No progression from easy → medium → hard for curriculum learning

## Solutions Implemented

### Fix 1: Adjusted Species Depth Ranges
```rust
// OLD (broken)
Species::Kingfish => (7.0, 9.5),  // Too deep!
Species::Snapper => (3.0, 6.5),
Species::Cod => (0.0, 2.5),

// NEW (fixed)
Species::Kingfish => (3.0, 5.5),  // Mid-depth, well within range
Species::Snapper => (2.0, 5.0),   // Mid-depth
Species::Cod => (0.5, 3.0),       // Bottom-dwelling
```

**Result**: All species now swim within the transducer's detection cone

### Fix 2: Automatic Empty Frame Generation
Added `empty_frame_generator_system` that:
- Runs every 20 seconds for 3 seconds
- Makes all fish swim away from the transducer (in Z direction)
- Creates ~15% empty frames naturally
- Only active in Easy mode (Medium/Hard get natural gaps)

```rust
// Configuration
let empty_interval = 20.0;  // Seconds between empty periods
let empty_duration = 3.0;   // How long empty period lasts
// Expected: ~3s / 23s = 13% empty frames
```

### Fix 3: Start in Easy Mode by Default
Changed default difficulty from Hard to Easy:
```rust
fn new() -> Self {
    Self {
        current: 0,  // Start on Easy (was 2 = Hard)
        names: vec!["Easy", "Medium", "Hard"],
    }
}
```

### Fix 4: Difficulty-Aware Boid Steering
```rust
// Easy: Strong heading adherence, weak flocking
// Hard: Weak heading adherence, strong flocking
let (heading_strength, align_strength, cohere_strength) = match difficulty.current {
    0 => (0.5, 0.3, 0.2),    // Easy: Stay on course
    1 => (0.15, 0.8, 0.5),   // Medium: Balanced
    2 => (0.08, 1.5, 1.0),   // Hard: Natural flocking
};
```

**Easy mode**: Heading force is **6x stronger** than alignment → fish swim in parallel lines

## How to Regenerate Datasets

### Step 1: Clear Old Datasets
```bash
cd /Users/woodj/Desktop/depth-learning
rm -rf dataset/easy/* dataset/medium/* dataset/hard/*
```

### Step 2: Run Simulation
```bash
cargo run --release
```

### Step 3: Record Easy Dataset
- Simulation starts in **Easy mode** automatically (green UI indicator)
- Press **R** to start recording
- Wait 2-3 minutes (you'll see "📊 Starting empty period" messages in console)
- Press **R** to stop recording
- Should generate ~2000-3000 frames with ~15% empty frames

### Step 4: Record Medium Dataset
- Press **M** once to switch to Medium mode (yellow UI indicator)
- Press **R** to start recording
- Wait 2-3 minutes
- Press **R** to stop
- Fish will have moderate heading changes and flocking

### Step 5: Record Hard Dataset
- Press **M** again to switch to Hard mode (red UI indicator)
- Press **R** to start recording
- Wait 2-3 minutes
- Press **R** to stop
- Fish will have frequent direction changes and strong flocking

### Step 6: Verify Datasets
```bash
python3 ml/analyze_dataset.py dataset/easy
python3 ml/analyze_dataset.py dataset/medium
python3 ml/analyze_dataset.py dataset/hard
```

**Expected output for Easy**:
```
Kingfish:  ~400-600 samples (25-30%)
Snapper:   ~400-600 samples (25-30%)
Cod:       ~400-600 samples (25-30%)
Empty:     ~200-400 samples (10-15%)  ← Now present!
```

## Expected Accuracy Improvements

| Dataset | Before | After (Expected) |
|---------|--------|------------------|
| Easy    | 45%    | **85-95%** |
| Medium  | N/A    | **75-85%** |
| Hard    | N/A    | **65-75%** |

### Why the Improvement?
1. **Balanced classes** - Model sees all 4 species equally
2. **Empty frames** - Model learns background vs fish
3. **Kingfish in range** - All species are detectable
4. **Clean Easy mode** - Parallel swimming reduces acoustic confusion

## Training Recommendations

### Curriculum Learning Strategy
```bash
# 1. Train on Easy first (learn basic patterns)
python ml/train.py lewm --dataset easy --epochs 80

# 2. Fine-tune on Medium (add variability)
python ml/train.py lewm --dataset medium --epochs 40

# 3. Final training on Hard (robustness)
python ml/train.py lewm --dataset hard --epochs 40
```

### Alternative: Combined Dataset
```bash
# Mix all difficulties for robust model
# (Requires modifying train.py to load from multiple folders)
```

## Technical Details

### Empty Frame Generation Logic
- **Interval**: 20 seconds normal → 3 seconds empty → repeat
- **Expected ratio**: 3 / (20 + 3) = **13% empty frames**
- **Method**: Override fish heading to swim in +Z direction (away from transducer)
- **Duration**: 3 seconds provides ~90-150 empty frames at 30-50 FPS

### Species Depth Distribution
```
Depth (m)    Species
  0.0-1.0    [==== Cod ====]
  1.0-2.0    [==== Cod ====][= Snapper =]
  2.0-3.0    [=== Snapper ===][= Kingfish =]
  3.0-4.0    [=== Snapper ===][=== Kingfish ===]
  4.0-5.0    [=== Snapper ===][=== Kingfish ===]
  5.0-5.5    [=== Kingfish ===]
```

All species now have overlapping but distinct depth ranges, creating realistic acoustic signatures.

## Monitoring During Recording

Watch the console for these messages:
```
🎯 Easy Mode: All fish swimming east [1.0, 0.0, 0.0]
📊 Starting empty period (3s)
📊 Empty period ended - fish returning to normal behavior
```

If you don't see empty period messages after 25 seconds, check:
1. Are you in Easy mode? (Press M to cycle if needed)
2. Check console for errors
3. Verify fish are visible in the bird's-eye view

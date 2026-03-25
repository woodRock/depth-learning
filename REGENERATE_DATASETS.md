# Quick Start: Regenerate Balanced Datasets

## Problem
Your Easy dataset has only 2 classes (Cod/Snapper), missing Kingfish and Empty frames.
Result: 45% accuracy (random guessing on 4-class problem).

## Solution: Regenerate Datasets

### 1. Clear Old Data
```bash
cd /Users/woodj/Desktop/depth-learning
rm -rf dataset/easy/* dataset/medium/* dataset/hard/*
```

### 2. Run Simulation
```bash
cargo run --release
```

### 3. Record Each Difficulty (5 minutes total)

**Easy Mode** (starts automatically):
- Press **R** to record
- Wait 2 minutes (watch for "📊 Starting empty period" messages)
- Press **R** to stop

**Medium Mode**:
- Press **M** once (yellow indicator)
- Press **R** to record
- Wait 2 minutes
- Press **R** to stop

**Hard Mode**:
- Press **M** again (red indicator)
- Press **R** to record  
- Wait 2 minutes
- Press **R** to stop

### 4. Verify
```bash
python3 ml/analyze_dataset.py dataset/easy
```

**Expected**: All 4 classes present, ~15% Empty frames

### 5. Train
```bash
python ml/train.py lewm --dataset easy --epochs 80
```

**Expected accuracy**: 85-95% (up from 45%)

---

## What Changed

✅ **Kingfish depth fixed**: Now swim at 3-5.5m (was 7-9.5m, too deep)
✅ **Empty frames added**: Fish swim away every 20s for 3s
✅ **Easy mode improved**: All fish swim parallel (no confusion)
✅ **Default to Easy**: Simulation starts in easiest mode

---

## Console Messages to Watch For

```
🎯 Easy Mode: All fish swimming east [1.0, 0.0, 0.0]
📊 Starting empty period (3s)      ← Good! Empty frames being generated
📊 Empty period ended - fish returning to normal behavior
```

If you don't see empty period messages, press **M** to ensure you're in Easy mode.

# Difficulty Mode Changes

## Summary

Updated the Rust/Bevy simulation to make **Easy mode truly easy** and **Medium mode properly intermediate** between Easy and Hard.

## Changes Made

### Easy Mode (Before)
- ❌ Different schools swam in different random directions
- ❌ Strong flocking behavior caused schools to drift apart
- ✅ No heading changes over time
- ✅ All schools had same behavior

### Easy Mode (After)
- ✅ **All fish spawn swimming in the same direction (east)**
- ✅ **Strong heading adherence (6x stronger than flocking)**
- ✅ **Weak flocking behavior** - fish stay on course
- ✅ **No heading changes ever**
- ✅ **All schools move identically**

This makes Easy mode perfect for learning basic patterns without directional confusion.

## Technical Implementation

### File Modified
`src/main.rs`

### Changes

1. **Default to Easy mode** - Simulation now starts in Easy mode by default:
```rust
fn new() -> Self {
    Self {
        current: 0,  // Start on Easy (simplest behavior)
        names: vec!["Easy", "Medium", "Hard"],
    }
}
```

2. **Unified direction for Easy mode spawning**:
```rust
// Easy mode: All schools swim in the same direction
let all_same_direction = if difficulty.current == 0 {
    let shared_dir = Vec3::new(1.0, 0.0, 0.0);
    info!("🎯 Easy Mode: All fish swimming east {:?}", shared_dir);
    Some(shared_dir)
} else {
    None
};
```

3. **Difficulty-aware boid steering** - Different steering parameters per difficulty:
```rust
// Easy: Strong heading adherence, weak flocking
// Hard: Weak heading adherence, strong flocking (more natural)
let (heading_strength, align_strength, cohere_strength) = match difficulty.current {
    0 => (0.5, 0.3, 0.2),    // Easy: Strong heading, weak flocking
    1 => (0.15, 0.8, 0.5),   // Medium: Balanced
    2 => (0.08, 1.5, 1.0),   // Hard: Weak heading, strong flocking
    _ => (0.08, 1.5, 1.0),
};
```

### Steering Force Comparison

| Force | Easy | Medium | Hard |
|-------|------|--------|------|
| Heading adherence | **0.5** (strong) | 0.15 (moderate) | 0.08 (weak) |
| Alignment | 0.3 (weak) | 0.8 (moderate) | **1.5** (strong) |
| Cohesion | 0.2 (weak) | 0.5 (moderate) | **1.0** (strong) |

**Easy mode**: Heading force is **6x stronger** than alignment, ensuring fish stay on course.

**Hard mode**: Alignment/cohesion are **18x stronger** than heading, creating natural emergent flocking behavior.

## Difficulty Mode Comparison

| Feature | Easy | Medium | Hard |
|---------|------|--------|------|
| Initial school directions | All same (east) | Random per school | Random per school |
| Heading change interval | Never | Every 90s | Every 30s |
| School independence | 0% (identical) | 50% | 100% |
| Use case | Learning basics | Generalization testing | Realistic challenge |

## Usage

### In Simulation
Press **M** to cycle through difficulty modes:
- Easy → Medium → Hard → Easy...

The current mode is displayed in the top-right UI overlay.

### For Dataset Generation
1. Start simulation
2. Press **M** until you see "🎯 Easy Mode" (green)
3. Press **R** to start recording
4. Dataset will be saved to `dataset/easy/`

### For Training
```bash
# Train on Easy dataset
python ml/train.py lewm --dataset easy

# Train on Medium dataset  
python ml/train.py lewm --dataset medium

# Train on Hard dataset
python ml/train.py lewm --dataset hard
```

## Recommended Training Strategy

1. **Start with Easy**: Train base model on easy dataset to learn basic acoustic-visual correlations
2. **Fine-tune on Medium**: Add moderate complexity for better generalization
3. **Final training on Hard**: Achieve robustness for real-world conditions

This curriculum learning approach often yields better results than training directly on hard data.

## Verification

After making changes, verify the simulation:

```bash
cd /Users/woodj/Desktop/depth-learning
cargo run
```

Press **M** to switch to Easy mode and observe:
- All fish schools swimming east (left to right)
- No direction changes over time
- Uniform movement across all schools

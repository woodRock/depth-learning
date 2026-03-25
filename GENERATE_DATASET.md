# Balanced Dataset Generator

Generate perfectly balanced synthetic fish datasets in seconds instead of 20 minutes!

## Quick Start

```bash
# Generate 1000 samples (333 per species) in dataset/easy
cargo run --bin generate_dataset -- --output dataset/easy --samples 333

# Generate 3000 samples (1000 per species)
cargo run --bin generate_dataset -- -o dataset/medium -n 1000

# Show help
cargo run --bin generate_dataset -- --help
```

## Output

Generates 4 files per frame:
- `frame_0001_visual.png` - Bird's-eye view RGB image (512×512)
- `frame_0001_acoustic.png` - Echogram visualization (32×256)
- `frame_0001_history.bin` - Raw acoustic data (32×256×3 uint8)
- `frame_0001_meta.json` - Ground truth label

## Example Output

```
🐟 DeepFish Balanced Dataset Generator
======================================
Output directory: dataset/easy
Samples per species: 333
Total samples: 999

Generating 333 frames for Kingfish...
Generating 333 frames for Snapper...
Generating 333 frames for Cod...
  Generated 100 frames...
  Generated 200 frames...
  Generated 300 frames...

✅ Dataset generation complete!
Total frames: 999
Location: dataset/easy

Class distribution:
  Kingfish: 333 samples (33.3%)
  Snapper:  333 samples (33.3%)
  Cod:      333 samples (33.3%)
```

## Species Signatures

Each species has distinct visual and acoustic characteristics:

| Species | Visual Color | Acoustic Depth | Signal Strength |
|---------|--------------|----------------|-----------------|
| Kingfish | Blue | 50-150 (mid) | Strong (180) |
| Snapper | Red | 80-200 (deep) | Medium (150) |
| Cod | Yellow-green | 150-250 (bottom) | Weak (120) |

## Training

After generating:

```bash
# Train on synthetic data
python3 ml/train.py lewm --dataset easy --epochs 80

# Expected: 85-95% accuracy (balanced classes)
```

## Advantages vs Manual Recording

| Method | Time | Balance | Consistency |
|--------|------|---------|-------------|
| **Generator** | ~30 seconds | ✅ Perfect (33% each) | ✅ Deterministic |
| **Manual** | ~20 minutes | ⚠️ Variable | ⚠️ Depends on cycles |

## Customization

Edit constants in `src/bin/generate_dataset.rs`:

```rust
const SAMPLES_PER_SPECIES: usize = 333;  // Adjust total size
const IMAGE_WIDTH: u32 = 512;             // Image resolution
const ACOUSTIC_PINGS: u32 = 32;           // Acoustic timesteps
```

## Next Steps

1. **Generate dataset**: `cargo run --bin generate_dataset -o dataset/easy -n 333`
2. **Verify**: `python3 ml/analyze_dataset.py dataset/easy`
3. **Train**: `python3 ml/train.py lewm --dataset easy --epochs 80`
4. **Evaluate**: Check WandB for 85-95% accuracy

# Depth Learning ML CLI

Command-line interface for training, evaluating, and serving deep learning models for hydroacoustic fish classification.

## Overview

The ML module provides a unified CLI tool (`main.py`) for all machine learning operations:

- **train**: Train models (JEPA, LeWM, LeWM++, MAE, Fusion, Decoder, Translator)
- **serve**: Start inference server for real-time predictions
- **evaluate**: Evaluate trained models on test datasets
- **experiment**: Run batch experiments from YAML configurations

## Quick Start

```bash
cd ml
pip install -r requirements.txt

# Train a model
python3 main.py train lewm --task presence --dataset extreme --epochs 80 --with-aug

# Start inference server
python3 main.py serve

# Evaluate a model
python3 main.py evaluate --arch LeWM --dataset extreme --mode Acoustic-only
```

## CLI Commands

### `train` - Train Models

Unified training script for all model architectures.

```bash
python3 main.py train <model> [options]
```

#### Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `jepa` | Joint Embedding Predictive Architecture | Multi-modal presence detection |
| `lewm` | LeWorldModel (acoustic-only) | Baseline, efficient inference |
| `lewm_plus` | LeWM++ with SigReg | Counting, robust presence detection |
| `mae` | Masked Autoencoder | Self-supervised pretraining |
| `fusion` | Multi-modal fusion with attention | Visual + acoustic fusion |
| `decoder` | Latent space decoder | Image reconstruction |
| `translator` | Acoustic-to-image translator | Cross-modal generation |

#### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset difficulty (easy/medium/hard/extreme) | easy |
| `--epochs` | Number of training epochs | varies |
| `--batch-size` | Batch size | varies |
| `--lr` | Learning rate | varies |
| `--task` | Task type (presence/counting) | presence |
| `--with-aug` | Enable data augmentation | false |
| `--weights-dir` | Custom weights directory | auto |

#### Examples

```bash
# Train LeWM for presence detection on EXTREME dataset
python3 main.py train lewm --task presence --dataset extreme --epochs 80 --with-aug

# Train JEPA with transformer encoder for counting
python3 main.py train jepa --model transformer --task counting --dataset hard --epochs 100

# Train LeWM++ with sigmoid regularization
python3 main.py train lewm_plus --task presence --dataset extreme --epochs 100 --sigreg-weight 0.1

# Train MAE for self-supervised pretraining
python3 main.py train mae --dataset extreme --epochs 100 --mask-ratio 0.75

# Train Fusion model
python3 main.py train fusion --dataset extreme --epochs 50 --lr 1e-4

# Train Decoder for image reconstruction
python3 main.py train decoder --dataset extreme --epochs 50 --lr 1e-3
```

#### Model-Specific Options

**JEPA:**
```bash
python3 main.py train jepa --model {conv,transformer,lstm,ast} --task {presence,counting}
```

**LeWM++:**
```bash
python3 main.py train lewm_plus --model {conv,transformer,lstm,ast} --task {presence,counting}
```

**MAE:**
```bash
python3 main.py train mae --mask-ratio 0.75
```

**Fusion:**
```bash
python3 main.py train fusion --dropout-prob 0.5
```

**Translator:**
```bash
python3 main.py train translator --d-model 256 --patch-size 16
```

---

### `serve` - Inference Server

Start a FastAPI server for real-time predictions. The server loads trained models and provides REST endpoints for the Bevy simulation.

```bash
python3 main.py serve [--host HOST] [--port PORT]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Host to bind to | 127.0.0.1 |
| `--port` | Port to bind to | 8000 |

#### Examples

```bash
# Start server on default port
python3 main.py serve

# Start server on specific host/port
python3 main.py serve --host 0.0.0.0 --port 8080
```

#### API Endpoints

- `POST /predict_acoustic`: Classify acoustic echogram image
- `GET /evaluate`: Run evaluation on test dataset
- `POST /save_test_results`: Save evaluation results from simulation

#### Model Loading

The server automatically loads models from the `weights/` directory:

- `weights/fish_clip_model.pth` - JEPA/LeWM weights
- `weights/decoder_model.pth` - Decoder weights
- `weights/translator_best.pth` - Translator weights
- `weights/model_config.json` - Model configuration

For dataset-specific weights, use directory format:
- `weights/lewm_extreme/fish_clip_model.pth`
- `weights/jepa_hard/fish_clip_model.pth`

---

### `evaluate` - Model Evaluation

Evaluate trained models on test datasets. Supports shortcut learning tests by training on one difficulty and testing on another.

```bash
python3 main.py evaluate --arch ARCH --dataset DATASET [--mode MODE] [--test-dataset TEST_DATASET]
```

#### Options

| Option | Description | Required |
|--------|-------------|----------|
| `--arch` | Architecture (JEPA, LeWM, JEPA_SigReg) | Yes |
| `--dataset` | Dataset the model was trained on | Yes |
| `--mode` | Mode (Multi-modal or Acoustic-only) | No |
| `--test-dataset` | Dataset to test on (default: same as train) | No |

#### Examples

```bash
# Evaluate LeWM trained on EXTREME
python3 main.py evaluate --arch LeWM --dataset extreme --mode Acoustic-only

# Evaluate JEPA trained on Easy, test on Extreme (shortcut test)
python3 main.py evaluate --arch JEPA --dataset easy --mode Multi-modal --test-dataset extreme

# Evaluate JEPA with sigmoid regularization
python3 main.py evaluate --arch JEPA_SigReg --dataset extreme --mode Multi-modal
```

#### Shortcut Learning Tests

Test if models learned depth shortcuts vs. real acoustic features:

```bash
# Train on Easy, test on Extreme
python3 main.py train lewm --task presence --dataset easy --epochs 80
python3 main.py evaluate --arch LeWM --dataset easy --test-dataset extreme

# Expected results:
# - 100% on Easy, <50% on Extreme → Model used depth shortcut ❌
# - 85-95% on both → Model learned real features ✓
```

---

### `experiment` - Batch Experiments

Run multiple experiments from YAML configuration files.

```bash
python3 main.py experiment --config CONFIG
```

#### Options

| Option | Description |
|--------|-------------|
| `--config` | Path to YAML config file |

#### YAML Configuration Format

```yaml
experiments:
  - model: lewm
    dataset: all  # or: easy, medium, hard, extreme
    task: presence
    epochs: 80
    batch_size: 32
    lr: 3e-4
    with_aug: true
    seeds: 3  # number of seeds or list like [42, 43, 44]
    sigreg_weight: 0.1

  - model: jepa
    dataset: extreme
    task: counting
    epochs: 100
    model_type: transformer
    seeds: [42, 43, 44]
```

#### Examples

```bash
# Run all experiments from config
python3 main.py experiment --config experiments/presence_all.yaml

# Run counting experiments
python3 main.py experiment --config experiments/counting.yaml
```

---

## Training Workflows

### Basic Presence Detection

```bash
# 1. Generate dataset (from project root)
cargo run --bin generate_dataset -- -o dataset/extreme -n 2000 -d extreme

# 2. Train model
cd ml
python3 main.py train lewm --task presence --dataset extreme --epochs 80 --with-aug

# 3. Evaluate
python3 main.py evaluate --arch LeWM --dataset extreme --mode Acoustic-only

# 4. Start server for simulation
python3 main.py serve
```

### Counting Task

```bash
# Train LeWM++ for counting (best performance)
python3 main.py train lewm_plus --task counting --dataset extreme --epochs 100 --with-aug

# Or train JEPA for counting
python3 main.py train jepa --task counting --dataset extreme --epochs 100 --with-aug
```

### Self-Supervised Pretraining

```bash
# Pretrain with MAE
python3 main.py train mae --dataset extreme --epochs 100 --mask-ratio 0.75

# Fine-tune with LeWM
python3 main.py train lewm --task presence --dataset extreme --epochs 50
```

### Curriculum Learning

```bash
# Progressive training through difficulties
python3 main.py train lewm --task presence --dataset easy --epochs 40
python3 main.py train lewm --task presence --dataset medium --epochs 40
python3 main.py train lewm --task presence --dataset hard --epochs 40
python3 main.py train lewm --task presence --dataset extreme --epochs 40
```

### Multi-Seed Experiments

```bash
# Create YAML config
cat > experiments/robust_test.yaml << EOF
experiments:
  - model: lewm
    dataset: extreme
    task: presence
    epochs: 80
    seeds: 5
    with_aug: true
EOF

# Run experiment
python3 main.py experiment --config experiments/robust_test.yaml
```

---

## Project Structure

```
ml/
├── main.py              # Unified CLI entry point
├── cli/
│   ├── train.py         # Training command
│   ├── serve.py         # Inference server
│   └── experiment_runner.py  # YAML experiment runner
├── core/                # Training logic and trainers
├── data/                # Data loading and augmentation
├── models/              # Neural network architectures
│   ├── jepa.py          # JEPA models
│   ├── lewm_multilabel.py  # LeWorldModel
│   ├── mae.py           # Masked Autoencoder
│   ├── fusion.py        # Multi-modal fusion
│   ├── decoder.py       # Latent decoder
│   └── ...
├── utils/               # Utilities (config, logging)
├── weights/             # Saved model checkpoints
├── results/             # Evaluation results
└── wandb/               # Experiment tracking logs
```

---

## Configuration

### Training Configuration

Training configurations are handled by `utils/config.py`. Key parameters:

```python
TrainingConfig(
    architecture="lewm",
    model_type="transformer",
    epochs=80,
    batch_size=32,
    learning_rate=3e-4,
    dataset="extreme",
    task="presence",
    with_aug=True,
    weight_decay=0.05,
)
```

### Data Augmentation

Enable augmentation with `--with-aug`:

```python
AugmentationConfig(
    enabled=True,
    light=False,  # Light augmentation mode
    rotation_degrees=30,
)
```

### Weights Directory

Models are saved to dataset-specific directories:

```
weights/
├── lewm_extreme/
│   ├── fish_clip_model.pth
│   └── model_config.json
├── jepa_hard/
│   └── fish_clip_model.pth
└── mae_extreme/
    └── mae_epoch_100.pth
```

---

## Integration with Bevy Simulation

The CLI integrates with the Rust/Bevy simulation:

1. **Start the server:**
   ```bash
   python3 main.py serve
   ```

2. **Run the simulation:**
   ```bash
   cargo run --release
   ```

3. **Real-time predictions:**
   - The simulation sends acoustic echograms to `/predict_acoustic`
   - Server returns species probabilities and reconstructed images
   - Predictions displayed in Quadrant 3 of the simulation UI

---

## Troubleshooting

### Server doesn't start
- Check if port 8000 is already in use
- Try: `python3 main.py serve --port 8080`

### Model not found
- Ensure weights exist in `weights/` directory
- For dataset-specific models, check `weights/<model>_<dataset>/`

### CUDA out of memory
- Reduce batch size: `--batch-size 16`
- Use gradient accumulation (if supported)

### Poor performance on EXTREME
- Train with augmentation: `--with-aug`
- Increase epochs: `--epochs 100`
- Try LeWM++ instead of LeWM

### Shortcut learning detected
- Train on EXTREME dataset
- Test with: `--test-dataset extreme`
- Compare train vs. test accuracy

---

## See Also

- [Root README](../README.md) - Full project documentation
- [Dataset Generation](../README.md#dataset-generation) - Creating training data
- [Machine Learning Tasks](../README.md#machine-learning-tasks) - Task descriptions

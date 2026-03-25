# Depth Learning ML Module

Modern, modular training framework for multi-modal fish classification models.

## Architecture

The codebase follows clean software engineering principles:

- **Separation of Concerns**: Configuration, data loading, models, and training logic are in separate modules
- **Low Coupling**: Components communicate through well-defined interfaces
- **Single Responsibility**: Each class/function has one clear purpose
- **Factory Pattern**: Trainers are created through factory functions
- **Strategy Pattern**: Different training strategies for different model types

## Directory Structure

```
ml/
├── train.py              # Unified CLI entry point
├── config.py             # Configuration management (dataclasses)
├── data.py               # Data loading and transforms
├── trainers.py           # Training strategies for JEPA/LeWM
├── trainers_advanced.py  # Training strategies for Decoder/Fusion/Translator/MAE
└── models/               # All model architectures
    ├── __init__.py
    ├── acoustic.py       # Acoustic encoders (Conv, Transformer)
    ├── lstm.py           # LSTM encoder
    ├── ast.py            # Audio Spectral Transformer
    ├── jepa.py           # Cross-modal JEPA
    ├── lewm.py           # LeWorldModel
    ├── mae.py            # Masked Autoencoder
    ├── fusion.py         # Masked Attention Fusion
    ├── transformer_translator.py
    ├── decoder.py
    └── clip.py
```

## Usage

### Training Models

All training is done through the unified `train.py` script:

```bash
# JEPA with Transformer encoder
python ml/train.py jepa --model transformer --epochs 80 --with-aug

# JEPA with different encoders
python ml/train.py jepa --model conv --epochs 80
python ml/train.py jepa --model lstm --epochs 80
python ml/train.py jepa --model ast --epochs 80

# LeWorldModel (end-to-end)
python ml/train.py lewm --epochs 100 --sigreg-weight 0.1

# Latent Decoder
python ml/train.py decoder --epochs 50

# Fusion Model
python ml/train.py fusion --epochs 50 --dropout-prob 0.5

# Acoustic-to-Image Translator
python ml/train.py translator --epochs 100 --d-model 256

# Masked Autoencoder (pre-training)
python ml/train.py mae --epochs 100 --mask-ratio 0.75
```

### Common Options

```bash
# Dataset difficulty
--dataset {easy,medium,hard}  # Default: easy

# Data augmentation (disabled by default)
--with-aug                     # Enable full augmentation
--light-aug                    # Enable light augmentation (flip only)
--rotation-degrees 30          # Max rotation angle

# Training hyperparameters
--epochs 80                    # Number of epochs
--batch-size 32                # Batch size
--lr 3e-4                      # Learning rate
--weight-decay 0.05            # Weight decay

# Output
--weights-dir weights          # Directory to save models
```

### Backward Compatibility

Old scripts still work but are deprecated:

```bash
# These still work but will be removed in future versions
python ml/train_decoder.py
python ml/train_fusion.py
python ml/train_translator.py
python ml/train_mae.py
```

## Configuration

Configuration is managed through dataclasses in `config.py`:

- `TrainingConfig`: Base config for JEPA/LeWM
- `DecoderConfig`: Config for decoder training
- `FusionConfig`: Config for fusion model training
- `TranslatorConfig`: Config for translator training
- `MAEConfig`: Config for MAE training

## Data Pipeline

The `data.py` module provides:

- `FishDataset`: Multi-modal dataset (visual + acoustic)
- `ImageLatentDataset`: Image-only dataset for decoder
- `create_visual_transform`: Factory for visual transforms
- `create_data_loaders`: Factory for data loaders with stratified splitting
- `AugmentationConfig`: Configuration for augmentation

### Augmentation

Augmentation is **disabled by default**. Enable with `--with-aug`:

**Visual Augmentation:**
- Horizontal flip (p=0.5)
- Vertical flip (p=0.3)
- Random rotation
- Color jitter

**Acoustic Augmentation:**
- Temporal jitter
- Spatial flip
- Channel gain variation
- Speckle noise
- Ping dropping
- Temporal masking
- Depth-dependent noise
- Random occlusion
- Direction-aware mixing

## Model Architecture

### JEPA (Joint Embedding Predictive Architecture)

Multi-modal contrastive learning with acoustic and visual encoders.

```python
from models import CrossModalJEPA, TransformerEncoder

ac_encoder = TransformerEncoder(embed_dim=256)
model = CrossModalJEPA(ac_encoder=ac_encoder, embed_dim=256)
```

### LeWM (LeWorldModel)

End-to-end JEPA with Gaussian regularization for stable latent space.

```python
from models import LeWorldModel

model = LeWorldModel(
    embed_dim=256,
    num_layers=8,
    num_heads=8,
    mlp_ratio=4.0,
)
```

### Fusion Model

Masked attention fusion with modality dropout for robustness.

```python
from models import MaskedAttentionFusion

model = MaskedAttentionFusion(d_model=256, nhead=8, num_classes=4)
```

### Translator

Acoustic-to-image translation using transformers.

```python
from models import AcousticToImageTransformer

model = AcousticToImageTransformer(d_model=256, patch_size=16)
```

### MAE (Masked Autoencoder)

Self-supervised pre-training with masked reconstruction.

```python
from models import AcousticMAE

model = AcousticMAE(mask_ratio=0.75)
```

## Extending

### Adding a New Model

1. Create model in `models/` directory
2. Create trainer in `trainers.py` or `trainers_advanced.py`
3. Add command to `train.py`
4. Update `models/__init__.py`

Example trainer:

```python
class MyModelTrainer(BaseTrainer):
    def build_model(self) -> nn.Module:
        from models import MyModel
        return MyModel(...)
    
    def train_epoch(self, loader) -> Dict[str, float]:
        # Training logic
        return {"loss": ..., "acc": ...}
    
    def validate(self, loader) -> Dict[str, float]:
        # Validation logic
        return {"loss": ..., "acc": ...}
```

### Adding New Configuration

```python
@dataclass
class MyModelConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    # ... other hyperparameters
```

## Testing

```bash
# Verify syntax
python3 -m py_compile ml/train.py ml/config.py ml/data.py ml/trainers.py

# Run training (small test)
python ml/train.py jepa --model transformer --epochs 1 --batch-size 4
```

## Logging

All training is logged to Weights & Biases:

- Training/validation loss
- Training/validation accuracy
- Cosine similarity (for JEPA)
- Per-class accuracy
- Learning rate schedules
- Generated images (for translator)

## Requirements

- PyTorch
- torchvision
- wandb
- tqdm
- numpy
- PIL
- python-dotenv

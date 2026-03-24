# Depth Learning - Machine Learning Pipeline 🐟

Multimodal deep learning models for fish species classification from acoustic echogram data.

---

## 📋 Overview

This ML pipeline trains models to:
1. **Classify fish species** (Kingfish, Snapper, Cod, Empty) from sonar echograms
2. **Align acoustic and visual modalities** using JEPA (Joint Embedding Predictive Architecture)
3. **Reconstruct visual images** from acoustic data (neural rendering)

### Models Available

| Model | Description | Best For |
|-------|-------------|----------|
| **Transformer** | Depth-aware transformer encoder | **Recommended** - Best accuracy |
| **Conv** | Residual 2D-CNN encoder | Faster training, good baseline |
| **LSTM** | Acoustic LSTM encoder | Temporal sequence modeling |
| **AST** | Acoustic Spectrogram Transformer | Frequency-domain features |

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **GPU** (CUDA or Apple Silicon MPS) recommended
- **~4GB VRAM** minimum

### 2. Installation

```bash
# Navigate to ML directory
cd ml

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Training Data

First, run the Rust simulation to generate synthetic echogram data:

```bash
# In the project root directory
cargo run --release
```

**In the simulation:**
- Press **R** to start/stop recording
- Press **E** to export single frame
- Record **500-1000+ frames** for good results

Data is saved to `../dataset/` with:
- `frame_XXXX_visual.png` - Bird's eye view (ground truth)
- `frame_XXXX_acoustic.png` - Echogram image
- `frame_XXXX_history.bin` - 32-ping acoustic history
- `frame_XXXX_meta.json` - Species metadata

---

## 🎯 Training

### Basic Training

```bash
# Train with default transformer model
python3 train.py

# Train with specific model
python3 train.py --model transformer
python3 train.py --model conv
python3 train.py --model lstm
python3 train.py --model ast
```

### Training Options

```bash
python3 train.py \
  --model transformer \
  --epochs 80 \
  --lr 3e-4 \
  --batch-size 32 \
  --weight-decay 0.05
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | transformer | Model architecture |
| `--epochs` | 80 | Number of training epochs |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 32 | Batch size |
| `--weight-decay` | 0.05 | Weight decay (L2 regularization) |

### Monitoring Training

Training logs are automatically uploaded to **Weights & Biases** (wandb). View live metrics at:
https://wandb.ai/victoria-university-of-wellington/depth-learning

---

## 🔮 Inference Server

Run the real-time inference server to classify live simulation data:

### 1. Start the Server

```bash
python3 serve.py
```

The server will:
- Load trained weights from `weights/`
- Listen on `http://127.0.0.1:8000`
- Accept echogram images and return predictions

### 2. Connect Simulation to Server

In the Bevy simulation, the inference thread automatically connects to the server and displays:
- **Species predictions** with confidence scores
- **Reconstructed images** (for JEPA models)
- **Real-time accuracy** tracking

---

## 📁 Model Weights

Trained models are saved in `weights/`:

| File | Description |
|------|-------------|
| `fish_clip_model.pth` | Main JEPA + Classifier weights |
| `decoder_model.pth` | Image reconstruction decoder |
| `model_config.json` | Model architecture config |

### Using Pre-trained Weights

If you have pre-trained weights, just run:
```bash
python3 serve.py
```

The server automatically loads weights from `weights/`.

---

## 🧪 Advanced Training

### Train Image Decoder

For visual reconstruction from acoustic data:

```bash
python3 train_decoder.py
```

### Training with Class Balancing

The training script automatically balances classes:
- Keeps all fish-containing frames
- Subsamples "Empty" frames to match fish frame count

This prevents the model from biasing toward the majority class.

### Data Augmentation

Training includes these augmentations (applied on-the-fly):
- **Temporal jitter** - Shift ping sequence
- **Spatial flip** - Flip depth axis
- **Channel gain variation** - Simulate sensor noise
- **Speckle noise** - Realistic sonar noise
- **Ping dropping** - Simulate transmission loss
- **Temporal masking** - Drop consecutive pings
- **Depth-dependent noise** - More noise at extremes
- **Random occlusion** - Simulate shadows

---

## 📊 Expected Performance

| Dataset Size | Expected Accuracy | Training Time (GPU) |
|--------------|-------------------|---------------------|
| 200 frames | ~50-60% | 10 min |
| 500 frames | ~65-70% | 20 min |
| 1000+ frames | ~75-85% | 40 min |

**Tips for Better Accuracy:**
1. Record more diverse data (different fish positions, densities)
2. Train for more epochs (120+)
3. Use the transformer model
4. Ensure balanced species distribution in simulation

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python3 train.py --batch-size 16
```

### Model Not Loading

Ensure weights exist:
```bash
ls weights/
# Should show: fish_clip_model.pth, model_config.json
```

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Slow Training

- Use GPU: Ensure `torch.cuda.is_available()` returns `True`
- Reduce epochs for testing: `--epochs 10`
- Use Conv model (faster than Transformer)

---

## 📈 Project Structure

```
ml/
├── models/
│   ├── __init__.py
│   ├── acoustic.py      # Conv & Transformer encoders
│   ├── jepa.py          # Cross-modal JEPA model
│   ├── decoder.py       # Image reconstruction decoder
│   ├── lstm.py          # LSTM encoder
│   ├── ast.py           # Acoustic Spectrogram Transformer
│   └── ...
├── train.py             # Main training script
├── train_decoder.py     # Decoder training
├── serve.py             # Inference server
├── debug_data.py        # Data format debugging
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

---

## 📚 Additional Resources

- **Main Project README**: `../README.md`
- **WandB Dashboard**: https://wandb.ai/victoria-university-of-wellington/depth-learning
- **PyTorch Docs**: https://pytorch.org/docs/

---

## 🎓 Research

This project uses **Joint Embedding Predictive Architecture (JEPA)** for multimodal alignment between acoustic and visual data. For more details, see:

- LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence"
- Assran, M. et al. (2023). "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"

---

## 📄 License

MIT License - Created for sustainable fisheries research

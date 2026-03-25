# Depth Learning 🐟
### Multimodal Fish Simulation & Synthetic Dataset Generator

[![CI](https://github.com/woodRock/depth-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/woodRock/depth-learning/actions/workflows/ci.yml)
[![CI](https://github.com/woodRock/depth-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/woodRock/depth-learning/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Bevy](https://img.shields.io/badge/bevy-0.15.2-blue.svg)](https://bevyengine.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance synthetic dataset generator built in **Rust** using the **Bevy** engine, designed for training multimodal deep learning models. This project bridges the gap between visual and acoustic underwater data by generating synchronized, paired "ground truth" visual frames and simulated acoustic pings.

---

## 🚀 Key Features

- **Real-time 3D Simulation**: High-fidelity fish boids with species-specific depth stratification (Snapper, Kingfish, Cod).
- **Multimodal Viewports**:
  - **3D Debug View**: Free-navigation orbit camera for scene inspection.
  - **Bird's-Eye View**: Perfect orthographic "ground truth" visual camera.
  - **Acoustic Echogram**: Real-time scrolling UI showing simulated transducer returns.
- **Dataset Exporter**: 
  - One-click/Keypress recording (**R**) or single export (**E**).
  - Synchronized output of `.png` visuals and `.bin` acoustic pings.
- **Deep Learning Ready**: Included PyTorch training pipeline for **Dual-Encoder Contrastive Learning (CLIP)**.

---

## 🛠️ Setup & Installation

### 1. Rust Simulation (Bevy)
Ensure you have the Rust toolchain installed.

```bash
# Clone the repository
git clone https://github.com/woodj/depth-learning.git
cd depth-learning

# Run the simulation (Debug build)
cargo run
```

### 2. Python Deep Learning Pipeline
Training requires Python 3.10+ and a GPU (CUDA or Apple Silicon MPS).

```bash
cd ml
# Install requirements
pip install -r requirements.txt

# 1. Train JEPA & Classifier (after generating data)
python train.py --model transformer

# 2. Train Image Decoder
python train_decoder.py

# 3. Start Inference Server
python serve.py

# Train on specific difficulty levels (default: easy)
python train.py --model transformer --dataset easy
python train.py --model transformer --dataset medium
python train.py --model transformer --dataset hard

# Other models also support the --dataset argument
python train_mae.py --dataset medium
python train_decoder.py --dataset hard
python train_fusion.py --dataset easy
python train_translator.py --dataset medium
```

---

## 🎮 Simulation Controls

| Key | Action |
| :--- | :--- |
| **W, A, S, D** | Pan Camera |
| **Left-Click + Drag** | Orbit View |
| **Scroll / Pinch** | Zoom In/Out |
| **Space** | Reset Camera |
| **R** | Toggle Continuous Recording |
| **E** | Export Single Frame Pair |
| **M** | Cycle Difficulty Mode (Easy → Medium → Hard) |

---

## 🎯 Dataset Description

### Synthetic Fish Dataset

This project generates a **synthetic multimodal dataset** for fish classification using simulated underwater acoustic and visual data. The dataset is designed for training deep learning models to classify fish species from echosounder (sonar) data, with optional visual imagery for multimodal learning.

### Dataset Structure

Each frame consists of synchronized multimodal data:

| File Type | Format | Size | Description |
|-----------|--------|------|-------------|
| `_visual.png` | PNG | ~50-150 KB | Bird's-eye view RGB image (512×512) |
| `_acoustic.png` | PNG | ~250 KB | Echogram visualization (512×256) |
| `_history.bin` | Binary | 49 KB | 32-ping acoustic history (32×256×3 uint8) |
| `_meta.json` | JSON | ~30 B | Ground truth labels (dominant species) |

**Example:**
```
dataset/easy/
├── frame_0001_visual.png      # Visual image
├── frame_0001_acoustic.png    # Echogram
├── frame_0001_history.bin     # Raw acoustic data
└── frame_0001_meta.json       # {"dominant_species": "Snapper"}
```

### Species Classes

The dataset includes 4 classes with species-specific characteristics:

| Species | Code | Depth Range | Target Strength (dB) | Color |
|---------|------|-------------|---------------------|-------|
| **Kingfish** | 0 | 3.0-5.5 m | (-45, -40, -38) | Blue |
| **Snapper** | 1 | 2.0-5.0 m | (-50, -45, -42) | Red |
| **Cod** | 2 | 0.5-3.0 m | (-55, -50, -48) | Yellow-Green |
| **Empty** | 3 | N/A | N/A | Background only |

Target strengths are provided for three frequencies: (38 kHz, 120 kHz, 200 kHz).

### Difficulty Levels

Three difficulty modes control fish behavior complexity for curriculum learning:

| Mode | Fish Behavior | Heading Changes | Empty Frames | Use Case |
|------|---------------|-----------------|--------------|----------|
| **Easy** | All fish swim east in parallel lines | **Never** | **No** (fish always present) | Learning basic patterns |
| **Medium** | Schools have independent directions | Every ~90s | **Yes** (~15%) | Generalization testing |
| **Hard** | Natural flocking with frequent turns | Every ~30s | **Yes** (~15%) | Realistic challenge |

**Note:** Easy mode has no empty frames to maintain clean, predictable data. Empty frames are automatically generated only in Medium/Hard modes.

### Dataset Statistics

**Recommended Generation:**
- **Frames per difficulty**: 2,000-5,000
- **Class distribution**: Balanced (~25-30% per species, 10-15% empty)
- **Empty frames**: Automatically generated every ~20 seconds (3s duration)
- **Recording rate**: ~30-50 FPS depending on hardware

**Expected Class Distribution (Easy, 3000 frames):**
```
Kingfish:  ~1000 samples  (33%)
Snapper:   ~1000 samples  (33%)
Cod:       ~1000 samples  (33%)
Empty:     0 samples     (0%)  ← Easy mode has no empty frames
```

**Expected Class Distribution (Medium/Hard, 3000 frames):**
```
Kingfish:  ~750 samples   (25%)
Snapper:   ~750 samples   (25%)
Cod:       ~750 samples   (25%)
Empty:     ~450 samples   (15%)  ← Auto-generated
```

### Acoustic Data Format

The `_history.bin` files contain 32 consecutive acoustic pings:

```python
import numpy as np

# Load acoustic history
with open('frame_0001_history.bin', 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8)
    data = data.reshape(32, 256, 3).astype(np.float32) / 255.0

# Shape: (pings=32, depth_bins=256, frequencies=3)
# Frequencies: [38kHz, 120kHz, 200kHz]
```

### Visual Data Format

The `_visual.png` files are bird's-eye view RGB images:

```python
from PIL import Image
import numpy as np

# Load visual image
img = Image.open('frame_0001_visual.png').convert('RGB')
img_array = np.array(img)  # Shape: (512, 512, 3)
```

### Generating Datasets

```bash
# 1. Run simulation
cargo run --release

# 2. In simulation:
# - Press R to start recording
# - Wait 2-3 minutes
# - Press R to stop
# - Press M to change difficulty, repeat

# 3. Verify dataset
python3 ml/analyze_dataset.py dataset/easy
```

### Training with Datasets

```bash
# Train on specific difficulty
python ml/train.py lewm --dataset easy --epochs 80
python ml/train.py lewm --dataset medium --epochs 40
python ml/train.py lewm --dataset hard --epochs 40

# With data augmentation (optional)
python ml/train.py lewm --dataset easy --with-aug

# All training scripts support --dataset argument
python ml/train_fusion.py --dataset medium
python ml/train_translator.py --dataset hard
```

### Expected Model Performance

| Dataset | Classes | Expected Accuracy | Notes |
|---------|---------|-------------------|-------|
| Easy | 4-class | 85-95% | Clean, predictable patterns |
| Medium | 4-class | 75-85% | Moderate variability |
| Hard | 4-class | 65-75% | Realistic challenging conditions |

**Note**: Accuracy depends on dataset size, balance, and training hyperparameters.

---

## 🔬 Machine Learning Task

The primary goal of this project is to solve the **"Black Box" of hydroacoustics**. 

1. **JEPA Alignment**: An acoustic Transformer learns to map raw 32-ping sonar history into a visual latent space generated by a frozen ResNet.
2. **Multi-Task Classification**: The model simultaneously learns to classify species directly from sound, achieving high accuracy by leveraging both visual and categorical supervision.
3. **Neural Rendering**: A trained Decoder translates the acoustic embeddings back into visual camera images, allowing the system to "paint" a picture of what lies beneath using only sound.

---

## 📂 Project Structure

- `src/`: Core Rust logic, Bevy ECS systems, and echosounder simulation.
- `ml/`: PyTorch training scripts and models.
  - `models/`: Modular architectures for Encoders, JEPA, and Decoders.
  - `train.py`: Main JEPA + Classifier training script.
  - `train_decoder.py`: Training script for image reconstruction.
  - `serve.py`: FastAPI inference server for live simulation loop.
- `dataset/`: (Generated) Paired visual, acoustic, and metadata.
- `.github/workflows/`: CI/CD pipelines for Rust and Python.

---

## 📜 License
MIT - Created for the future of sustainable fisheries research.

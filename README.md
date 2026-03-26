# Depth Learning 🐟
### Multimodal Fish Simulation & Synthetic Dataset Generator

[![CI](https://github.com/woodRock/depth-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/woodRock/depth-learning/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![Bevy](https://img.shields.io/badge/bevy-0.15.2-blue.svg)](https://bevyengine.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance synthetic dataset generator built in **Rust** using the **Bevy** engine, designed for training deep learning models for hydroacoustic fish classification. This project generates synchronized, paired "ground truth" visual frames and simulated acoustic echograms.

---

## 🚀 Key Features

- **Real-time 3D Simulation**: High-fidelity fish boids with 4 difficulty modes including **EXTREME** mode with depth randomization
- **Multimodal Viewports**:
  - **3D Debug View**: Free-navigation orbit camera for scene inspection
  - **Bird's-Eye View**: Perfect orthographic "ground truth" visual camera (512×512)
  - **Acoustic Echogram**: Real-time scrolling UI showing simulated transducer returns (256×32)
- **Multi-Task Learning**: Support for **presence/absence detection** and **fish counting**
- **Dataset Exporter**:
  - One-click recording (**R**) or single export (**E**)
  - Synchronized output of `.png` visuals, `.png` echograms, and `.bin` acoustic history
- **Deep Learning Pipeline**: Complete PyTorch training with JEPA, LeWM, MAE, and more

---

## 🛠️ Quick Start

### 1. Rust Simulation (Bevy)

```bash
# Clone and run
git clone https://github.com/woodj/depth-learning.git
cd depth-learning
cargo run --release
```

### 2. Python Deep Learning Pipeline

```bash
cd ml
pip install -r requirements.txt

# Generate dataset first (see below)
# Then train
python3 train.py lewm --task presence --dataset easy --epochs 80

# Start inference server
python3 serve.py
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
| **M** | Cycle Difficulty: Easy → Medium → Hard → **EXTREME** |

---

## 📊 Dataset Generation

### Command-Line Generation (Recommended)

```bash
# Generate datasets for all difficulty levels
cargo run --bin generate_dataset -- -o dataset/easy -n 1000 -d easy
cargo run --bin generate_dataset -- -o dataset/medium -n 1000 -d medium
cargo run --bin generate_dataset -- -o dataset/hard -n 1000 -d hard
cargo run --bin generate_dataset -- -o dataset/extreme -n 1000 -d extreme
```

### In-Simulation Recording

```bash
cargo run --release

# In simulation:
# 1. Press M to select difficulty
# 2. Press R to start recording
# 3. Wait 2-3 minutes (~1000-2000 frames)
# 4. Press R to stop
# 5. Press M to change difficulty, repeat
```

### Dataset Structure

Each frame includes:

| File | Format | Description |
|------|--------|-------------|
| `_visual.png` | PNG (512×512) | Bird's-eye view RGB image |
| `_acoustic.png` | PNG (256×32) | Echogram visualization |
| `_history.bin` | Binary (24.6 KB) | 32-ping acoustic history (32×256×3) |
| `_meta.json` | JSON | Labels with species counts |

**Example metadata:**
```json
{
  "dominant_species": "Kingfish",
  "species_present": ["Kingfish", "Snapper"],
  "species_counts": {"Kingfish": 5, "Snapper": 2, "Cod": 0, "Empty": 0}
}
```

---

## 🎯 Difficulty Modes

| Mode | Depth Stratification | Heading Changes | School Behavior | Use Case |
|------|---------------------|-----------------|-----------------|----------|
| **Easy** | ✓ Strict | Never | All swim east together | Learning basic patterns |
| **Medium** | ✓ Strict | Every ~90s | Independent schools | Generalization testing |
| **Hard** | ✓ Strict | Every ~30s | Full flocking behavior | Realistic conditions |
| **EXTREME** | ❌ **Randomized** | Every ~15s | Chaotic (1.5× independence) | **Testing for shortcut learning** |

### Why EXTREME Mode Matters

In Easy/Medium/Hard modes, species have fixed depth ranges:
- **Kingfish**: 3.0-5.5 m (deep)
- **Snapper**: 2.0-5.0 m (mid)
- **Cod**: 0.5-3.0 m (shallow)

A model could achieve 100% accuracy by just learning **"deep = Kingfish"** without looking at acoustic features!

**EXTREME mode** randomizes depths across the full water column (0.5-9.5 m), forcing the model to learn **real acoustic signatures** instead of depth shortcuts.

**Test for shortcut learning:**
```
Train on Easy → Test on Extreme
├─ 100% on Easy, <50% on Extreme → Model cheated (used depth) ❌
└─ 85-95% on both → Model learned acoustic features ✓
```

---

## 🧠 Machine Learning Tasks

### 1. Presence/Absence Detection (Recommended)

**Task:** Which species are present in the echogram?

```bash
# Train JEPA
python3 train.py jepa --task presence --dataset extreme --epochs 80 --with-aug

# Train LeWM
python3 train.py lewm --task presence --dataset extreme --epochs 80 --with-aug
```

**Expected Performance:**
- Easy/Medium/Hard: 85-95% F1
- **EXTREME: 85-95% F1** (if model learned real features)

### 2. Fish Counting (Advanced)

**Task:** How many fish of each species are present?

```bash
python3 train.py lewm --task counting --dataset extreme --epochs 80 --with-aug
```

**Expected Performance:**
- MAE: 1-3 fish per species
- RMSE: 2-5 fish per species

---

## 🏋️ Training Examples

### Basic Training (Presence Detection)

```bash
cd ml

# Train on Easy dataset
python3 train.py lewm --task presence --dataset easy --epochs 80

# Train with augmentation (recommended)
python3 train.py lewm --task presence --dataset easy --epochs 80 --with-aug

# Train on EXTREME (best for generalization)
python3 train.py lewm --task presence --dataset extreme --epochs 80 --with-aug
```

### Curriculum Learning

```bash
# Progressive difficulty training
python3 train.py lewm --task presence --dataset easy --epochs 40
python3 train.py lewm --task presence --dataset medium --epochs 40
python3 train.py lewm --task presence --dataset hard --epochs 40
python3 train.py lewm --task presence --dataset extreme --epochs 40
```

### Available Models

```bash
# JEPA (Joint Embedding Predictive Architecture)
python3 train.py jepa --task presence --dataset extreme --epochs 80

# LeWM (LeWorldModel - recommended)
python3 train.py lewm --task presence --dataset extreme --epochs 80

# MAE (Masked Autoencoder - self-supervised pretraining)
python3 train.py mae --dataset extreme --epochs 100

# Fusion (multimodal visual+acoustic)
python3 train.py fusion --dataset extreme --epochs 50

# Decoder (image reconstruction from acoustic)
python3 train.py decoder --dataset extreme --epochs 50
```

---

## 🔬 Inference & Live Testing

### Start Inference Server

```bash
cd ml
python3 serve.py
```

### Test in Simulation

```bash
# In another terminal
cargo run --release

# The simulation will automatically connect to the server
# and display real-time predictions in Quadrant 3
```

**UI Display:**
- **Presence mode:** Shows probability bars with ✓ for detected species
- **Counting mode:** Shows estimated vs actual counts with error margins

---

## 📈 Expected Results

### Good Model (Learning Acoustic Features)

| Train Dataset | Test Easy | Test Extreme | Conclusion |
|--------------|-----------|--------------|------------|
| Easy | 95% | 50% ❌ | Used depth shortcut |
| Easy + Aug | 90% | 75% ✓ | Learning features |
| **Extreme** | **90%** | **90% ✓** | **Robust features** |

### Species Detection Performance

| Species | Easy | Medium | Hard | Extreme |
|---------|------|--------|------|---------|
| Kingfish | 90-95% | 85-90% | 80-85% | 85-90% |
| Snapper | 90-95% | 85-90% | 80-85% | 85-90% |
| Cod | 90-95% | 85-90% | 80-85% | 85-90% |
| Empty | 95-98% | 90-95% | 85-90% | 90-95% |

---

## 📂 Project Structure

```
depth-learning/
├── src/
│   ├── main.rs              # Bevy simulation
│   └── bin/
│       └── generate_dataset.rs  # Headless dataset generator
├── ml/
│   ├── models/
│   │   ├── jepa.py          # JEPA architecture
│   │   ├── lewm.py          # LeWorldModel
│   │   ├── lewm_multilabel.py  # Multi-task LeWM
│   │   ├── mae.py           # Masked Autoencoder
│   │   └── ...
│   ├── train.py             # Unified training script
│   ├── trainers.py          # Training loops
│   ├── data.py              # Data loading with augmentation
│   ├── serve.py             # FastAPI inference server
│   └── config.py            # Training configurations
├── dataset/
│   ├── easy/                # Easy difficulty data
│   ├── medium/              # Medium difficulty data
│   ├── hard/                # Hard difficulty data
│   └── extreme/             # EXTREME difficulty data (depth randomized)
└── weights/                 # Saved model checkpoints
```

---

## 🧪 Testing for Shortcut Learning

### The Depth Shortcut Problem

Hydroacoustic datasets often have **depth stratification**:
- Species live at specific depths
- Model can cheat by learning "depth = species"
- Works in lab, fails in real world

### The EXTREME Solution

Our **EXTREME mode** breaks depth stratification:
1. Fish spawn at random depths
2. Continuous vertical jitter during swimming
3. Depth teleportation when hitting bounds

### Validation Protocol

```bash
# 1. Generate EXTREME dataset
cargo run --bin generate_dataset -- -o dataset/extreme -n 2000 -d extreme

# 2. Train on EXTREME
python3 train.py lewm --task presence --dataset extreme --epochs 80 --with-aug

# 3. Test in simulation (press M to EXTREME)
cargo run --release

# 4. Compare accuracies:
#    - Easy/Medium/Hard: ~90%
#    - EXTREME: ~85-90% ← Model learned real features! ✓
```

---

## 🛠️ Troubleshooting

### Model gets 100% on training but <50% on Extreme test
**Problem:** Model learned depth shortcut  
**Solution:** Train on EXTREME dataset with `--with-aug`

### F1 score > 100%
**Problem:** Bug in metric calculation (should be fixed)  
**Solution:** Pull latest changes

### Simulation doesn't connect to server
**Problem:** Server not running or wrong port  
**Solution:** Run `python3 ml/serve.py` first

---

## 📜 License

MIT License - Created for sustainable fisheries research and conservation.

---

## 🙏 Acknowledgments

Built with:
- **Bevy** (Rust game engine) for 3D simulation
- **PyTorch** for deep learning
- **FastAPI** for inference serving
- **wandb** for experiment tracking

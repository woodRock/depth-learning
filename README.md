# 🐟 Multimodal Fish Simulation & Synthetic Dataset Generator

[![CI](https://github.com/woodj/depth-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/woodj/depth-learning/actions/workflows/ci.yml)
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
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run training (after generating data in the sim)
python train.py
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

---

## 🔬 Machine Learning Task

The primary goal of this project is to solve the **"Black Box" of hydroacoustics**. By training a **Contrastive CLIP-style model**, the acoustic encoder learns to "understand" the visual representation of fish, allowing for zero-shot species identification from raw echogram data.

---

## 📂 Project Structure

- `src/`: Core Rust logic, Bevy ECS systems, and echosounder simulation.
- `ml/`: PyTorch training scripts, dataset loaders, and model architectures.
- `dataset/`: (Generated) Paired visual and acoustic frames.
- `.github/workflows/`: CI/CD pipelines for Rust and Python.

---

## 📜 License
MIT - Created for the future of sustainable fisheries research.

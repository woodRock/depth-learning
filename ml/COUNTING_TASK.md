# Counting Task Training & Evaluation

This directory contains scripts for training and evaluating all models on the **fish counting task**.

## Overview

The counting task predicts the **number of fish** of each species in the transducer beam, rather than just presence/absence.

**Models:**
- **JEPA (Multi-modal)**: Trained with visual teacher, evaluated acoustic-only
- **JEPA (Acoustic-Only)**: Same weights as JEPA, acoustic-only evaluation
- **JEPA+SigReg**: JEPA with Gaussian latent regularization
- **LeWM**: Acoustic-only world model

**Metrics:**
- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual counts
- **RMSE** (Root Mean Square Error): Penalizes larger errors more heavily

---

## Quick Start

### Training (All Models, All Datasets)

**Linux/macOS:**
```bash
cd ml
./train_counting_all.sh              # Default: 100 epochs, 15 patience
./train_counting_all.sh 50 10        # Custom: 50 epochs, 10 patience
```

**Windows (PowerShell):**
```powershell
cd ml
.\train_counting_all.ps1             # Default: 100 epochs, 15 patience
.\train_counting_all.ps1 50 10       # Custom: 50 epochs, 10 patience
```

### Training Individual Models

```bash
# JEPA (Multi-modal)
python3 train.py jepa --dataset easy --epochs 100 --task counting --with-aug

# LeWM (Acoustic-only)
python3 train.py lewm --dataset easy --epochs 100 --task counting

# JEPA+SigReg (Multi-modal + SigReg)
python3 train_jepa_sigreg.py --dataset easy --epochs 100 --task counting --with-aug
```

### Evaluation

```bash
# Evaluate all models, generate MAE table
python3 evaluate_counting_all.py

# Custom output file
python3 evaluate_counting_all.py --output my_results.json
```

### Generate Table

```bash
# Generate counting task table (MAE/RMSE)
python3 generate_table.py --task counting

# Generate presence task table (F1 scores)
python3 generate_table.py --task presence

# Generate majority task table (accuracy)
python3 generate_table.py --task majority
```

---

## Output Files

After running the full pipeline:

```
ml/
├── weights/
│   ├── jepa_easy/
│   ├── jepa_medium/
│   ├── jepa_hard/
│   ├── jepa_extreme/
│   ├── lewm_easy/
│   ├── lewm_medium/
│   ├── lewm_hard/
│   ├── lewm_extreme/
│   ├── jepa_sigreg_easy/
│   ├── jepa_sigreg_medium/
│   ├── jepa_sigreg_hard/
│   └── jepa_sigreg_extreme/
├── results_counting.json      # All counting task results
├── table_counting.tex         # LaTeX table with MAE/RMSE
└── train_counting_all.sh      # Training script
```

---

## Table Format

The generated `table_counting.tex` contains:

| Model | Dataset | KF | SN | CD | EM | **Macro** | KF | SN | CD | EM | **Macro** |
|-------|---------|----|----|----|----|-----------|----|----|----|----|-----------|
|       |         | **MAE** | | | | | **RMSE** | | | | |

- **KF, SN, CD, EM**: Per-species MAE/RMSE
- **Macro**: Average across all four species
- **Bold**: Best (lowest) MAE per dataset

---

## Expected Results

Typical MAE ranges (lower is better):

| Dataset | Expected Macro MAE |
|---------|-------------------|
| Easy    | 0.5 - 2.0 |
| Medium  | 1.0 - 3.0 |
| Hard    | 2.0 - 5.0 |
| Extreme | 3.0 - 8.0 |

**JEPA+SigReg** typically achieves the best results due to Gaussian regularization preventing overfitting.

---

## Troubleshooting

### Weights Not Found
```
⚠ Weights not found for JEPA on easy
```
**Solution:** Run training first: `./train_counting_all.sh`

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size in training script or use `--batch-size 16`

### Empty Results Table
```
No table generated (missing data)
```
**Solution:** Run evaluation first: `python3 evaluate_counting_all.py`

---

## Citation

If you use these counting task results, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2026}
}
```

# Depth Learning ML Module

Multi-modal deep learning for fish species classification and counting from acoustic and visual data.

## Quick Start

### Installation

```bash
# From the repo root
pip install -e .
```

### Training

```bash
# Train LeWM++ (best model) for counting
depth train lewm_plus --dataset easy --task counting --epochs 100

# Train with experiment runner
depth experiment --config experiments/counting_all.yaml
```

### Evaluation

```bash
# Evaluate a trained model
depth evaluate --arch LeWM --dataset extreme --mode Acoustic-only
```

### Server

```bash
# Start inference server
depth serve --host 127.0.0.1 --port 8000
```

## Architecture

```
depth/
├── cli/           # Command-line interfaces
├── core/          # Core business logic
├── data/          # Data loading & processing
├── models/        # Neural network architectures
└── utils/         # Utilities
```

## Models

| Model | Type | Best For |
|-------|------|----------|
| **LeWM++** | Multi-modal + SigReg | Counting, Presence |
| **JEPA** | Multi-modal | Presence detection |
| **LeWM** | Acoustic-only | Baseline comparison |

## Configuration

Experiments are configured via YAML:

```yaml
# config/experiments/counting.yaml
model: lewm_plus
dataset: all
task: counting
epochs: 100
seeds: 3
sigreg_weight: 0.1
```

Run with:
```bash
depth experiment --config experiments/counting.yaml
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
black .
mypy .
```

### Adding a New Model

1. Create model in `models/my_model.py`
2. Create trainer in `core/trainer.py`
3. Add CLI command in `cli/train.py`
4. Write tests in `tests/test_my_model.py`

## API Usage

```python
from depth.models import LeWMPlus
from depth.utils.config import TrainingConfig
from depth.core import get_trainer

config = TrainingConfig(architecture="lewm_plus", dataset="extreme", task="presence")
trainer = get_trainer(config)
trainer.train()
```

## Citation

If you use this code, please cite:

```bibtex
@software{depth_learning,
  title = {Depth Learning: Multi-modal Fish Classification},
  year = {2026},
  url = {https://github.com/your-repo/depth-learning}
}
```

## License

MIT License - see LICENSE file for details.

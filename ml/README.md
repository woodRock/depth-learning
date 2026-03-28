# Depth Learning ML Module

Multi-modal deep learning for fish species classification and counting from acoustic and visual data.

## Quick Start

### Installation

```bash
cd ml
pip install -r requirements.txt
```

### Training

```bash
# Train LeWM++ (best model) for counting
python -m depth_learning.cli.train lewm_plus --dataset easy --task counting --epochs 100

# Train with experiment runner
python -m depth_learning.cli.experiment run --config config/experiments/counting_all.yaml
```

### Evaluation

```bash
# Evaluate all trained models
python -m depth_learning.cli.evaluate --all

# Generate results table
python -m depth_learning.cli.table --task counting
```

### Server

```bash
# Start inference server
python -m depth_learning.cli.serve --host 127.0.0.1 --port 8000
```

## Architecture

```
depth_learning/
├── cli/           # Command-line interfaces
├── core/          # Core business logic
├── data/          # Data loading & processing
├── models/        # Neural network architectures
├── server/        # API server
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
python -m depth_learning.cli.experiment run --config config/experiments/counting.yaml
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
from depth_learning import Trainer, Evaluator
from depth_learning.models import LeWMPlus

# Training
trainer = Trainer.from_config("config.yaml")
results = trainer.train()

# Evaluation
evaluator = Evaluator(results)
metrics = evaluator.compute_metrics()

# Export
from depth_learning.utils import export_latex_table
export_latex_table(metrics, "results.tex")
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

# ML Module Architecture Refactoring Plan

## Current State
```
ml/
├── train.py (551 lines)
├── trainers.py (1256 lines)
├── serve.py (603 lines)
├── experiment.py (471 lines)
├── evaluate_*.py (multiple files)
├── models/
└── ...
```

## Target Architecture

```
ml/
├── __init__.py                 # Package initialization
├── pyproject.toml              # Dependencies & metadata
├── README.md                   # Module documentation
│
├── cli/                        # Command-line interfaces
│   ├── __init__.py
│   ├── train.py                # Training CLI
│   ├── evaluate.py             # Evaluation CLI
│   ├── experiment.py           # Experiment runner CLI
│   └── serve.py                # Server CLI
│
├── core/                       # Core business logic
│   ├── __init__.py
│   ├── trainer.py              # Base trainer + implementations
│   ├── evaluator.py            # Evaluation logic
│   └── experiment.py           # Experiment orchestration
│
├── models/                     # Neural network architectures
│   ├── __init__.py
│   ├── encoders/               # Acoustic encoders
│   │   ├── __init__.py
│   │   ├── conv.py
│   │   ├── transformer.py
│   │   ├── lstm.py
│   │   └── ast.py
│   ├── jepa.py                 # JEPA model
│   ├── lewm.py                 # LeWM model
│   ├── lewm_plus.py            # LeWM++ model
│   └── ...
│
├── data/                       # Data loading & processing
│   ├── __init__.py
│   ├── dataset.py              # FishDataset class
│   ├── transforms.py           # Data transforms
│   └── splits.py               # Train/val/test splitting
│
├── server/                     # API server
│   ├── __init__.py
│   ├── app.py                  # FastAPI application
│   ├── routes.py               # API routes
│   └── inference.py            # Model inference logic
│
├── utils/                      # Utilities
│   ├── __init__.py
│   ├── logging.py              # Logging configuration
│   ├── config.py               # Configuration utilities
│   ├── metrics.py              # Metric calculations
│   └── io.py                   # File I/O utilities
│
├── config/                     # Configuration files
│   ├── default.yaml            # Default configuration
│   └── experiments/            # Experiment configurations
│
├── tests/                      # Unit & integration tests
│   ├── __init__.py
│   ├── test_trainers.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_experiment.py
│
└── scripts/                    # One-off scripts (not CLI)
    ├── analyze_dataset.py
    └── generate_table.py
```

## Key Improvements

### 1. **Proper Python Package**
- `__init__.py` files throughout
- Importable as `from depth_learning import trainer`
- Version management in `pyproject.toml`

### 2. **Separation of Concerns**
- **CLI** (`cli/`): Command-line interfaces only
- **Core** (`core/`): Business logic
- **Models** (`models/`): Neural network architectures
- **Data** (`data/`): Data loading
- **Server** (`server/`): API server
- **Utils** (`utils/`): Shared utilities

### 3. **File Size Limits**
- No file > 500 lines
- Trainers split by model type
- Routes separated from app logic

### 4. **Logging**
```python
# utils/logging.py
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Configure handlers...
    return logger

# Usage in trainers.py
from ..utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Training started")
```

### 5. **Type Hints**
```python
# Before
def train_epoch(self, loader):
    return {"loss": 0.5, "f1": 0.8}

# After
def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
    return {"loss": 0.5, "f1": 0.8}
```

### 6. **Error Handling**
```python
# Before
try:
    model.load_state_dict(weights)
except:
    print("Error loading weights")

# After
class ModelLoadError(Exception):
    pass

def load_model_weights(model: nn.Module, path: Path) -> nn.Module:
    if not path.exists():
        raise ModelLoadError(f"Weights not found: {path}")
    try:
        model.load_state_dict(torch.load(path))
        return model
    except Exception as e:
        raise ModelLoadError(f"Failed to load weights: {e}") from e
```

### 7. **Configuration Validation**
```yaml
# config/default.yaml
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0003
  early_stopping:
    enabled: true
    patience: 15
```

```python
# utils/config.py
from pydantic import BaseModel, validator

class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    
    @validator('learning_rate')
    def check_lr(cls, v):
        if v <= 0 or v > 1:
            raise ValueError('Learning rate must be in (0, 1]')
        return v
```

### 8. **Testing**
```python
# tests/test_trainers.py
import pytest
from depth_learning.core.trainer import JEPATrainer

def test_jepa_trainer_init():
    trainer = JEPATrainer(config, device)
    assert trainer is not None

def test_training_step():
    # Test single training step
    pass
```

## Migration Plan

### Phase 1: Foundation (1-2 days)
1. Create directory structure
2. Add `__init__.py` files
3. Create `pyproject.toml`
4. Set up logging utility

### Phase 2: Core Refactoring (3-4 days)
1. Split `trainers.py` into model-specific files
2. Move evaluation logic to `core/evaluator.py`
3. Extract data loading to `data/` module
4. Split `serve.py` into server module

### Phase 3: CLI Separation (1-2 days)
1. Move CLI code to `cli/`
2. Use `click` or `typer` for better CLI
3. Ensure CLI imports from core

### Phase 4: Quality Improvements (2-3 days)
1. Add type hints throughout
2. Add comprehensive logging
3. Write unit tests
4. Add configuration validation

### Phase 5: Documentation (1 day)
1. Docstrings for all public APIs
2. README for each module
3. Usage examples

## Benefits

| Benefit | Impact |
|---------|--------|
| **Maintainability** | Easier to find and fix bugs |
| **Testability** | Can test components in isolation |
| **Extensibility** | Easy to add new models/features |
| **Onboarding** | New developers can understand quickly |
| **Reliability** | Type hints catch errors early |
| **Debugging** | Logging helps diagnose issues |
| **Reusability** | Core logic can be imported elsewhere |

## Immediate Actions (Quick Wins)

1. **Delete `old_scripts/`** - Archive to separate repo if needed
2. **Add `__init__.py`** to ml/ directory
3. **Create `utils/logging.py`** - Add logging everywhere
4. **Split `trainers.py`** - By model type (JEPATrainer, LeWMTrainer, LeWMPlusTrainer)
5. **Add type hints** to function signatures
6. **Create `requirements.txt`** - Document dependencies

## Long-term Vision

```python
# Clean, professional usage
from depth_learning import Trainer, ExperimentRunner
from depth_learning.models import LeWMPlus
from depth_learning.data import FishDataset

# Training
trainer = Trainer.from_config("config/experiments/counting.yaml")
results = trainer.run()

# Evaluation
evaluator = Evaluator(results)
metrics = evaluator.compute_metrics()

# Export
report = Report(metrics)
report.to_latex("table.tex")
```

# Experiment Runner

A unified experiment orchestration system for depth learning research.

## Features

- ✅ **Single Python Script** - Replaces bash/ps1/bat duplicates
- ✅ **YAML Configuration** - Declarative experiment definitions
- ✅ **Multiple Seeds** - N runs with different random seeds
- ✅ **Model Comparison** - Evaluate different models on same datasets
- ✅ **Dry Run Mode** - Preview what would be executed
- ✅ **Automatic Aggregation** - Collect results from all runs

## Quick Start

### Run Single Experiment

```bash
cd ml

# Train LeWM++ on easy dataset for counting task
python experiment.py run --model lewm_plus --dataset easy --task counting --epochs 100

# Train with 5 different seeds
python experiment.py run --model lewm_plus --dataset easy --task counting --seeds 5
```

### Run Multiple Experiments (YAML Config)

```bash
# Run all counting experiments (3 models × 4 datasets × 3 seeds = 36 runs)
python experiment.py run --config experiments/counting_all.yaml

# Dry run first to see what would be executed
python experiment.py run --config experiments/counting_all.yaml --dry-run
```

### Evaluate & Generate Tables

```bash
# Evaluate all trained models
python experiment.py evaluate --all

# Generate LaTeX table
python experiment.py table --task counting
```

## YAML Configuration Format

```yaml
name: counting_all
experiments:
  - model: lewm_plus
    dataset: all  # or: easy, medium, hard, extreme
    task: counting
    epochs: 100
    seeds: 3  # Number of runs with different seeds
    sigreg_weight: 0.1
    
  - model: jepa
    dataset: all
    task: counting
    epochs: 100
    seeds: 3
    
  - model: lewm
    dataset: all
    task: counting
    epochs: 100
    seeds: 3
```

## Command Reference

### `run` - Execute Experiments

```bash
# Single experiment
python experiment.py run --model <model> --dataset <dataset> --task <task> [options]

# From YAML config
python experiment.py run --config <config.yaml> [options]

# Options:
  --model       jepa | lewm | lewm_plus
  --dataset     easy | medium | hard | extreme | all
  --task        presence | counting | single_label
  --epochs      Number of epochs (default: 100)
  --seeds       Number of runs with different seeds (default: 1)
  --dry-run     Show what would be executed
```

### `evaluate` - Evaluate Trained Models

```bash
# Evaluate all trained models
python experiment.py evaluate --all

# Options:
  --all         Evaluate all models
  --dry-run     Show what would be executed
```

### `table` - Generate Results Table

```bash
# Generate LaTeX table
python experiment.py table --task counting

# Options:
  --task        presence | counting | majority
  --dry-run     Show what would be executed
```

### `init` - Create Config Templates

```bash
# Create experiment config templates
python experiment.py init --output experiments

# Creates:
#   experiments/counting_all.yaml
#   experiments/presence_all.yaml
```

## Pre-configured Experiments

### Counting Task (All Models)

```bash
python experiment.py run --config experiments/counting_all.yaml
```

Trains:
- JEPA on easy/medium/hard/extreme (3 seeds each)
- LeWM on easy/medium/hard/extreme (3 seeds each)
- LeWM++ on easy/medium/hard/extreme (3 seeds each)

**Total:** 36 training runs

### Presence Task (All Models)

```bash
python experiment.py run --config experiments/presence_all.yaml
```

Same as above but for presence/absence detection task.

## Advanced Usage

### Custom Seed Configuration

```yaml
experiments:
  - model: lewm_plus
    dataset: easy
    task: counting
    seeds: [42, 43, 44, 123, 456]  # Specific seeds
```

### Custom Hyperparameters

```yaml
experiments:
  - model: lewm_plus
    dataset: all
    task: counting
    epochs: 150
    batch_size: 16
    learning_rate: 0.0001
    sigreg_weight: 0.5
    patience: 20
```

### Mixed Experiments

```yaml
experiments:
  # Quick test on easy
  - model: lewm_plus
    dataset: easy
    task: counting
    epochs: 10
    seeds: 1
    
  # Full evaluation on extreme
  - model: lewm_plus
    dataset: extreme
    task: counting
    epochs: 100
    seeds: 5
```

## Output Structure

```
ml/
├── experiment.py              # Experiment runner
├── experiments/               # YAML configurations
│   ├── counting_all.yaml
│   └── presence_all.yaml
├── weights/                   # Trained models
│   ├── lewm_plus_easy_seed42/
│   ├── lewm_plus_easy_seed43/
│   └── ...
└── results.json              # Aggregated results
```

## Error Handling

- **Timeout:** Experiments timeout after 12 hours
- **Failed Runs:** Logged but don't stop other experiments
- **Dry Run:** Always use `--dry-run` first to verify configuration

## Best Practices

1. **Always dry-run first:**
   ```bash
   python experiment.py run --config experiments/counting_all.yaml --dry-run
   ```

2. **Start small:**
   ```bash
   # Test with 1 seed on 1 dataset
   python experiment.py run --model lewm_plus --dataset easy --seeds 1
   ```

3. **Use config files for large experiments:**
   ```bash
   # Don't run 36 experiments from command line
   python experiment.py run --config experiments/counting_all.yaml
   ```

4. **Monitor disk space:**
   - Each model: ~350MB
   - Full experiment (36 runs): ~12GB

## Migration from Shell Scripts

**Old (bash):**
```bash
./train_counting_all.sh 100 15
```

**New (Python):**
```bash
python experiment.py run --config experiments/counting_all.yaml
```

**Benefits:**
- Cross-platform (no bash/ps1/bat duplicates)
- Type-safe configuration
- Better error handling
- Automatic result aggregation
- Easier to extend

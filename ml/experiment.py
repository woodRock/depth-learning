#!/usr/bin/env python3
"""
Experiment Runner for Depth Learning

A unified experiment orchestration system that supports:
- Multiple models (JEPA, LeWM, LeWM++)
- Multiple tasks (presence, counting, single_label)
- Multiple datasets (easy, medium, hard, extreme)
- N runs with different random seeds
- YAML configuration files
- Command-line arguments
- Automatic result aggregation

Usage:
    # Run single experiment
    python experiment.py run --model lewm_plus --dataset easy --task counting --epochs 100
    
    # Run with config file
    python experiment.py run --config experiments/counting_all.yaml
    
    # Run multiple seeds
    python experiment.py run --model lewm_plus --dataset all --seeds 5
    
    # Evaluate all trained models
    python experiment.py evaluate --all
    
    # Generate results table
    python experiment.py table --task counting

Author: Depth Learning Team
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import hashlib


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    model: str
    dataset: str
    task: str = "presence"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    patience: int = 15
    sigreg_weight: float = 0.1
    with_aug: bool = True
    seeds: List[int] = field(default_factory=lambda: [42])
    
    def to_args(self, seed: int, weights_dir: str) -> List[str]:
        """Convert config to command line arguments."""
        args = [
            sys.executable, "train.py", self.model,
            "--dataset", self.dataset,
            "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size),
            "--lr", str(self.learning_rate),
            "--weight-decay", str(self.weight_decay),
            "--patience", str(self.patience),
            "--sigreg-weight", str(self.sigreg_weight),
            "--task", self.task,
            "--weights-dir", weights_dir,
            "--seed", str(seed),  # Note: requires --seed support in train.py
        ]
        
        if self.with_aug:
            args.append("--with-aug")
        
        return args


@dataclass
class ExperimentRunnerConfig:
    """Configuration for the experiment runner."""
    name: str = "experiment"
    base_dir: str = "experiments"
    ml_dir: str = "ml"
    results_file: str = "results.json"
    default_epochs: int = 100
    default_patience: int = 15
    default_seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])


# =============================================================================
# Experiment Runner
# =============================================================================

class ExperimentRunner:
    """Orchestrates machine learning experiments."""
    
    def __init__(self, config: ExperimentRunnerConfig):
        self.config = config
        self.ml_path = Path(config.ml_dir)
        self.results_path = self.ml_path / config.results_file
        
        # Create directories
        Path(config.base_dir).mkdir(exist_ok=True)
        
    def run_experiment(self, exp_config: ExperimentConfig, dry_run: bool = False) -> Dict[str, Any]:
        """Run a single experiment configuration."""
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_config.model} on {exp_config.dataset} ({exp_config.task})")
        print(f"{'='*70}")
        
        results = {
            "model": exp_config.model,
            "dataset": exp_config.dataset,
            "task": exp_config.task,
            "runs": [],
        }
        
        for seed in exp_config.seeds:
            run_result = self._run_single_seed(exp_config, seed, dry_run)
            results["runs"].append(run_result)
            
            if run_result["success"] and not dry_run:
                print(f"  ✓ Seed {seed}: Complete")
            elif not dry_run:
                print(f"  ✗ Seed {seed}: Failed")
        
        return results
    
    def _run_single_seed(self, exp_config: ExperimentConfig, seed: int, dry_run: bool) -> Dict[str, Any]:
        """Run experiment with a single seed."""
        # Generate weights directory name with seed hash
        seed_hash = hashlib.md5(f"{exp_config.model}_{exp_config.dataset}_{seed}".encode()).hexdigest()[:8]
        weights_dir = f"weights/{exp_config.model}_{exp_config.dataset}_seed{seed}"
        
        run_result = {
            "seed": seed,
            "weights_dir": weights_dir,
            "success": False,
            "duration": None,
        }
        
        if dry_run:
            print(f"  [DRY RUN] Seed {seed}: {exp_config.model} --dataset {exp_config.dataset}")
            run_result["success"] = True
            return run_result
        
        # Change to ml directory
        original_dir = os.getcwd()
        os.chdir(self.ml_path)
        
        try:
            start_time = datetime.datetime.now()
            
            # Build command
            cmd = exp_config.to_args(seed, weights_dir)
            
            # Run training
            print(f"  Running seed {seed}...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 12,  # 12 hour timeout
            )
            
            end_time = datetime.datetime.now()
            run_result["duration"] = (end_time - start_time).total_seconds()
            
            if result.returncode == 0:
                run_result["success"] = True
            else:
                print(f"    Error: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            print(f"    Timeout after 12 hours")
        except Exception as e:
            print(f"    Exception: {e}")
        finally:
            os.chdir(original_dir)
        
        return run_result
    
    def run_from_yaml(self, yaml_path: str, dry_run: bool = False) -> List[Dict[str, Any]]:
        """Run experiments from YAML configuration file."""
        print(f"\nLoading configuration from: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        all_results = []
        
        # Parse experiments from config
        experiments = config.get("experiments", [])
        
        for exp_data in experiments:
            # Handle dataset: "all" expands to all datasets
            datasets = exp_data.get("dataset", "easy")
            if datasets == "all":
                datasets = ["easy", "medium", "hard", "extreme"]
            elif isinstance(datasets, str):
                datasets = [datasets]
            
            # Handle seeds
            seeds = exp_data.get("seeds", self.config.default_seeds)
            if isinstance(seeds, int):
                # N runs with different seeds
                seeds = self.config.default_seeds[:seeds]
            
            for dataset in datasets:
                exp_config = ExperimentConfig(
                    model=exp_data.get("model", "lewm"),
                    dataset=dataset,
                    task=exp_data.get("task", "presence"),
                    epochs=exp_data.get("epochs", self.config.default_epochs),
                    seeds=seeds,
                    sigreg_weight=exp_data.get("sigreg_weight", 0.1),
                )
                
                results = self.run_experiment(exp_config, dry_run)
                all_results.append(results)
        
        return all_results
    
    def evaluate_all(self, dry_run: bool = False) -> None:
        """Evaluate all trained models."""
        print("\n" + "="*70)
        print("Evaluating All Trained Models")
        print("="*70)
        
        if dry_run:
            print("[DRY RUN] Would evaluate all models")
            return
        
        # Change to ml directory
        original_dir = os.getcwd()
        os.chdir(self.ml_path)
        
        try:
            # Run counting evaluation
            print("\nRunning counting task evaluation...")
            subprocess.run([sys.executable, "evaluate_counting_all.py"], check=True)
            
            # Generate table
            print("\nGenerating results table...")
            subprocess.run([sys.executable, "generate_table.py", "--task", "counting"], check=True)
            
            print("\n✓ Evaluation complete!")
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Evaluation failed: {e}")
        finally:
            os.chdir(original_dir)
    
    def generate_table(self, task: str = "counting", dry_run: bool = False) -> None:
        """Generate LaTeX table from results."""
        print(f"\nGenerating {task} task table...")
        
        if dry_run:
            print(f"[DRY RUN] python generate_table.py --task {task}")
            return
        
        # Change to ml directory
        original_dir = os.getcwd()
        os.chdir(self.ml_path)
        
        try:
            subprocess.run([sys.executable, "generate_table.py", "--task", task], check=True)
            print(f"✓ Table generated: table_{task}.tex")
        except subprocess.CalledProcessError as e:
            print(f"✗ Table generation failed: {e}")
        finally:
            os.chdir(original_dir)


# =============================================================================
# Pre-configured Experiment Templates
# =============================================================================

def create_counting_all_config() -> Dict:
    """Create configuration for counting task on all models/datasets."""
    return {
        "name": "counting_all",
        "experiments": [
            {
                "model": "jepa",
                "dataset": "all",
                "task": "counting",
                "epochs": 100,
                "seeds": 3,  # Use first 3 default seeds
            },
            {
                "model": "lewm",
                "dataset": "all",
                "task": "counting",
                "epochs": 100,
                "seeds": 3,
            },
            {
                "model": "lewm_plus",
                "dataset": "all",
                "task": "counting",
                "epochs": 100,
                "seeds": 3,
                "sigreg_weight": 0.1,
            },
        ]
    }


def create_presence_all_config() -> Dict:
    """Create configuration for presence task on all models/datasets."""
    return {
        "name": "presence_all",
        "experiments": [
            {
                "model": "jepa",
                "dataset": "all",
                "task": "presence",
                "epochs": 100,
                "seeds": 3,
            },
            {
                "model": "lewm",
                "dataset": "all",
                "task": "presence",
                "epochs": 100,
                "seeds": 3,
            },
            {
                "model": "lewm_plus",
                "dataset": "all",
                "task": "presence",
                "epochs": 100,
                "seeds": 3,
            },
        ]
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment Runner for Depth Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python experiment.py run --model lewm_plus --dataset easy --task counting
  
  # Run with config file
  python experiment.py run --config experiments/counting_all.yaml
  
  # Run multiple seeds
  python experiment.py run --model lewm_plus --dataset all --seeds 5
  
  # Dry run (show what would be executed)
  python experiment.py run --model lewm_plus --dataset easy --dry-run
  
  # Evaluate all trained models
  python experiment.py evaluate --all
  
  # Generate results table
  python experiment.py table --task counting
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--config", type=str, help="YAML configuration file")
    run_parser.add_argument("--model", type=str, choices=["jepa", "lewm", "lewm_plus"],
                           help="Model to train")
    run_parser.add_argument("--dataset", type=str, 
                           choices=["easy", "medium", "hard", "extreme", "all"],
                           help="Dataset to train on")
    run_parser.add_argument("--task", type=str, 
                           choices=["presence", "counting", "single_label"],
                           default="presence", help="Task type")
    run_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    run_parser.add_argument("--seeds", type=int, default=1, 
                           help="Number of runs with different seeds")
    run_parser.add_argument("--dry-run", action="store_true", 
                           help="Show what would be executed")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument("--all", action="store_true", help="Evaluate all models")
    eval_parser.add_argument("--dry-run", action="store_true", help="Dry run")
    
    # Table command
    table_parser = subparsers.add_parser("table", help="Generate results table")
    table_parser.add_argument("--task", type=str, 
                             choices=["presence", "counting", "majority"],
                             default="counting", help="Task type for table")
    table_parser.add_argument("--dry-run", action="store_true", help="Dry run")
    
    # Init command (create config templates)
    init_parser = subparsers.add_parser("init", help="Create experiment config templates")
    init_parser.add_argument("--output", type=str, default="experiments",
                            help="Output directory for config files")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner_config = ExperimentRunnerConfig()
    runner = ExperimentRunner(runner_config)
    
    if args.command == "run":
        if args.config:
            # Run from YAML config
            runner.run_from_yaml(args.config, args.dry_run)
        elif args.model:
            # Run single experiment
            datasets = ["easy", "medium", "hard", "extreme"] if args.dataset == "all" else [args.dataset]
            seeds = runner_config.default_seeds[:args.seeds]
            
            for dataset in datasets:
                exp_config = ExperimentConfig(
                    model=args.model,
                    dataset=dataset,
                    task=args.task,
                    epochs=args.epochs,
                    seeds=seeds,
                )
                runner.run_experiment(exp_config, args.dry_run)
        else:
            run_parser.print_help()
    
    elif args.command == "evaluate":
        if args.all:
            runner.evaluate_all(args.dry_run)
        else:
            eval_parser.print_help()
    
    elif args.command == "table":
        runner.generate_table(args.task, args.dry_run)
    
    elif args.command == "init":
        # Create config templates
        Path(args.output).mkdir(exist_ok=True)
        
        configs = {
            "counting_all.yaml": create_counting_all_config(),
            "presence_all.yaml": create_presence_all_config(),
        }
        
        for filename, config_data in configs.items():
            filepath = Path(args.output) / filename
            with open(filepath, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"Created: {filepath}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

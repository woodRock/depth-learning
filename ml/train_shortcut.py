#!/usr/bin/env python3
"""
Training script for shortcut test evaluation.

This script trains models and saves them to dataset-specific directories
for use with shortcut_test.py.

Usage:
    # Train JEPA on Easy for 10 epochs
    python train_shortcut.py jepa --dataset easy --epochs 10
    
    # Train LeWM on Extreme for 10 epochs
    python train_shortcut.py lewm --dataset extreme --epochs 10
    
    # Train all 4 combinations
    python train_shortcut.py all --epochs 10
"""

import os
import sys
import subprocess
from pathlib import Path


def train_model(architecture: str, dataset: str, epochs: int = 10):
    """Train a model and save to dataset-specific directory."""
    
    # Create dataset-specific weights directory
    weights_dir = Path(__file__).parent / "weights" / f"{architecture.lower()}_{dataset}"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training {architecture} on {dataset} for {epochs} epochs")
    print(f"Weights will be saved to: {weights_dir}")
    print(f"{'='*60}\n")
    
    # Build training command
    cmd = [
        sys.executable, str(Path(__file__).parent / "train.py"),
        architecture.lower(),
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--weights-dir", str(weights_dir),
    ]
    
    # Add architecture-specific options
    if architecture == "JEPA":
        cmd.extend(["--model", "transformer"])
        cmd.extend(["--with-aug"])  # Enable augmentation for JEPA
    
    print(f"Running: {' '.join(cmd)}\n")
    
    # Run training
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print(f"\n✓ Training complete for {architecture} on {dataset}")
        print(f"  Weights saved to: {weights_dir}")
    else:
        print(f"\n✗ Training failed for {architecture} on {dataset}")
    
    return result.returncode


def train_all(epochs: int = 10):
    """Train all 4 combinations for shortcut testing."""
    
    combinations = [
        ("JEPA", "easy"),
        ("JEPA", "extreme"),
        ("LeWM", "easy"),
        ("LeWM", "extreme"),
    ]
    
    results = {}
    for arch, dataset in combinations:
        retcode = train_model(arch, dataset, epochs)
        results[f"{arch}_{dataset}"] = retcode
    
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    for key, retcode in results.items():
        status = "✓ Success" if retcode == 0 else "✗ Failed"
        print(f"  {key}: {status}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train models for shortcut test evaluation"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Training mode")
    
    # JEPA parser
    jepa_parser = subparsers.add_parser("jepa", help="Train JEPA model")
    jepa_parser.add_argument("--dataset", type=str, required=True,
                            choices=["easy", "medium", "hard", "extreme"])
    jepa_parser.add_argument("--epochs", type=int, default=10)
    jepa_parser.set_defaults(arch="JEPA")
    
    # LeWM parser
    lewm_parser = subparsers.add_parser("lewm", help="Train LeWM model")
    lewm_parser.add_argument("--dataset", type=str, required=True,
                            choices=["easy", "medium", "hard", "extreme"])
    lewm_parser.add_argument("--epochs", type=int, default=10)
    lewm_parser.set_defaults(arch="LeWM")
    
    # All parser
    all_parser = subparsers.add_parser("all", help="Train all 4 combinations")
    all_parser.add_argument("--epochs", type=int, default=10)
    
    args = parser.parse_args()
    
    if args.command == "all":
        train_all(args.epochs)
    elif args.command in ["jepa", "lewm"]:
        train_model(args.arch, args.dataset, args.epochs)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python train_shortcut.py jepa --dataset easy --epochs 10")
        print("  python train_shortcut.py lewm --dataset extreme --epochs 10")
        print("  python train_shortcut.py all --epochs 10")


if __name__ == "__main__":
    main()

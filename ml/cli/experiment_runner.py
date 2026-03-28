"""
Experiment runner for YAML-based batch training.
"""

import os
import yaml
import argparse
import torch
from typing import List, Dict, Any
from pathlib import Path

from cli.train import run_training
from utils.config import (
    TrainingConfig,
    DecoderConfig,
    FusionConfig,
    TranslatorConfig,
    MAEConfig,
)
from utils.logging import get_logger

logger = get_logger(__name__)

def run_experiments_from_yaml(yaml_path: str):
    """Parse YAML and run experiments."""
    if not os.path.exists(yaml_path):
        logger.error(f"Config file not found: {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)

    experiments = config_data.get("experiments", [])
    logger.info(f"Loaded {len(experiments)} experiment templates from {yaml_path}")

    for exp in experiments:
        # Expand datasets if "all"
        datasets = exp.get("dataset", "easy")
        if datasets == "all":
            datasets = ["easy", "medium", "hard", "extreme"]
        elif isinstance(datasets, str):
            datasets = [datasets]
        
        # Expand seeds
        seeds = exp.get("seeds", [42])
        if isinstance(seeds, int):
            # If a number is provided, generate that many seeds
            seeds = [42 + i for i in range(seeds)]
        
        model_name = exp.get("model")
        if not model_name:
            logger.error("Experiment missing 'model' field")
            continue

        for dataset in datasets:
            for seed in seeds:
                logger.info(f"Running experiment: {model_name} on {dataset} (seed {seed})")
                
                # Construct mock args and config
                mock_args = argparse.Namespace()
                mock_args.command = model_name
                mock_args.dataset = dataset
                mock_args.seed = seed
                mock_args.task = exp.get("task", "presence")
                mock_args.weights_dir = f"weights/{model_name}_{dataset}_seed{seed}"
                
                # Create specific config object based on model name
                if model_name in ["jepa", "lewm", "lewm_plus"]:
                    # These use TrainingConfig
                    # Map exp keys to config fields
                    conf = TrainingConfig(
                        model_type=exp.get("model_type", "transformer"),
                        epochs=exp.get("epochs", 80),
                        batch_size=exp.get("batch_size", 32),
                        learning_rate=exp.get("lr", 3e-4),
                        dataset=dataset,
                        sigreg_weight=exp.get("sigreg_weight", 0.1),
                        with_aug=exp.get("with_aug", True)
                    )
                    job_type = f"exp-{model_name}-{dataset}-{mock_args.task}"
                elif model_name == "decoder":
                    conf = DecoderConfig(
                        dataset=dataset,
                        epochs=exp.get("epochs", 50),
                        batch_size=exp.get("batch_size", 16),
                        learning_rate=exp.get("lr", 1e-3)
                    )
                    job_type = "exp-decoder"
                elif model_name == "fusion":
                    conf = FusionConfig(
                        dataset=dataset,
                        epochs=exp.get("epochs", 50),
                        batch_size=exp.get("batch_size", 32),
                        learning_rate=exp.get("lr", 1e-4),
                        dropout_prob=exp.get("dropout_prob", 0.5)
                    )
                    job_type = "exp-fusion"
                elif model_name == "translator":
                    conf = TranslatorConfig(
                        dataset=dataset,
                        epochs=exp.get("epochs", 100),
                        batch_size=exp.get("batch_size", 16),
                        learning_rate=exp.get("lr", 1e-4)
                    )
                    job_type = "exp-translator"
                elif model_name == "mae":
                    conf = MAEConfig(
                        dataset=dataset,
                        epochs=exp.get("epochs", 100),
                        batch_size=exp.get("batch_size", 64),
                        learning_rate=exp.get("lr", 1e-3)
                    )
                    job_type = "exp-mae"
                else:
                    logger.warning(f"Unknown model type in experiment: {model_name}")
                    continue

                # Run training
                try:
                    run_training(mock_args, conf, job_type)
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    import traceback
                    traceback.print_exc()

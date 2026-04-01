#!/usr/bin/env python3
"""
SNR Ablation Study script.
"""

import os
import sys
import argparse
import torch
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli.train import run_training
from utils.config import TrainingConfig, TranslatorConfig
from utils.logging import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="SNR Ablation Study")
    parser.add_argument("-n", "--epochs", type=int, default=10, help="Number of epochs per model")
    parser.add_argument("--dataset", type=str, default="easy", choices=["easy", "medium", "hard", "extreme"], help="Dataset to use")
    parser.add_argument("--task", type=str, default="presence", choices=["presence", "majority", "counting"], help="Task to perform")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    snr_levels = [0, 5, 10, 15, 20]
    models = ["lewm", "translator"]
    
    for model_name in models:
        for snr in snr_levels:
            logger.info(f"--- Ablation Study: Model={model_name}, SNR={snr}dB, Task={args.task}, Epochs={args.epochs} ---")
            
            # Create a mock args namespace for run_training
            mock_args = argparse.Namespace()
            mock_args.command = model_name
            mock_args.dataset = args.dataset
            mock_args.task = args.task
            mock_args.epochs = args.epochs
            mock_args.batch_size = args.batch_size
            mock_args.lr = 1e-4 if model_name == "translator" else 3e-4
            mock_args.with_aug = False 
            mock_args.light_aug = False
            mock_args.rotation_degrees = 30
            mock_args.seed = 42
            mock_args.snr_db = float(snr)
            mock_args.weights_dir = f"weights/ablation_{model_name}_{args.task}_snr{snr}"
            
            if model_name == "lewm":
                mock_args.model = "lewm"
                config = TrainingConfig.from_args(mock_args)
                job_type = f"ablation-lewm-{args.task}-snr{snr}"
            else: # translator
                config = TranslatorConfig(
                    dataset=args.dataset,
                    with_aug=False,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=mock_args.lr,
                    d_model=256,
                    patch_size=16,
                    task=args.task,
                    snr_db=float(snr),
                    weights_dir=mock_args.weights_dir
                )
                job_type = f"ablation-translator-{args.task}-snr{snr}"
            
            # Run the training
            try:
                run_training(mock_args, config, job_type)
            except Exception as e:
                logger.error(f"Error during ablation for {model_name} at {snr}dB: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Unified training script for all depth learning models.
"""

import os
import sys
import argparse
import torch
import wandb
from typing import Optional, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    TrainingConfig,
    DecoderConfig,
    FusionConfig,
    TranslatorConfig,
    MAEConfig,
    add_common_args,
)
from data import create_visual_transform, AugmentationConfig, create_data_loaders, ImageLatentDataset
from core import get_trainer
from utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def setup_wandb(config: Any, job_type: str, architecture: str, task: str) -> None:
    """Initialize wandb with configuration."""
    wandb.init(
        entity=getattr(config, 'wandb_entity', 'victoria-university-of-wellington'),
        project=getattr(config, 'wandb_project', 'depth-learning'),
        job_type=job_type,
        reinit=True,
        resume="allow",
        config={
            **vars(config),
            "architecture": architecture,
            "task": task,
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        },
    )


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _get_dataset_path(dataset_name: str) -> str:
    """Get absolute path to dataset."""
    ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(ml_dir)
    return os.path.join(project_root, "dataset", dataset_name)


def run_training(args: argparse.Namespace, config: Any, job_type: str) -> None:
    """Generic training runner."""
    device = _get_device()
    
    # Set weights directory
    if hasattr(args, 'weights_dir') and args.weights_dir is not None:
        config.weights_dir = args.weights_dir
    elif not hasattr(config, 'weights_dir') or config.weights_dir == "weights":
        # Default specialized directory
        model_name = args.command
        config.weights_dir = f"weights/{model_name}_{getattr(config, 'dataset', 'default')}"

    print(f"--- Starting {args.command.upper()} Training on {device} ---")
    task = getattr(args, 'task', 'presence')
    setup_wandb(config, job_type=job_type, architecture=args.command, task=task)

    # Create data loaders
    aug_config = AugmentationConfig(
        enabled=getattr(config, 'with_aug', False),
        light=getattr(config, 'light_aug', False),
        rotation_degrees=getattr(config, 'rotation_degrees', 30),
    )
    transform = create_visual_transform(aug_config)
    dataset_path = _get_dataset_path(getattr(config, 'dataset', 'easy'))

    # Special dataset for Decoder
    if args.command == "decoder":
        from torch.utils.data import DataLoader, Subset
        full_ds = ImageLatentDataset(dataset_path, transform=transform)
        train_size = int(0.9 * len(full_ds))
        train_ds = Subset(full_ds, range(train_size))
        val_ds = Subset(full_ds, range(train_size, len(full_ds)))
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    else:
        train_loader, val_loader = create_data_loaders(
            dataset_path, 
            transform=transform, 
            batch_size=config.batch_size,
            seed=getattr(args, 'seed', 42)
        )

    # Get trainer and setup optimizer
    trainer = get_trainer(config, device)
    trainer.model = trainer.build_model() # Some trainers use task inside build_model, others don't
    if hasattr(trainer.model, 'task'):
        trainer.model.task = task
        
    trainer.optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.model.parameters()),
        lr=config.learning_rate,
        weight_decay=getattr(config, 'weight_decay', 0.05),
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=config.epochs
    )

    trainer.train(train_loader, val_loader)
    wandb.finish()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified training script for depth learning models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Model type to train")
    
    # JEPA
    jepa_parser = subparsers.add_parser("jepa", help="Train JEPA model")
    jepa_parser.add_argument("--model", type=str, default="transformer", choices=["conv", "transformer", "lstm", "ast"])
    jepa_parser.add_argument("--epochs", type=int, default=80)
    jepa_parser.add_argument("--lr", type=float, default=3e-4)
    jepa_parser.add_argument("--batch-size", type=int, default=32)
    jepa_parser.add_argument("--weight-decay", type=float, default=0.05)
    jepa_parser.add_argument("--task", type=str, default="presence", choices=["presence", "single_label", "counting"])
    add_common_args(jepa_parser)
    
    # LeWM
    lewm_parser = subparsers.add_parser("lewm", help="Train LeWorldModel")
    lewm_parser.add_argument("--epochs", type=int, default=100)
    lewm_parser.add_argument("--lr", type=float, default=3e-4)
    lewm_parser.add_argument("--batch-size", type=int, default=32)
    lewm_parser.add_argument("--task", type=str, default="presence", choices=["presence", "counting"])
    add_common_args(lewm_parser)

    # LeWM++
    lewm_plus_parser = subparsers.add_parser("lewm_plus", help="Train LeWM++")
    lewm_plus_parser.add_argument("--model", type=str, default="transformer", choices=["conv", "transformer", "lstm", "ast"])
    lewm_plus_parser.add_argument("--epochs", type=int, default=100)
    lewm_plus_parser.add_argument("--lr", type=float, default=3e-4)
    lewm_plus_parser.add_argument("--batch-size", type=int, default=32)
    lewm_plus_parser.add_argument("--task", type=str, default="presence", choices=["presence", "counting"])
    add_common_args(lewm_plus_parser)

    # Decoder
    decoder_parser = subparsers.add_parser("decoder", help="Train Latent Decoder")
    decoder_parser.add_argument("--epochs", type=int, default=50)
    decoder_parser.add_argument("--lr", type=float, default=1e-3)
    decoder_parser.add_argument("--batch-size", type=int, default=16)
    add_common_args(decoder_parser)

    # Fusion
    fusion_parser = subparsers.add_parser("fusion", help="Train Fusion Model")
    fusion_parser.add_argument("--epochs", type=int, default=50)
    fusion_parser.add_argument("--batch-size", type=int, default=32)
    fusion_parser.add_argument("--lr", type=float, default=1e-4)
    fusion_parser.add_argument("--dropout-prob", type=float, default=0.5)
    add_common_args(fusion_parser)

    # Translator
    translator_parser = subparsers.add_parser("translator", help="Train Translator")
    translator_parser.add_argument("--epochs", type=int, default=100)
    translator_parser.add_argument("--batch-size", type=int, default=16)
    translator_parser.add_argument("--lr", type=float, default=1e-4)
    translator_parser.add_argument("--d-model", type=int, default=256)
    translator_parser.add_argument("--patch-size", type=int, default=16)
    add_common_args(translator_parser)

    # MAE
    mae_parser = subparsers.add_parser("mae", help="Train MAE")
    mae_parser.add_argument("--epochs", type=int, default=100)
    mae_parser.add_argument("--batch-size", type=int, default=64)
    mae_parser.add_argument("--lr", type=float, default=1e-3)
    mae_parser.add_argument("--mask-ratio", type=float, default=0.75)
    add_common_args(mae_parser)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Map command to config and job type
    if args.command == "jepa":
        config = TrainingConfig.from_args(args)
        job_type = f"jepa-{config.model_type}-{args.task}"
    elif args.command == "lewm":
        args.model = "lewm" # Hack for from_args
        config = TrainingConfig.from_args(args)
        job_type = f"lewm-{args.task}"
    elif args.command == "lewm_plus":
        config = TrainingConfig.from_args(args)
        job_type = f"lewm-plus-{config.model_type}-{args.task}"
    elif args.command == "decoder":
        config = DecoderConfig(
            dataset=args.dataset, with_aug=args.with_aug, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr
        )
        job_type = "decoder-train"
    elif args.command == "fusion":
        config = FusionConfig(
            dataset=args.dataset, with_aug=args.with_aug, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr, dropout_prob=args.dropout_prob
        )
        job_type = "fusion-train"
    elif args.command == "translator":
        config = TranslatorConfig(
            dataset=args.dataset, with_aug=args.with_aug, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr, d_model=args.d_model, patch_size=args.patch_size
        )
        job_type = "translator-train"
    elif args.command == "mae":
        config = MAEConfig(
            dataset=args.dataset, with_aug=args.with_aug, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.lr, mask_ratio=args.mask_ratio
        )
        job_type = "mae-pretrain"
    
    run_training(args, config, job_type)


if __name__ == "__main__":
    main()

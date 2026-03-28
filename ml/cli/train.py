#!/usr/bin/env python3
"""
Unified training script for all depth learning models.

This script serves as the single entry point for training any model in the system.
It uses a command-based architecture to route to appropriate training pipelines.

Usage:
    python train.py jepa --model transformer --epochs 80
    python train.py lewm --epochs 100 --task presence
    python train.py lewm_plus --dataset easy --epochs 100

Author: Depth Learning Team
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import wandb
from typing import Optional, Literal
from dotenv import load_dotenv

from utils.config import (
    TrainingConfig,
    DecoderConfig,
    FusionConfig,
    TranslatorConfig,
    MAEConfig,
    add_common_args,
)
from data.data import create_visual_transform, AugmentationConfig
from core import get_trainer, LeWMPlusTrainer
from core.trainers_advanced import DecoderTrainer, FusionTrainer, TranslatorTrainer, MAETrainer
from utils.logging import setup_logging, get_logger

# Load environment variables
load_dotenv()

# Initialize logging
setup_logging()
logger = get_logger(__name__)


def setup_wandb(config: object, job_type: str) -> None:
    """
    Initialize wandb with configuration.
    
    Args:
        config: Training configuration object
        job_type: Type of job (e.g., 'jepa-transformer-presence')
    """
    wandb.init(
        entity=getattr(config, 'wandb_entity', 'victoria-university-of-wellington'),
        project=getattr(config, 'wandb_project', 'depth-learning'),
        job_type=job_type,
        config={
            **vars(config),
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        },
    )


def train_jepa(args: argparse.Namespace) -> None:
    """
    Train JEPA model (Joint Embedding Predictive Architecture).
    
    Args:
        args: Command-line arguments
    """
    config = TrainingConfig.from_args(args)
    # Set dataset-specific weights directory if not specified
    if args.weights_dir is None:
        config.weights_dir = f"weights/jepa_{config.dataset}"
    else:
        config.weights_dir = args.weights_dir
    # Set early stopping patience
    config.early_stop_patience = args.patience
    device = _get_device()

    logger.info(f"Starting JEPA Training ({config.model_type.upper()}) on {device}")
    logger.info(f"Task: {args.task.upper()}")
    logger.info(f"Dataset: {config.dataset} | Augmentation: {'enabled' if config.with_aug else 'disabled'}")
    logger.info(f"Epochs: {config.epochs} | Batch size: {config.batch_size} | LR: {config.learning_rate}")

    print(f"--- Starting JEPA Training ({config.model_type.upper()}) on {device} ---")
    print(f"--- Task: {args.task.upper()} ---")
    print(f"--- Dataset: {config.dataset} | Augmentation: {'enabled' if config.with_aug else 'disabled'} ---")

    setup_wandb(config, job_type=f"jepa-{config.model_type}-{args.task}")

    # Create data loaders
    aug_config = AugmentationConfig(
        enabled=config.with_aug,
        light=config.light_aug,
        rotation_degrees=config.rotation_degrees,
    )
    transform = create_visual_transform(aug_config)
    dataset_path = _get_dataset_path(config.dataset)

    from data.data import FishDataset
    from torch.utils.data import Subset

    # Create dataset with task-specific labels
    multi_label = (args.task == "presence")
    
    # Create unbalanced dataset for proper stratified splitting
    # This ensures Empty class is represented in both train and val
    unbalanced_dataset = FishDataset(dataset_path, transform=transform, mode="val", multi_label=multi_label, task=args.task)

    # Split into train/val using stratified split (preserves class ratios from unbalanced data)
    from data.data import create_stratified_split
    train_indices, val_indices = create_stratified_split(unbalanced_dataset)

    # Use unbalanced dataset for both train and val
    # The training will be on unbalanced data, but this ensures Empty class is in validation
    train_ds = Subset(unbalanced_dataset, train_indices)
    val_ds = Subset(unbalanced_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Get trainer and train
    trainer = get_trainer(config, device)
    trainer.model = trainer.build_model(task=args.task)
    trainer.optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=config.epochs
    )

    trainer.train(train_loader, val_loader)
    wandb.finish()


def train_jepa_sigreg(args: argparse.Namespace) -> None:
    """Train Multi-modal JEPA with SigReg regularization."""
    config = TrainingConfig.from_args(args)
    # Set dataset-specific weights directory
    if args.weights_dir is None:
        config.weights_dir = f"weights/jepa_sigreg_{config.dataset}"
    else:
        config.weights_dir = args.weights_dir
    config.early_stop_patience = args.patience
    device = _get_device()

    print(f"--- Starting JEPA+SigReg Training ({config.model_type.upper()}) on {device} ---")
    print(f"--- Task: {args.task.upper()} ---")
    print(f"--- Dataset: {config.dataset} | Augmentation: {'enabled' if config.with_aug else 'disabled'} ---")
    print(f"--- SigReg Weight: {config.sigreg_weight} ---")

    setup_wandb(config, job_type=f"jepa-sigreg-{config.model_type}-{args.task}")

    # Create data loaders
    aug_config = AugmentationConfig(
        enabled=config.with_aug,
        light=config.light_aug,
        rotation_degrees=config.rotation_degrees,
    )
    transform = create_visual_transform(aug_config)
    dataset_path = _get_dataset_path(config.dataset)

    from data.data import FishDataset
    from torch.utils.data import Subset

    multi_label = (args.task == "presence")
    unbalanced_dataset = FishDataset(dataset_path, transform=transform, mode="val", multi_label=multi_label, task=args.task)

    from data.data import create_stratified_split
    train_indices, val_indices = create_stratified_split(unbalanced_dataset)

    train_ds = Subset(unbalanced_dataset, train_indices)
    val_ds = Subset(unbalanced_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Import and create trainer
    from trainers_jepa_sigreg import JEPASigRegTrainer
    trainer = JEPASigRegTrainer(config, device)
    trainer.model = trainer.build_model(task=args.task)
    trainer.optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=config.epochs
    )

    trainer.train(train_loader, val_loader)
    wandb.finish()


def train_lewm_plus(args: argparse.Namespace) -> None:
    """Train LeWM++ (Multi-modal JEPA with SigReg regularization)."""
    args.model = "lewm_plus"
    config = TrainingConfig.from_args(args)
    if args.weights_dir is None:
        config.weights_dir = f"weights/lewm_plus_{config.dataset}"
    else:
        config.weights_dir = args.weights_dir
    config.early_stop_patience = args.patience
    device = _get_device()

    print(f"--- Starting LeWM++ Training ({config.model_type.upper()}) on {device} ---")
    print(f"--- Task: {args.task.upper()} ---")
    print(f"--- Dataset: {config.dataset} | Augmentation: {'enabled' if config.with_aug else 'disabled'} ---")
    print(f"--- SigReg Weight: {config.sigreg_weight} ---")

    setup_wandb(config, job_type=f"lewm-plus-{config.model_type}-{args.task}")

    # Create data loaders
    aug_config = AugmentationConfig(
        enabled=config.with_aug,
        light=config.light_aug,
        rotation_degrees=config.rotation_degrees,
    )
    transform = create_visual_transform(aug_config)
    dataset_path = _get_dataset_path(config.dataset)

    from data.data import FishDataset
    from torch.utils.data import Subset

    multi_label = (args.task == "presence")
    unbalanced_dataset = FishDataset(
        dataset_path,
        transform=transform,
        mode="val",
        multi_label=multi_label,
        task=args.task,
    )

    from data.data import create_stratified_split
    train_indices, val_indices = create_stratified_split(unbalanced_dataset)

    train_ds = Subset(unbalanced_dataset, train_indices)
    val_ds = Subset(unbalanced_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Import and create trainer
    from core import LeWMPlusTrainer
    trainer = LeWMPlusTrainer(config, device)
    trainer.model = trainer.build_model(task=args.task)
    trainer.optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=config.epochs
    )

    trainer.train(train_loader, val_loader)
    wandb.finish()


def train_lewm(args: argparse.Namespace) -> None:
    """Train LeWorldModel with multi-label presence/absence or counting."""
    # Set model type for args before creating config
    args.model = "lewm"
    config = TrainingConfig.from_args(args)
    # Set dataset-specific weights directory if not specified
    if args.weights_dir is None:
        config.weights_dir = f"weights/lewm_{config.dataset}"
    else:
        config.weights_dir = args.weights_dir
    # Set early stopping patience
    config.early_stop_patience = args.patience
    device = _get_device()

    print(f"--- Starting LeWorldModel Training on {device} ---")
    print(f"--- Task: {args.task.upper()} ---")
    print(f"--- Dataset: {config.dataset} | Augmentation: {'enabled' if config.with_aug else 'disabled'} ---")

    setup_wandb(config, job_type=f"lewm-{args.task}")

    # Create data loaders with multi_label=True and task-specific format
    aug_config = AugmentationConfig(
        enabled=config.with_aug,
        light=config.light_aug,
        rotation_degrees=config.rotation_degrees,
    )
    transform = create_visual_transform(aug_config)
    dataset_path = _get_dataset_path(config.dataset)

    from data.data import FishDataset
    from torch.utils.data import Subset

    # Create unbalanced dataset for proper stratified splitting
    # This ensures Empty class is represented in both train and val
    unbalanced_dataset = FishDataset(
        dataset_path,
        transform=transform,
        mode="val",
        multi_label=True,
        task=args.task
    )

    # Split into train/val using stratified split (preserves class ratios from unbalanced data)
    from data.data import create_stratified_split
    train_indices, val_indices = create_stratified_split(unbalanced_dataset)

    # Use unbalanced dataset for both train and val
    # The training will be on unbalanced data, but this ensures Empty class is in validation
    train_ds = Subset(unbalanced_dataset, train_indices)
    val_ds = Subset(unbalanced_dataset, val_indices)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Get trainer and train
    trainer = get_trainer(config, device)
    trainer.model = trainer.build_model(task=args.task)
    trainer.optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=config.epochs
    )

    trainer.train(train_loader, val_loader)
    wandb.finish()


def train_decoder(args: argparse.Namespace) -> None:
    """Train Latent Decoder model."""
    config = DecoderConfig(
        dataset=args.dataset,
        with_aug=args.with_aug,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weights_dir=args.weights_dir,
    )
    device = _get_device()
    
    print(f"--- Starting Decoder Training on {device} ---")
    print(f"--- Dataset: {config.dataset} | Augmentation: {'enabled' if config.with_aug else 'disabled'} ---")
    
    setup_wandb(config, job_type="decoder-train")
    trainer = DecoderTrainer(config, device)
    trainer.train()
    wandb.finish()


def train_fusion(args: argparse.Namespace) -> None:
    """Train Masked Attention Fusion model."""
    config = FusionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout_prob=args.dropout_prob,
        dataset=args.dataset,
        with_aug=args.with_aug,
        weights_dir=args.weights_dir,
    )
    device = _get_device()
    
    print(f"--- Starting Fusion Model Training on {device} ---")
    print(f"--- Dataset: {config.dataset} | Augmentation: {'enabled' if config.with_aug else 'disabled'} ---")
    
    setup_wandb(config, job_type="fusion-train")
    trainer = FusionTrainer(config, device)
    trainer.train()
    wandb.finish()


def train_translator(args: argparse.Namespace) -> None:
    """Train Acoustic-to-Image Translator model."""
    config = TranslatorConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dataset=args.dataset,
        with_aug=args.with_aug,
        d_model=args.d_model,
        patch_size=args.patch_size,
        weights_dir=args.weights_dir,
    )
    device = _get_device()
    
    print(f"--- Starting Translator Training on {device} ---")
    print(f"--- Dataset: {config.dataset} | Augmentation: {'enabled' if config.with_aug else 'disabled'} ---")
    
    setup_wandb(config, job_type="translator-train")
    trainer = TranslatorTrainer(config, device)
    trainer.train()
    wandb.finish()


def train_mae(args: argparse.Namespace) -> None:
    """Train Acoustic Masked Autoencoder."""
    config = MAEConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mask_ratio=args.mask_ratio,
        dataset=args.dataset,
        with_aug=args.with_aug,
        weights_dir=args.weights_dir,
    )
    device = _get_device()
    
    print(f"--- Starting MAE Training on {device} ---")
    print(f"--- Dataset: {config.dataset} | Augmentation: {'enabled' if config.with_aug else 'disabled'} ---")
    
    setup_wandb(config, job_type="mae-pretrain")
    trainer = MAETrainer(config, device)
    trainer.train()
    wandb.finish()


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def _get_dataset_path(dataset: str) -> str:
    """Get absolute path to dataset."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", dataset)
    )


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified training script for depth learning models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Model type to train")
    
    # JEPA parser
    jepa_parser = subparsers.add_parser("jepa", help="Train JEPA model")
    jepa_parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["conv", "transformer", "lstm", "ast"],
        help="Acoustic encoder type (default: transformer)"
    )
    jepa_parser.add_argument("--epochs", type=int, default=80)
    jepa_parser.add_argument("--lr", type=float, default=3e-4)
    jepa_parser.add_argument("--batch-size", type=int, default=32)
    jepa_parser.add_argument("--weight-decay", type=float, default=0.05)
    jepa_parser.add_argument("--label-smoothing", type=float, default=0.1)
    jepa_parser.add_argument("--use-focal-loss", action="store_true", default=True)
    jepa_parser.add_argument("--task", type=str, default="presence", choices=["presence", "single_label", "counting"],
                            help="Task type: presence (multi-label), single_label, or counting (default: presence)")
    jepa_parser.add_argument("--rotation-degrees", type=int, default=30)
    jepa_parser.add_argument("--n-chunks", type=int, default=10)
    jepa_parser.add_argument("--sigreg-weight", type=float, default=0.1)
    jepa_parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    # --with-aug is in add_common_args, just set default to True
    add_common_args(jepa_parser)
    jepa_parser.set_defaults(func=train_jepa, with_aug=True)  # Enable aug by default for JEPA

    # JEPA + SigReg parser
    jepa_sigreg_parser = subparsers.add_parser("jepa_sigreg", help="Train Multi-modal JEPA with SigReg")
    jepa_sigreg_parser.add_argument(
        "--model",
        type=str,
        default="transformer",
        choices=["conv", "transformer", "lstm", "ast"],
        help="Acoustic encoder type (default: transformer)"
    )
    jepa_sigreg_parser.add_argument("--epochs", type=int, default=100)
    jepa_sigreg_parser.add_argument("--lr", type=float, default=3e-4)
    jepa_sigreg_parser.add_argument("--batch-size", type=int, default=32)
    jepa_sigreg_parser.add_argument("--weight-decay", type=float, default=0.05)
    jepa_sigreg_parser.add_argument("--task", type=str, default="presence", choices=["presence", "single_label"])
    jepa_sigreg_parser.add_argument("--sigreg-weight", type=float, default=0.1)
    jepa_sigreg_parser.add_argument("--patience", type=int, default=15)
    add_common_args(jepa_sigreg_parser)
    jepa_sigreg_parser.set_defaults(func=train_jepa_sigreg, with_aug=True)

    # LeWM parser (multi-label presence/absence detection or counting)
    lewm_parser = subparsers.add_parser("lewm", help="Train LeWorldModel with multi-label tasks")
    lewm_parser.add_argument("--epochs", type=int, default=100)
    lewm_parser.add_argument("--lr", type=float, default=3e-4)
    lewm_parser.add_argument("--batch-size", type=int, default=32)
    lewm_parser.add_argument("--weight-decay", type=float, default=0.05)
    lewm_parser.add_argument("--task", type=str, default="presence", choices=["presence", "counting"],
                            help="Task type: presence (presence/absence) or counting (fish counts)")
    lewm_parser.add_argument("--rotation-degrees", type=int, default=30)
    lewm_parser.add_argument("--n-chunks", type=int, default=10)
    lewm_parser.add_argument("--sigreg-weight", type=float, default=0.1)
    lewm_parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15)")
    add_common_args(lewm_parser)
    lewm_parser.set_defaults(func=train_lewm)

    # LeWM++ parser (Multi-modal JEPA + SigReg)
    lewm_plus_parser = subparsers.add_parser("lewm_plus", help="Train LeWM++ (JEPA + SigReg)")
    lewm_plus_parser.add_argument("--model", type=str, default="transformer",
                                  choices=["conv", "transformer", "lstm", "ast"],
                                  help="Acoustic encoder type (default: transformer)")
    lewm_plus_parser.add_argument("--epochs", type=int, default=100)
    lewm_plus_parser.add_argument("--lr", type=float, default=3e-4)
    lewm_plus_parser.add_argument("--batch-size", type=int, default=32)
    lewm_plus_parser.add_argument("--weight-decay", type=float, default=0.05)
    lewm_plus_parser.add_argument("--task", type=str, default="presence",
                                  choices=["presence", "single_label", "counting"],
                                  help="Task type (default: presence)")
    lewm_plus_parser.add_argument("--sigreg-weight", type=float, default=0.1)
    lewm_plus_parser.add_argument("--patience", type=int, default=15)
    add_common_args(lewm_plus_parser)
    lewm_plus_parser.set_defaults(func=train_lewm_plus, with_aug=True)

    # Decoder parser
    decoder_parser = subparsers.add_parser("decoder", help="Train Latent Decoder")
    decoder_parser.add_argument("--epochs", type=int, default=50)
    decoder_parser.add_argument("--lr", type=float, default=1e-3)
    decoder_parser.add_argument("--batch-size", type=int, default=16)
    add_common_args(decoder_parser)
    decoder_parser.set_defaults(func=train_decoder)
    
    # Fusion parser
    fusion_parser = subparsers.add_parser("fusion", help="Train Fusion Model")
    fusion_parser.add_argument("--epochs", type=int, default=50)
    fusion_parser.add_argument("--batch-size", type=int, default=32)
    fusion_parser.add_argument("--lr", type=float, default=1e-4)
    fusion_parser.add_argument("--dropout-prob", type=float, default=0.5)
    add_common_args(fusion_parser)
    fusion_parser.set_defaults(func=train_fusion)
    
    # Translator parser
    translator_parser = subparsers.add_parser("translator", help="Train Acoustic-to-Image Translator")
    translator_parser.add_argument("--epochs", type=int, default=100)
    translator_parser.add_argument("--batch-size", type=int, default=16)
    translator_parser.add_argument("--lr", type=float, default=1e-4)
    translator_parser.add_argument("--d-model", type=int, default=256)
    translator_parser.add_argument("--patch-size", type=int, default=16)
    add_common_args(translator_parser)
    translator_parser.set_defaults(func=train_translator)
    
    # MAE parser
    mae_parser = subparsers.add_parser("mae", help="Train Masked Autoencoder")
    mae_parser.add_argument("--epochs", type=int, default=100)
    mae_parser.add_argument("--batch-size", type=int, default=64)
    mae_parser.add_argument("--lr", type=float, default=1e-3)
    mae_parser.add_argument("--mask-ratio", type=float, default=0.75)
    add_common_args(mae_parser)
    mae_parser.set_defaults(func=train_mae)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()

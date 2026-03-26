"""Configuration management for training pipelines."""

from dataclasses import dataclass, field
from typing import Optional, Literal
import argparse


@dataclass
class TrainingConfig:
    """Base configuration for all training pipelines."""

    # Model settings
    model_type: str = "transformer"
    embed_dim: int = 256

    # Training settings
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05

    # Dataset settings
    dataset: Literal["easy", "medium", "hard", "extreme"] = "easy"
    n_chunks: int = 10

    # Augmentation settings
    with_aug: bool = False
    light_aug: bool = False
    rotation_degrees: int = 30

    # Loss settings
    label_smoothing: float = 0.1
    use_focal_loss: bool = True

    # LeWM specific
    sigreg_weight: float = 0.1

    # Logging
    wandb_entity: str = "victoria-university-of-wellington"
    wandb_project: str = "depth-learning"

    # Paths
    weights_dir: str = "weights"
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create config from argparse namespace."""
        return cls(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            dataset=args.dataset,
            n_chunks=args.n_chunks,
            with_aug=args.with_aug,
            light_aug=args.light_aug,
            rotation_degrees=args.rotation_degrees,
            label_smoothing=getattr(args, 'label_smoothing', 0.1),
            use_focal_loss=getattr(args, 'use_focal_loss', True),
            sigreg_weight=args.sigreg_weight,
        )


@dataclass
class DecoderConfig:
    """Configuration for decoder training."""
    
    dataset: Literal["easy", "medium", "hard"] = "easy"
    with_aug: bool = False
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    weights_dir: str = "weights"
    wandb_project: str = "depth-learning"


@dataclass
class FusionConfig:
    """Configuration for fusion model training."""
    
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    dropout_prob: float = 0.5
    dataset: Literal["easy", "medium", "hard"] = "easy"
    with_aug: bool = False
    weights_dir: str = "weights"
    wandb_project: str = "depth-learning"


@dataclass
class TranslatorConfig:
    """Configuration for acoustic-to-image translator training."""
    
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    dataset: Literal["easy", "medium", "hard"] = "easy"
    with_aug: bool = False
    d_model: int = 256
    patch_size: int = 16
    weights_dir: str = "weights"
    wandb_entity: str = "victoria-university-of-wellington"
    wandb_project: str = "depth-learning"


@dataclass
class MAEConfig:
    """Configuration for masked autoencoder training."""
    
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    mask_ratio: float = 0.75
    dataset: Literal["easy", "medium", "hard"] = "easy"
    with_aug: bool = False
    weights_dir: str = "weights"
    wandb_project: str = "depth-learning"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to argument parser."""
    parser.add_argument(
        "--dataset",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "extreme"],
        help="Dataset difficulty level (default: easy)"
    )
    parser.add_argument(
        "--with-aug", 
        action="store_true", 
        default=False,
        help="Enable data augmentation (default: disabled)"
    )
    parser.add_argument(
        "--light-aug", 
        action="store_true", 
        default=False,
        help="Use light augmentation (only horizontal flip)"
    )
    parser.add_argument(
        "--weights-dir", 
        type=str, 
        default="weights",
        help="Directory to save model weights"
    )

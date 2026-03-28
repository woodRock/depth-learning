"""Configuration management for training pipelines."""

from dataclasses import dataclass, field
from typing import Optional, Literal
import argparse


@dataclass
class TrainingConfig:
    """Base configuration for all training pipelines."""

    # Model settings
    architecture: str = "jepa"
    model_type: str = "transformer"
    embed_dim: int = 256

    # Training settings
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05

    # Dataset settings
    dataset: Literal["easy", "medium", "hard", "extreme"] = "easy"
    task: str = "presence"
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

    # Early stopping
    early_stop_patience: int = 15  # Stop if no improvement for N epochs
    early_stop_min_delta: float = 0.001  # Minimum improvement to count as progress

    # Logging
    wandb_entity: str = "victoria-university-of-wellington"
    wandb_project: str = "depth-learning"

    # Paths
    weights_dir: str = "weights"
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        """Create config from argparse namespace."""
        return cls(
            architecture=getattr(args, 'command', "jepa"),
            model_type=getattr(args, 'model', "transformer"),
            epochs=getattr(args, 'epochs', 80),
            batch_size=getattr(args, 'batch_size', 32),
            learning_rate=getattr(args, 'lr', 3e-4),
            weight_decay=getattr(args, 'weight_decay', 0.05),
            dataset=getattr(args, 'dataset', "easy"),
            task=getattr(args, 'task', "presence"),
            n_chunks=getattr(args, 'n_chunks', 10),
            with_aug=getattr(args, 'with_aug', False),
            light_aug=getattr(args, 'light_aug', False),
            rotation_degrees=getattr(args, 'rotation_degrees', 30),
            label_smoothing=getattr(args, 'label_smoothing', 0.1),
            use_focal_loss=getattr(args, 'use_focal_loss', True),
            sigreg_weight=getattr(args, 'sigreg_weight', 0.1),
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
        default=None,
        help="Directory to save model weights"
    )
    parser.add_argument(
        "--rotation-degrees",
        type=int,
        default=30,
        help="Max rotation angle for augmentation (default: 30)"
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=10,
        help="Number of chunks for acoustic processing (default: 10)"
    )
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        default=0.1,
        help="Weight for SigReg regularization (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

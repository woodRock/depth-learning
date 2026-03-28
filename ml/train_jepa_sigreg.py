#!/usr/bin/env python3
"""
Train Multi-modal JEPA with SigReg Regularization.

This script trains the fused JEPA+SigReg model on a specified dataset.

Usage:
    python train_jepa_sigreg.py --dataset easy --epochs 100
    python train_jepa_sigreg.py --dataset extreme --sigreg-weight 0.5

Arguments:
    --dataset       Dataset difficulty (easy, medium, hard, extreme)
    --epochs        Number of training epochs (default: 100)
    --batch-size    Batch size (default: 32)
    --lr            Learning rate (default: 3e-4)
    --sigreg-weight SigReg regularization weight (default: 0.1)
    --patience      Early stopping patience (default: 15)
    --weights-dir   Directory to save weights (default: weights/jepa_sigreg_<dataset>)
"""

import os
import sys
import argparse
import torch
import wandb
from dotenv import load_dotenv

# Add ml directory to path
sys.path.insert(0, os.path.dirname(__file__))

from config import TrainingConfig
from data import create_visual_transform, AugmentationConfig, FishDataset, create_stratified_split
from torch.utils.data import Subset
from trainers_jepa_sigreg import JEPASigRegTrainer


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def setup_wandb(config: TrainingConfig, job_type: str) -> None:
    """Initialize wandb with config."""
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


def get_dataset_path(dataset: str) -> str:
    """Get absolute path to dataset."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", dataset)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train Multi-modal JEPA with SigReg regularization"
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["easy", "medium", "hard", "extreme"],
        help="Dataset difficulty level"
    )
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay (default: 0.05)")
    
    # Model settings
    parser.add_argument("--model", type=str, default="transformer",
                       choices=["conv", "transformer", "lstm", "ast"],
                       help="Acoustic encoder type (default: transformer)")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension (default: 256)")
    parser.add_argument("--task", type=str, default="presence",
                       choices=["presence", "single_label", "counting"],
                       help="Task type (default: presence)")
    
    # SigReg settings
    parser.add_argument("--sigreg-weight", type=float, default=0.1,
                       help="SigReg regularization weight (default: 0.1)")
    
    # Early stopping
    parser.add_argument("--patience", type=int, default=15,
                       help="Early stopping patience (default: 15)")
    
    # Augmentation
    parser.add_argument("--with-aug", action="store_true", default=True,
                       help="Enable data augmentation (default: enabled)")
    parser.add_argument("--light-aug", action="store_true", default=False,
                       help="Use light augmentation (default: False)")
    parser.add_argument("--rotation-degrees", type=int, default=30,
                       help="Max rotation angle (default: 30)")
    
    # Output
    parser.add_argument("--weights-dir", type=str, default=None,
                       help="Directory to save weights (default: weights/jepa_sigreg_<dataset>)")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_type=args.model,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dataset=args.dataset,
        sigreg_weight=args.sigreg_weight,
        with_aug=args.with_aug,
        light_aug=args.light_aug,
        rotation_degrees=args.rotation_degrees,
        weights_dir=args.weights_dir if args.weights_dir else f"weights/jepa_sigreg_{args.dataset}",
    )
    
    # Set early stopping patience
    config.early_stop_patience = args.patience
    
    # Get device
    device = get_device()
    
    # Print configuration
    print("="*70)
    print("🚀 Training Multi-modal JEPA with SigReg")
    print("="*70)
    print(f"Dataset:        {args.dataset}")
    print(f"Model:          {args.model}")
    print(f"Embed dim:      {args.embed_dim}")
    print(f"Task:           {args.task}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.lr}")
    print(f"SigReg weight:  {args.sigreg_weight}")
    print(f"Patience:       {args.patience}")
    print(f"Augmentation:   {'enabled' if args.with_aug else 'disabled'}")
    print(f"Weights dir:    {config.weights_dir}")
    print(f"Device:         {device}")
    print("="*70)
    
    # Setup wandb
    setup_wandb(config, job_type=f"jepa-sigreg-{args.model}-{args.task}")
    
    # Create data loaders
    aug_config = AugmentationConfig(
        enabled=args.with_aug,
        light=args.light_aug,
        rotation_degrees=args.rotation_degrees,
    )
    transform = create_visual_transform(aug_config)
    dataset_path = get_dataset_path(args.dataset)
    
    # Create dataset
    multi_label = (args.task == "presence")
    unbalanced_dataset = FishDataset(
        dataset_path,
        transform=transform,
        mode="val",
        multi_label=multi_label,
        task=args.task,
    )
    
    # Stratified split
    train_indices, val_indices = create_stratified_split(unbalanced_dataset)
    
    # Create subsets
    train_ds = Subset(unbalanced_dataset, train_indices)
    val_ds = Subset(unbalanced_dataset, val_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val samples:   {len(val_ds)}")
    
    # Create trainer and train
    trainer = JEPASigRegTrainer(config, device)
    trainer.model = trainer.build_model(task=args.task)
    trainer.optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trainer.model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer,
        T_max=config.epochs,
    )
    
    print(f"\n📈 Starting Training...")
    trainer.train(train_loader, val_loader)
    
    # Finish wandb
    wandb.finish()
    
    print("\n" + "="*70)
    print("✅ Training Complete!")
    print("="*70)
    print(f"Weights saved to: {config.weights_dir}")
    print(f"\nNext steps:")
    print(f"  1. Start Python server:  python3 serve.py")
    print(f"  2. Run simulation:       cargo run --release")
    print(f"  3. Evaluate in sim UI:   Select 'JEPA' + '{args.dataset}' + 'Multi-modal'")
    print("="*70)


if __name__ == "__main__":
    main()

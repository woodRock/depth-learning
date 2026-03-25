#!/usr/bin/env python3
"""
Bisection search to find optimal sigreg_weight for LeWorldModel.

This script quickly finds a good sigreg_weight by:
1. Training for 1 epoch at each evaluation point
2. Using bisection search to narrow down the best range
3. Selecting the weight with highest validation accuracy

Usage:
    python3 tune_sigreg_weight.py --dataset easy --epochs-per-search 1 --bisection-steps 4
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from dotenv import load_dotenv

from config import TrainingConfig
from data import create_data_loaders, create_visual_transform, AugmentationConfig
from trainers import LeWMTrainer, get_trainer
from models.lewm import LeWorldModel

load_dotenv()


def evaluate_sigreg_weight(
    sigreg_weight: float,
    dataset_path: str,
    batch_size: int = 32,
    epochs: int = 1,
    device: torch.device = None,
) -> float:
    """
    Train LeWM for specified epochs and return best validation accuracy.
    
    Args:
        sigreg_weight: Weight for SIGReg regularization
        dataset_path: Path to dataset
        batch_size: Training batch size
        epochs: Number of epochs to train
        device: Torch device
        
    Returns:
        Best validation accuracy achieved
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create config
    config = TrainingConfig(
        model_type="lewm",
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=3e-4,
        weight_decay=0.05,
        dataset="easy",
        sigreg_weight=sigreg_weight,
    )
    
    # Create data loaders
    aug_config = AugmentationConfig(enabled=False)
    transform = create_visual_transform(aug_config)
    
    train_loader, val_loader = create_data_loaders(
        dataset_path=dataset_path,
        transform=transform,
        batch_size=batch_size,
        n_chunks=10,
    )
    
    # Build model
    model = LeWorldModel(
        embed_dim=256,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.1,
        n_classes=4,
        use_classifier=True,
        use_decoder=True,
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for vis, ac, labels in train_loader:
            vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
            
            optimizer.zero_grad()
            pred_emb, goal_emb, species_logits, recon_img = model(ac)
            
            target_img = vis
            loss, pred_loss, sigreg_loss, loss_cls, recon_loss = model.compute_loss(
                pred_emb, goal_emb, species_logits, labels,
                recon_img=recon_img,
                target_img=target_img,
                sigreg_weight=sigreg_weight,
                recon_weight=0.01,
            )
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(species_logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for vis, ac, labels in val_loader:
                vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
                
                pred_emb, goal_emb, species_logits, recon_img = model(ac)
                target_img = vis
                
                loss, pred_loss, sigreg_loss, loss_cls, recon_loss = model.compute_loss(
                    pred_emb, goal_emb, species_logits, labels,
                    recon_img=recon_img,
                    target_img=target_img,
                    sigreg_weight=sigreg_weight,
                    recon_weight=0.01,
                )
                
                val_loss += loss.item()
                preds = torch.argmax(species_logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        best_val_acc = max(best_val_acc, val_acc)
        
        scheduler.step()
    
    return best_val_acc


def bisection_search(
    dataset_path: str,
    min_weight: float = 0.001,
    max_weight: float = 1.0,
    steps: int = 4,
    batch_size: int = 32,
    epochs_per_eval: int = 1,
) -> tuple:
    """
    Perform bisection search to find optimal sigreg_weight.
    
    At each step, evaluates 3 points: left, middle, right
    Keeps the best 2 points and narrows the search range.
    
    Args:
        dataset_path: Path to dataset
        min_weight: Minimum sigreg_weight to search
        max_weight: Maximum sigreg_weight to search
        steps: Number of bisection steps
        batch_size: Batch size for training
        epochs_per_eval: Epochs to train at each evaluation
        
    Returns:
        (best_weight, best_accuracy, history)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Bisection Search for Optimal sigreg_weight")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_path}")
    print(f"Search range: [{min_weight:.4f}, {max_weight:.4f}]")
    print(f"Bisection steps: {steps}")
    print(f"Epochs per evaluation: {epochs_per_eval}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    history = []
    best_weight = min_weight
    best_acc = 0.0
    
    left = min_weight
    right = max_weight
    
    for step in range(steps):
        print(f"\n{'='*70}")
        print(f"Step {step+1}/{steps}")
        print(f"{'='*70}")
        print(f"Search range: [{left:.4f}, {right:.4f}]")
        
        # Evaluate 3 points: left, middle, right
        mid = (left + right) / 2
        test_points = [left, mid, right]
        
        # Remove duplicates
        test_points = sorted(list(set(test_points)))
        
        results = []
        for weight in test_points:
            print(f"\nEvaluating sigreg_weight={weight:.4f}...")
            acc = evaluate_sigreg_weight(
                sigreg_weight=weight,
                dataset_path=dataset_path,
                batch_size=batch_size,
                epochs=epochs_per_eval,
                device=device,
            )
            results.append((weight, acc))
            print(f"  → Validation accuracy: {acc:.4f}")
            
            history.append({
                'step': step + 1,
                'weight': weight,
                'accuracy': acc,
            })
        
        # Find best
        results.sort(key=lambda x: x[1], reverse=True)
        best_in_step = results[0]
        
        print(f"\nBest in step: sigreg_weight={best_in_step[0]:.4f}, acc={best_in_step[1]:.4f}")
        
        if best_in_step[1] > best_acc:
            best_acc = best_in_step[1]
            best_weight = best_in_step[0]
        
        # Narrow search: keep best 2 points
        if results[0][0] == mid:
            # Middle is best, search around it
            left = results[1][0] if results[1][0] < mid else mid
            right = results[1][0] if results[1][0] > mid else mid
        elif results[0][0] == left:
            # Left is best, search left half
            right = mid
        else:
            # Right is best, search right half
            left = mid
        
        # Ensure we don't collapse to same point
        if right - left < 0.001:
            print(f"Search range too small, stopping early")
            break
    
    print(f"\n{'='*70}")
    print(f"Search Complete!")
    print(f"{'='*70}")
    print(f"Best sigreg_weight: {best_weight:.4f}")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"\nFull history:")
    for h in history:
        print(f"  Step {h['step']}: weight={h['weight']:.4f}, acc={h['accuracy']:.4f}")
    print(f"{'='*70}\n")
    
    return best_weight, best_acc, history


def main():
    parser = argparse.ArgumentParser(description="Bisection search for optimal sigreg_weight")
    parser.add_argument("--dataset", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--min-weight", type=float, default=0.001, help="Minimum sigreg_weight")
    parser.add_argument("--max-weight", type=float, default=1.0, help="Maximum sigreg_weight")
    parser.add_argument("--steps", type=int, default=4, help="Number of bisection steps")
    parser.add_argument("--epochs-per-eval", type=int, default=1, help="Epochs per evaluation")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", type=str, default="sigreg_search_results.txt", help="Output file")
    args = parser.parse_args()
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", args.dataset))
    
    # Run bisection search
    best_weight, best_acc, history = bisection_search(
        dataset_path=dataset_path,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        steps=args.steps,
        batch_size=args.batch_size,
        epochs_per_eval=args.epochs_per_eval,
    )
    
    # Save results
    with open(args.output, "w") as f:
        f.write(f"Bisection Search Results for sigreg_weight\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Search range: [{args.min_weight}, {args.max_weight}]\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Epochs per evaluation: {args.epochs_per_eval}\n\n")
        f.write(f"Best sigreg_weight: {best_weight:.4f}\n")
        f.write(f"Best validation accuracy: {best_acc:.4f}\n\n")
        f.write(f"Full history:\n")
        for h in history:
            f.write(f"  Step {h['step']}: weight={h['weight']:.4f}, acc={h['accuracy']:.4f}\n")
    
    print(f"Results saved to {args.output}")
    
    # Print recommended command for full training
    print(f"\n{'='*70}")
    print(f"Recommended command for full training:")
    print(f"{'='*70}")
    print(f"python3 train.py lewm --dataset {args.dataset} --sigreg-weight {best_weight:.4f} --epochs 80")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

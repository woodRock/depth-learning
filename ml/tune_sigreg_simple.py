#!/usr/bin/env python3
"""
Quick sigreg_weight search WITHOUT reconstruction (faster, cleaner signal).

Reconstruction can interfere with finding good classification hyperparameters.
This script finds sigreg_weight using only prediction + classification loss.

Usage:
    python3 tune_sigreg_simple.py --dataset easy --epochs 5
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from config import TrainingConfig
from data import create_data_loaders, create_visual_transform, AugmentationConfig
from models.lewm import LeWorldModel

load_dotenv()


def evaluate_sigreg_weight(
    sigreg_weight: float,
    dataset_path: str,
    epochs: int = 5,
    batch_size: int = 32,
    device: torch.device = None,
) -> float:
    """Train LeWM (no reconstruction) and return final validation accuracy."""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create data loaders
    aug_config = AugmentationConfig(enabled=False)
    transform = create_visual_transform(aug_config)
    
    train_loader, val_loader = create_data_loaders(
        dataset_path=dataset_path,
        transform=transform,
        batch_size=batch_size,
        n_chunks=10,
    )
    
    # Build model WITHOUT decoder (faster, cleaner)
    model = LeWorldModel(
        embed_dim=256,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop=0.1,
        n_classes=4,
        use_classifier=True,
        use_decoder=False,  # No reconstruction for tuning
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for vis, ac, labels in train_loader:
            vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (no reconstruction)
            pred_emb, goal_emb, species_logits, _ = model(ac)
            
            # Loss (no reconstruction)
            loss, pred_loss, sigreg_loss, loss_cls, _ = model.compute_loss(
                pred_emb, goal_emb, species_logits, labels,
                sigreg_weight=sigreg_weight,
                recon_weight=0.0,  # No reconstruction
            )
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(species_logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for vis, ac, labels in val_loader:
                vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
                
                pred_emb, goal_emb, species_logits, _ = model(ac)
                preds = torch.argmax(species_logits, dim=1)
                val_correct += (preds == labels).sum().item()  # Fixed: compare to labels
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        scheduler.step()
        
        print(f"  Epoch {epoch+1}/{epochs}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
    
    return val_acc


def main():
    parser = argparse.ArgumentParser(description="Simple sigreg_weight search (no reconstruction)")
    parser.add_argument("--dataset", type=str, default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", args.dataset))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"Simple sigreg_weight Search (No Reconstruction)")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    print(f"{'='*70}\n")
    
    # Test a range of weights
    test_weights = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    results = []
    for weight in test_weights:
        print(f"\nTesting sigreg_weight={weight:.4f}...")
        acc = evaluate_sigreg_weight(
            sigreg_weight=weight,
            dataset_path=dataset_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        results.append((weight, acc))
        print(f"  → Final val_acc: {acc:.4f}")
    
    # Sort by accuracy
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"Results (sorted by accuracy):")
    print(f"{'='*70}")
    for weight, acc in results:
        marker = "← BEST" if weight == results[0][0] else ""
        print(f"  sigreg_weight={weight:.4f}: val_acc={acc:.4f} {marker}")
    print(f"{'='*70}\n")
    
    best_weight = results[0][0]
    print(f"Recommended command:")
    print(f"  python3 train.py lewm --dataset {args.dataset} --sigreg-weight {best_weight:.4f} --epochs 80")
    print()


if __name__ == "__main__":
    main()

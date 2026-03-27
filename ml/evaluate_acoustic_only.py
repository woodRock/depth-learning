#!/usr/bin/env python3
"""
Evaluate models in acoustic-only mode for real-world deployment comparison.

This script evaluates:
1. JEPA: Trained with images (joint embedding), evaluated on acoustic-only
2. LeWM: Trained on acoustic-only, evaluated on acoustic-only

Results are saved to results.json with mode="acoustic_only" marker.
Uses the same stratified train/val split as training.
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# Add ml directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data import FishDataset, create_stratified_split
from models.acoustic import ConvEncoder, TransformerEncoder
from models.jepa import CrossModalJEPA
from models.lewm_multilabel import LeWorldModelMultiLabel


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_jepa_model(dataset: str, device: torch.device):
    """Load trained JEPA model."""
    weights_dir = Path(__file__).parent / "weights" / f"jepa_{dataset}"
    config_path = weights_dir / "model_config.json"
    weights_path = weights_dir / "fish_clip_model.pth"
    
    if not weights_path.exists():
        print(f"Warning: JEPA weights not found for {dataset}")
        return None, None
    
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    model_type = config.get("model_type", "transformer")
    embed_dim = config.get("config", {}).get("embed_dim", 256)
    
    # Build model
    if model_type == "conv":
        ac_encoder = ConvEncoder(embed_dim=embed_dim)
    else:
        ac_encoder = TransformerEncoder(embed_dim=embed_dim)
    
    model = CrossModalJEPA(
        ac_encoder=ac_encoder,
        embed_dim=embed_dim,
        use_focal_loss=True,
        task="presence",
    ).to(device)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, model_type


def load_lewm_model(dataset: str, device: torch.device):
    """Load trained LeWM model."""
    weights_dir = Path(__file__).parent / "weights" / f"lewm_{dataset}"
    config_path = weights_dir / "model_config.json"
    weights_path = weights_dir / "fish_clip_model.pth"
    
    if not weights_path.exists():
        print(f"Warning: LeWM weights not found for {dataset}")
        return None
    
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    task = config.get("task", "presence")
    lewm_config = config.get("config", {})
    embed_dim = lewm_config.get("embed_dim", 256)
    num_layers = lewm_config.get("num_layers", 8)
    num_heads = lewm_config.get("num_heads", 8)
    
    model = LeWorldModelMultiLabel(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop=0.1,
        n_classes=4,
        use_classifier=True,
        use_decoder=True,
        task=task,
    ).to(device)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def evaluate_acoustic_only(model, model_type: str, dataset: str, split: str, device: torch.device):
    """Evaluate model in acoustic-only mode on a specific split.
    
    Uses the same stratified splitting logic as train.py:
    1. Create full dataset with mode="train" (balances classes)
    2. Create stratified split to get train/val indices
    3. For val split, create fresh dataset with mode="val" using val_indices
    """
    
    dataset_path = Path(__file__).parent.parent / "dataset" / dataset
    if not dataset_path.exists():
        return None
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    task = getattr(model, 'task', 'presence') if model_type == "lewm" else "presence"
    multi_label = (task == "presence") or (model_type == "lewm")
    
    # Step 1: Create full balanced dataset (same as train.py)
    full_dataset = FishDataset(
        str(dataset_path),
        transform=eval_transform,
        mode="train",  # Use train mode to get balanced dataset
        multi_label=multi_label,
        task=task,
    )
    
    if len(full_dataset) == 0:
        return None
    
    # Step 2: Create stratified split (same as train.py)
    train_indices, val_indices = create_stratified_split(full_dataset)
    
    # Step 3: Select appropriate split
    if split == "train":
        # Train split uses the balanced full_dataset
        split_dataset = Subset(full_dataset, train_indices)
    elif split == "val":
        # Val split creates fresh dataset with mode="val" (no augmentation)
        val_dataset = FishDataset(
            str(dataset_path),
            transform=eval_transform,
            mode="val",  # No augmentation, no balancing
            multi_label=multi_label,
            task=task,
        )
        split_dataset = Subset(val_dataset, val_indices)
    else:  # test - use validation indices as proxy
        val_dataset = FishDataset(
            str(dataset_path),
            transform=eval_transform,
            mode="val",
            multi_label=multi_label,
            task=task,
        )
        split_dataset = Subset(val_dataset, val_indices)
    
    if len(split_dataset) == 0:
        return None
    
    loader = DataLoader(split_dataset, batch_size=32, shuffle=False)
    
    # Per-class metrics
    class_tp = torch.zeros(4)
    class_fp = torch.zeros(4)
    class_fn = torch.zeros(4)
    
    model.eval()
    with torch.no_grad():
        for _, ac, labels in loader:
            ac, labels = ac.to(device), labels.to(device)
            
            if model_type == "lewm":
                _, _, species_logits, _ = model(ac)
            else:
                # JEPA acoustic-only
                _, species_logits = model.forward_ac_to_vis_latent(ac)
            
            # Multi-label evaluation
            probs = torch.sigmoid(species_logits)
            preds = (probs > 0.5).float()
            
            for i in range(labels.shape[0]):
                tp = preds[i] * labels[i]
                fp = preds[i] * (1 - labels[i])
                fn = (1 - preds[i]) * labels[i]
                for c in range(4):
                    class_tp[c] += tp[c].item()
                    class_fp[c] += fp[c].item()
                    class_fn[c] += fn[c].item()
    
    # Calculate F1 scores
    class_precision = class_tp / (class_tp + class_fp + 1e-8)
    class_recall = class_tp / (class_tp + class_fn + 1e-8)
    class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall + 1e-8)
    
    return {
        "kingfish_f1": class_f1[0].item(),
        "snapper_f1": class_f1[1].item(),
        "cod_f1": class_f1[2].item(),
        "empty_f1": class_f1[3].item(),
        "avg_f1": class_f1.mean().item(),
    }


def evaluate_all_splits(model, model_type: str, dataset: str, device: torch.device):
    """Evaluate model on all splits (train, val, test)."""
    results = {}
    
    for split in ["train", "val", "test"]:
        metrics = evaluate_acoustic_only(model, model_type, dataset, split, device)
        if metrics:
            results[split] = metrics
            print(f"  {split.capitalize()}: Macro F1 = {metrics['avg_f1']*100:.1f}%")
        else:
            print(f"  {split.capitalize()}: No data")
    
    return results


def save_acoustic_only_results(results: dict, results_path: str):
    """Save acoustic-only evaluation results to results.json."""
    
    # Load existing results
    existing = []
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            existing = json.load(f)
    
    # Remove old acoustic_only entries
    existing = [e for e in existing if e.get("mode") != "acoustic_only"]
    
    # Add new results
    for entry in results:
        existing.append(entry)
    
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models in acoustic-only mode"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["easy", "medium", "hard", "extreme"],
        choices=["easy", "medium", "hard", "extreme"],
        help="Datasets to evaluate"
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        default=["JEPA", "LeWM"],
        choices=["JEPA", "LeWM"],
        help="Architectures to evaluate"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results.json"),
        help="Path to results.json"
    )
    
    args = parser.parse_args()
    device = get_device()
    
    print(f"{'='*60}")
    print(f"Acoustic-Only Evaluation")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    results = []
    
    for dataset in args.datasets:
        print(f"\n{'='*40}")
        print(f"Evaluating {dataset.upper()} dataset")
        print(f"{'='*40}")
        
        entry = {
            "architecture": None,
            "model_type": None,
            "dataset": dataset,
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": "acoustic_only",
            "train": None,
            "val": None,
            "test": None,
        }
        
        for arch in args.architectures:
            print(f"\n{arch} (acoustic-only):")
            
            if arch == "JEPA":
                model, model_type = load_jepa_model(dataset, device)
                if model is None:
                    print(f"  Skipping - no weights found")
                    continue
            else:  # LeWM
                model = load_lewm_model(dataset, device)
                model_type = "lewm"
                if model is None:
                    print(f"  Skipping - no weights found")
                    continue
            
            split_results = evaluate_all_splits(model, model_type, dataset, device)
            
            # Store results
            if arch == "JEPA":
                entry["architecture"] = "JEPA"
                entry["model_type"] = model_type
                entry["train"] = split_results.get("train")
                entry["val"] = split_results.get("val")
                entry["test"] = split_results.get("test")
            else:
                # For LeWM, create separate entry
                lewm_entry = entry.copy()
                lewm_entry["architecture"] = "LeWM"
                lewm_entry["model_type"] = "lewm"
                lewm_entry["train"] = split_results.get("train")
                lewm_entry["val"] = split_results.get("val")
                lewm_entry["test"] = split_results.get("test")
                lewm_entry["timestamp"] = datetime.datetime.now().isoformat()
                results.append(lewm_entry)
        
        results.append(entry.copy())
    
    save_acoustic_only_results(results, args.results_path)
    
    print(f"\n{'='*60}")
    print("Evaluation Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

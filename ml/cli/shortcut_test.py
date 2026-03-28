#!/usr/bin/env python3
"""
Shortcut Test Evaluation Script

This script evaluates models trained on one difficulty level and tested on another
to detect depth-shortcut exploitation vs genuine acoustic feature learning.

Two key tests:
1. Easy→Extreme: Model trained on Easy, tested on Extreme
   - Expected: ~50% F1 (collapse) confirms depth-shortcut exploitation
2. Extreme→Easy: Model trained on Extreme, tested on Easy
   - Expected: >88% F1 (retention) confirms genuine acoustic feature learning

Usage:
    python shortcut_test.py --train-dataset easy --test-dataset extreme --architecture JEPA
    python shortcut_test.py --train-dataset extreme --test-dataset easy --architecture LeWM
    python shortcut_test.py --all  # Run all 4 combinations (JEPA/LeWM x both directions)
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add ml directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data import FishDataset
from models.acoustic import ConvEncoder, TransformerEncoder
from models.jepa import CrossModalJEPA
from models.lewm_multilabel import LeWorldModelMultiLabel
from models.lstm import AcousticLSTM
from models.ast import AcousticAST


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_model(architecture: str, train_dataset: str, device: torch.device) -> torch.nn.Module:
    """Load trained model weights."""
    # Models are stored in dataset-specific subdirectories
    weights_dir = Path(__file__).parent / "weights" / f"{architecture.lower()}_{train_dataset}"
    config_path = weights_dir / "model_config.json"
    weights_path = weights_dir / "fish_clip_model.pth"
    
    # Fallback to main weights directory if dataset-specific not found
    if not weights_path.exists():
        weights_dir = Path(__file__).parent / "weights"
        config_path = weights_dir / "model_config.json"
        weights_path = weights_dir / "fish_clip_model.pth"
    
    # Load config to get model hyperparameters
    config = {}
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    
    model = None
    model_type = None
    
    if architecture == "JEPA":
        model_type = config.get("model_type", "transformer")
        embed_dim = config.get("config", {}).get("embed_dim", 256)
        
        if model_type == "conv":
            ac_encoder = ConvEncoder(embed_dim=embed_dim)
        elif model_type == "lstm":
            ac_encoder = AcousticLSTM(input_dim=768, hidden_dim=256, num_classes=embed_dim)
        elif model_type == "ast":
            ac_encoder = AcousticAST(d_model=embed_dim, num_classes=embed_dim)
        else:
            ac_encoder = TransformerEncoder(embed_dim=embed_dim)
        
        model = CrossModalJEPA(
            ac_encoder=ac_encoder,
            embed_dim=embed_dim,
            use_focal_loss=True,
            task="presence",
        )
        
    elif architecture == "LeWM":
        model_type = "lewm"
        task = config.get("task", "presence")
        # Get LeWM-specific config with defaults
        lewm_config = config.get("config", {})
        embed_dim = lewm_config.get("embed_dim", 256)
        num_layers = lewm_config.get("num_layers", 8)
        num_heads = lewm_config.get("num_heads", 8)
        
        # Try to detect embed_dim from checkpoint if config doesn't have it
        if weights_path.exists() and embed_dim == 256:
            try:
                state_dict = torch.load(weights_path, map_location=device, weights_only=False)
                # Check classifier weight shape to detect embed_dim
                if "classifier.0.weight" in state_dict:
                    detected_embed = state_dict["classifier.0.weight"].shape[1]
                    if detected_embed != 256:
                        print(f"Detected embed_dim={detected_embed} from checkpoint (config has {embed_dim})")
                        embed_dim = detected_embed
            except Exception as e:
                print(f"Warning: Could not detect embed_dim from checkpoint: {e}")
        
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
        )
    
    if model is None:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Load weights with strict=False to handle minor architecture differences
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded {architecture} weights from {weights_path}")
    else:
        print(f"Warning: No weights found at {weights_path}, using random initialization")
    
    model.to(device)
    model.eval()
    return model, model_type


def evaluate_cross_dataset(
    model: torch.nn.Module,
    model_type: str,
    test_dataset: str,
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Evaluate model on test dataset and return per-class F1 scores."""
    
    dataset_path = Path(__file__).parent.parent / "dataset" / test_dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {dataset_path}")
    
    # Evaluation transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset - use val mode for evaluation
    task = getattr(model, 'task', 'presence') if model_type == "lewm" else "presence"
    multi_label = (task == "presence") or (model_type == "lewm")
    
    dataset = FishDataset(
        str(dataset_path),
        transform=eval_transform,
        mode="val",  # Use val mode to get validation split
        multi_label=multi_label,
        task=task,
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Per-class confusion metrics
    class_tp = torch.zeros(4)
    class_fp = torch.zeros(4)
    class_fn = torch.zeros(4)
    
    model.eval()
    with torch.no_grad():
        for _, ac, labels in tqdm(loader, desc=f"Evaluating on {test_dataset}"):
            ac, labels = ac.to(device), labels.to(device)
            
            if model_type == "lewm":
                # LeWM forward returns: pred_emb, goal_emb, species_logits, recon_img
                outputs = model(ac)
                species_logits = outputs[2]  # Get classification logits
            elif model_type in ["conv", "transformer"]:
                # JEPA models
                _, species_logits = model.forward_ac_to_vis_latent(ac)
            else:
                # Other models (lstm, ast, etc.)
                species_logits = model(ac)
            
            if task == "presence" or model_type == "lewm":
                # Multi-label: sigmoid + threshold
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
            else:
                # Single-label: argmax
                preds_idx = torch.argmax(species_logits, dim=1)
                for i in range(labels.shape[0]):
                    label_idx = labels[i].item()
                    pred_idx = preds_idx[i].item()
                    if label_idx == pred_idx:
                        class_tp[label_idx] += 1
                    else:
                        class_fp[pred_idx] += 1
                        class_fn[label_idx] += 1
    
    # Calculate per-class F1
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


def run_shortcut_test(
    train_dataset: str,
    test_dataset: str,
    architecture: str,
    results_path: str,
) -> dict:
    """Run a single shortcut test and log results."""
    
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Shortcut Test: {architecture} trained on {train_dataset}, tested on {test_dataset}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load model
    model, model_type = load_model(architecture, train_dataset, device)
    
    # Evaluate on test dataset
    test_metrics = evaluate_cross_dataset(model, model_type, test_dataset, device)
    
    # Log results
    results_entry = {
        "architecture": architecture,
        "model_type": model_type,
        "dataset": train_dataset,
        "test_dataset": test_dataset,  # Cross-dataset evaluation
        "timestamp": datetime.datetime.now().isoformat(),
        "shortcut_test": True,
        "train": None,  # Not applicable for shortcut test
        "val": None,    # Not applicable for shortcut test
        "test": test_metrics,
    }
    
    # Append to results.json
    results = []
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except Exception:
            pass
    
    results.append(results_entry)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"\nTest Metrics:")
    print(f"  Kingfish F1: {test_metrics['kingfish_f1']*100:.1f}%")
    print(f"  Snapper F1:  {test_metrics['snapper_f1']*100:.1f}%")
    print(f"  Cod F1:      {test_metrics['cod_f1']*100:.1f}%")
    print(f"  Empty F1:    {test_metrics['empty_f1']*100:.1f}%")
    print(f"  Macro F1:    {test_metrics['avg_f1']*100:.1f}%")
    
    return test_metrics


def run_all_shortcut_tests(results_path: str):
    """Run all 4 shortcut test combinations."""
    
    combinations = [
        ("easy", "extreme", "JEPA"),
        ("easy", "extreme", "LeWM"),
        ("extreme", "easy", "JEPA"),
        ("extreme", "easy", "LeWM"),
    ]
    
    all_results = {}
    for train_ds, test_ds, arch in combinations:
        try:
            metrics = run_shortcut_test(train_ds, test_ds, arch, results_path)
            key = f"{arch}_{train_ds}_to_{test_ds}"
            all_results[key] = metrics
        except Exception as e:
            print(f"Error running {arch} {train_ds}→{test_ds}: {e}")
            all_results[f"{arch}_{train_ds}_to_{test_ds}"] = {"error": str(e)}
    
    print(f"\n{'='*60}")
    print("Summary of All Shortcut Tests")
    print(f"{'='*60}")
    for key, metrics in all_results.items():
        if "error" in metrics:
            print(f"{key}: ERROR - {metrics['error']}")
        else:
            print(f"{key}: Macro F1 = {metrics['avg_f1']*100:.1f}%")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run shortcut tests to detect depth-shortcut exploitation"
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        choices=["easy", "medium", "hard", "extreme"],
        help="Dataset to train on"
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        choices=["easy", "medium", "hard", "extreme"],
        help="Dataset to test on (cross-dataset evaluation)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["JEPA", "LeWM"],
        help="Model architecture"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 4 shortcut test combinations"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results.json"),
        help="Path to results.json file"
    )
    
    args = parser.parse_args()
    
    if args.all:
        run_all_shortcut_tests(args.results_path)
    elif args.train_dataset and args.test_dataset and args.architecture:
        run_shortcut_test(
            args.train_dataset,
            args.test_dataset,
            args.architecture,
            args.results_path,
        )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python shortcut_test.py --train-dataset easy --test-dataset extreme --architecture JEPA")
        print("  python shortcut_test.py --all")


if __name__ == "__main__":
    main()

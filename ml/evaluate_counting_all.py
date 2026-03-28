#!/usr/bin/env python3
"""
Evaluate All Models for Counting Task

Evaluates all model architectures on the counting task across all datasets
and generates a comprehensive MAE table.

Models evaluated:
  - JEPA (Multi-modal)
  - JEPA (Acoustic-Only)
  - LeWM (Acoustic-Only)
  - JEPA+SigReg (Multi-modal)

Usage:
    python3 evaluate_counting_all.py [--output results_counting.json]
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# Add ml directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data import FishDataset, create_stratified_split
from torch.utils.data import Subset


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_jepa_model(dataset, device, acoustic_only=False):
    """Load JEPA model."""
    # Windows-compatible path handling
    script_dir = Path(__file__).parent
    weights_dir = script_dir / "weights" / f"jepa_{dataset}"
    config_path = weights_dir / "model_config.json"
    weights_path = weights_dir / "fish_clip_model.pth"
    
    # Debug: print path for troubleshooting
    print(f"  Looking for weights at: {weights_path.absolute()}")
    
    if not weights_path.exists():
        print(f"  ✗ Weights not found at: {weights_path}")
        return None, None
    
    with open(config_path, "r") as f:
        config = json.load(f)

    from models.acoustic import TransformerEncoder
    from models.jepa import CrossModalJEPA

    embed_dim = config.get("config", {}).get("embed_dim", 256)
    ac_encoder = TransformerEncoder(embed_dim=embed_dim)

    model = CrossModalJEPA(
        ac_encoder=ac_encoder,
        embed_dim=embed_dim,
        use_focal_loss=True,
        task="counting",
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, "jepa"


def load_jepa_sigreg_model(dataset, device):
    """Load JEPA+SigReg model."""
    # Windows-compatible path handling using os.path
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, "weights", f"jepa_sigreg_{dataset}")
    config_path = os.path.join(weights_dir, "model_config.json")
    weights_path = os.path.join(weights_dir, "fish_clip_model.pth")
    
    # Debug: print path for troubleshooting
    print(f"  Looking for weights at: {weights_path}")
    print(f"  File exists: {os.path.exists(weights_path)}")
    
    if not os.path.exists(weights_path):
        print(f"  ✗ Weights not found!")
        # List what IS in the weights directory
        if os.path.exists(os.path.dirname(weights_path)):
            files = os.listdir(os.path.dirname(weights_path))
            print(f"  Files in directory: {files}")
        return None, None

    with open(config_path, "r") as f:
        config = json.load(f)

    from models.acoustic import TransformerEncoder
    from models.jepa_sigreg import MultimodalJEPASigReg

    embed_dim = config.get("config", {}).get("embed_dim", 256)
    ac_encoder = TransformerEncoder(embed_dim=embed_dim)

    model = MultimodalJEPASigReg(
        ac_encoder=ac_encoder,
        embed_dim=embed_dim,
        use_focal_loss=True,
        task="counting",
        use_sigreg=True,
        sigreg_weight=config.get("config", {}).get("sigreg_weight", 0.1),
    ).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, "jepa_sigreg"


def load_lewm_model(dataset, device):
    """Load LeWM model."""
    weights_dir = Path(__file__).parent / "weights" / f"lewm_{dataset}"
    config_path = weights_dir / "model_config.json"
    weights_path = weights_dir / "fish_clip_model.pth"
    
    if not weights_path.exists():
        return None, None
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    from models.lewm_multilabel import LeWorldModelMultiLabel
    
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
        task="counting",
    ).to(device)
    
    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, "lewm"


def evaluate_model(model, model_type, dataset, device):
    """Evaluate model on counting task."""
    dataset_path = Path(__file__).parent.parent / "dataset" / dataset
    
    if not dataset_path.exists():
        return None
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset and split
    full_dataset = FishDataset(
        str(dataset_path),
        transform=eval_transform,
        mode="val",
        multi_label=True,
        task="counting",
    )
    
    if len(full_dataset) == 0:
        return None
    
    train_indices, val_indices = create_stratified_split(full_dataset)
    val_dataset = Subset(full_dataset, val_indices)
    loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    model.eval()
    
    # Per-class MAE
    class_mae = [0.0, 0.0, 0.0, 0.0]  # KF, SN, CD, EM
    class_mse = [0.0, 0.0, 0.0, 0.0]
    total_samples = 0
    
    with torch.no_grad():
        for _, ac, labels in loader:
            ac, labels = ac.to(device), labels.to(device)
            
            if model_type in ["jepa", "jepa_sigreg"]:
                _, species_logits = model.forward_ac_to_vis_latent(ac)
            else:  # lewm
                _, _, species_logits, _ = model(ac)
            
            # For counting, output is already scaled (tanh * 30, clamped to 0)
            pred_counts = species_logits.clamp(min=0)
            true_counts = labels.clamp(min=0)
            
            # Per-class MAE and MSE
            for i in range(4):
                class_mae[i] += F.l1_loss(pred_counts[:, i], true_counts[:, i]).item() * len(labels)
                class_mse[i] += F.mse_loss(pred_counts[:, i], true_counts[:, i]).item() * len(labels)
            
            total_samples += len(labels)
    
    # Average MAE
    class_mae = [m / total_samples for m in class_mae]
    class_rmse = [torch.sqrt(torch.tensor(m / total_samples)).item() for m in class_mse]
    macro_mae = sum(class_mae) / 4
    macro_rmse = sum(class_rmse) / 4
    
    return {
        "kingfish_mae": class_mae[0],
        "snapper_mae": class_mae[1],
        "cod_mae": class_mae[2],
        "empty_mae": class_mae[3],
        "macro_mae": macro_mae,
        "kingfish_rmse": class_rmse[0],
        "snapper_rmse": class_rmse[1],
        "cod_rmse": class_rmse[2],
        "empty_rmse": class_rmse[3],
        "macro_rmse": macro_rmse,
        "num_samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models for counting task")
    parser.add_argument("--output", type=str, default="results_counting.json",
                       help="Output file for results (default: results_counting.json)")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    print("="*70)
    
    datasets = ["easy", "medium", "hard", "extreme"]
    models = {
        "JEPA (Multi-modal)": load_jepa_model,
        "JEPA (Acoustic-Only)": lambda d, dev: load_jepa_model(d, dev, acoustic_only=True),
        "JEPA+SigReg": load_jepa_sigreg_model,
        "LeWM": load_lewm_model,
    }
    
    results = []
    
    for model_name, load_fn in models.items():
        print(f"\n{'='*70}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*70}")

        for dataset in datasets:
            print(f"\n  Dataset: {dataset}...")
            print(f"  Calling load function: {load_fn.__name__ if hasattr(load_fn, '__name__') else 'lambda'}")

            model, model_type = load_fn(dataset, device)
            
            print(f"  Returned: model={model}, model_type={model_type}")

            if model is None:
                print(f"    ⚠ Weights not found for {model_name} on {dataset}")
                continue
            
            metrics = evaluate_model(model, model_type, dataset, device)
            
            if metrics is None:
                print(f"    ⚠ Evaluation failed for {model_name} on {dataset}")
                continue
            
            entry = {
                "architecture": model_name,
                "dataset": dataset,
                "task": "counting",
                "timestamp": datetime.datetime.now().isoformat(),
                "test": metrics,
            }
            results.append(entry)
            
            print(f"    ✓ Macro MAE: {metrics['macro_mae']:.3f}")
            print(f"      KF: {metrics['kingfish_mae']:.3f}, SN: {metrics['snapper_mae']:.3f}, "
                  f"CD: {metrics['cod_mae']:.3f}, EM: {metrics['empty_mae']:.3f}")
    
    # Save results
    output_path = Path(__file__).parent / args.output
    
    # Load existing results and merge
    existing = []
    if output_path.exists():
        with open(output_path, "r") as f:
            existing = json.load(f)
    
    # Remove old counting entries
    existing = [r for r in existing if r.get("task") != "counting"]
    existing.extend(results)
    
    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*70}")
    
    # Generate table
    generate_table(results)


def generate_table(results):
    """Generate LaTeX table for counting task results."""
    print("\n" + "="*70)
    print("GENERATING COUNTING TASK TABLE")
    print("="*70)
    
    # Organize results
    organized = defaultdict(dict)
    for entry in results:
        arch = entry["architecture"]
        dataset = entry["dataset"]
        organized[arch][dataset] = entry.get("test", {})
    
    # Model order
    model_order = [
        "JEPA (Multi-modal)",
        "JEPA (Acoustic-Only)",
        "JEPA+SigReg",
        "LeWM",
    ]
    
    # Generate LaTeX
    latex = r"""\begin{table*}[t]
\centering
\caption{%
  Counting task results: Mean Absolute Error (MAE) for fish count estimation.
  Per-species MAE shown for Kingfish (KF), Snapper (SN), Cod (CD), and Empty (EM).
  The overall macro MAE across all four classes is shown in the \textbf{Macro} column.
  Lower is better. Bold entries indicate the best (lowest) MAE per difficulty level.
}
\label{tab:counting_results}
\setlength{\tabcolsep}{4.5pt}
\small
\begin{tabular}{@{}ll cccc|cccc@{}}
\toprule
& & \multicolumn{4}{c|}{\textbf{MAE}} & \multicolumn{4}{c}{\textbf{RMSE}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10}
\textbf{Model} & \textbf{Dataset} 
  & KF & SN & CD & EM & \textbf{Macro}
  & KF & SN & CD & EM & \textbf{Macro} \\
\midrule
"""
    
    for model_name in model_order:
        if model_name not in organized:
            continue
        
        for dataset in ["easy", "medium", "hard", "extreme"]:
            if dataset not in organized[model_name]:
                continue
            
            metrics = organized[model_name][dataset]
            
            # Find best MAE for this dataset
            best_mae = float('inf')
            for m in model_order:
                if m in organized and dataset in organized[m]:
                    mae = organized[m][dataset].get("macro_mae", float('inf'))
                    if mae < best_mae:
                        best_mae = mae
            
            # Format values
            def fmt(value, is_best):
                if is_best:
                    return f"\\textbf{{{value:.3f}}}"
                return f"{value:.3f}"
            
            is_best = metrics.get("macro_mae", float('inf')) == best_mae
            
            latex += f"{model_name} & {dataset} "
            latex += f"& {fmt(metrics.get('kingfish_mae', 0), False)} "
            latex += f"& {fmt(metrics.get('snapper_mae', 0), False)} "
            latex += f"& {fmt(metrics.get('cod_mae', 0), False)} "
            latex += f"& {fmt(metrics.get('empty_mae', 0), False)} "
            latex += f"& {fmt(metrics.get('macro_mae', 0), is_best)} "
            latex += f"& {metrics.get('kingfish_rmse', 0):.3f} "
            latex += f"& {metrics.get('snapper_rmse', 0):.3f} "
            latex += f"& {metrics.get('cod_rmse', 0):.3f} "
            latex += f"& {metrics.get('empty_rmse', 0):.3f} "
            latex += f"& {metrics.get('macro_rmse', 0):.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    
    # Save table
    table_path = Path(__file__).parent / "table_counting.tex"
    with open(table_path, "w") as f:
        f.write(latex)
    
    print(f"Table saved to: {table_path}")
    print("\n" + latex)


if __name__ == "__main__":
    main()

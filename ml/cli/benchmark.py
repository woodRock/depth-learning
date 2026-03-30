#!/usr/bin/env python3
"""
Computational cost benchmarking for depth learning models.
Measures training time, VRAM usage, throughput, and FLOPs.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.jepa import CrossModalJEPA
from models.lewm_plus import LeWMPlus
from models.lewm_multilabel import LeWorldModelMultiLabel
from models.acoustic import ConvEncoder, TransformerEncoder
from models.transformer_translator import AcousticToImageTransformer
from models.fusion import MaskedAttentionFusion
from utils.logging import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, input_size_vis, input_size_ac):
    """
    Rough estimation of FLOPs for the given model.
    """
    params = count_parameters(model)
    # This is a very rough heuristic: FLOPs ~= 2 * Params (per forward pass)
    # We multiply by 3 to account for backward pass and optimizer steps
    return params * 2 * 3 / 1e9  # GFLOPs

def benchmark_model(model_name: str, model: nn.Module, device: torch.device, 
                   batch_size: int = 32, epochs: int = 10):
    """Benchmark a single model."""
    model.to(device)
    model.train()
    
    # Dummy data
    vis = torch.randn(batch_size, 3, 224, 224).to(device)
    vis_feats = torch.randn(batch_size, 2048).to(device) # For Fusion
    ac = torch.randn(batch_size, 32 * 256 * 3).to(device) # Flattened acoustic
    # Multi-hot labels for presence/absence
    labels = torch.randint(0, 2, (batch_size, 4)).float().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        if model_name == "Translator":
            _ = model(ac)
        elif model_name == "JEPA":
            _ = model(vis, ac)
        elif model_name == "LeWM++":
            _ = model(vis, ac, labels)
        elif model_name == "LeWM":
            _ = model(ac)
        elif model_name == "Fusion":
            _ = model(vis_feats, ac)
        torch.cuda.synchronize() if device.type == "cuda" else None

    # Reset VRAM tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        if model_name == "Translator":
            gen_img, species_logits = model(ac)
            # Dummy loss combining reconstruction and classification
            loss = F.mse_loss(gen_img, vis) + F.binary_cross_entropy_with_logits(species_logits, labels)
            
        elif model_name == "JEPA":
            predicted_target, target_latent, species_logits = model(vis, ac)
            loss, _, _ = model.compute_loss(predicted_target, target_latent, species_logits, labels)
            
        elif model_name == "LeWM++":
            predicted_target, target_latent, species_logits, recon_img, sigreg_loss = model(vis, ac, labels)
            loss, _, _, _, _ = model.compute_loss(
                predicted_target, target_latent, species_logits, labels,
                recon_img=recon_img, target_img=vis, sigreg_loss=sigreg_loss
            )
            
        elif model_name == "LeWM":
            pred_emb, goal_emb, species_logits, recon_img = model(ac)
            loss, _, _, _, _ = model.compute_loss(
                pred_emb, goal_emb, species_logits, labels,
                recon_img=recon_img, target_img=vis
            )

        elif model_name == "Fusion":
            logits = model(vis_feats, ac)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
        loss.backward()
        optimizer.step()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
            
    end_time = time.time()
    total_time = end_time - start_time
    
    # Throughput
    total_samples = epochs * batch_size
    throughput = total_samples / total_time
    
    # VRAM
    vram_usage = 0
    if device.type == "cuda":
        vram_usage = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
    
    # Params
    params_m = count_parameters(model) / 1e6 # Millions
    
    # FLOPs (Estimation)
    flops_g = estimate_flops(model, (batch_size, 3, 224, 224), (batch_size, 32 * 256 * 3))
    
    return {
        "Model": model_name,
        "Time (s)": total_time,
        "VRAM (MB)": vram_usage,
        "Throughput (s/s)": throughput,
        "Params (M)": params_m,
        "GFLOPs": flops_g
    }

def generate_latex_table(results: List[Dict[str, Any]]):
    """Generate a LaTeX table from benchmarking results."""
    header = ["Model", "Time (10 ep)", "VRAM (MB)", "Throughput", "Params (M)", "GFLOPs"]
    
    latex = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lccccc}\n\\hline\n"
    latex += " & ".join(header) + " \\\\\n\\hline\n"
    
    for r in results:
        row = [
            r["Model"],
            f"{r['Time (s)']:.2f}s",
            f"{r['VRAM (MB)']:.0f}",
            f"{r['Throughput (s/s)']:.1f}",
            f"{r['Params (M)']:.2f}",
            f"{r['GFLOPs']:.2f}"
        ]
        latex += " & ".join(row) + " \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n\\caption{Computational Cost Benchmark (10 Epochs)}\n\\end{table}"
    return latex

def main():
    parser = argparse.ArgumentParser(description="Benchmark model computational costs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for benchmarking")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Benchmarking on {device}...")

    # Instantiate models
    models_to_bench = [
        ("Translator", AcousticToImageTransformer(d_model=256, patch_size=16)),
        ("JEPA", CrossModalJEPA(ac_encoder=TransformerEncoder(embed_dim=256), embed_dim=256)),
        ("LeWM", LeWorldModelMultiLabel(embed_dim=256, use_decoder=True)),
        ("LeWM++", LeWMPlus(ac_encoder=TransformerEncoder(embed_dim=256), embed_dim=256, use_decoder=True)),
        ("Fusion", MaskedAttentionFusion(d_model=256, nhead=8, num_classes=4))
    ]

    results = []
    for name, model in models_to_bench:
        logger.info(f"Benchmarking {name}...")
        try:
            res = benchmark_model(name, model, device, batch_size=args.batch_size, epochs=args.epochs)
            results.append(res)
        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Print results
    print("\nBenchmark Results:")
    for r in results:
        print(f"{r['Model']}: {r['Time (s)']:.2f}s, {r['VRAM (MB)']:.0f} MB, {r['Throughput (s/s)']:.1f} samples/s")

    # Generate LaTeX
    latex_table = generate_latex_table(results)
    print("\nLaTeX Table:")
    print(latex_table)

if __name__ == "__main__":
    main()

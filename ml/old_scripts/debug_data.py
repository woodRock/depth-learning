#!/usr/bin/env python3
"""Debug script to verify data format consistency between training and inference."""

import os
import numpy as np
from glob import glob
from PIL import Image
import torch

def check_history_files():
    """Check _history.bin files in dataset."""
    # Try both relative paths
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        return
    
    history_files = sorted(glob(os.path.join(dataset_path, "*_history.bin")))
    
    if not history_files:
        print(f"❌ No _history.bin files found in {dataset_path}")
        return
    
    print(f"✓ Found {len(history_files)} _history.bin files")
    
    # Check first file
    with open(history_files[0], "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
    
    print(f"  File size: {len(raw_data)} bytes")
    print(f"  Expected: 32 * 256 * 3 = {32 * 256 * 3} bytes")
    
    if len(raw_data) >= 32 * 256 * 3:
        data = raw_data[:32 * 256 * 3].reshape(32, 256, 3).astype(np.float32) / 255.0
        print(f"  Shape after reshape: {data.shape}")
        print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  Mean per channel: R={data[:,:,0].mean():.3f}, G={data[:,:,1].mean():.3f}, B={data[:,:,2].mean():.3f}")
    else:
        print(f"❌ File too small!")

def check_echogram_extraction():
    """Check echogram PNG extraction matches _history.bin."""
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    # Find matching pairs
    history_files = sorted(glob(os.path.join(dataset_path, "*_history.bin")))
    
    if not history_files:
        print("\n❌ No history files for comparison")
        return
    
    print("\n--- Comparing _history.bin vs _acoustic.png ---")
    
    for h_path in history_files[:3]:  # Check first 3
        base = h_path.replace("_history.bin", "")
        png_path = base + "_acoustic.png"
        
        if not os.path.exists(png_path):
            print(f"  Missing PNG for {h_path}")
            continue
        
        # Load history
        with open(h_path, "rb") as f:
            hist_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(32, 256, 3)
        
        # Load PNG and extract last 32 columns
        png = np.array(Image.open(png_path).convert("RGB"))  # Ensure RGB
        png_extracted = png[:, -32:, :].transpose(1, 0, 2)  # (32, 256, 3)
        
        print(f"\n  File: {os.path.basename(base)}")
        print(f"    History shape: {hist_data.shape}, PNG extracted: {png_extracted.shape}")
        
        # Compare
        diff = np.abs(hist_data.astype(float) - png_extracted.astype(float)).mean()
        print(f"    Mean absolute difference: {diff:.3f}")
        
        if diff < 1.0:
            print(f"    ✓ Match!")
        else:
            print(f"    ❌ MISMATCH! Data formats differ.")

def check_model_input():
    """Verify model receives correct input format."""
    print("\n--- Model Input Check ---")
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    history_files = sorted(glob(os.path.join(dataset_path, "*_history.bin")))
    
    if not history_files:
        return
    
    with open(history_files[0], "rb") as f:
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
    
    # Training format
    data = raw_data[:32 * 256 * 3].reshape(32, 256, 3).astype(np.float32) / 255.0
    flat = torch.from_numpy(data.flatten()).float()
    
    print(f"  Training input shape: {flat.shape}")
    print(f"  Expected: 24576 (32 * 256 * 3)")
    
    # Simulate inference from echogram PNG
    png_path = history_files[0].replace("_history.bin", "_acoustic.png")
    if os.path.exists(png_path):
        png = np.array(Image.open(png_path).convert("RGB"))  # Match serve.py
        png_extracted = png[:, -32:, :].transpose(1, 0, 2).astype(np.float32) / 255.0
        png_flat = torch.from_numpy(png_extracted.flatten()).float()
        
        print(f"  Inference input shape: {png_flat.shape}")
        
        if flat.shape == png_flat.shape:
            print(f"  ✓ Shapes match!")
        else:
            print(f"  ❌ Shape mismatch!")

if __name__ == "__main__":
    print("=" * 60)
    print("DATA FORMAT DEBUG")
    print("=" * 60)
    check_history_files()
    check_echogram_extraction()
    check_model_input()
    print("\n" + "=" * 60)

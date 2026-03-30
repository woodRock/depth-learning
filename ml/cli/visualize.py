#!/usr/bin/env python3
"""
Visualization script to compare Translator and JEPA reconstructions.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Tuple
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.jepa import CrossModalJEPA
from models.acoustic import ConvEncoder, TransformerEncoder
from models.decoder import LatentDecoder
from models.transformer_translator import AcousticToImageTransformer
from data.data import FishDataset, create_visual_transform, AugmentationConfig
from utils.logging import setup_logging, get_logger

# Initialize logging
setup_logging()
logger = get_logger(__name__)

def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def _get_dataset_path(dataset_name: str) -> str:
    """Get absolute path to dataset."""
    ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(ml_dir)
    return os.path.join(project_root, "dataset", dataset_name)

def load_models(dataset: str, device: torch.device, seed: int = 42):
    """Load Translator, JEPA, and Decoder models with their weights."""
    weights_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    
    # 1. Load Translator
    translator = AcousticToImageTransformer(d_model=256, patch_size=16).to(device)
    translator_weights = os.path.join(weights_base, f"Translator_{dataset}_seed{seed}", "fish_clip_model.pth")
    if os.path.exists(translator_weights):
        logger.info(f"Loading Translator weights from {translator_weights}")
        translator.load_state_dict(torch.load(translator_weights, map_location=device, weights_only=True))
    else:
        logger.warning(f"Translator weights not found at {translator_weights}")
    translator.eval()

    # 2. Load JEPA (using TransformerEncoder by default as it's common)
    # We'll check for model_config.json if possible
    ac_encoder = TransformerEncoder(embed_dim=256)
    jepa = CrossModalJEPA(ac_encoder=ac_encoder, embed_dim=256).to(device)
    jepa_weights = os.path.join(weights_base, f"JEPA_{dataset}_seed{seed}", "fish_clip_model.pth")
    if os.path.exists(jepa_weights):
        logger.info(f"Loading JEPA weights from {jepa_weights}")
        # Need to check if the weights match the encoder. 
        # If it fails, we might need to try ConvEncoder.
        try:
            jepa.load_state_dict(torch.load(jepa_weights, map_location=device, weights_only=True))
        except Exception as e:
            logger.warning(f"Failed to load JEPA with TransformerEncoder: {e}. Trying ConvEncoder.")
            ac_encoder = ConvEncoder(embed_dim=256)
            jepa = CrossModalJEPA(ac_encoder=ac_encoder, embed_dim=256).to(device)
            jepa.load_state_dict(torch.load(jepa_weights, map_location=device, weights_only=True))
    else:
        logger.warning(f"JEPA weights not found at {jepa_weights}")
    jepa.eval()

    # 3. Load Decoder
    decoder = LatentDecoder(in_channels=512).to(device)
    decoder_weights = os.path.join(weights_base, f"Decoder_{dataset}_seed{seed}", "fish_clip_model.pth")
    if os.path.exists(decoder_weights):
        logger.info(f"Loading Decoder weights from {decoder_weights}")
        decoder.load_state_dict(torch.load(decoder_weights, map_location=device, weights_only=True))
    else:
        logger.warning(f"Decoder weights not found at {decoder_weights}")
    decoder.eval()

    return translator, jepa, decoder

def save_image(tensor, path):
    """Save a [3, H, W] tensor as an image."""
    # Assume tensor is in [0, 1] range
    img = tensor.cpu().detach().permute(1, 2, 0).numpy()
    img = (img * 255).astype('uint8')
    Image.fromarray(img).save(path)

def main():
    parser = argparse.ArgumentParser(description="Visualize model comparisons")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g. extreme)")
    parser.add_argument("-n", type=int, default=10, help="Number of examples to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    args = parser.parse_args()

    device = _get_device()
    dataset_path = _get_dataset_path(args.dataset)
    
    # Create figures directory
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Load models
    translator, jepa, decoder = load_models(args.dataset, device, args.seed)

    # Load dataset (no augmentation)
    transform = create_visual_transform(AugmentationConfig(enabled=False))
    
    # We'll manually sample from train and val folders if possible, or use FishDataset
    # and its internal splitting.
    full_dataset = FishDataset(dataset_path, transform=transform, mode="val", seed=args.seed)
    
    # Stratified split to get indices
    from data.data import create_stratified_split
    train_indices, val_indices = create_stratified_split(full_dataset)
    
    # Sample n/2 from each
    n_train = args.n // 2
    n_val = args.n - n_train
    
    selected_samples = [] # List of (index, set_name)
    
    # Deterministic sampling
    import numpy as np
    rng = np.random.RandomState(args.seed)
    
    if len(train_indices) >= n_train:
        chosen_train = rng.choice(train_indices, n_train, replace=False)
        for idx in chosen_train:
            selected_samples.append((idx, "train"))
    
    if len(val_indices) >= n_val:
        chosen_val = rng.choice(val_indices, n_val, replace=False)
        for idx in chosen_val:
            selected_samples.append((idx, "val"))

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225],
    )

    logger.info(f"Generating {len(selected_samples)} comparisons...")

    with torch.no_grad():
        for idx, set_name in selected_samples:
            vis_tensor, ac_tensor, _ = full_dataset[idx]
            vis_path = full_dataset.visual_files[idx]
            frame_id = vis_path.name.split("_visual.png")[0] # e.g. frame_0001
            
            ac_tensor = ac_tensor.unsqueeze(0).to(device)
            
            # Ground Truth
            gt_img = inv_normalize(vis_tensor).clamp(0, 1)
            
            # Translator reconstruction
            trans_img, _ = translator(ac_tensor)
            trans_img = trans_img.squeeze(0).clamp(0, 1)
            
            # JEPA + Decoder reconstruction
            latent, _ = jepa.forward_ac_to_vis_latent(ac_tensor)
            jepa_img = decoder(latent)
            jepa_img = jepa_img.squeeze(0).clamp(0, 1)
            
            # Save images
            save_image(gt_img, os.path.join(figures_dir, f"{frame_id}_{set_name}_gt.png"))
            save_image(trans_img, os.path.join(figures_dir, f"{frame_id}_{set_name}_translator.png"))
            save_image(jepa_img, os.path.join(figures_dir, f"{frame_id}_{set_name}_jepa.png"))
            
            logger.info(f"Saved {frame_id} ({set_name})")

    logger.info(f"Done! Figures saved to {figures_dir}")

if __name__ == "__main__":
    main()

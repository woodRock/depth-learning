#!/usr/bin/env python3
"""
Visualization script to compare Translator, JEPA, and LeWM reconstructions.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Optional
from pathlib import Path
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.jepa import CrossModalJEPA
from models.lewm_plus import LeWMPlus
from models.lewm_multilabel import LeWorldModelMultiLabel
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
    """Load all relevant models with their weights."""
    weights_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    
    # 1. Load Translator
    translator = AcousticToImageTransformer(d_model=256, patch_size=16).to(device)
    # Check both naming conventions
    t_paths = [
        os.path.join(weights_base, f"Translator_{dataset}_seed{seed}", "fish_clip_model.pth"),
        os.path.join(weights_base, f"translator_{dataset}", "fish_clip_model.pth")
    ]
    t_loaded = False
    for p in t_paths:
        if os.path.exists(p):
            logger.info(f"Loading Translator weights from {p}")
            translator.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            t_loaded = True
            break
    if not t_loaded:
        logger.warning(f"Translator weights not found.")
    translator.eval()

    # 2. Load JEPA + Decoder
    ac_encoder = TransformerEncoder(embed_dim=256)
    jepa = CrossModalJEPA(ac_encoder=ac_encoder, embed_dim=256).to(device)
    j_paths = [
        os.path.join(weights_base, f"JEPA_{dataset}_seed{seed}", "fish_clip_model.pth"),
        os.path.join(weights_base, f"jepa_{dataset}", "fish_clip_model.pth")
    ]
    j_loaded = False
    for p in j_paths:
        if os.path.exists(p):
            logger.info(f"Loading JEPA weights from {p}")
            try:
                jepa.load_state_dict(torch.load(p, map_location=device, weights_only=True))
                j_loaded = True
                break
            except:
                logger.info("Retrying JEPA with ConvEncoder...")
                jepa = CrossModalJEPA(ac_encoder=ConvEncoder(embed_dim=256), embed_dim=256).to(device)
                jepa.load_state_dict(torch.load(p, map_location=device, weights_only=True))
                j_loaded = True
                break
    jepa.eval()

    decoder = LatentDecoder(in_channels=512).to(device)
    d_paths = [
        os.path.join(weights_base, f"Decoder_{dataset}_seed{seed}", "fish_clip_model.pth"),
        os.path.join(weights_base, f"decoder_{dataset}", "fish_clip_model.pth")
    ]
    d_loaded = False
    for p in d_paths:
        if os.path.exists(p):
            logger.info(f"Loading Decoder weights from {p}")
            decoder.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            d_loaded = True
            break
    decoder.eval()

    # 3. Load LeWM
    lewm = LeWorldModelMultiLabel(embed_dim=256, use_decoder=True).to(device)
    l_paths = [
        os.path.join(weights_base, f"LeWM_{dataset}_seed{seed}", "fish_clip_model.pth"),
        os.path.join(weights_base, f"lewm_{dataset}", "fish_clip_model.pth")
    ]
    lewm_loaded = False
    for p in l_paths:
        if os.path.exists(p):
            logger.info(f"Loading LeWM weights from {p}")
            lewm.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            lewm_loaded = True
            break
    lewm.eval()

    # 4. Load LeWM++
    lewm_plus = LeWMPlus(ac_encoder=TransformerEncoder(embed_dim=256), embed_dim=256, use_decoder=True).to(device)
    lp_paths = [
        os.path.join(weights_base, f"LeWMPlus_{dataset}_seed{seed}", "fish_clip_model.pth"),
        os.path.join(weights_base, f"lewm_plus_{dataset}", "fish_clip_model.pth")
    ]
    lewm_plus_loaded = False
    for p in lp_paths:
        if os.path.exists(p):
            logger.info(f"Loading LeWM++ weights from {p}")
            lewm_plus.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            lewm_plus_loaded = True
            break
    lewm_plus.eval()

    return {
        "translator": translator if t_loaded else None,
        "jepa": jepa if j_loaded else None,
        "decoder": decoder if d_loaded else None,
        "lewm": lewm if lewm_loaded else None,
        "lewm_plus": lewm_plus if lewm_plus_loaded else None
    }

def save_image(tensor, path):
    """Save a [3, H, W] tensor as an image."""
    if tensor is None: return
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
    models_dict = load_models(args.dataset, device, args.seed)

    # Load dataset (no augmentation)
    transform = create_visual_transform(AugmentationConfig(enabled=False))
    full_dataset = FishDataset(dataset_path, transform=transform, mode="val", seed=args.seed)
    
    # Stratified split to get indices
    from data.data import create_stratified_split
    train_indices, val_indices = create_stratified_split(full_dataset)
    
    # Sample n/2 from each
    n_train = args.n // 2
    n_val = args.n - n_train
    
    selected_samples = []
    import numpy as np
    rng = np.random.RandomState(args.seed)
    
    if len(train_indices) >= n_train:
        chosen_train = rng.choice(train_indices, n_train, replace=False)
        for idx in chosen_train: selected_samples.append((idx, "train"))
    
    if len(val_indices) >= n_val:
        chosen_val = rng.choice(val_indices, n_val, replace=False)
        for idx in chosen_val: selected_samples.append((idx, "val"))

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225],
    )

    logger.info(f"Generating {len(selected_samples)} comparisons...")

    with torch.no_grad():
        for idx, set_name in selected_samples:
            vis_tensor, ac_tensor, _ = full_dataset[idx]
            vis_path = full_dataset.visual_files[idx]
            frame_id = vis_path.name.split("_visual.png")[0]
            
            ac_tensor = ac_tensor.unsqueeze(0).to(device)
            
            # Ground Truth
            gt_img = inv_normalize(vis_tensor).clamp(0, 1)
            save_image(gt_img, os.path.join(figures_dir, f"{frame_id}_{set_name}_gt.png"))
            
            # Translator
            if models_dict["translator"]:
                trans_img, _ = models_dict["translator"](ac_tensor)
                save_image(trans_img.squeeze(0).clamp(0, 1), os.path.join(figures_dir, f"{frame_id}_{set_name}_translator.png"))
            
            # JEPA + Decoder
            if models_dict["jepa"] and models_dict["decoder"]:
                latent, _ = models_dict["jepa"].forward_ac_to_vis_latent(ac_tensor)
                jepa_img = models_dict["decoder"](latent)
                save_image(jepa_img.squeeze(0).clamp(0, 1), os.path.join(figures_dir, f"{frame_id}_{set_name}_jepa.png"))
            
            # LeWM
            if models_dict["lewm"]:
                # LeWM Multi-label forward returns: pred_emb, goal_emb, logits, recon_img
                _, _, _, lewm_img = models_dict["lewm"](ac_tensor)
                if lewm_img is not None:
                    save_image(lewm_img.squeeze(0).clamp(0, 1), os.path.join(figures_dir, f"{frame_id}_{set_name}_lewm.png"))

            # LeWM++
            if models_dict["lewm_plus"]:
                # LeWMPlus needs a dummy vis for the forward pass if we use the main forward
                # or we can just call the decoder if we have the context latent
                context_latent = models_dict["lewm_plus"].context_encoder(ac_tensor)
                lewm_plus_img = models_dict["lewm_plus"].decoder(context_latent)
                save_image(lewm_plus_img.squeeze(0).clamp(0, 1), os.path.join(figures_dir, f"{frame_id}_{set_name}_lewm_plus.png"))
            
            logger.info(f"Saved {frame_id} ({set_name})")

    logger.info(f"Done! Figures saved to {figures_dir}")

if __name__ == "__main__":
    main()

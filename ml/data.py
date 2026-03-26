"""Data loading and transformation utilities."""

import os
import json
from typing import Optional, Callable, Dict, List, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


class AugmentationConfig:
    """Configuration for data augmentation."""
    
    def __init__(
        self,
        enabled: bool = False,
        light: bool = False,
        rotation_degrees: int = 30,
    ):
        self.enabled = enabled
        self.light = light
        self.rotation_degrees = rotation_degrees


def create_visual_transform(aug_config: AugmentationConfig) -> transforms.Compose:
    """Create visual transformation pipeline based on augmentation config."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    
    if aug_config.light:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
    elif aug_config.enabled:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=aug_config.rotation_degrees),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


class FishDataset(Dataset):
    """Dataset for fish classification with visual and acoustic modalities."""

    SPECIES_MAP: Dict[str, int] = {"Kingfish": 0, "Snapper": 1, "Cod": 2, "Empty": 3}
    NUM_CLASSES = 4

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        mode: str = "train",
        seed: int = 42,
        multi_label: bool = False,  # New: support multi-label format
        task: str = "presence",  # "presence" or "counting"
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        self.seed = seed
        self.multi_label = multi_label  # True for presence/absence detection
        self.task = task  # "presence" or "counting"

        self.visual_files = self._load_valid_samples()

        # Only balance if specifically requested (usually for training)
        if mode == "train":
            self.visual_files = self._balance_classes()
    
    def _load_valid_samples(self) -> List[Path]:
        """Load all valid samples with complete modalities."""
        all_visuals = sorted(self.data_dir.glob("*_visual.png"))
        valid_samples = []

        for v_path in all_visuals:
            # Check for _acoustic.png instead of _history.bin
            acoustic_path = v_path.with_name(v_path.name.replace("_visual.png", "_acoustic.png"))
            m_path = v_path.with_name(v_path.name.replace("_visual.png", "_meta.json"))

            if acoustic_path.exists() and m_path.exists():
                valid_samples.append(v_path)

        return valid_samples
    
    def _balance_classes(self) -> List[Path]:
        """Balance classes by downsampling to the size of the smallest class."""
        class_samples: Dict[int, List[Path]] = {0: [], 1: [], 2: [], 3: []}
        
        for v_path in self.visual_files:
            m_path = v_path.with_name(v_path.name.replace("_visual.png", "_meta.json"))
            with open(m_path, "r") as f:
                meta = json.load(f)
                label = self.SPECIES_MAP.get(meta["dominant_species"], 3)
                class_samples[label].append(v_path)
        
        # Find the size of the smallest non-empty class
        counts = {label: len(samples) for label, samples in class_samples.items() if len(samples) > 0}
        if not counts:
            return self.visual_files
            
        min_samples = min(counts.values())
        
        print(f"Balancing dataset to {min_samples} samples per class.")
        
        # Deterministic RNG for balancing
        rng = np.random.RandomState(self.seed)
        
        balanced_files = []
        for label, samples in class_samples.items():
            if not samples: continue
            rng.shuffle(samples)
            balanced_files.extend(samples[:min_samples])
            print(f"  Class {label} ({next(k for k, v in self.SPECIES_MAP.items() if v == label)}): {min_samples} samples")
            
        return balanced_files

    def __len__(self) -> int:
        return len(self.visual_files)
    
    def __getitem__(self, idx: int) -> tuple:
        vis_path = self.visual_files[idx]
        # Load from _acoustic.png instead of _history.bin to match inference pipeline
        acoustic_path = vis_path.with_name(vis_path.name.replace("_visual.png", "_acoustic.png"))
        meta_path = vis_path.with_name(vis_path.name.replace("_visual.png", "_meta.json"))

        # Load visual image
        vis_img = Image.open(vis_path).convert("RGB")

        # Load acoustic history from echogram PNG
        data = self._load_acoustic_data(acoustic_path)

        # Apply acoustic augmentation during training
        if self.mode == "train":
            data = self._apply_acoustic_augmentation(data)

        history_tensor = torch.from_numpy(data.flatten()).float()

        # Load label
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
            # Counting task: read species counts
            if self.task == "counting":
                if "species_counts" in meta:
                    # New format with counts
                    counts = meta["species_counts"]
                    count_tensor = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
                    for species_name, count in counts.items():
                        label_idx = self.SPECIES_MAP.get(species_name, 3)
                        count_tensor[label_idx] = float(count)
                    label_tensor = count_tensor
                else:
                    # Fallback: use presence as proxy for count
                    if "species_present" in meta:
                        species_present = meta["species_present"]
                        count_tensor = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
                        for species_name in species_present:
                            label_idx = self.SPECIES_MAP.get(species_name, 3)
                            count_tensor[label_idx] = 1.0
                        label_tensor = count_tensor
                    else:
                        # Legacy: single species = 1
                        label = self.SPECIES_MAP.get(meta.get("dominant_species", "Empty"), 3)
                        count_tensor = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
                        count_tensor[label] = 1.0
                        label_tensor = count_tensor
            
            # Presence/absence task OR multi_label=True: read species present
            elif self.task == "presence" or self.multi_label:
                if "species_present" in meta:
                    # New multi-label format
                    species_present = meta["species_present"]
                    multi_hot = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
                    for species_name in species_present:
                        label = self.SPECIES_MAP.get(species_name, 3)
                        multi_hot[label] = 1.0
                    label_tensor = multi_hot
                else:
                    # Legacy single-label format: convert to multi-hot
                    label = self.SPECIES_MAP.get(meta.get("dominant_species", "Empty"), 3)
                    multi_hot = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
                    multi_hot[label] = 1.0
                    label_tensor = multi_hot
            else:
                # Single-label classification (class indices)
                label = self.SPECIES_MAP.get(meta.get("dominant_species", "Empty"), 3)
                label_tensor = torch.tensor(label, dtype=torch.long)

        # Apply visual transform
        if self.transform:
            vis_img = self.transform(vis_img)

        return vis_img, history_tensor, label_tensor
    
    def _load_acoustic_data(self, path: Path) -> np.ndarray:
        """Load and preprocess acoustic data from echogram PNG image.
        
        This matches the inference pipeline which sends rendered echogram images.
        The echogram PNG is (256 height, 512 width, 3 channels) where:
        - Height = depth bins (256)
        - Width = pings (512, we take last 32 columns = most recent pings)
        - Channels = RGB (representing 3 frequencies)
        
        Returns: (32, 256, 3) array normalized to [0, 1]
        """
        from PIL import Image
        
        # Load echogram PNG
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)  # (256, 512, 3)
        
        # Extract last 32 pings (rightmost columns)
        # Shape: (256 depth, 32 pings, 3 channels)
        last_32_pings = img_np[:, -32:, :]
        
        # Transpose to match training format: (32 pings, 256 depth, 3 channels)
        data = last_32_pings.transpose(1, 0, 2).astype(np.float32) / 255.0
        
        return data

    def _apply_acoustic_augmentation(self, data: np.ndarray) -> np.ndarray:
        """Apply acoustic data augmentation."""
        # Use a local RNG to avoid global state issues
        # But for training, we WANT variety, so we don't use a fixed seed here

        # 1. Temporal Jitter
        shift = np.random.randint(-3, 4)
        data = np.roll(data, shift, axis=0)

        # 2. Spatial Flip (depth flip - makes model depth-invariant!)
        if np.random.random() > 0.5:
            data = np.flip(data, axis=1).copy()

        # 3. Channel Gain Variation
        gain = 1.0 + (np.random.rand(3) - 0.5) * 0.15
        data = (data * gain).clip(0, 1)

        # 4. Speckle Noise
        noise_scale = np.random.uniform(0.01, 0.04)
        noise = np.random.normal(0, noise_scale, data.shape)
        data = (data + noise).clip(0, 1)

        # 5. Ping Dropping
        drop_count = np.random.randint(0, 5)
        if drop_count > 0:
            indices = np.random.choice(32, drop_count, replace=False)
            data[indices, :, :] = 0.0

        # 6. Temporal Masking
        if np.random.random() > 0.6:
            mask_start = np.random.randint(0, 28)
            mask_len = np.random.randint(2, 5)
            data[mask_start:mask_start + mask_len, :, :] = 0.0

        # 7. Depth-dependent noise (REDUCED to prevent depth memorization)
        depth_gradient = np.linspace(0.5, 1.5, 256).reshape(1, -1, 1)
        depth_noise = np.random.normal(0, 0.02, data.shape) * depth_gradient
        data = (data + depth_noise).clip(0, 1)

        # 8. Random occlusion
        if np.random.random() > 0.7:
            occlude_depth = np.random.randint(50, 200)
            occlude_height = np.random.randint(10, 30)
            data[occlude_depth:occlude_depth + occlude_height, :, :] *= 0.3

        # 9. Direction-aware mixing
        if np.random.random() > 0.5:
            if np.random.random() > 0.5:
                data = np.flip(data, axis=0).copy()
            else:
                stride = np.random.choice([1, 2, 3])
                if stride > 1:
                    data = data[::stride, :, :]
                    pad_count = 32 - data.shape[0]
                    if pad_count > 0:
                        data = np.pad(data, ((0, pad_count), (0, 0), (0, 0)), mode='edge')
        
        # 10. Random depth shift (prevents depth memorization!)
        if np.random.random() > 0.5:
            depth_shift = np.random.randint(-30, 31)  # Shift up to 30 depth bins
            data = np.roll(data, depth_shift, axis=1)
            # Zero out the rolled edges
            if depth_shift > 0:
                data[:, :depth_shift, :] = 0
            elif depth_shift < 0:
                data[:, depth_shift:, :] = 0

        return data


class ImageLatentDataset(Dataset):
    """Dataset for image-only data (used in decoder training)."""
    
    def __init__(self, data_dir: str, transform: Optional[Callable] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.visual_files = self._load_valid_samples()
    
    def _load_valid_samples(self) -> list:
        """Load all valid samples with metadata."""
        all_visuals = sorted(self.data_dir.glob("*_visual.png"))
        valid_samples = []
        
        for v_path in all_visuals:
            m_path = v_path.with_name(v_path.name.replace("_visual.png", "_meta.json"))
            if m_path.exists():
                valid_samples.append(v_path)
        
        return valid_samples
    
    def __len__(self) -> int:
        return len(self.visual_files)
    
    def __getitem__(self, idx: int):
        img = Image.open(self.visual_files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def create_stratified_split(
    dataset: FishDataset,
    train_ratio: float = 0.8,
) -> tuple:
    """Create stratified train/val split that preserves class ratios."""
    total_frames = len(dataset)
    
    # Group by class
    class_indices: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for idx in range(total_frames):
        vis_path = dataset.visual_files[idx]
        meta_path = vis_path.with_name(vis_path.name.replace("_visual.png", "_meta.json"))
        with open(meta_path, "r") as f:
            meta = json.load(f)
            label = dataset.SPECIES_MAP.get(meta["dominant_species"], 3)
            class_indices[label].append(idx)
    
    # Stratified split: maintain same ratio in train/val for each class
    train_indices = []
    val_indices = []
    rng = np.random.RandomState(dataset.seed) # Use dataset's seed
    
    for class_label, indices in class_indices.items():
        if not indices: continue
        rng.shuffle(indices)
        
        # Split this class's samples into train/val
        split_point = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_point])
        val_indices.extend(indices[split_point:])
    
    # Shuffle final indices
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    
    return train_indices, val_indices


def create_data_loaders(
    dataset_path: str,
    transform: transforms.Compose,
    batch_size: int,
    n_chunks: int = 10,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple:
    """Create train and validation data loaders with stratified splitting."""
    # 1. Create one full dataset without balancing to get consistent indices
    full_dataset = FishDataset(dataset_path, transform=transform, mode="val", seed=seed)
    
    if len(full_dataset) < batch_size:
        raise ValueError(f"Not enough data. Found {len(full_dataset)} samples.")
    
    # 2. Split indices using the un-balanced full_dataset
    train_indices, val_indices = create_stratified_split(full_dataset)
    
    print(f"Total samples: {len(full_dataset)} | Train indices: {len(train_indices)} | Val indices: {len(val_indices)}")
    
    # 3. Create training dataset (balanced)
    # Note: We can't easily balance AFTER splitting without changing indices.
    # However, if we balance the WHOLE dataset first (deterministically), it works.
    balanced_dataset = FishDataset(dataset_path, transform=transform, mode="train", seed=seed)
    
    # Now we need to split the BALANCED dataset
    train_indices, val_indices = create_stratified_split(balanced_dataset)
    
    print(f"Balanced samples: {len(balanced_dataset)} | Train: {len(train_indices)} | Val: {len(val_indices)}")
    
    train_ds = Subset(balanced_dataset, train_indices)
    
    # Val dataset uses the SAME balanced dataset but with val_indices
    # This ensures no overlap and consistent indexing
    val_ds = Subset(balanced_dataset, val_indices)
    
    # Set mode="val" for val_ds to disable augmentations
    # Subset doesn't have mode, so we'd need to handle it in FishDataset.__getitem__
    # But FishDataset.mode is set for the whole balanced_dataset.
    # We can create a separate dataset for validation with the same balanced files.
    
    val_ds_base = FishDataset(dataset_path, transform=transform, mode="val", seed=seed)
    val_ds_base.visual_files = balanced_dataset.visual_files # COPY the balanced file list!
    val_ds = Subset(val_ds_base, val_indices)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

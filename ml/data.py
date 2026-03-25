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
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        mode: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.transform = transform
        
        self.visual_files = self._load_valid_samples()
        
        if mode == "train":
            self.visual_files = self._balance_classes()
    
    def _load_valid_samples(self) -> List[Path]:
        """Load all valid samples with complete modalities."""
        all_visuals = sorted(self.data_dir.glob("*_visual.png"))
        valid_samples = []
        
        for v_path in all_visuals:
            h_path = v_path.with_name(v_path.name.replace("_visual.png", "_history.bin"))
            m_path = v_path.with_name(v_path.name.replace("_visual.png", "_meta.json"))
            
            if h_path.exists() and m_path.exists():
                valid_samples.append(v_path)
        
        return valid_samples
    
    def _balance_classes(self) -> List[Path]:
        """Balance classes by downsampling majority class (Empty)."""
        fish_samples = []
        empty_samples = []
        
        for v_path in self.visual_files:
            m_path = v_path.with_name(v_path.name.replace("_visual.png", "_meta.json"))
            with open(m_path, "r") as f:
                if json.load(f)["dominant_species"] == "Empty":
                    empty_samples.append(v_path)
                else:
                    fish_samples.append(v_path)
        
        np.random.shuffle(empty_samples)
        num_to_keep = min(len(empty_samples), len(fish_samples))
        
        print(f"Balanced Dataset: {len(fish_samples)} Fish frames, {num_to_keep} Empty frames.")
        return fish_samples + empty_samples[:num_to_keep]
    
    def __len__(self) -> int:
        return len(self.visual_files)
    
    def __getitem__(self, idx: int) -> tuple:
        vis_path = self.visual_files[idx]
        history_path = vis_path.with_name(vis_path.name.replace("_visual.png", "_history.bin"))
        meta_path = vis_path.with_name(vis_path.name.replace("_visual.png", "_meta.json"))
        
        # Load visual image
        vis_img = Image.open(vis_path).convert("RGB")
        
        # Load acoustic history
        data = self._load_acoustic_data(history_path)
        
        # Apply acoustic augmentation during training (ALWAYS on for training)
        if self.mode == "train":
            data = self._apply_acoustic_augmentation(data)
        
        history_tensor = torch.from_numpy(data.flatten()).float()
        
        # Load label
        with open(meta_path, "r") as f:
            meta = json.load(f)
            label = self.SPECIES_MAP.get(meta["dominant_species"], 3)
        
        # Apply visual transform
        if self.transform:
            vis_img = self.transform(vis_img)
        
        return vis_img, history_tensor, label
    
    def _load_acoustic_data(self, path: Path) -> np.ndarray:
        """Load and preprocess acoustic data from binary file."""
        with open(path, "rb") as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
            expected = 32 * 256 * 3
            data = raw_data[:expected].reshape(32, 256, 3).astype(np.float32) / 255.0
        return data
    
    def _apply_acoustic_augmentation(self, data: np.ndarray) -> np.ndarray:
        """Apply acoustic data augmentation."""
        # 1. Temporal Jitter
        shift = np.random.randint(-3, 4)
        data = np.roll(data, shift, axis=0)
        
        # 2. Spatial Flip
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
        
        # 7. Depth-dependent noise
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
    n_chunks: int = 10,
    train_ratio: float = 0.8,
) -> tuple:
    """Create stratified train/val split with class balancing."""
    total_frames = len(dataset)
    
    # Group by class - use dataset's visual_files directly to get labels
    class_indices: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
    for idx in range(total_frames):
        vis_path = dataset.visual_files[idx]
        meta_path = vis_path.with_name(vis_path.name.replace("_visual.png", "_meta.json"))
        with open(meta_path, "r") as f:
            meta = json.load(f)
            label = dataset.SPECIES_MAP.get(meta["dominant_species"], 3)
            class_indices[label].append(idx)
    
    # Remove classes with too few samples
    class_indices = {k: v for k, v in class_indices.items() if len(v) > 10}
    
    if not class_indices:
        raise ValueError("Not enough samples in any class (need >10 per class)")
    
    # Balance classes
    min_class_size = min(len(indices) for indices in class_indices.values())
    max_class_size = max(len(indices) for indices in class_indices.values())
    
    if max_class_size > min_class_size * 10:
        target_per_class = max(min_class_size, max_class_size // 3)
    else:
        target_per_class = min_class_size
    
    train_indices = []
    val_indices = []
    rng = np.random.RandomState(42)
    
    for class_label, indices in class_indices.items():
        rng.shuffle(indices)
        balanced_indices = indices[:target_per_class]
        chunk_size = len(balanced_indices) // n_chunks
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = len(balanced_indices) if i == n_chunks - 1 else start_idx + chunk_size
            chunk = balanced_indices[start_idx:end_idx]
            
            split_point = int(len(chunk) * train_ratio)
            train_indices.extend(chunk[:split_point])
            val_indices.extend(chunk[split_point:])
    
    # Subsample training for diversity
    rng.shuffle(train_indices)
    train_indices = train_indices[::2]
    
    return train_indices, val_indices


def create_data_loaders(
    dataset_path: str,
    transform: transforms.Compose,
    batch_size: int,
    n_chunks: int = 10,
    num_workers: int = 4,
) -> tuple:
    """Create train and validation data loaders with stratified splitting."""
    # Note: use_augmentation is not needed - acoustic augmentation is always on for training
    # Visual augmentation is controlled by the transform parameter
    full_dataset = FishDataset(dataset_path, transform=transform)
    
    if len(full_dataset) < batch_size:
        raise ValueError(f"Not enough data. Found {len(full_dataset)} samples.")
    
    train_indices, val_indices = create_stratified_split(full_dataset, n_chunks)
    
    print(f"Train samples: {len(train_indices)} | Val samples: {len(val_indices)}")
    
    train_ds = Subset(
        FishDataset(dataset_path, transform=transform, mode="train"),
        train_indices,
    )
    val_ds = Subset(
        FishDataset(dataset_path, transform=transform, mode="val"),
        val_indices,
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

"""Tests for depth.data.data — FishDataset and transforms."""

import json
import pytest
import torch
import numpy as np
from PIL import Image
from data.data import (
    FishDataset,
    AugmentationConfig,
    create_visual_transform,
)


# ---------------------------------------------------------------------------
# AugmentationConfig + create_visual_transform
# ---------------------------------------------------------------------------

class TestCreateVisualTransform:
    def _make_img(self):
        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def test_no_aug_returns_tensor(self):
        cfg = AugmentationConfig(enabled=False)
        t = create_visual_transform(cfg)
        out = t(self._make_img())
        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 224, 224)

    def test_light_aug_returns_tensor(self):
        cfg = AugmentationConfig(enabled=False, light=True)
        t = create_visual_transform(cfg)
        out = t(self._make_img())
        assert out.shape == (3, 224, 224)

    def test_full_aug_returns_tensor(self):
        cfg = AugmentationConfig(enabled=True)
        t = create_visual_transform(cfg)
        out = t(self._make_img())
        assert out.shape == (3, 224, 224)

    def test_output_is_normalized(self):
        # ImageNet normalization means non-black images will have values outside [0,1]
        cfg = AugmentationConfig(enabled=False)
        t = create_visual_transform(cfg)
        img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
        out = t(img)
        # After normalization the values are typically in roughly [-2.5, 2.5]
        assert out.min().item() < 0.5


# ---------------------------------------------------------------------------
# FishDataset — basic loading
# ---------------------------------------------------------------------------

class TestFishDataset:
    def test_len(self, sample_dataset):
        ds = FishDataset(str(sample_dataset), mode="val", multi_label=True)
        assert len(ds) == 12

    def test_getitem_returns_three_elements(self, sample_dataset):
        ds = FishDataset(str(sample_dataset), mode="val", multi_label=True)
        item = ds[0]
        assert len(item) == 3

    def test_visual_shape_without_transform(self, sample_dataset):
        ds = FishDataset(str(sample_dataset), mode="val")
        _, history, _ = ds[0]
        # history_tensor is flattened: 32 × 256 × 3 = 24576
        assert history.shape == (24576,)

    def test_visual_shape_with_transform(self, sample_dataset):
        cfg = AugmentationConfig(enabled=False)
        t = create_visual_transform(cfg)
        ds = FishDataset(str(sample_dataset), transform=t, mode="val")
        vis, _, _ = ds[0]
        assert vis.shape == (3, 224, 224)

    def test_presence_label_is_multihot(self, sample_dataset):
        ds = FishDataset(str(sample_dataset), mode="val", task="presence")
        _, _, label = ds[0]
        assert label.shape == (4,)
        assert label.sum() >= 1.0  # at least one species present

    def test_counting_label_shape(self, sample_dataset):
        ds = FishDataset(str(sample_dataset), mode="val", task="counting")
        _, _, label = ds[0]
        assert label.shape == (4,)

    def test_counting_label_values_reasonable(self, sample_dataset):
        ds = FishDataset(str(sample_dataset), mode="val", task="counting")
        for i in range(len(ds)):
            _, _, label = ds[i]
            assert (label >= 0).all(), "Counts must be non-negative"

    def test_majority_label_is_scalar(self, sample_dataset):
        ds = FishDataset(str(sample_dataset), mode="val", task="majority")
        _, _, label = ds[0]
        assert label.ndim == 0
        assert label.dtype == torch.long

    def test_species_map_coverage(self, sample_dataset):
        ds = FishDataset(str(sample_dataset), mode="val", task="presence")
        seen_labels = set()
        for i in range(len(ds)):
            _, _, label = ds[i]
            idx = label.argmax().item()
            seen_labels.add(idx)
        assert seen_labels == {0, 1, 2, 3}

    def test_snr_noise_changes_data(self, sample_dataset):
        ds_clean = FishDataset(str(sample_dataset), mode="val")
        ds_noisy = FishDataset(str(sample_dataset), mode="val", snr_db=0.0)
        _, h_clean, _ = ds_clean[0]
        _, h_noisy, _ = ds_noisy[0]
        assert not torch.equal(h_clean, h_noisy)

    def test_history_tensor_dtype(self, sample_dataset):
        ds = FishDataset(str(sample_dataset), mode="val")
        _, history, _ = ds[0]
        assert history.dtype == torch.float32

    def test_missing_acoustic_file_excluded(self, tmp_path):
        # Create a visual + meta but NO acoustic — should not appear in dataset
        vis_arr = np.zeros((512, 512, 3), dtype=np.uint8)
        Image.fromarray(vis_arr).save(tmp_path / "frame_0001_visual.png")
        meta = {"dominant_species": "Cod", "species_present": ["Cod"],
                "species_counts": {"Kingfish": 0, "Snapper": 0, "Cod": 32, "Empty": 0}}
        with open(tmp_path / "frame_0001_meta.json", "w") as f:
            json.dump(meta, f)
        ds = FishDataset(str(tmp_path), mode="val")
        assert len(ds) == 0

    def test_empty_directory(self, tmp_path):
        ds = FishDataset(str(tmp_path), mode="val")
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# Multi-species labels
# ---------------------------------------------------------------------------

class TestMultiSpeciesLabels:
    def test_multi_label_has_multiple_ones(self, multi_species_dataset):
        ds = FishDataset(str(multi_species_dataset), mode="val", task="presence")
        _, _, label = ds[0]
        assert label.sum().item() == pytest.approx(2.0)  # Kingfish + Snapper

    def test_kingfish_index(self, multi_species_dataset):
        ds = FishDataset(str(multi_species_dataset), mode="val", task="presence")
        _, _, label = ds[0]
        assert label[FishDataset.SPECIES_MAP["Kingfish"]] == 1.0
        assert label[FishDataset.SPECIES_MAP["Snapper"]] == 1.0
        assert label[FishDataset.SPECIES_MAP["Cod"]] == 0.0

"""Shared pytest fixtures for the depth test suite."""

import json
import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal balanced synthetic dataset (3 samples × 4 species = 12 total)."""
    species_list = ["Kingfish", "Snapper", "Cod", "Empty"]
    for i in range(12):
        species = species_list[i % 4]

        # Visual: 512×512 RGB
        vis_arr = np.random.randint(0, 200, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(vis_arr).save(tmp_path / f"frame_{i:04d}_visual.png")

        # Acoustic: height=256 (depth bins), width=512 (ping columns), RGB
        ac_arr = np.random.randint(0, 200, (256, 512, 3), dtype=np.uint8)
        Image.fromarray(ac_arr).save(tmp_path / f"frame_{i:04d}_acoustic.png")

        # Metadata
        counts = {"Kingfish": 0, "Snapper": 0, "Cod": 0, "Empty": 0}
        counts[species] = 32  # 32 cumulative detections across 32 pings → 1 fish
        meta = {
            "dominant_species": species,
            "species_present": [species],
            "species_counts": counts,
        }
        with open(tmp_path / f"frame_{i:04d}_meta.json", "w") as f:
            json.dump(meta, f)

    return tmp_path


@pytest.fixture
def multi_species_dataset(tmp_path):
    """Dataset where every sample has multiple species present."""
    for i in range(8):
        vis_arr = np.random.randint(0, 200, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(vis_arr).save(tmp_path / f"frame_{i:04d}_visual.png")

        ac_arr = np.random.randint(0, 200, (256, 512, 3), dtype=np.uint8)
        Image.fromarray(ac_arr).save(tmp_path / f"frame_{i:04d}_acoustic.png")

        meta = {
            "dominant_species": "Kingfish",
            "species_present": ["Kingfish", "Snapper"],
            "species_counts": {"Kingfish": 64, "Snapper": 32, "Cod": 0, "Empty": 0},
        }
        with open(tmp_path / f"frame_{i:04d}_meta.json", "w") as f:
            json.dump(meta, f)

    return tmp_path

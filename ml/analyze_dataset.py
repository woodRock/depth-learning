#!/usr/bin/env python3
"""Debug script to analyze dataset quality."""

import json
from pathlib import Path
from collections import defaultdict

def analyze_dataset(dataset_path: str):
    """Analyze class distribution and data quality."""
    path = Path(dataset_path)
    
    if not path.exists():
        print(f"Dataset not found: {dataset_path}")
        return
    
    # Count classes
    species_count = defaultdict(int)
    total_frames = 0
    
    # Track acoustic signal strength
    acoustic_sizes = []
    
    for meta_file in path.glob('*_meta.json'):
        total_frames += 1
        with open(meta_file) as f:
            data = json.load(f)
            species = data.get('dominant_species', 'Unknown')
            species_count[species] += 1
        
        # Check acoustic file size (proxy for signal strength)
        acoustic_file = meta_file.with_name(meta_file.name.replace('_meta.json', '_history.bin'))
        if acoustic_file.exists():
            acoustic_sizes.append(acoustic_file.stat().st_size)
    
    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {dataset_path}")
    print(f"{'='*60}")
    print(f"Total frames: {total_frames}")
    print(f"\nClass distribution:")
    
    for species in ['Kingfish', 'Snapper', 'Cod', 'Empty']:
        count = species_count.get(species, 0)
        pct = (count / total_frames * 100) if total_frames > 0 else 0
        bar = '█' * int(pct / 2)
        print(f"  {species:12} {count:5} ({pct:5.1f}%) {bar}")
    
    other = sum(c for s, c in species_count.items() if s not in ['Kingfish', 'Snapper', 'Cod', 'Empty'])
    if other > 0:
        print(f"  Other        {other:5} ({other/total_frames*100:5.1f}%)")
    
    print(f"\nAcoustic file sizes:")
    print(f"  Min: {min(acoustic_sizes):>6} bytes")
    print(f"  Max: {max(acoustic_sizes):>6} bytes")
    print(f"  Avg: {sum(acoustic_sizes)/len(acoustic_sizes):>6.0f} bytes")
    
    # Check for potential issues
    print(f"\n{'='*60}")
    print("Potential Issues:")
    print(f"{'='*60}")
    
    if species_count.get('Kingfish', 0) < 50:
        print("⚠️  WARNING: Very few Kingfish samples (<50)")
        print("   → Kingfish may be swimming outside transducer range")
    
    if species_count.get('Empty', 0) == 0:
        print("⚠️  WARNING: No Empty frames")
        print("   → Transducer always detects fish (no background class)")
    
    total_fish = species_count.get('Kingfish', 0) + species_count.get('Snapper', 0) + species_count.get('Cod', 0)
    if total_fish == total_frames:
        print("⚠️  WARNING: Every frame has fish")
        print("   → Consider reducing fish density or adding empty periods")
    
    if total_frames < 1000:
        print(f"⚠️  WARNING: Small dataset ({total_frames} frames)")
        print("   → Recommend at least 2000-5000 frames for training")
    
    print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "dataset/easy"
    
    analyze_dataset(dataset)

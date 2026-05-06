import echopype as ep
import xarray as xr
import numpy as np
from PIL import Image
import os
import json
import cv2
from pathlib import Path
import sys

def process_file(raw_path, output_dir):
    print(f"Processing {raw_path}...")
    try:
        ed = ep.open_raw(raw_path, sonar_model='EK60')
        sv_ds = ep.calibrate.compute_Sv(ed)
    except Exception as e:
        print(f"Error processing {raw_path}: {e}")
        return

    # Check available frequencies
    available_freqs = sv_ds.frequency_nominal.values
    print(f"Available frequencies: {available_freqs}")
    
    target_freqs = [38000, 120000, 200000]
    channels = []
    for f in target_freqs:
        # Find closest available frequency
        idx = np.argmin(np.abs(available_freqs - f))
        channels.append(sv_ds.channel.values[idx])
        print(f"  Mapping {f}Hz -> {available_freqs[idx]}Hz (Channel: {sv_ds.channel.values[idx]})")

    # Crop to 10-100m range
    range_m = sv_ds.echo_range.isel(channel=0, ping_time=0).values
    crop_idx = np.where((range_m > 10.0) & (range_m < 100.0))[0]
    if len(crop_idx) == 0:
        crop_idx = np.arange(len(range_m))
    
    start_idx, end_idx = crop_idx[0], crop_idx[-1]
    
    sv_data = []
    for ch in channels:
        data = sv_ds.Sv.sel(channel=ch).values[:, start_idx:end_idx+1]
        data = np.nan_to_num(data, nan=-999.0)
        sv_data.append(data)
    
    num_pings = sv_data[0].shape[0]
    num_samples = sv_data[0].shape[1]
    
    os.makedirs(output_dir, exist_ok=True)
    window_size = 32
    
    # Use filename as prefix to avoid overwriting frames from different files
    file_prefix = Path(raw_path).stem

    count = 0
    for start_ping in range(0, num_pings, window_size):
        end_ping = min(start_ping + window_size, num_pings)
        actual_pings = end_ping - start_ping

        segment = np.zeros((window_size, num_samples, 3))
        for i in range(3):
            segment[:actual_pings, :, i] = sv_data[i][start_ping:end_ping, :]
            if actual_pings < window_size:
                segment[actual_pings:, :, i] = -999.0

        # Detection logic - analyze each frequency channel for fish detections
        # Channel 0 (38kHz): good for larger fish like Kingfish
        # Channel 1 (120kHz): medium range, good for Snapper
        # Channel 2 (200kHz): high frequency, smaller fish detection
        
        # Calculate detection metrics for the entire window
        # Use multiple thresholds to estimate fish abundance
        
        # 38kHz channel (Kingfish indicator)
        ch0_valid = segment[:actual_pings, :, 0].flatten()
        ch0_valid = ch0_valid[(ch0_valid > -100) & (ch0_valid < 0)]
        
        # 120kHz channel (Snapper indicator)  
        ch1_valid = segment[:actual_pings, :, 1].flatten()
        ch1_valid = ch1_valid[(ch1_valid > -100) & (ch1_valid < 0)]
        
        # Count fish echoes at different strength levels
        # Strong echoes (> -50 dB) likely indicate fish
        # Medium echoes (-50 to -60 dB)
        # Weak echoes (-60 to -70 dB)
        
        ch0_strong = np.sum(ch0_valid > -50)
        ch0_medium = np.sum((ch0_valid > -60) & (ch0_valid <= -50))
        ch0_weak = np.sum((ch0_valid > -70) & (ch0_valid <= -60))
        
        ch1_strong = np.sum(ch1_valid > -50)
        ch1_medium = np.sum((ch1_valid > -60) & (ch1_valid <= -50))
        ch1_weak = np.sum((ch1_valid > -70) & (ch1_valid <= -60))
        
        # Estimate fish counts based on echo strength distribution
        # Strong echoes weighted more heavily
        # Scale to get reasonable fish counts (0-10 range)
        normalization = (actual_pings / window_size) * (num_samples / 100.0)
        
        kingfish_score = (ch0_strong * 3 + ch0_medium * 2 + ch0_weak * 1) / 1000.0 * normalization
        snapper_score = (ch1_strong * 3 + ch1_medium * 2 + ch1_weak * 1) / 1000.0 * normalization
        
        # Convert scores to fish counts (0-10 range typically)
        kingfish_count = max(0, min(10, int(kingfish_score)))
        snapper_count = max(0, min(10, int(snapper_score)))
        
        # Determine if frame is empty based on overall echo strength
        max_sv = np.max(segment[(segment > -100) & (segment < 0)]) if np.any((segment > -100) & (segment < 0)) else -999.0
        is_empty = max_sv < -55.0 or (kingfish_count + snapper_count) < 2
        
        # Build species present list and counts
        if is_empty:
            dominant_species = "Empty"
            species_present = ["Empty"]
            species_counts = {"Empty": 0}
        else:
            species_present = []
            species_counts = {}
            
            if kingfish_count > 0:
                species_present.append("Kingfish")
                species_counts["Kingfish"] = kingfish_count
            
            if snapper_count > 0:
                species_present.append("Snapper")
                species_counts["Snapper"] = snapper_count
            
            # If no specific detections but not empty, use low counts
            if not species_present:
                # Use max_sv to determine likely species
                if max_sv > -50:
                    species_present = ["Kingfish", "Snapper"]
                    species_counts = {"Kingfish": 2, "Snapper": 1}
                    dominant_species = "Kingfish"
                else:
                    species_present = ["Kingfish"]
                    species_counts = {"Kingfish": 1}
                    dominant_species = "Kingfish"
            else:
                # Dominant species is the one with higher count
                dominant_species = max(species_counts, key=species_counts.get)

        # Normalize and Resize
        sv_min, sv_max = -80.0, -30.0
        norm_segment = np.clip((segment - sv_min) / (sv_max - sv_min), 0, 1)

        resized_segment = np.zeros((window_size, 256, 3))
        for i in range(3):
            temp = norm_segment[:, :, i].T
            resized_temp = cv2.resize(temp, (window_size, 256), interpolation=cv2.INTER_LINEAR)
            resized_segment[:, :, i] = resized_temp.T

        # Prepare for pipeline
        full_echogram = np.zeros((256, 512, 3), dtype=np.uint8)
        full_echogram[:, -32:, :] = (resized_segment.transpose(1, 0, 2) * 255).astype(np.uint8)

        # Save files
        frame_name = f"{file_prefix}_f{count:03}"
        Image.fromarray(full_echogram).save(Path(output_dir) / f"{frame_name}_acoustic.png")
        Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(Path(output_dir) / f"{frame_name}_visual.png")

        meta = {
            "dominant_species": dominant_species,
            "species_present": species_present,
            "species_counts": species_counts,
            "source": "NOAA_REAL",
            "original_file": str(raw_path),
            "max_sv": float(max_sv)
        }
        with open(Path(output_dir) / f"{frame_name}_meta.json", "w") as f:
            json.dump(meta, f)
        count += 1

    print(f"  Generated {count} frames")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 real/process_to_pipeline.py <file_or_dir> [output_dir]")
        return

    input_path = Path(sys.argv[1])
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset/real_data"

    if input_path.is_file():
        process_file(input_path, output_dir)
    elif input_path.is_dir():
        for raw_file in sorted(input_path.glob("**/*.raw")):
            process_file(raw_file, output_dir)
    else:
        print(f"Error: {input_path} not found")

if __name__ == "__main__":
    main()

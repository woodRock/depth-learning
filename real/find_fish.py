import echopype as ep
import numpy as np

raw_path = "real/RL1606/EK60/1606RL-D20160628-T205440.raw"
ed = ep.open_raw(raw_path, sonar_model='EK60')
sv_ds = ep.calibrate.compute_Sv(ed)

# Only look below 10m range to avoid surface noise
# Range samples to depth
range_ds = sv_ds.echo_range.isel(channel=0, ping_time=0).values
valid_range_idx = np.where(range_ds > 10.0)[0]

if len(valid_range_idx) > 0:
    min_idx = valid_range_idx[0]
    sub_sv = sv_ds.Sv.isel(range_sample=slice(min_idx, None))
    max_sv_sub = sub_sv.max(dim='range_sample')
    
    # Find pings where max Sv > -60 dB below 10m
    # Use mean instead of max to avoid single-pixel noise
    mean_sv_sub = sub_sv.mean(dim='range_sample')
    
    print(f"Max Sv (below 10m): {max_sv_sub.values.min()} to {max_sv_sub.values.max()}")
    print(f"Mean Sv (below 10m): {mean_sv_sub.values.min()} to {mean_sv_sub.values.max()}")
    
    fish_pings = np.where(max_sv_sub.values > -60.0)[0]
    print(f"Found {len(fish_pings)} pings with Sv > -60dB below 10m")

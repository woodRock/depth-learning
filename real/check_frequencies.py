import echopype as ep
import xarray as xr
import sys

raw_path = "real/RL1606/EK60/1606RL-D20160628-T212740.raw"
try:
    ed = ep.open_raw(raw_path, sonar_model='EK60')
    print(f"Sonar model: {ed.sonar_model}")
    # Inspect groups
    for group in ['beam', 'vendor', 'platform']:
        try:
            ds = ed[group]
            if 'frequency' in ds.coords or 'channel_id' in ds.coords:
                print(f"Group '{group}' has frequencies/channels")
                if 'frequency' in ds:
                    print(f"Frequencies in {group}: {ds.frequency.values}")
        except:
            pass
except Exception as e:
    print(f"Error: {e}")

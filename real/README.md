# NOAA Real-World Echosounder Data

This directory contains tools and documentation for downloading and processing real-world hydroacoustic data from the NOAA [National Centers for Environmental Information (NCEI)](https://www.ncei.noaa.gov/products/water-column-sonar-data).

## Prerequisites

To download and process the data, you need the following tools:

```bash
# For downloading data from AWS S3
pip install awscli

# For parsing Simrad .raw files and calibration
pip install echopype xarray netcdf4

# For image processing and resizing
pip install opencv-python Pillow
```

## 1. Downloading Data

Data is hosted in the `noaa-wcsd-pds` AWS bucket. You can browse the [Data Archive CSV](./Fisheries%20Resources%20Division_Echosounder_data_archive.csv) to find specific cruises.

### Example: Download from Reuben Lasker (RL1606)
The following command downloads a small segment of data from the Summer 2016 California Current Ecosystem Survey:

```bash
# List available files for a specific cruise and instrument
aws s3 ls s3://noaa-wcsd-pds/data/raw/Reuben_Lasker/RL1606/EK60/ --no-sign-request

# Download a .raw file and its index/bottom files
aws s3 cp s3://noaa-wcsd-pds/data/raw/Reuben_Lasker/RL1606/EK60/1606RL-D20160628-T205440.raw ./real/RL1606/EK60/ --no-sign-request
```

## 2. Processing for the Pipeline

The project pipeline expects data in a specific format:
- **Dimensions:** 32 pings (time) x 256 samples (depth).
- **Format:** RGB PNG images where R=38kHz, G=120kHz, B=200kHz.
- **Normalization:** Sv (Volume Backscattering Strength) normalized from [-80dB, -30dB] to [0, 1].

### Using the Processing Script
Run the provided Python script to convert `.raw` files into pipeline-ready frames:

```bash
python3 real/process_to_pipeline.py real/RL1606/EK60/1606RL-D20160628-T205440.raw
```

This will generate:
- `dataset/real_test/frame_XXXX_acoustic.png`: The multi-frequency echogram.
- `dataset/real_test/frame_XXXX_meta.json`: Labels mapped to synthetic classes (e.g., Sardine -> Kingfish).
- `dataset/real_test/frame_XXXX_visual.png`: Placeholder black images (real data lacks visual GT).

## 3. Class Mapping

Since the deep learning models are trained on synthetic species (Kingfish, Snapper, Cod), the real-world data is mapped as follows for testing:

| Real Species (RL1606) | Synthetic Class | Pipeline Index |
|-----------------------|-----------------|----------------|
| Pacific Sardine       | Kingfish        | 0              |
| Northern Anchovy      | Snapper         | 1              |
| Low Backscatter       | Empty           | 3              |

## 4. Analysis Tools

- `real/check_frequencies.py`: Identifies the frequencies available in a `.raw` file.
- `real/find_fish.py`: Scans a file for significant backscatter (Sv > -60dB) below the surface noise zone (>10m).

---
**Note:** Real-world data often contains surface noise, bubbles, and bottom reflections. The processing script automatically crops the range to 10m–100m to focus on pelagic fish targets.

# Bike Route & Vibration Analysis

This small project is set up to:

1. Read **Garmin route files** (`.fit`, `.gpx`, `.tcx`) and compute:
   - Total distance ridden.
   - Distance per surface type (currently everything is just "unknown",
     but this is where you can later add proper surface labels).

2. Read **phone sensor CSV files** and compute:
   - Acceleration magnitude from `accelerometer_x/y/z`.
   - Per-file RMS vibration.
   - An averaged vibration spectrum across all rides.

3. Save a compact JSON file with the results so you can plot them separately.

## Folder structure

```text
project_root/
  analyze_bike_data.py       # main analysis script
  plot_bike_data.py          # plotting only (you can tweak titles here)
  config.json                # optional: describes folder locations
  data/
    gps/                     # put .fit/.gpx/.tcx files here
    csv/                     # put sensor CSVs here
  outputs/
    analysis_results.json    # created by analyse_bike_data.py
```

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib
# Optional, if you want to read .fit files directly:
pip install fitparse
```

## How to run the analysis

From the project root:

```bash
python analyze_bike_data.py
```

This will:

- Scan `data/gps` for `.fit`, `.gpx`, `.tcx`.
- Scan `data/csv` for `.csv`.
- Print summaries to the terminal.
- Write `outputs/analysis_results.json`.

## How to plot the results

```bash
python plot_bike_data.py
```

You can also customise titles and, if you later know the true sampling rate,
convert the spectrum x-axis from *cycles per sample* to **Hz**:

```bash
python plot_bike_data.py --fs_hz 100 \
  --title_spectrum "Average Spectrum (100 Hz sampling)" \
  --title_rms "Vibration RMS per Ride" \
  --title_surface "Surface Distance Breakdown"
```

## Troubleshooting tips

- If the analysis script says "too few samples for spectrum":
  - That CSV did not have enough rows (less than 32 usable acceleration samples).
- If you get a `ValueError` about missing `accelerometer_x/y/z`:
  - Check the column names in your CSV and either:
    - Rename them to `accelerometer_x`, `accelerometer_y`, `accelerometer_z`, or
    - Adapt `load_csv_sensor()` in `analyse_bike_data.py` to your exact headers.

You can now commit this whole folder to GitHub and iterate on it as your project evolves.

# analyze_bike_data.py
"""
Analysis script for:
- Computing total distance from Garmin route files (FIT/GPX/TCX).
- Computing an average vibration spectrum from accelerometer data CSVs.
- Printing summaries to the terminal.
- Saving a compact JSON file for separate plotting.

Folder structure expected:

project_root/
  analyze_bike_data.py
  plot_bike_data.py
  config.json              (optional helper)
  data/
    gps/                   (.fit / .gpx / .tcx files from Garmin)
    csv/                   (phone sensor CSVs)
  outputs/
    analysis_results.json  (auto-created)

This version:
- Uses accelerometer_x / y / z to compute acceleration magnitude.
- Does NOT rely on timestamps for vibration spectra (index-based FFT).
- For now, all distance is counted as "unknown" surface. This can be
  extended later once you have surface labels.
"""

import os
import glob
import math
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET

# Optional FIT support: install locally with "pip install fitparse"
try:
    from fitparse import FitFile
    HAVE_FITPARSE = True
except Exception:
    HAVE_FITPARSE = False


# ---------------------------
# Small helpers
# ---------------------------

def haversine(lat1, lon1, lat2, lon2):
    """Distance in metres between two lat/lon points."""
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


# ---------------------------
# GPS / route parsers: GPX / TCX / FIT
# ---------------------------

def parse_gpx(path: str) -> pd.DataFrame:
    """Parse a GPX file into a DataFrame with columns [time, lat, lon, ele]."""
    ns = {
        "default": "http://www.topografix.com/GPX/1/1",
        "gpxtpx": "http://www.garmin.com/xmlschemas/TrackPointExtension/v1",
        "gpxx": "http://www.garmin.com/xmlschemas/GpxExtensions/v3",
    }
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return pd.DataFrame(columns=["time", "lat", "lon", "ele"])

    pts = []
    for trkpt in root.findall(".//default:trkpt", ns):
        lat = float(trkpt.attrib.get("lat"))
        lon = float(trkpt.attrib.get("lon"))
        ele_el = trkpt.find("default:ele", ns)
        t_el = trkpt.find("default:time", ns)

        ele = float(ele_el.text) if ele_el is not None else np.nan
        t = pd.to_datetime(t_el.text, utc=True, errors="coerce") if t_el is not None else pd.NaT
        pts.append((t, lat, lon, ele))

    df = pd.DataFrame(pts, columns=["time", "lat", "lon", "ele"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def parse_tcx(path: str) -> pd.DataFrame:
    """Parse a TCX file into a DataFrame with columns [time, lat, lon, ele]."""
    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return pd.DataFrame(columns=["time", "lat", "lon", "ele"])

    pts = []
    for tp in root.findall(".//tcx:Trackpoint", ns):
        t_el = tp.find("tcx:Time", ns)
        pos = tp.find("tcx:Position", ns)
        ele_el = tp.find("tcx:AltitudeMeters", ns)

        if pos is None:
            continue
        lat_el = pos.find("tcx:LatitudeDegrees", ns)
        lon_el = pos.find("tcx:LongitudeDegrees", ns)
        if lat_el is None or lon_el is None:
            continue

        lat = float(lat_el.text)
        lon = float(lon_el.text)
        t = pd.to_datetime(t_el.text, utc=True, errors="coerce") if t_el is not None else pd.NaT
        ele = float(ele_el.text) if ele_el is not None else np.nan

        pts.append((t, lat, lon, ele))

    df = pd.DataFrame(pts, columns=["time", "lat", "lon", "ele"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def semicircles_to_deg(s: float) -> float:
    """Convert Garmin 'semicircles' coordinate format to degrees."""
    return s * (180.0 / 2 ** 31)


def parse_fit(path: str) -> pd.DataFrame:
    """Parse a FIT file into a DataFrame with columns [time, lat, lon, ele]."""
    if not HAVE_FITPARSE:
        print(f"[INFO] fitparse not installed → skipping FIT: {Path(path).name}")
        return pd.DataFrame(columns=["time", "lat", "lon", "ele"])

    try:
        fit = FitFile(path)
        fit.parse()
    except Exception:
        return pd.DataFrame(columns=["time", "lat", "lon", "ele"])

    pts = []
    for msg in fit.get_messages("record"):
        vals = {d.name: d.value for d in msg}
        lat = vals.get("position_lat", None)
        lon = vals.get("position_long", None)
        if lat is None or lon is None:
            continue

        lat = semicircles_to_deg(lat)
        lon = semicircles_to_deg(lon)
        ele = vals.get("altitude", np.nan)
        t = vals.get("timestamp", None)
        t = pd.to_datetime(t, utc=True, errors="coerce") if t is not None else pd.NaT

        pts.append((t, lat, lon, float(ele) if ele is not None else np.nan))

    df = pd.DataFrame(pts, columns=["time", "lat", "lon", "ele"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def parse_route_file(path: str) -> pd.DataFrame:
    """Dispatch to the correct route parser based on file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".gpx":
        return parse_gpx(path)
    if ext == ".tcx":
        return parse_tcx(path)
    if ext == ".fit":
        return parse_fit(path)

    print(f"[WARN] Unsupported route format: {Path(path).name}")
    return pd.DataFrame(columns=["time", "lat", "lon", "ele"])


# ---------------------------
# Distance from route
# ---------------------------

def compute_track_distances(track_df: pd.DataFrame) -> float:
    """Compute per-point and total distance along the route in metres."""
    if track_df.empty:
        track_df["dist_m"] = []
        track_df["cum_dist_m"] = []
        return 0.0

    dists = [0.0]
    for i in range(1, len(track_df)):
        d = haversine(
            track_df.lat.iat[i - 1], track_df.lon.iat[i - 1],
            track_df.lat.iat[i], track_df.lon.iat[i]
        )
        dists.append(d)

    track_df["dist_m"] = dists
    track_df["cum_dist_m"] = track_df["dist_m"].cumsum()
    return float(track_df["dist_m"].sum())


# ---------------------------
# CSV parsing (tailored to your sample)
# ---------------------------

def load_csv_sensor(csv_path: str):
    """
    Parse a phone/IMU CSV.

    Expected columns (from your sample):
      - 'seconds_elapsed'
      - 'accelerometer_x', 'accelerometer_y', 'accelerometer_z'

    The function computes:
      - 'acc_mag' = sqrt(ax^2 + ay^2 + az^2)

    Returns:
      df          → DataFrame with acc_mag column
      has_surface → bool (currently False unless you add a 'surface' column)
      signal_col  → name of the column to use for spectral analysis ('acc_mag')
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Optional time index (not used for spectrum, just kept if needed later)
    if "seconds_elapsed" in df.columns:
        df["seconds_elapsed"] = pd.to_numeric(df["seconds_elapsed"], errors="coerce")

    # Acceleration axes (required)
    for c in ["accelerometer_x", "accelerometer_y", "accelerometer_z"]:
        if c not in df.columns:
            raise ValueError(f"{csv_path}: missing expected column '{c}'")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute magnitude
    df["acc_mag"] = np.sqrt(
        df["accelerometer_x"] ** 2 +
        df["accelerometer_y"] ** 2 +
        df["accelerometer_z"] ** 2
    )

    # Optional surface column – if you later add one by hand
    has_surface = "surface" in df.columns

    signal_col = "acc_mag"
    return df, has_surface, signal_col


# ---------------------------
# Placeholder surface mapping
# ---------------------------

def align_surface_to_route(route_df: pd.DataFrame, sensor_df: pd.DataFrame | None):
    """
    For the moment, we do not really have a 'surface' column in the CSV,
    and we do not align by time. Everything is labelled as 'unknown'.

    Later, once you have surface labels per segment or per time-window,
    this function can be expanded.
    """
    n = len(route_df)
    return ["unknown"] * max(0, n - 1)


# ---------------------------
# Spectral analysis (index-based)
# ---------------------------

def rms(x: np.ndarray) -> float:
    """Root-mean-square of a 1D signal."""
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    return float(np.sqrt(np.nanmean(x ** 2)))


def index_fft(signal_1d, target_nfft: int = 4096):
    """
    Compute a single-sided FFT versus 'cycles per sample'.

    - Does NOT use timestamps, only the sample index.
    - Pads or truncates to 'target_nfft' so spectra can be averaged
      across multiple files.
    """
    x = np.asarray(signal_1d, dtype=float)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)
    n = len(x)
    if n < 32:
        return np.array([]), np.array([])

    if n >= target_nfft:
        x = x[:target_nfft]
        nfft = target_nfft
    else:
        nfft = target_nfft
        x = np.pad(x, (0, nfft - n))

    X = np.fft.rfft(x, nfft)
    bins = np.fft.rfftfreq(nfft, d=1.0)  # cycles per sample
    asd = (2.0 / nfft) * np.abs(X)
    return bins, asd


# ---------------------------
# Main
# ---------------------------

def main(route_dir: str = "data/gps", csv_dir: str = "data/csv", out_dir: str = "outputs"):
    """
    Main entry point:
      - Reads all route files (.fit / .gpx / .tcx) from 'route_dir'.
      - Reads all CSV sensor files from 'csv_dir'.
      - Prints summaries to the terminal.
      - Writes 'analysis_results.json' to 'out_dir'.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Route files: FIT + GPX + TCX all in the same folder
    route_paths = sorted(
        glob.glob(os.path.join(route_dir, "*.fit")) +
        glob.glob(os.path.join(route_dir, "*.gpx")) +
        glob.glob(os.path.join(route_dir, "*.tcx"))
    )
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    # Load sensor CSVs
    csv_objects = []
    for p in csv_paths:
        df, has_surface, signal_col = load_csv_sensor(p)
        csv_objects.append({
            "path": p,
            "stem": Path(p).stem,
            "df": df,
            "has_surface": has_surface,
            "signal_col": signal_col,
        })

    total_km = 0.0
    surface_dist_m = Counter()

    target_nfft = 4096
    sum_asd = None
    count_asd = 0
    common_bins = None
    per_file_vib = []

    print("\n=== ANALYSIS START ===\n")

    # --- Distance from routes ---
    for rpath in route_paths:
        r_df = parse_route_file(rpath)
        if r_df.empty:
            print(f"[WARN] Empty/unsupported route: {Path(rpath).name}")
            continue

        total_m = compute_track_distances(r_df)
        total_km += total_m / 1000.0
        print(f"[ROUTE] {Path(rpath).name}  distance = {total_m / 1000.0:,.2f} km")

        # For now, everything goes to 'unknown' surface
        surfaces = align_surface_to_route(r_df, None)
        for i in range(1, len(r_df)):
            d = r_df["dist_m"].iat[i]
            lab = surfaces[i - 1]
            surface_dist_m[lab] += float(d)

    # --- Vibration per CSV (index-based spectra) ---
    for obj in csv_objects:
        sig = obj["signal_col"]
        s = pd.to_numeric(obj["df"][sig], errors="coerce").dropna().values
        if s.size < 32:
            print(f"[SKIP] {Path(obj['path']).name}: too few samples for spectrum.")
            continue

        bins, asd = index_fft(s, target_nfft=target_nfft)
        if bins.size == 0:
            print(f"[SKIP] {Path(obj['path']).name}: spectrum failed.")
            continue

        if common_bins is None:
            common_bins = bins
            sum_asd = np.zeros_like(asd)

        sum_asd += asd
        count_asd += 1

        per_file_vib.append({
            "file": Path(obj["path"]).name,
            "rms": rms(s),
        })

    # --- Terminal output ---
    print("\n--- DISTANCE SUMMARY ---")
    print(f"Total distance (all routes): {total_km:,.2f} km")
    if surface_dist_m:
        tot = sum(surface_dist_m.values())
        if tot > 0:
            print("Road-surface breakdown (by distance):")
            for k, v in sorted(surface_dist_m.items(), key=lambda kv: -kv[1]):
                print(f"  - {k}: {v / 1000.0:,.2f} km  ({100.0 * v / tot:5.1f}%)")
        else:
            print("  No segment distances accumulated.")
    else:
        print("  No surface mapping available.")

    print("\n--- VIBRATION SUMMARY (index-based) ---")
    if count_asd == 0:
        print("No spectra computed.")
        avg_spectrum = None
    else:
        avg_asd = (sum_asd / count_asd).tolist()
        avg_bins = common_bins.tolist()
        print(f"Averaged spectra across {count_asd} file(s).")
        arr = np.array(avg_asd)
        if len(arr):
            arr[0] = 0.0  # ignore DC
        idx = np.argsort(arr)[-5:][::-1]
        print("Top 5 spectral peaks (bin in cycles/sample):")
        for i in idx:
            print(f"  - bin={avg_bins[i]:.5f}  amplitude={arr[i]:.6g}")
        avg_spectrum = {"bins_cyc_per_sample": avg_bins, "amplitude": avg_asd}

    # --- Save JSON for plotting script (separate file) ---
    results = {
        "distance_km_total": total_km,
        "surface_breakdown": {k: v for k, v in surface_dist_m.items()},
        "vibration": {
            "averaged": avg_spectrum,
            "per_file": per_file_vib,
            "nfft": target_nfft,
        },
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_json = Path(out_dir) / "analysis_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Saved results → {out_json}")
    print("\n=== ANALYSIS END ===\n")


if __name__ == "__main__":
    # You can override these via environment variables if you like.
    main(
        route_dir=os.environ.get("ROUTE_DIR", "data/gps"),
        csv_dir=os.environ.get("CSV_DIR", "data/csv"),
        out_dir=os.environ.get("OUT_DIR", "outputs"),
    )

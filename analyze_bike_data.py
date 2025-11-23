# analyze_bike_data.py
"""
Analysis script for:
- Computing total distance from Garmin route files (FIT/GPX/TCX).
- Estimating road-surface distance breakdown using OpenStreetMap (OSM) from GPS tracks.
- Computing vibration metrics from accelerometer CSVs (Sensor Logger).
- Extracting gear shifting information directly from FIT files (Garmin + SRAM AXS).
- Computing overall average speed, max speed, and elevation gain/loss.
- Printing concise summaries to the terminal.
- Saving a compact JSON file for separate plotting.

CONFIG
------
Uses a config.json file in the project root, e.g.:

{
  "route_dir": "D:/Files/TUe/Tesi/Data/Garmin",
  "csv_dir": "D:/Files/TUe/Tesi/Data/Logger Sensor",
  "out_dir": "outputs",
  "smoothen_signal": true,
  "freq": 120,
  "mincutoff": 0.1,
  "beta": 0.1,
  "verbose": false
}
"""

import os
import glob
import math
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET

import osmnx as ox
from osmnx import graph as ox_graph

# Optional FIT support: install locally with "pip install fitparse"
try:
    from fitparse import FitFile
    HAVE_FITPARSE = True
except Exception:
    HAVE_FITPARSE = False


# ---------------------------
# Config loader
# ---------------------------

def load_config(path: str = "config.json") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------
# Small helpers
# ---------------------------

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in metres between two lat/lon points."""
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


# ---------------------------
# Route parsers: GPX / TCX / FIT
# ---------------------------

def parse_gpx(path: str) -> pd.DataFrame:
    ns = {
        "default": "http://www.topografix.com/GPX/1/1",
        "gpxtpx": "http://www.garmin.com/xmlschemas/TrackPointExtension/v1",
        "gpxx": "http://www.garmin.com/xmlschemas/GpxExtensions/v3",
    }
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return pd.DataFrame(columns=["time", "lat", "lon", "ele", "speed"])

    pts = []
    for trkpt in root.findall(".//default:trkpt", ns):
        lat = float(trkpt.attrib.get("lat"))
        lon = float(trkpt.attrib.get("lon"))
        ele_el = trkpt.find("default:ele", ns)
        t_el = trkpt.find("default:time", ns)

        ele = float(ele_el.text) if ele_el is not None else np.nan
        t = pd.to_datetime(t_el.text, utc=True, errors="coerce") if t_el is not None else pd.NaT

        speed = np.nan  # GPX usually has no per-point speed

        pts.append((t, lat, lon, ele, speed))

    df = pd.DataFrame(pts, columns=["time", "lat", "lon", "ele", "speed"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def parse_tcx(path: str) -> pd.DataFrame:
    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return pd.DataFrame(columns=["time", "lat", "lon", "ele", "speed"])

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

        speed = np.nan

        pts.append((t, lat, lon, ele, speed))

    df = pd.DataFrame(pts, columns=["time", "lat", "lon", "ele", "speed"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def semicircles_to_deg(s: float) -> float:
    """Convert Garmin 'semicircles' to degrees."""
    return s * (180.0 / 2 ** 31)


def parse_fit(path: str) -> pd.DataFrame:
    """
    FIT parser for GPS track:
    - lat / lon from semicircles
    - ele from enhanced_altitude (metres)
    - speed from enhanced_speed (m/s)
    Shifting is analysed separately by re-reading the FIT file.
    """
    if not HAVE_FITPARSE:
        print(f"[INFO] fitparse not installed → skipping FIT: {Path(path).name}")
        return pd.DataFrame(columns=["time", "lat", "lon", "ele", "speed"])

    try:
        fit = FitFile(path)
        fit.parse()
    except Exception:
        return pd.DataFrame(columns=["time", "lat", "lon", "ele", "speed"])

    pts = []
    for msg in fit.get_messages("record"):
        vals = {f.name: f.value for f in msg}

        lat_raw = vals.get("position_lat", None)
        lon_raw = vals.get("position_long", None)
        if lat_raw is None or lon_raw is None:
            continue

        lat = semicircles_to_deg(lat_raw)
        lon = semicircles_to_deg(lon_raw)

        # Elevation: device stores enhanced_altitude in metres.
        ele_val = vals.get("enhanced_altitude", None)
        if ele_val is None:
            ele_val = vals.get("altitude", np.nan)

        # Speed: device uses enhanced_speed in m/s.
        sp_val = vals.get("enhanced_speed", None)
        if sp_val is None:
            sp_val = vals.get("speed", np.nan)

        t = vals.get("timestamp", None)
        t = pd.to_datetime(t, utc=True, errors="coerce") if t is not None else pd.NaT

        pts.append(
            (
                t,
                lat,
                lon,
                float(ele_val) if ele_val is not None else np.nan,
                float(sp_val) if sp_val is not None else np.nan,
            )
        )

    if not pts:
        return pd.DataFrame(columns=["time", "lat", "lon", "ele", "speed"])

    df = pd.DataFrame(pts, columns=["time", "lat", "lon", "ele", "speed"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def parse_route_file(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".gpx":
        return parse_gpx(path)
    if ext == ".tcx":
        return parse_tcx(path)
    if ext == ".fit":
        return parse_fit(path)
    print(f"[WARN] Unsupported route format: {Path(path).name}")
    return pd.DataFrame(columns=["time", "lat", "lon", "ele", "speed"])


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
            track_df.lat.iat[i],     track_df.lon.iat[i]
        )
        dists.append(d)

    track_df["dist_m"] = dists
    track_df["cum_dist_m"] = track_df["dist_m"].cumsum()
    return float(track_df["dist_m"].sum())


# ---------------------------
# OSM: map surface categories
# ---------------------------

def map_osm_surface_category(row):
    """
    Map raw OSM tags (surface, highway, tracktype) to high-level categories:
    - 'asphalt'
    - 'gravel'
    - 'unpaved'
    - 'paved'
    - 'natural'
    - 'unknown'
    """
    surf = row.get("surface", None)
    hw = row.get("highway", None)
    track = row.get("tracktype", None)

    if isinstance(surf, (list, tuple)):
        surf = surf[0] if surf else None

    surf_str = str(surf).strip().lower() if surf not in (None, float("nan")) else ""
    hw_str = str(hw).strip().lower() if hw not in (None, float("nan")) else ""
    track_str = str(track).strip().lower() if track not in (None, float("nan")) else ""

    # 1) explicit surface tag
    if surf_str:
        if surf_str in {"asphalt"}:
            return "asphalt"
        if surf_str in {"gravel", "fine_gravel"}:
            return "gravel"
        if surf_str in {
            "paved",
            "concrete",
            "concrete:plates",
            "concrete:lanes",
            "paving_stones",
            "sett",
            "cobblestone"
        }:
            return "paved"
        if surf_str in {"grass", "forest", "wood", "meadow"}:
            return "natural"
        if surf_str in {"ground", "dirt", "earth", "mud", "sand"}:
            return "unpaved"
        return "unpaved"

    # 2) infer from highway + tracktype
    if hw_str in {
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link",
        "residential", "living_street",
        "service", "unclassified",
        "trunk", "trunk_link"
    }:
        return "asphalt"

    if hw_str in {"cycleway"}:
        return "paved"

    if hw_str == "track":
        if track_str in {"grade1"}:
            return "paved"
        if track_str in {"grade2", "grade3", "grade4", "grade5"}:
            return "unpaved"
        return "unpaved"

    if hw_str in {"path", "footway", "bridleway"}:
        return "natural"

    return "unknown"


def classify_surface_osm(track_df: pd.DataFrame, sample_step: int = 1, verbose: bool = False) -> dict:
    """
    Given a route track_df with columns ['lat', 'lon', 'dist_m'], use OpenStreetMap
    to estimate how much distance was ridden on each surface type.

    sample_step:
        1 -> use every segment (most accurate)
        >1 -> sub-sample for speed, distances then approximated
    """
    if track_df.empty or "lat" not in track_df or "lon" not in track_df or "dist_m" not in track_df:
        return {}

    margin = 0.001
    north = track_df["lat"].max() + margin
    south = track_df["lat"].min() - margin
    east = track_df["lon"].max() + margin
    west = track_df["lon"].min() - margin

    try:
        bbox = (west, south, east, north)
        G = ox_graph.graph_from_bbox(
            bbox=bbox,
            network_type="bike",
        )
    except Exception as e:
        if verbose:
            print(f"[OSM] Failed to download graph for this track: {e}")
        return {"unknown": float(track_df["dist_m"].sum())}

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

    surface_dist = Counter()
    n = len(track_df)

    mids_lat = []
    mids_lon = []
    seg_dists = []

    for i in range(1, n):
        d = float(track_df["dist_m"].iat[i])
        if d <= 0:
            continue
        if i % sample_step != 0:
            continue

        lat1, lon1 = track_df["lat"].iat[i - 1], track_df["lon"].iat[i - 1]
        lat2, lon2 = track_df["lat"].iat[i], track_df["lon"].iat[i]

        mid_lat = 0.5 * (lat1 + lat2)
        mid_lon = 0.5 * (lon1 + lon2)

        mids_lat.append(mid_lat)
        mids_lon.append(mid_lon)
        seg_dists.append(d)

    if not mids_lat:
        return {"unknown": float(track_df["dist_m"].sum())}

    try:
        nearest = ox.distance.nearest_edges(G, mids_lon, mids_lat)
    except Exception as e:
        if verbose:
            print(f"[OSM] nearest_edges failed: {e}")
        return {"unknown": float(track_df["dist_m"].sum())}

    for (u, v, k), seg_d in zip(nearest, seg_dists):
        try:
            row = edges.loc[(u, v, k)]
        except KeyError:
            surface_cat = "unknown"
        else:
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            surface_cat = map_osm_surface_category(row)

        surface_dist[surface_cat] += seg_d

    return surface_dist


# ---------------------------
# CSV parsing (phone / IMU) – UPDATED VIBRATION HANDLING
# ---------------------------

def load_csv_sensor(csv_path: str):
    """
    Load a Sensor Logger CSV and compute a linear-acceleration magnitude signal.

    - Uses accelerometer_x/y/z (in m/s^2).
    - If gravity_x/y/z are present, subtract them to get linear acceleration.
    - Otherwise, subtract the mean from each axis.
    - Returns the DataFrame and the name of the vibration column: 'acc_lin_mag'.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # optional time column
    if "seconds_elapsed" in df.columns:
        df["seconds_elapsed"] = pd.to_numeric(df["seconds_elapsed"], errors="coerce")

    # base accelerometer (required)
    for c in ["accelerometer_x", "accelerometer_y", "accelerometer_z"]:
        if c not in df.columns:
            raise ValueError(f"{csv_path}: missing expected column '{c}'")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- build linear acceleration ---
    has_gravity = all(col in df.columns for col in ["gravity_x", "gravity_y", "gravity_z"])

    if has_gravity:
        df["gravity_x"] = pd.to_numeric(df["gravity_x"], errors="coerce")
        df["gravity_y"] = pd.to_numeric(df["gravity_y"], errors="coerce")
        df["gravity_z"] = pd.to_numeric(df["gravity_z"], errors="coerce")

        # Best case: subtract gravity vector (platform-neutral)
        df["ax_lin"] = df["accelerometer_x"] - df["gravity_x"]
        df["ay_lin"] = df["accelerometer_y"] - df["gravity_y"]
        df["az_lin"] = df["accelerometer_z"] - df["gravity_z"]
    else:
        # Fallback: remove mean from each axis to remove constant 1g
        df["ax_lin"] = df["accelerometer_x"] - df["accelerometer_x"].mean()
        df["ay_lin"] = df["accelerometer_y"] - df["accelerometer_y"].mean()
        df["az_lin"] = df["accelerometer_z"] - df["accelerometer_z"].mean()

    # Vibration magnitude in m/s^2
    df["acc_lin_mag"] = np.sqrt(
        df["ax_lin"]**2 + df["ay_lin"]**2 + df["az_lin"]**2
    )

    signal_col = "acc_lin_mag"
    return df, signal_col


# ---------------------------
# Spectral analysis
# ---------------------------

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    return float(np.sqrt(np.nanmean(x ** 2)))


def one_euro_filter(signal, freq, mincutoff, beta):
    if freq <= 0:
        return signal

    def alpha(cutoff):
        te = 1.0 / freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    x = np.asarray(signal, dtype=float)
    if x.size == 0:
        return x

    dx = np.zeros_like(x)
    dx[1:] = freq * (x[1:] - x[:-1])

    edx = np.zeros_like(x)
    a_d = alpha(mincutoff)
    edx[0] = dx[0]
    for i in range(1, len(x)):
        edx[i] = a_d * dx[i] + (1.0 - a_d) * edx[i - 1]

    out = np.zeros_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        cutoff = mincutoff + beta * abs(edx[i])
        a = alpha(cutoff)
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]

    return out

def index_fft(signal_1d, sample_rate: float = 1.0, target_nfft: int = 4096):
    """
    FFT helper.

    - If sample_rate=1.0 (default), the returned 'freqs' are in 'cycles per sample'
      (exactly like the original index-based version).
    - If you pass the real sample_rate in Hz, 'freqs' will be in Hz.
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

    # if sample_rate = 1.0 → freq axis is effectively "cycles per sample"
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)

    asd = (2.0 / nfft) * np.abs(X)
    return freqs, asd



# ---------------------------
# Shifting analysis (direct from FIT)
# ---------------------------

def summarize_shifting_from_fit(fit_path: str, verbose: bool = False) -> dict:
    """
    Directly read the FIT file and derive a gear index sequence.

    Priority:
      1) If 'rear_gear_num' exists (SRAM index), use that (values 1–12).
      2) Else if 'rear_gear' exists (SRAM teeth), sort unique teeth and map
         to indices 1..N.
      3) Else fall back to generic 'gear*-like' field detection.

    Then:
      - count how often the gear index changes  -> shifting events
      - count how many samples in each gear    -> gear_usage
    """
    fname = Path(fit_path).name

    if not HAVE_FITPARSE:
        if verbose:
            print(f"[SHIFT] {fname}: fitparse not installed.")
        return {"file": fname, "num_shifts": 0, "gear_usage": {}, "gear_column": None}

    try:
        fit = FitFile(fit_path)
        fit.parse()
    except Exception as e:
        if verbose:
            print(f"[SHIFT] {fname}: failed to parse FIT ({e}).")
        return {"file": fname, "num_shifts": 0, "gear_usage": {}, "gear_column": None}

    gear_values_by_field = {}

    for msg in fit.get_messages():
        vals = {f.name: f.value for f in msg}
        for k, v in vals.items():
            if "gear" not in k.lower():
                continue
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            if math.isnan(val):
                continue
            gear_values_by_field.setdefault(k, []).append(val)

    if not gear_values_by_field:
        if verbose:
            print(f"[SHIFT] {fname}: no 'gear' fields in FIT.")
        return {"file": fname, "num_shifts": 0, "gear_usage": {}, "gear_column": None}

    # Case 1: rear_gear_num present
    if "rear_gear_num" in gear_values_by_field:
        seq_raw = np.array(gear_values_by_field["rear_gear_num"], dtype=float)
        seq = [int(round(x)) for x in seq_raw if not math.isnan(x)]
        uniq_vals = sorted(set(seq))
        if verbose:
            print(
                f"[SHIFT] {fname}: using 'rear_gear_num' as gear index "
                f"(values={uniq_vals}) with {len(seq)} samples."
            )
        gear_column_used = "rear_gear_num"

    # Case 2: use rear_gear teeth mapped to indices
    elif "rear_gear" in gear_values_by_field:
        seq_teeth = np.array(gear_values_by_field["rear_gear"], dtype=float)
        seq_teeth = [int(round(x)) for x in seq_teeth if not math.isnan(x)]
        uniq_teeth = sorted(set(seq_teeth))
        tooth_to_idx = {t: i + 1 for i, t in enumerate(uniq_teeth)}
        seq = [tooth_to_idx[t] for t in seq_teeth]
        if verbose:
            print(
                f"[SHIFT] {fname}: using 'rear_gear' TEETH mapped to indices "
                f"(teeth={uniq_teeth} -> 1..{len(uniq_teeth)}) with {len(seq)} samples."
            )
        gear_column_used = "rear_gear(teeth_mapped)"

    # Case 3: generic fall-back
    else:
        candidates = []
        for col, arr in gear_values_by_field.items():
            s = np.array(arr, dtype=float)
            s = s[~np.isnan(s)]
            if s.size == 0:
                continue
            uniq = np.unique(np.round(s).astype(int))
            if len(uniq) >= 2 and len(uniq) <= 30 and uniq.min() >= 1 and uniq.max() <= 30:
                candidates.append((col, len(s), uniq))

        if not candidates:
            if verbose:
                print(f"[SHIFT] {fname}: gear fields exist but none look like a 1–30 index.")
            return {"file": fname, "num_shifts": 0, "gear_usage": {}, "gear_column": None}

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_col, n_valid, uniq_vals = candidates[0]
        if verbose:
            print(
                f"[SHIFT] {fname}: using column '{best_col}' as gear index "
                f"(values={list(uniq_vals)}) with {n_valid} samples."
            )
        seq_raw = np.array(gear_values_by_field[best_col], dtype=float)
        seq = [int(round(x)) for x in seq_raw if not math.isnan(x)]
        gear_column_used = best_col

    # Now compute shifts and usage
    shifts = 0
    usage = Counter()
    prev = None
    for g in seq:
        usage[g] += 1
        if prev is None:
            prev = g
            continue
        if g != prev:
            shifts += 1
            prev = g

    if verbose:
        print(f"[SHIFT] {fname}: detected {shifts} shifting events.")
        for g, c in usage.most_common(10):
            print(f"        gear {g}: {c} samples")

    return {
        "file": fname,
        "num_shifts": int(shifts),
        "gear_usage": {int(k): int(v) for k, v in usage.items()},
        "gear_column": gear_column_used,
    }

# ---------------------------
# Questionnaire analysis
# ---------------------------

def load_questionnaire(path: str) -> pd.DataFrame:
    """
    Load a questionnaire CSV.

    Assumptions (generic):
    - One row per participant.
    - Optional column 'participant_id' or 'id'.
    - All Likert / slider answers are numeric columns (0–100 etc.).
    Non-numeric columns are kept but not analysed statistically.
    """
    p = Path(path)
    if not p.exists():
        print(f"[SURVEY] File not found: {p}")
        return pd.DataFrame()

    df = pd.read_csv(p)
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df


def summarise_questionnaire(df: pd.DataFrame) -> dict:
    """
    Summarise all questionnaire columns.

    - Numeric / Likert-style columns:
        type = "numeric", with mean, std, min, max, n
    - Categorical / text columns (e.g. value ranges, gender, km/year):
        type = "categorical", with counts and percentages per category
    """
    summary: dict[str, dict] = {}

    # Work on a copy to avoid side effects
    df = df.copy()

    for col in df.columns:
        s = df[col]

        # Drop empty / pure-NaN responses
        s_non_null = s.dropna()
        if s_non_null.empty:
            continue

        # Try to interpret as numeric
        numeric = pd.to_numeric(s_non_null, errors="coerce")
        n_numeric = numeric.notna().sum()

        # Heuristic: numeric if at least half of the non-null values are numeric
        # and at least 3 numeric responses
        if n_numeric >= max(3, int(0.5 * len(s_non_null))):
            vals = numeric[numeric.notna()]
            if vals.empty:
                continue

            summary[col] = {
                "type": "numeric",
                "n": int(vals.size),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        else:
            # Treat as categorical: value ranges, text labels, etc.
            vc = s_non_null.astype(str).value_counts(dropna=False)
            total = float(len(s_non_null))
            categories = []
            for label, count in vc.items():
                categories.append({
                    "label": str(label),
                    "count": int(count),
                    "percent": float(100.0 * count / total),
                })

            summary[col] = {
                "type": "categorical",
                "n": int(total),
                "categories": categories,
            }

    return summary




# ---------------------------
# Main
# ---------------------------

def main(route_dir: str = "data/gps", csv_dir: str = "data/csv", out_dir: str = "outputs"):
    cfg = load_config("config.json")
    route_dir = cfg.get("route_dir", route_dir)
    csv_dir = cfg.get("csv_dir", csv_dir)
    out_dir = cfg.get("out_dir", out_dir)

    # --------------------------------------------------------------
    # LOAD PHONE–GARMIN TIME OFFSETS (activity_start.txt file)
    # --------------------------------------------------------------
    offset_map: dict[int, float] = {}
    offset_path = cfg.get("activity_start_file", None)

    if offset_path and Path(offset_path).exists():
        df_off = pd.read_csv(offset_path)
        df_off["phone_start_iso"] = pd.to_datetime(df_off["phone_start_iso"])
        df_off["fit_start_iso"] = pd.to_datetime(df_off["fit_start_iso"])

        # offset = phone_start - fit_start
        # So that: t_aligned = seconds_elapsed + offset  → time since FIT start
        for _, row in df_off.iterrows():
            pid = int(row["participant_id"])
            offset_sec = (row["phone_start_iso"] - row["fit_start_iso"]).total_seconds()
            offset_map[pid] = float(offset_sec)

        print("[OFFSET] Loaded offsets for participants:", offset_map)
    else:
        print("[OFFSET] No valid activity_start_file in config, skipping alignment.")

    smoothen_signal = bool(cfg.get("smoothen_signal", False))
    freq = float(cfg.get("freq", 0)) if cfg.get("freq", 0) is not None else 0.0
    mincutoff = float(cfg.get("mincutoff", 0.1))
    beta = float(cfg.get("beta", 0.0))
    verbose = bool(cfg.get("verbose", False))

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    route_paths = sorted(
        glob.glob(os.path.join(route_dir, "*.fit")) +
        glob.glob(os.path.join(route_dir, "*.gpx")) +
        glob.glob(os.path.join(route_dir, "*.tcx"))
    )
    csv_paths = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))

    csv_objects = []
    for p in csv_paths:
        df, signal_col = load_csv_sensor(p)
        csv_objects.append({
            "path": p,
            "stem": Path(p).stem,
            "df": df,
            "signal_col": signal_col,
        })

    total_km = 0.0
    surface_dist_m = Counter()
    surface_by_route = []          # per-route surface table
    shifting_summaries = []

    # Speed & elevation accumulators
    total_time_s = 0.0
    global_max_speed_kmh = 0.0
    total_elev_gain_m = 0.0
    total_elev_loss_m = 0.0
    num_routes_with_ele = 0
    route_stats = []   # per-route statistics (for JSON)

    # Garmin speed time series per participant (for alignment)
    garmin_time_by_pid: dict[int, np.ndarray] = {}
    garmin_speed_by_pid: dict[int, np.ndarray] = {}

    # Vibration (index-based spectrum, global)
    target_nfft = 4096
    sum_asd = None
    count_asd = 0
    common_bins = None
    per_file_vib = []

    # NEW: vibration vs speed (time-domain RMS + spectral windows)
    vibration_speed_summary: list[dict[str, float]] = []
    vibration_freq_speed_records: list[dict[str, Any]] = []

    print("\n=== ANALYSIS START ===\n")
    if verbose:
        print(f"[CONFIG] route_dir = {route_dir}")
        print(f"[CONFIG] csv_dir   = {csv_dir}")
        print(f"[CONFIG] out_dir   = {out_dir}")
        print(f"[CONFIG] smoothen_signal = {smoothen_signal}, freq = {freq}, "
              f"mincutoff = {mincutoff}, beta = {beta}, verbose = {verbose}\n")

    # --------------------------------------------------------------
    # 1) Distance + OSM surface + shifting from routes (Garmin)
    # --------------------------------------------------------------
    for rpath in route_paths:
        r_df = parse_route_file(rpath)
        if r_df.empty:
            if verbose:
                print(f"[WARN] Empty/unsupported route: {Path(rpath).name}")
            continue

        total_m = compute_track_distances(r_df)
        total_km += total_m / 1000.0

        if verbose:
            print(f"[ROUTE] {Path(rpath).name}  distance = {total_m / 1000.0:,.2f} km")

        # --- speed statistics for this route ---
        duration_s = np.nan
        avg_speed_kmh = np.nan
        max_speed_kmh = np.nan

        t_fit_s = None
        speed_mps_series = None

        if "time" in r_df.columns:
            t_valid = r_df["time"].dropna()
            if len(t_valid) >= 2:
                duration_s = (t_valid.iloc[-1] - t_valid.iloc[0]).total_seconds()
                # relative time in seconds since FIT start (for alignment)
                t0 = t_valid.iloc[0]
                t_fit_s = (t_valid - t0).dt.total_seconds().to_numpy()

                if duration_s > 0:
                    avg_speed_kmh = (total_m / 1000.0) / (duration_s / 3600.0)

        # max speed + speed series
        if "speed" in r_df.columns and r_df["speed"].notna().any():
            sp_mps = r_df["speed"].to_numpy(dtype=float)
            sp_kmh = sp_mps * 3.6
            max_speed_kmh = float(np.nanmax(sp_kmh))
            speed_mps_series = sp_mps
        else:
            # fallback: compute from GPS distance / dt (more noisy)
            if "time" in r_df.columns and "dist_m" in r_df.columns:
                dt = r_df["time"].diff().dt.total_seconds().to_numpy()
                dist = r_df["dist_m"].to_numpy()
                speed_mps = np.zeros_like(dist, dtype=float)
                mask = dt > 0
                speed_mps[mask] = dist[mask] / dt[mask]
                if mask.any():
                    max_speed_kmh = float(speed_mps.max() * 3.6)
                speed_mps_series = speed_mps

        # --- elevation statistics for this route ---
        elev_gain_m = np.nan
        elev_loss_m = np.nan
        if "ele" in r_df.columns and r_df["ele"].notna().any():
            ele = r_df["ele"].to_numpy(dtype=float)

            ele_s = pd.Series(ele).rolling(window=5, center=True, min_periods=1).median().to_numpy()
            dele = np.diff(ele_s)

            elev_min_step = 0.5
            if dele.size:
                gain = dele[dele > elev_min_step].sum()
                loss = -dele[dele < -elev_min_step].sum()
            else:
                gain = loss = 0.0

            elev_gain_m = float(gain)
            elev_loss_m = float(loss)
        else:
            gain = loss = 0.0

        # accumulate global stats
        if duration_s == duration_s and duration_s > 0:  # not NaN
            total_time_s += duration_s
        if max_speed_kmh == max_speed_kmh:  # not NaN
            global_max_speed_kmh = max(global_max_speed_kmh, max_speed_kmh)
        if elev_gain_m == elev_gain_m:  # not NaN
            total_elev_gain_m += elev_gain_m
            total_elev_loss_m += elev_loss_m
            num_routes_with_ele += 1

        # store per-route stats for JSON
        route_stats.append({
            "file": Path(rpath).name,
            "distance_km": total_m / 1000.0,
            "duration_s": duration_s,
            "avg_speed_kmh": avg_speed_kmh,
            "max_speed_kmh": max_speed_kmh,
            "elev_gain_m": elev_gain_m,
            "elev_loss_m": elev_loss_m,
        })

        # --- surface ---
        surf_dist_this_route = classify_surface_osm(r_df, sample_step=1, verbose=verbose)
        if verbose:
            if surf_dist_this_route:
                print("  [SURFACE OSM] Breakdown for this route:")
                tot_route = sum(surf_dist_this_route.values())
                for lab, dist_m in sorted(surf_dist_this_route.items(), key=lambda kv: -kv[1]):
                    print(f"    - {lab}: {dist_m/1000.0:6.2f} km ({100.0*dist_m/tot_route:5.1f}%)")
            else:
                print("  [SURFACE OSM] No OSM result, counting as 'unknown'.")

        if not surf_dist_this_route:
            surf_dist_this_route = {"unknown": total_m}

        # extract participant id from the filename: match "User_0" or "User 0"
        fname = Path(rpath).name
        stem = Path(rpath).stem
        pid = None

        m = re.search(r"User[_ ](\d+)", stem)
        if m:
            pid = int(m.group(1))   # direct match
        else:
            if verbose:
                print(f"[WARN] Could not extract PID from filename: {stem}")

        for lab, dist_m in surf_dist_this_route.items():
            surface_dist_m[lab] += float(dist_m)

            # per-route surface row
            surface_by_route.append({
                "participant_id": pid,
                "file": fname,
                "surface": lab,
                "distance_km": float(dist_m) / 1000.0,
            })

        # store Garmin time & speed for this participant (for later alignment)
        if pid is not None and t_fit_s is not None and speed_mps_series is not None:
            n = min(len(t_fit_s), len(speed_mps_series))
            if n > 1:
                garmin_time_by_pid[pid] = np.asarray(t_fit_s[:n], dtype=float)
                garmin_speed_by_pid[pid] = np.asarray(speed_mps_series[:n], dtype=float)

        # --- shifting (FIT only) ---
        if Path(rpath).suffix.lower() == ".fit":
            shift_info = summarize_shifting_from_fit(rpath, verbose=verbose)
            shifting_summaries.append(shift_info)

    # --------------------------------------------------------------
    # 2) Global vibration per CSV (index-based spectrum, as before)
    # --------------------------------------------------------------
    for obj in csv_objects:
        sig = obj["signal_col"]
        s_raw = pd.to_numeric(obj["df"][sig], errors="coerce").dropna().values
        if s_raw.size < 32:
            if verbose:
                print(f"[SKIP] {Path(obj['path']).name}: too few samples for spectrum.")
            continue

        if smoothen_signal and freq > 0:
            s = one_euro_filter(s_raw, freq=freq, mincutoff=mincutoff, beta=beta)
        else:
            s = s_raw

        # 🔧 NEW: estimate sample rate for this file from 'seconds_elapsed' if present
        df_phone = obj["df"]
        if "seconds_elapsed" in df_phone.columns:
            t_phone_glob = pd.to_numeric(df_phone["seconds_elapsed"], errors="coerce").to_numpy(dtype=float)
            dt_glob = np.diff(t_phone_glob)
            dt_glob = dt_glob[dt_glob > 0]
            if dt_glob.size > 0:
                sample_rate_global = 1.0 / float(np.nanmedian(dt_glob))
            else:
                sample_rate_global = freq if freq > 0 else 100.0
        else:
            sample_rate_global = freq if freq > 0 else 100.0

        # pass sample_rate into index_fft (now returns Hz)
        bins, asd = index_fft(s, sample_rate_global, target_nfft=target_nfft)
        if bins.size == 0:
            if verbose:
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

    # --------------------------------------------------------------
    # 3) VIBRATION vs SPEED (aligned phone + Garmin, in Hz)
    # --------------------------------------------------------------
    if offset_map and garmin_time_by_pid:
        for obj in csv_objects:
            stem = obj["stem"]
            m = re.search(r"User[_ ](\d+)", stem)
            if not m:
                continue
            pid = int(m.group(1))

            if pid not in offset_map:
                if verbose:
                    print(f"[VIB] No offset for pid={pid}, skipping vibration-speed alignment.")
                continue
            if pid not in garmin_time_by_pid:
                if verbose:
                    print(f"[VIB] No Garmin speed for pid={pid}, skipping vibration-speed alignment.")
                continue

            df_phone = obj["df"]
            if "seconds_elapsed" not in df_phone.columns:
                if verbose:
                    print(f"[VIB] CSV {obj['path']} missing 'seconds_elapsed', skipping.")
                continue

            # phone time in seconds (since app started)
            t_phone = pd.to_numeric(df_phone["seconds_elapsed"], errors="coerce").to_numpy(dtype=float)
            acc_vals = pd.to_numeric(df_phone[obj["signal_col"]], errors="coerce").to_numpy(dtype=float)

            # estimate sampling rate from seconds_elapsed
            dt = np.diff(t_phone)
            dt = dt[dt > 0]
            if dt.size == 0:
                if verbose:
                    print(f"[VIB] Could not estimate sample rate for {obj['path']}, skipping.")
                continue
            sample_rate = 1.0 / float(np.nanmedian(dt))   # Hz

            # align phone times to FIT-relative seconds
            offset_sec = offset_map[pid]
            t_aligned = t_phone + offset_sec  # now ~seconds since FIT start

            t_fit = garmin_time_by_pid[pid]
            sp_mps = garmin_speed_by_pid[pid]
            if len(t_fit) < 2:
                continue

            # interpolate Garmin speed to phone timestamps
            speed_interp = np.interp(t_aligned, t_fit, sp_mps)

            # windowed RMS + FFT vs speed (2-second windows)
            window_s = 2.0
            N = int(window_s * sample_rate)
            if N <= 0 or N > len(acc_vals):
                continue

            n_max = len(acc_vals) - N

            for i in range(0, n_max, N):
                win_acc = acc_vals[i:i + N]
                win_speed = speed_interp[i:i + N]

                if np.all(np.isnan(win_acc)) or np.all(np.isnan(win_speed)):
                    continue

                vib_rms = float(rms(win_acc))
                v_mean_kmh = float(np.nanmean(win_speed) * 3.6)  # m/s → km/h

                # --- FFT for this window (returns Hz) ---
                try:
                    freqs, asd = index_fft(win_acc, sample_rate=sample_rate, target_nfft=4096)

                    idx_max = int(np.nanargmax(asd))
                    peak_hz = float(freqs[idx_max])
                    peak_amp = float(asd[idx_max])

                    vibration_freq_speed_records.append({
                        "participant_id": pid,
                        "speed_kmh": v_mean_kmh,
                        "peak_hz": peak_hz,
                        "peak_amp": peak_amp
                    })
                except Exception as e:
                    print(f"[WARN] FFT failed for pid={pid}, window={i}: {e}")

                # RMS-level summary
                vibration_speed_summary.append({
                    "participant_id": pid,
                    "speed_kmh": v_mean_kmh,
                    "rms_m_s2": vib_rms,
                })

        if verbose and vibration_speed_summary:
            print(f"[VIB] Computed vibration_vs_speed records: {len(vibration_speed_summary)}")
    else:
        if verbose:
            print("[VIB] Skipping vibration_vs_speed (no offsets or no Garmin data).")

    # --------------------------------------------------------------
    # 4) Global distance summary
    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    # 5) Global speed & elevation summary
    # --------------------------------------------------------------
    print("\n--- SPEED & ELEVATION SUMMARY ---")
    if total_time_s > 0:
        overall_avg_speed_kmh = total_km / (total_time_s / 3600.0)
        print(f"Overall average speed: {overall_avg_speed_kmh:5.1f} km/h")
    else:
        overall_avg_speed_kmh = None
        print("Overall average speed: n/a (no valid time data)")

    if global_max_speed_kmh > 0:
        print(f"Maximum instantaneous speed: {global_max_speed_kmh:5.1f} km/h")
    else:
        print("Maximum instantaneous speed: n/a")

    if num_routes_with_ele > 0:
        avg_gain_per_ride = total_elev_gain_m / num_routes_with_ele
        avg_loss_per_ride = total_elev_loss_m / num_routes_with_ele
        print(f"Total elevation gain (all rides): {total_elev_gain_m:7.1f} m")
        print(f"Total elevation loss (all rides): {total_elev_loss_m:7.1f} m")
        print(f"Average elevation gain per ride: {avg_gain_per_ride:6.1f} m")
        print(f"Average elevation loss per ride: {avg_loss_per_ride:6.1f} m")
    else:
        avg_gain_per_ride = None
        avg_loss_per_ride = None
        print("Elevation data: n/a")

    # --------------------------------------------------------------
    # 6) Global gear summary
    # --------------------------------------------------------------
    global_gear_usage = Counter()
    total_shifts_all = 0
    for info in shifting_summaries:
        total_shifts_all += info.get("num_shifts", 0)
        for g, c in info.get("gear_usage", {}).items():
            global_gear_usage[int(g)] += int(c)

    print("\n--- GEAR SUMMARY (all participants) ---")
    print(f"Total shifting events (all files): {total_shifts_all}")
    if global_gear_usage:
        print("Gear usage (index → samples):")
        for g in sorted(global_gear_usage.keys()):
            print(f"  - gear {g}: {global_gear_usage[g]} samples")
    else:
        print("  No gear data available.")

    # --------------------------------------------------------------
    # 7) Vibration summary (index-based)
    # --------------------------------------------------------------
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
            arr[0] = 0.0
        idx = np.argsort(arr)[-5:][::-1]
        print("Top 5 spectral peaks (freq in Hz):")   # <- text updated
        for i in idx:
            print(f"  - f={avg_bins[i]:.3f} Hz  amplitude={arr[i]:.6g}")
        avg_spectrum = {"bins_cyc_per_sample": avg_bins, "amplitude": avg_asd}

        # Time-domain RMS summary (nice for the thesis)
        if per_file_vib:
            rms_vals = np.array([v["rms"] for v in per_file_vib], dtype=float)
            mean_rms = float(np.nanmean(rms_vals))
            min_rms = float(np.nanmin(rms_vals))
            max_rms = float(np.nanmax(rms_vals))
            print("\n--- VIBRATION RMS (time-domain) ---")
            print(f"Mean RMS vibration across rides: {mean_rms:.3f} m/s²")
            print(f"Min / max RMS across rides:      {min_rms:.3f} / {max_rms:.3f} m/s²")

    # --------------------------------------------------------------
    # 8) QUESTIONNAIRE ANALYSIS
    # --------------------------------------------------------------
    pre_survey_path = cfg.get("pre_survey_csv", None)
    post_survey_path = cfg.get("post_survey_csv", None)

    survey_results = {
        "pre_summary": None,
        "post_summary": None,
    }

    if pre_survey_path:
        pre_df = load_questionnaire(pre_survey_path)
        if not pre_df.empty:
            print(f"[SURVEY] Loaded pre-questionnaire: {pre_survey_path}")
            survey_results["pre_summary"] = summarise_questionnaire(pre_df)
        else:
            pre_df = pd.DataFrame()
    else:
        pre_df = pd.DataFrame()

    if post_survey_path:
        post_df = load_questionnaire(post_survey_path)
        if not post_df.empty:
            print(f"[SURVEY] Loaded post-questionnaire: {post_survey_path}")
            survey_results["post_summary"] = summarise_questionnaire(post_df)
        else:
            post_df = pd.DataFrame()
    else:
        post_df = pd.DataFrame()

    # --------------------------------------------------------------
    # 9) Pack results for JSON
    # --------------------------------------------------------------
    results = {
        "distance_km_total": total_km,
        "surface_breakdown": {k: v for k, v in surface_dist_m.items()},
        "surface_by_route": surface_by_route,
        "routes": route_stats,
        "speed_elevation_summary": {
            "total_time_s": total_time_s,
            "overall_avg_speed_kmh": overall_avg_speed_kmh,
            "max_speed_kmh": global_max_speed_kmh,
            "total_elev_gain_m": total_elev_gain_m,
            "total_elev_loss_m": total_elev_loss_m,
            "avg_gain_per_ride_m": avg_gain_per_ride,
            "avg_loss_per_ride_m": avg_loss_per_ride,
            "num_routes_with_ele": num_routes_with_ele,
        },
        "vibration": {
            "averaged": avg_spectrum,
            "per_file": per_file_vib,
            "nfft": target_nfft,
        },
        # simple compatibility key if plot_bike_data.py still expects this:
        "vibration_vs_speed": vibration_speed_summary,
        # richer structure with spectra:
        "vibration_speed": {
            "windows_rms": vibration_speed_summary,
            "windows_spectra": vibration_freq_speed_records,
        },
        "shifting": {
            "per_file": shifting_summaries,
            "combined": {
                "total_shifts": total_shifts_all,
                "gear_usage": {str(k): int(v) for k, v in global_gear_usage.items()},
            },
        },
        "questionnaires": survey_results,
        "config_used": {
            "smoothen_signal": smoothen_signal,
            "freq": freq,
            "mincutoff": mincutoff,
            "beta": beta,
            "verbose": verbose,
        },
    }

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_json = Path(out_dir) / "analysis_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Saved results → {out_json}")
    print("\n=== ANALYSIS END ===\n")


if __name__ == "__main__":
    main(
        route_dir=os.environ.get("ROUTE_DIR", "data/gps"),
        csv_dir=os.environ.get("CSV_DIR", "data/csv"),
        out_dir=os.environ.get("OUT_DIR", "outputs"),
    )

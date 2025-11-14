# analyze_bike_data.py
"""
Analysis script for:
- Computing total distance from Garmin route files (FIT/GPX/TCX).
- Estimating road-surface distance breakdown using OpenStreetMap (OSM) based on GPS track.
- Computing an average vibration spectrum from accelerometer data CSVs.
- Printing summaries to the terminal.
- Saving a compact JSON file for separate plotting.

We keep the FIT files so that you can later extend this script to extract shifting,
power, cadence or any other Garmin metrics.

CONFIG
------
Uses a config.json file in the project root, e.g.:

{
  "route_dir": "data/gps",
  "csv_dir": "data/csv",
  "out_dir": "outputs",
  "smoothen_signal": true,
  "freq": 120,
  "mincutoff": 0.1,
  "beta": 0.1
}
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

import osmnx as ox
import geopandas as gpd
from osmnx import graph as ox_graph  # NEW: proper graph_from_bbox import

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
    """Convert Garmin 'semicircles' to degrees."""
    return s * (180.0 / 2 ** 31)


def parse_fit(path: str) -> pd.DataFrame:
    """
    Minimal FIT parser: only GPS track.
    Surface is not taken from FIT, but inferred from OSM later.
    """
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
            track_df.lat.iat[i],     track_df.lon.iat[i]
        )
        dists.append(d)

    track_df["dist_m"] = dists
    track_df["cum_dist_m"] = track_df["dist_m"].cumsum()
    return float(track_df["dist_m"].sum())


# ---------------------------
def map_osm_surface_category(row):
    """
    Map raw OSM tags (surface, highway, tracktype) to 6 high-level categories:
    - 'asphalt'
    - 'gravel'
    - 'unpaved'
    - 'paved'
    - 'natural'
    - 'unknown'

    All decisions are still based on OpenStreetMap tags.
    """

    surf = row.get("surface", None)
    hw = row.get("highway", None)
    track = row.get("tracktype", None)

    # normalise values
    if isinstance(surf, (list, tuple)):
        surf = surf[0] if surf else None

    surf_str = str(surf).strip().lower() if surf not in (None, float("nan")) else ""
    hw_str = str(hw).strip().lower() if hw not in (None, float("nan")) else ""
    track_str = str(track).strip().lower() if track not in (None, float("nan")) else ""

    # 1) EXPLICIT SURFACE TAG WINS
    if surf_str:
        # Asphalt (pure asphalt)
        if surf_str in {"asphalt"}:
            return "asphalt"

        # Gravel-like
        if surf_str in {"gravel", "fine_gravel"}:
            return "gravel"

        # Paved, but not explicitly asphalt
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

        # Clearly natural surfaces (grass, forest floor, etc.)
        if surf_str in {"grass", "forest", "wood", "meadow"}:
            return "natural"

        # Other unsealed / loose / dirt-like
        if surf_str in {"ground", "dirt", "earth", "mud", "sand"}:
            return "unpaved"

        # If we get something exotic, keep it but group as unpaved
        return "unpaved"

    # 2) NO SURFACE TAG → INFER FROM HIGHWAY + TRACKTYPE

    # Typical paved roads
    if hw_str in {
        "primary", "primary_link",
        "secondary", "secondary_link",
        "tertiary", "tertiary_link",
        "residential", "living_street",
        "service", "unclassified",
        "trunk", "trunk_link"
    }:
        # we assume paved tarmac here
        return "asphalt"

    # Cycleways in urban areas are usually paved
    if hw_str in {"cycleway"}:
        return "paved"

    # Agricultural / forest tracks
    if hw_str == "track":
        # OSM convention: grade1 = solid / paved / compacted
        if track_str in {"grade1"}:
            return "paved"
        # grades 2–5 are progressively rougher unpaved
        if track_str in {"grade2", "grade3", "grade4", "grade5"}:
            return "unpaved"
        # if no grade, assume unpaved track
        return "unpaved"

    # Paths / footways / bridleways: usually natural or unpaved
    if hw_str in {"path", "footway", "bridleway"}:
        return "natural"

    # If we really cannot infer anything → Unknown
    return "unknown"

# ---------------------------

def classify_surface_osm(track_df: pd.DataFrame, sample_step: int = 1) -> dict:
    """
    Given a route track_df with columns ['lat', 'lon', 'dist_m'], use OpenStreetMap
    to estimate how much distance was ridden on each surface type.

    sample_step:
        1 -> use every segment (most accurate, still fine for 10 rides)
        >1 -> sub-sample for speed, distances then approximated
    """
    if track_df.empty or "lat" not in track_df or "lon" not in track_df or "dist_m" not in track_df:
        return {}

    # Bounding box around the ride (+ small margin, ~100 m)
    margin = 0.001
    north = track_df["lat"].max() + margin
    south = track_df["lat"].min() - margin
    east  = track_df["lon"].max() + margin
    west  = track_df["lon"].min() - margin

    try:
        # OSMnx 2.x: use graph_from_bbox from osmnx.graph with a bbox tuple
        # bbox = (left, bottom, right, top) = (west, south, east, north)
        bbox = (west, south, east, north)
        G = ox_graph.graph_from_bbox(
            bbox=bbox,
            network_type="bike",  # or "all" if you want everything
        )
    except Exception as e:
        print(f"[OSM] Failed to download graph for this track: {e}")
        return {"unknown": float(track_df["dist_m"].sum())}

    # Convert edges to GeoDataFrame (for 'surface', 'highway', 'tracktype')
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

        # Sub-sampling if sample_step > 1
        if i % sample_step != 0:
            continue

        lat1, lon1 = track_df["lat"].iat[i - 1], track_df["lon"].iat[i - 1]
        lat2, lon2 = track_df["lat"].iat[i],     track_df["lon"].iat[i]

        mid_lat = 0.5 * (lat1 + lat2)
        mid_lon = 0.5 * (lon1 + lon2)

        mids_lat.append(mid_lat)
        mids_lon.append(mid_lon)
        seg_dists.append(d)

    if not mids_lat:
        # No samples taken: count everything as unknown
        return {"unknown": float(track_df["dist_m"].sum())}

    try:
        nearest = ox.distance.nearest_edges(G, mids_lon, mids_lat)
    except Exception as e:
        print(f"[OSM] nearest_edges failed: {e}")
        return {"unknown": float(track_df["dist_m"].sum())}

    # IMPORTANT: if sample_step > 1, you could scale distances,
    # but we now default to sample_step = 1 (full coverage).
    for (u, v, k), seg_d in zip(nearest, seg_dists):
        try:
            row = edges.loc[(u, v, k)]
        except KeyError:
            surface_cat = "unknown"
        else:
            if isinstance(row, pd.DataFrame):
                # Multi-match, use first row
                row = row.iloc[0]
            surface_cat = map_osm_surface_category(row)

        surface_dist[surface_cat] += seg_d

    return surface_dist




# ---------------------------
# CSV parsing (phone / IMU)
# ---------------------------

def load_csv_sensor(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    if "seconds_elapsed" in df.columns:
        df["seconds_elapsed"] = pd.to_numeric(df["seconds_elapsed"], errors="coerce")

    for c in ["accelerometer_x", "accelerometer_y", "accelerometer_z"]:
        if c not in df.columns:
            raise ValueError(f"{csv_path}: missing expected column '{c}'")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["acc_mag"] = np.sqrt(
        df["accelerometer_x"] ** 2 +
        df["accelerometer_y"] ** 2 +
        df["accelerometer_z"] ** 2
    )

    signal_col = "acc_mag"
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


def index_fft(signal_1d, target_nfft: int = 4096):
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
    bins = np.fft.rfftfreq(nfft, d=1.0)
    asd = (2.0 / nfft) * np.abs(X)
    return bins, asd


# ---------------------------
# Main
# ---------------------------

def main(route_dir: str = "data/gps", csv_dir: str = "data/csv", out_dir: str = "outputs"):
    cfg = load_config("config.json")
    route_dir = cfg.get("route_dir", route_dir)
    csv_dir = cfg.get("csv_dir", csv_dir)
    out_dir = cfg.get("out_dir", out_dir)

    smoothen_signal = bool(cfg.get("smoothen_signal", False))
    freq = float(cfg.get("freq", 0)) if cfg.get("freq", 0) is not None else 0.0
    mincutoff = float(cfg.get("mincutoff", 0.1))
    beta = float(cfg.get("beta", 0.0))

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

    target_nfft = 4096
    sum_asd = None
    count_asd = 0
    common_bins = None
    per_file_vib = []

    print("\n=== ANALYSIS START ===\n")
    print(f"[CONFIG] route_dir = {route_dir}")
    print(f"[CONFIG] csv_dir   = {csv_dir}")
    print(f"[CONFIG] out_dir   = {out_dir}")
    print(f"[CONFIG] smoothen_signal = {smoothen_signal}, freq = {freq}, "
          f"mincutoff = {mincutoff}, beta = {beta}\n")

    # --- Distance + OSM surface from routes ---
    for rpath in route_paths:
        r_df = parse_route_file(rpath)
        if r_df.empty:
            print(f"[WARN] Empty/unsupported route: {Path(rpath).name}")
            continue

        total_m = compute_track_distances(r_df)
        total_km += total_m / 1000.0
        print(f"[ROUTE] {Path(rpath).name}  distance = {total_m / 1000.0:,.2f} km")

        surf_dist_this_route = classify_surface_osm(r_df)

        if surf_dist_this_route:
            print("  [SURFACE OSM] Breakdown for this route:")
            tot_route = sum(surf_dist_this_route.values())
            for lab, dist_m in sorted(surf_dist_this_route.items(), key=lambda kv: -kv[1]):
                print(f"    - {lab}: {dist_m/1000.0:6.2f} km ({100.0*dist_m/tot_route:5.1f}%)")
        else:
            print("  [SURFACE OSM] No OSM result, counting as 'unknown'.")
            surf_dist_this_route = {"unknown": total_m}

        for lab, dist_m in surf_dist_this_route.items():
            surface_dist_m[lab] += float(dist_m)

    # --- Vibration per CSV ---
    for obj in csv_objects:
        sig = obj["signal_col"]
        s_raw = pd.to_numeric(obj["df"][sig], errors="coerce").dropna().values
        if s_raw.size < 32:
            print(f"[SKIP] {Path(obj['path']).name}: too few samples for spectrum.")
            continue

        if smoothen_signal and freq > 0:
            s = one_euro_filter(s_raw, freq=freq, mincutoff=mincutoff, beta=beta)
        else:
            s = s_raw

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
            arr[0] = 0.0
        idx = np.argsort(arr)[-5:][::-1]
        print("Top 5 spectral peaks (bin in cycles/sample):")
        for i in idx:
            print(f"  - bin={avg_bins[i]:.5f}  amplitude={arr[i]:.6g}")
        avg_spectrum = {"bins_cyc_per_sample": avg_bins, "amplitude": avg_asd}

    results = {
        "distance_km_total": total_km,
        "surface_breakdown": {k: v for k, v in surface_dist_m.items()},
        "vibration": {
            "averaged": avg_spectrum,
            "per_file": per_file_vib,
            "nfft": target_nfft,
        },
        "config_used": {
            "smoothen_signal": smoothen_signal,
            "freq": freq,
            "mincutoff": mincutoff,
            "beta": beta,
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

# plot_bike_data.py
"""
Plotting script for analyze_bike_data.py results, using Plotly.

Generates:
- Vibration spectrum (in Hz, 0–30 Hz)
- RMS vibration per ride (bar chart)
- Road-surface breakdown (bar chart with percentages)

Usage examples:

    # With manually specified sampling rate:
    python plot_bike_data.py --fs_hz 94.44

    # Let the script estimate sampling rate from CSV timestamps:
    python plot_bike_data.py --auto_fs
"""

import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ---------------------------------------------------
# CONFIG LOADERS
# ---------------------------------------------------

def load_config(path="config.json"):
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------
# AUTO SAMPLING RATE ESTIMATION
# ---------------------------------------------------

def auto_sampling_rate(csv_dir: str):
    """
    Estimate sampling rate from the first CSV file with a 'seconds_elapsed' column.
    Returns fs in Hz or None if estimation fails.
    """
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print("[AUTO_FS] No CSV files found.")
        return None

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if "seconds_elapsed" not in df.columns:
            continue

        t = pd.to_numeric(df["seconds_elapsed"], errors="coerce").dropna().values
        if len(t) < 5:
            continue

        dt = np.diff(t)
        dt = dt[dt > 0]
        if len(dt) == 0:
            continue

        fs = 1.0 / np.median(dt)
        print(f"[AUTO_FS] Estimated sampling rate from {csv_path.name}: {fs:.2f} Hz")
        return fs

    print("[AUTO_FS] Could not estimate sampling rate from CSV files.")
    return None


# ---------------------------------------------------
# PLOTTING HELPERS
# ---------------------------------------------------

def make_figure_layout(title, x_label, y_label, cfg):
    template = cfg.get("plotly_template", "plotly_white")
    width = cfg.get("stimulus_width", 1280)
    height = cfg.get("stimulus_height", 720)
    font_family = cfg.get("font_family", "Open Sans, verdana, arial, sans-serif")
    font_size = cfg.get("font_size", 20)

    layout = dict(
        title=title,
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        template=template,
        width=width,
        height=height,
        font=dict(family=font_family, size=font_size),
        margin=dict(l=80, r=40, t=80, b=80),
    )
    return layout


def save_figure(fig: go.Figure, out_html: Path, out_png: Path, save_figures: bool):
    if not save_figures:
        print("[SAVE] Skipping save (save_figures = False). Showing interactively only.")
        return

    out_html.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(out_html))
    print(f"[SAVE] Saved HTML → {out_html}")

    try:
        fig.write_image(str(out_png))
        print(f"[SAVE] Saved PNG  → {out_png}")
    except Exception as e:
        print(f"[WARN] PNG save failed: {e}. Install 'kaleido' if missing.")


# ---------------------------------------------------
# MAIN PLOT LOGIC
# ---------------------------------------------------

def main(results_path: str,
         title_spectrum: str,
         title_rms: str,
         title_surface: str,
         fs_hz: float | None,
         auto_fs: bool):

    cfg = load_config("config.json")

    out_dir = Path(cfg.get("out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    save_figures = bool(cfg.get("save_figures", True))

    results = load_results(results_path)

    vib = results.get("vibration", {})
    surface_breakdown = results.get("surface_breakdown", {})

    # ------------------------------------------------
    # Determine sampling rate
    # ------------------------------------------------
    if auto_fs:
        csv_dir = cfg.get("csv_dir", "data/csv")
        fs_auto = auto_sampling_rate(csv_dir)
        if fs_auto:
            fs_hz = fs_auto

    if fs_hz is None:
        print("[INFO] No sampling rate provided → spectrum in cycles/sample.")
        convert_to_hz = False
    else:
        convert_to_hz = True
        print(f"[INFO] Using sampling rate fs = {fs_hz:.2f} Hz")

    # ------------------------------------------------
    # 1) VIBRATION SPECTRUM
    # ------------------------------------------------
    avg_spec = vib.get("averaged")
    if avg_spec:
        bins = np.array(avg_spec["bins_cyc_per_sample"])
        amp = np.array(avg_spec["amplitude"])

        if convert_to_hz:
            freq = bins * fs_hz
            mask = (freq >= 5) & (freq <= 70)
            freq = freq[mask]
            amp = amp[mask]
            x_label = "Frequency (Hz)"
        else:
            freq = bins
            x_label = "Frequency (cycles/sample)"

        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(
            x=freq,
            y=amp,
            mode="lines",
            line=dict(width=3),
            name="Averaged spectrum"
        ))

        fig_spec.update_layout(
            **make_figure_layout(title_spectrum, x_label, "Amplitude (m/s²)", cfg)
        )

        save_figure(
            fig_spec,
            out_html=out_dir / "vibration_spectrum.html",
            out_png=out_dir / "vibration_spectrum.png",
            save_figures=save_figures,
        )
    else:
        print("[PLOT] No vibration spectrum found.")

    # ------------------------------------------------
    # 2) RMS VIBRATION PER FILE
    # ------------------------------------------------
    per_file = vib.get("per_file", [])
    if per_file:
        names = [entry["file"] for entry in per_file]
        rms_vals = [entry["rms"] for entry in per_file]

        order = np.argsort(rms_vals)[::-1]
        names = [names[i] for i in order]
        rms_vals = [rms_vals[i] for i in order]

        fig_rms = go.Figure()
        fig_rms.add_trace(go.Bar(
            x=names,
            y=rms_vals,
            text=[f"{v:.2f}" for v in rms_vals],
            textposition="outside"
        ))

        fig_rms.update_layout(
            **make_figure_layout(title_rms, "File", "RMS (m/s²)", cfg)
        )
        # Set tick angle separately to avoid passing xaxis twice
        fig_rms.update_xaxes(tickangle=-60)

        save_figure(
            fig_rms,
            out_html=out_dir / "vibration_rms_per_file.html",
            out_png=out_dir / "vibration_rms_per_file.png",
            save_figures=save_figures,
        )
    else:
        print("[PLOT] No RMS data.")

    # ------------------------------------------------
    # 3) SURFACE BREAKDOWN
    # ------------------------------------------------
    if surface_breakdown:
        labels = list(surface_breakdown.keys())
        vals_m = [surface_breakdown[k] for k in labels]
        vals_km = [v / 1000 for v in vals_m]
        total_m = sum(vals_m)
        pct = [100 * v / total_m for v in vals_m] if total_m > 0 else [0 for _ in vals_m]

        fig_surf = go.Figure()
        fig_surf.add_trace(go.Bar(
            x=labels,
            y=vals_km,
            text=[f"{p:.1f}%" for p in pct],
            textposition="outside"
        ))

        fig_surf.update_layout(
            **make_figure_layout(title_surface, "Surface type", "Distance (km)", cfg)
        )

        save_figure(
            fig_surf,
            out_html=out_dir / "surface_breakdown.html",
            out_png=out_dir / "surface_breakdown.png",
            save_figures=save_figures,
        )
    else:
        print("[PLOT] No surface data.")

    print("[PLOT] Completed.")


# ---------------------------------------------------
# CLI
# ---------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot bike vibration & surface data using Plotly")
    parser.add_argument("--results", default="outputs/analysis_results.json")
    parser.add_argument("--title_spectrum", default="Vibration Spectrum (0–30 Hz)")
    parser.add_argument("--title_rms", default="RMS Vibration per Ride")
    parser.add_argument("--title_surface", default="Road Surface Breakdown")
    parser.add_argument("--fs_hz", type=float, default=None, help="Sampling rate in Hz")
    parser.add_argument("--auto_fs", action="store_true", help="Automatically estimate sampling rate")

    args = parser.parse_args()

    main(
        results_path=args.results,
        title_spectrum=args.title_spectrum,
        title_rms=args.title_rms,
        title_surface=args.title_surface,
        fs_hz=args.fs_hz,
        auto_fs=args.auto_fs,
    )

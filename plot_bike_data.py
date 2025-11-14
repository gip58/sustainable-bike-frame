# plot_bike_data.py
"""
Plotting script for the analysis results produced by analyze_bike_data.py.

Integrates basic settings from config.json for:
- out_dir (for saving figure images)
- font_family, font_size
- stimulus_width / stimulus_height
- save_figures flag

Reads outputs/analysis_results.json and plots:
  * Average vibration spectrum (cycles/sample, or Hz if you supply fs_hz).
  * Per-file RMS vibration.
  * Road-surface distance breakdown.

Usage:

    python plot_bike_data.py

Or with custom titles and sampling rate:

    python plot_bike_data.py --fs_hz 120 \
        --title_spectrum "Average Spectrum (120 Hz sampling)" \
        --title_rms "Vibration RMS per Ride" \
        --title_surface "Surface Distance Breakdown"
"""

import json
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Config loader
# ---------------------------

def load_config(path="config.json"):
    """
    Load configuration settings from a JSON file if it exists.
    Returns an empty dict if not found or invalid.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_results(path):
    """Load the JSON results produced by analyse_bike_data.py."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(results_path,
         title_spectrum,
         title_rms,
         title_surface,
         hz_if_known=None):
    # Load config
    cfg = load_config("config.json")

    out_dir = cfg.get("out_dir", "outputs")
    save_figures = bool(cfg.get("save_figures", False))

    # Plot styling
    font_family = cfg.get("font_family", None)
    font_size = cfg.get("font_size", None)
    if font_family is not None:
        plt.rcParams["font.family"] = font_family
    if font_size is not None:
        plt.rcParams["font.size"] = font_size

    # Figure size (convert pixels to inches assuming 100 dpi)
    dpi = 100.0
    stim_w = cfg.get("stimulus_width", 1280)
    stim_h = cfg.get("stimulus_height", 720)
    fig_w_inches = stim_w / dpi
    fig_h_inches = stim_h / dpi

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    R = load_results(results_path)

    # --- Average spectrum ---
    vib = R.get("vibration", {})
    avg = vib.get("averaged")
    if avg:
        x = np.array(avg["bins_cyc_per_sample"], dtype=float)
        y = np.array(avg["amplitude"], dtype=float)

        fig1 = plt.figure(figsize=(fig_w_inches, fig_h_inches), dpi=dpi)
        if hz_if_known and hz_if_known > 0:
            # Convert cycles/sample to Hz: f = (cycles/sample) * fs
            xf = x * hz_if_known
            plt.plot(xf, y)
            plt.xlabel("Frequency (Hz)")
        else:
            plt.plot(x, y)
            plt.xlabel("Frequency (cycles/sample)")

        plt.ylabel("Amplitude (arb. units)")
        plt.title(title_spectrum)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_figures:
            fig_path = out_dir_path / "spectrum.png"
            fig1.savefig(fig_path, dpi=dpi)
            print(f"[SAVE] Spectrum figure saved to {fig_path}")

    # --- Per-file RMS ---
    per_file = vib.get("per_file", [])
    if per_file:
        files = [p["file"] for p in per_file]
        rms_vals = [float(p["rms"]) for p in per_file]

        # Sort by RMS descending
        order = np.argsort(rms_vals)[::-1]
        files = [files[i] for i in order]
        rms_vals = [rms_vals[i] for i in order]

        fig2 = plt.figure(figsize=(fig_w_inches, fig_h_inches), dpi=dpi)
        x_pos = np.arange(len(files))
        plt.bar(x_pos, rms_vals)
        plt.xticks(x_pos, files, rotation=75, ha="right", fontsize=8)
        plt.ylabel("RMS (same units as acc_mag)")
        plt.title(title_rms)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        if save_figures:
            fig_path = out_dir_path / "rms_per_file.png"
            fig2.savefig(fig_path, dpi=dpi)
            print(f"[SAVE] RMS figure saved to {fig_path}")

    # --- Surface breakdown ---
    surf = R.get("surface_breakdown", {})
    if surf:
        labels = list(surf.keys())
        vals_m = [surf[k] for k in labels]
        total = sum(vals_m) if vals_m else 0.0
        km = [v / 1000.0 for v in vals_m]
        pct = [100.0 * v / total if total > 0 else 0.0 for v in vals_m]

        fig3 = plt.figure(figsize=(fig_w_inches, fig_h_inches), dpi=dpi)
        positions = np.arange(len(labels))
        plt.bar(positions, km)
        plt.xticks(positions, labels, rotation=30, ha="right")
        plt.ylabel("Distance (km)")
        plt.title(title_surface)
        for i, v in enumerate(km):
            plt.text(i, v, f"{pct[i]:.1f}%", ha="center", va="bottom", fontsize=8)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        if save_figures:
            fig_path = out_dir_path / "surface_breakdown.png"
            fig3.savefig(fig_path, dpi=dpi)
            print(f"[SAVE] Surface breakdown figure saved to {fig_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot bike analysis results.")
    parser.add_argument(
        "--results",
        default="outputs/analysis_results.json",
        help="Path to analysis_results.json (default: outputs/analysis_results.json)",
    )
    parser.add_argument(
        "--title_spectrum",
        default="Average Vibration Spectrum",
        help="Title for the spectrum plot",
    )
    parser.add_argument(
        "--title_rms",
        default="Per-file Vibration RMS",
        help="Title for the RMS bar chart",
    )
    parser.add_argument(
        "--title_surface",
        default="Road-surface Distance Breakdown",
        help="Title for the surface breakdown chart",
    )
    parser.add_argument(
        "--fs_hz",
        type=float,
        default=None,
        help="Common sampling rate (Hz) if you want x-axis in Hz instead of cycles/sample.",
    )
    args = parser.parse_args()

    main(
        results_path=args.results,
        title_spectrum=args.title_spectrum,
        title_rms=args.title_rms,
        title_surface=args.title_surface,
        hz_if_known=args.fs_hz,
    )

# plot_bike_data.py
"""
Plotting script for the analysis results produced by analyse_bike_data.py.

- Reads outputs/analysis_results.json.
- Plots:
    * Average vibration spectrum (cycles/sample, or Hz if you supply fs_hz).
    * Per-file RMS vibration.
    * Road-surface distance breakdown (currently only 'unknown').

You can safely tweak titles and labels here without touching the analysis code.

Usage (from terminal):

    python plot_bike_data.py

Or with custom titles and sampling rate:

    python plot_bike_data.py --fs_hz 100 \
        --title_spectrum "Average Spectrum (100 Hz basis)" \
        --title_rms "Vibration RMS per Ride" \
        --title_surface "Surface Distance Breakdown"

"""

import json
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str):
    """Load the JSON results produced by analyse_bike_data.py."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(results_path: str,
         title_spectrum: str,
         title_rms: str,
         title_surface: str,
         hz_if_known: float | None = None):
    R = load_results(results_path)

    # --- Average spectrum ---
    vib = R.get("vibration", {})
    avg = vib.get("averaged")
    if avg:
        x = np.array(avg["bins_cyc_per_sample"], dtype=float)
        y = np.array(avg["amplitude"], dtype=float)

        plt.figure()
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

    # --- Per-file RMS ---
    per_file = vib.get("per_file", [])
    if per_file:
        files = [p["file"] for p in per_file]
        rms_vals = [float(p["rms"]) for p in per_file]

        # Sort by RMS descending
        order = np.argsort(rms_vals)[::-1]
        files = [files[i] for i in order]
        rms_vals = [rms_vals[i] for i in order]

        plt.figure()
        x_pos = np.arange(len(files))
        plt.bar(x_pos, rms_vals)
        plt.xticks(x_pos, files, rotation=75, ha="right", fontsize=8)
        plt.ylabel("RMS (same units as acc_mag)")
        plt.title(title_rms)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

    # --- Surface breakdown ---
    surf = R.get("surface_breakdown", {})
    if surf:
        labels = list(surf.keys())
        vals_m = [surf[k] for k in labels]
        total = sum(vals_m) if vals_m else 0.0
        km = [v / 1000.0 for v in vals_m]
        pct = [100.0 * v / total if total > 0 else 0.0 for v in vals_m]

        plt.figure()
        positions = np.arange(len(labels))
        plt.bar(positions, km)
        plt.xticks(positions, labels, rotation=30, ha="right")
        plt.ylabel("Distance (km)")
        plt.title(title_surface)
        for i, v in enumerate(km):
            plt.text(i, v, f"{pct[i]:.1f}%", ha="center", va="bottom", fontsize=8)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

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

# plot_bike_data.py
"""
Plotting script for the analysis results produced by analyze_bike_data.py.

Integrates basic settings from config.json for:
- out_dir (for saving figure images)
- font_family, font_size
- stimulus_width / stimulus_height
- plotly_template
- save_figures flag

Reads outputs/analysis_results.json and plots:
  * Average vibration spectrum (cycles/sample, or Hz if you supply fs_hz or --auto_fs).
  * Per-file RMS vibration.
  * Road-surface distance breakdown.
  * Pre-questionnaire:
        - Gender (pie chart)
        - Age (histogram)
        - Diverging stacked bar chart for the "Before buying, how important..."
          Likert block.

Usage examples
--------------
    python plot_bike_data.py
    python plot_bike_data.py --fs_hz 100
    python plot_bike_data.py --auto_fs

"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import math
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import re


# ---------------------------
# Config loader
# ---------------------------

def load_config(path: str = "config.json") -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ---------------------------
# Helpers for Plotly styling / saving
# ---------------------------

def get_plot_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "template": cfg.get("plotly_template", "plotly_white"),
        "font_family": cfg.get("font_family", "Open Sans, verdana, arial, sans-serif"),
        "font_size": cfg.get("font_size", 20),
        "width": cfg.get("stimulus_width", 1280),
        "height": cfg.get("stimulus_height", 720),
        "save_figures": bool(cfg.get("save_figures", True)),
        "out_dir": Path(cfg.get("out_dir", "outputs")),
    }


def apply_layout(fig: go.Figure, title: str, x_label: str, y_label: str, cfg: Dict[str, Any]) -> go.Figure:
    ps = get_plot_settings(cfg)
    fig.update_layout(
        template=ps["template"],
        title=title,
        font=dict(
            family=ps["font_family"],
            size=ps["font_size"],
        ),
        width=ps["width"],
        height=ps["height"],
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=80, r=40, t=80, b=80),
    )
    return fig


def save_figure(fig: go.Figure, filename_stem: str, cfg: Dict[str, Any]) -> None:
    ps = get_plot_settings(cfg)
    out_dir = ps["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"{filename_stem}.html"
    png_path = out_dir / f"{filename_stem}.png"

    # Optional vector format for LaTeX / Overleaf (e.g. "svg" or "eps")
    vector_fmt = cfg.get("vector_format", None)  # "svg", "eps", or None
    vector_path = None
    if vector_fmt:
        vector_fmt = vector_fmt.lower().strip()
        if vector_fmt in {"svg", "eps"}:
            vector_path = out_dir / f"{filename_stem}.{vector_fmt}"
        else:
            print(f"[WARN] Unsupported vector_format '{vector_fmt}'. Use 'svg' or 'eps'.")
            vector_fmt = None

    if ps["save_figures"]:
        # Always save HTML (interactive)
        fig.write_html(str(html_path))
        print(f"[SAVE] Saved HTML → {html_path}")

        # Raster PNG (for quick viewing / reports)
        try:
            fig.write_image(str(png_path))
            print(f"[SAVE] Saved PNG  → {png_path}")
        except Exception as e:
            print(f"[WARN] Could not save PNG ({e}). HTML is still saved.")

        # Optional vector export for Overleaf
        if vector_fmt and vector_path is not None:
            try:
                fig.write_image(str(vector_path), format=vector_fmt)
                print(f"[SAVE] Saved {vector_fmt.upper()} → {vector_path}")
            except Exception as e:
                print(f"[WARN] Could not save {vector_fmt.upper()} ({e}).")

# ---------------------------
# Load analysis results
# ---------------------------

def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# VIBRATION PLOTS
# ---------------------------


def plot_vibration_spectrum(cfg: Dict[str, Any]) -> None:
    """
    Plot the average vibration spectrum (5–70 Hz) using:
        results['vibration']['averaged']
    Automatically detects if bins are cycles/sample or already in Hz.
    Uses full-resolution FFT bins (THOUSANDS of points) + optional smoothing.

    Output saved using apply_layout() + save_figure().
    """
    # ----------------------------------------------------------
    # Load results.json
    # ----------------------------------------------------------
    results_path = cfg.get("results_path", "outputs/analysis_results.json")
    if not Path(results_path).exists():
        print(f"[VIB] results_path not found: {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    vib = results.get("vibration", {})
    av = vib.get("averaged", None)

    if av is None:
        print("[VIB] No averaged spectrum found.")
        return

    # raw FFT bins
    bins_raw = np.array(av.get("bins_cyc_per_sample", []), dtype=float)
    amp_raw = np.array(av.get("amplitude", []), dtype=float)

    if bins_raw.size == 0 or amp_raw.size == 0:
        print("[VIB] Empty arrays.")
        return

    # ----------------------------------------------------------
    # Detect whether bins are cycles/sample or already Hz
    # ----------------------------------------------------------
    if bins_raw.max() <= 0.6:
        # old-style bins (0–0.5 cycles/sample)
        fs = cfg.get("freq", 100.0)   # fallback to 100 Hz
        freq_hz = bins_raw * fs
    else:
        # already in Hz (new analysis pipeline)
        freq_hz = bins_raw

    # ----------------------------------------------------------
    # Frequency cut from config (default 5–70 Hz)
    # ----------------------------------------------------------
    fmin = float(cfg.get("vibration_freq_min_hz", 5.0))
    fmax = float(cfg.get("vibration_freq_max_hz", 70.0))
    mask = (freq_hz >= fmin) & (freq_hz <= fmax)

    if not mask.any():
        print(f"[VIB] No data in {fmin}–{fmax} Hz.")
        return

    freq_hz = freq_hz[mask]
    amp = amp_raw[mask]

    # ----------------------------------------------------------
    # Optional smoothing (moving average) — keeps ALL points
    # ----------------------------------------------------------
    smooth_bins = int(cfg.get("vibration_spectrum_smooth_bins", 9))  # 9 pts ≈ ~0.1 Hz at 100Hz
    if smooth_bins > 1:
        kernel = np.ones(smooth_bins) / smooth_bins
        amp_smooth = np.convolve(amp, kernel, mode="same")
    else:
        amp_smooth = amp

    # ----------------------------------------------------------
    # Plot full-resolution spectrum
    # ----------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freq_hz,
        y=amp_smooth,
        mode="lines",
        line=dict(width=2),
        name="Averaged spectrum"
    ))

    # optional raw underlay
    if bool(cfg.get("vibration_plot_raw_underlay", False)):
        fig.add_trace(go.Scatter(
            x=freq_hz,
            y=amp,
            mode="lines",
            opacity=0.3,
            line=dict(width=1),
            name="Raw spectrum"
        ))

    apply_layout(
        fig,
        "Average vibration spectrum",
        "Frequency (Hz)",
        "Amplitude (a.u.)",
        cfg
    )

    save_figure(fig, "vibration_spectrum", cfg)
    print("[VIB] Saved → vibration_spectrum")





def plot_vibration_rms(results: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    vib = results.get("vibration", {})
    per_file = vib.get("per_file", [])
    if not per_file:
        print("[VIB] No per-file RMS data.")
        return

    files = [item["file"] for item in per_file]
    rms_vals = [item["rms"] for item in per_file]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=files,
        y=rms_vals,
        name="RMS (m/s²)"
    ))
    fig.update_xaxes(tickangle=-60)
    apply_layout(fig, "RMS vibration per file", "File", "RMS (m/s²)", cfg)
    save_figure(fig, "vibration_rms_per_file", cfg)

def plot_vibration_vs_speed(results: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    Scatter plot of vibration (RMS m/s²) versus speed (km/h),
    using the 'vibration_vs_speed' entries computed in analyze_bike_data.py.
    Each point is one time window (e.g. 2 seconds) for one participant.
    """
    vib_vs_speed = results.get("vibration_vs_speed", None)
    if not vib_vs_speed:
        print("[PLOT] No 'vibration_vs_speed' data found in results – skipping.")
        return

    df = pd.DataFrame(vib_vs_speed)

    # Basic sanity: keep only finite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["speed_kmh", "rms_m_s2"])
    if df.empty:
        print("[PLOT] 'vibration_vs_speed' contains no valid rows – skipping.")
        return

    # Optional: lightly clip extreme values to keep the plot readable
    # (tweak thresholds if needed)
    df = df[(df["speed_kmh"] >= 0) & (df["speed_kmh"] <= 80)]
    df = df[(df["rms_m_s2"] >= 0)]

    if df.empty:
        print("[PLOT] 'vibration_vs_speed' all filtered out – skipping.")
        return

    fig = px.scatter(
        df,
        x="speed_kmh",
        y="rms_m_s2",
        color="participant_id",
        hover_data=["participant_id", "speed_kmh", "rms_m_s2"],
    )

    apply_layout(
        fig,
        "Vibration versus speed (RMS over time windows)",
        "Speed (km/h)",
        "RMS acceleration (m/s²)",
        cfg,
    )

    # Make axes slightly nicer
    try:
        max_v = float(df["speed_kmh"].max())
        fig.update_xaxes(range=[0, max_v * 1.05])
    except Exception:
        pass

    save_figure(fig, "vibration_vs_speed", cfg)




def plot_vibration_rms_vs_speed_binned(cfg: dict,
                                       results_path: str = "outputs/analysis_results.json",
                                       bin_width_kmh: float = 5.0):

    results_path = Path(results_path)
    if not results_path.exists():
        print(f"[VIB-SPEED] results file not found: {results_path}")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        res = json.load(f)

    vib_vs_speed = res.get("vibration_speed", {}).get("windows_rms", [])
    if not vib_vs_speed:
        print("[VIB-SPEED] No vibration_speed/windows_rms data found.")
        return

    df = pd.DataFrame(vib_vs_speed).dropna(subset=["speed_kmh", "rms_m_s2"])
    if df.empty:
        print("[VIB-SPEED] No valid RMS/speed rows.")
        return

    max_speed = df["speed_kmh"].max()
    bins = np.arange(0, max_speed + bin_width_kmh, bin_width_kmh)

    df["speed_bin"] = pd.cut(df["speed_kmh"], bins=bins, include_lowest=True)

    grouped = (
        df.groupby("speed_bin")["rms_m_s2"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )

    def bin_center(interval):
        return 0.5 * (interval.left + interval.right)

    grouped["speed_center"] = grouped["speed_bin"].apply(bin_center)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped["speed_center"],
        y=grouped["mean"],
        mode="lines+markers",
        name="RMS vibration",
        error_y=dict(
            type="data",
            array=grouped["std"].fillna(0.0),
            visible=True,
        ),
    ))

    apply_layout(
        fig,
        f"Vibration vs speed (RMS, {bin_width_kmh:g} km/h bins)",
        "Speed [km/h]",
        "RMS acceleration [m/s²]",
        cfg,
    )

    save_figure(fig, "vibration_rms_vs_speed_binned", cfg)
    print("[VIB-SPEED] Saved → vibration_rms_vs_speed_binned")




def plot_vibration_peak_freq_vs_speed_by_surface(results: dict, cfg: dict) -> None:
    """
    Subplots: dominant vibration frequency (Hz) vs cycling speed (km/h),
    split by SURFACE TYPE.
    ALL subplots forced to UNIFORM Plotly default blue using colorway override.
    """

    print("[VIB-FREQ-SURF] RUNNING FINAL SCATTER PLOT UNIFORM BLUE VERSION (COLORWAY OVERRIDE)")

    # ---- 0) Imports (Assumed available) ----
    import pandas as pd
    import numpy as np
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # ---- 1) Get data ----
    vib_speed = results.get("vibration_speed", {})
    records = vib_speed.get("windows_spectra", [])

    if not records:
        print("[VIB-FREQ-SURF] No data – skipping.")
        return

    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["speed_kmh", "peak_hz", "participant_id"])

    if df.empty:
        print("[VIB-FREQ-SURF] Empty dataframe – skipping.")
        return

    # ---- 2) Dominant surface per participant ----
    surf_df = pd.DataFrame(results.get("surface_by_route", []))
    if surf_df.empty:
        print("[VIB-FREQ-SURF] No surface data – skipping.")
        return

    grouped = (
        surf_df.groupby(["participant_id", "surface"], as_index=False)["distance_km"]
        .sum()
    )
    idx = grouped.groupby("participant_id")["distance_km"].idxmax()
    dom_surface = grouped.loc[idx]
    df["surface"] = df["participant_id"].map(
        dict(zip(dom_surface["participant_id"], dom_surface["surface"]))
    )

    # ---- 3) Surfaces to plot ----
    breakdown = results.get("surface_breakdown", {})
    surfaces = (
        sorted(breakdown, key=breakdown.get, reverse=True)[:3]
        if breakdown else df["surface"].value_counts().head(3).index.tolist()
    )

    # ---- 4) Subplots ----
    fig = make_subplots(
        rows=1,
        cols=len(surfaces),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[s.title() for s in surfaces],
        horizontal_spacing=0.06,
    )

    # ---- 5) Marker settings ----
    BLUE = "#636EFA"   # Plotly default blue
    OPACITY = 1.0      # Full opacity to prevent overlap shadowing

    if "peak_amp" in df.columns:
        amp = df["peak_amp"].astype(float)
        size = 6 + 18 * (amp - amp.min()) / (amp.max() - amp.min() + 1e-9)
    else:
        size = 10

    for i, s in enumerate(surfaces, start=1):
        dfi = df[df["surface"] == s]
        if dfi.empty:
            continue

        fig.add_trace(
            go.Scattergl(
                x=dfi["speed_kmh"],
                y=dfi["peak_hz"],
                mode="markers",
                name="Vibration Data",       # Group all traces under one name
                legendgroup="data",          # Group all traces under one legend group
                marker=dict(
                    size=size.loc[dfi.index] if hasattr(size, "loc") else size,
                    color=BLUE,              # Explicitly set color
                    opacity=OPACITY,
                ),
                showlegend=False,
            ),
            row=1,
            col=i,
        )

    # ---- 6) APPLY LAYOUT WITH COLORWAY OVERRIDE ----
    fig.update_layout(
        template=None,
        title="Dominant vibration frequency vs cycling speed (split by surface type)",
        margin=dict(l=90, r=30, t=80, b=90),
        font=dict(
            family=cfg.get("font_family", "Arial"),
            size=cfg.get("font_size", 18),
        ),
        # 🟢 CRITICAL CHANGE: Force the color cycle to only contain the desired BLUE
        colorway=[BLUE],
    )

    # ---- 7) Global axis labels ----
    fig.add_annotation(
        text="Speed (km/h)",
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.add_annotation(
        text="Dominant vibration frequency (Hz)",
        x=-0.10,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
    )

    save_figure(fig, "vibration_peak_freq_vs_speed_by_surface", cfg)
    print("[VIB-FREQ-SURF] Saved vibration_peak_freq_vs_speed_by_surface.")


def plot_vibration_peak_freq_vs_speed(results: dict, cfg: dict):
    """
    Scatter plot: dominant vibration frequency (Hz) vs speed (km/h),
    using the per-window FFT peak stored in:
        results["vibration_speed"]["windows_spectra"]
    """

    vib_speed = results.get("vibration_speed", {})
    records = vib_speed.get("windows_spectra", [])

    if not records:
        print("[VIB-FREQ] No vibration_speed/windows_spectra data found – skipping.")
        return

    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["speed_kmh", "peak_hz"])

    if df.empty:
        print("[VIB-FREQ] Dataframe empty after cleaning – skipping.")
        return

    fig = px.scatter(
        df,
        x="speed_kmh",
        y="peak_hz",
        color="participant_id",
        size="peak_amp",
        opacity=0.55,
        labels={
            "speed_kmh": "Speed (km/h)",
            "peak_hz": "Dominant vibration frequency (Hz)",
            "participant_id": "Participant",
            "peak_amp": "Amplitude"
        },
    )

    # use your global template + axes formatting
    apply_layout(
        fig,
        "Dominant vibration frequency vs speed by participant ",
        "Speed (km/h)",
        "Frequency (Hz)",
        cfg
    )

    # use your central saving system
    save_figure(fig, "vibration_peak_freq_vs_speed", cfg)

    print("[VIB-FREQ] Saved vibration_peak_freq_vs_speed.")


# ---------------------------
# SURFACE BREAKDOWN PLOT
# ---------------------------
def plot_vibration_peak_freq_vs_speed_by_surface(results: dict, cfg: dict) -> None:
    """
    Subplots: dominant vibration frequency (Hz) vs cycling speed (km/h),
    split by SURFACE TYPE (participant's dominant surface).
    """

    # ---- 1) Get window-level spectra ----
    vib_speed = results.get("vibration_speed", {})
    records = vib_speed.get("windows_spectra", [])

    if not records:
        print("[VIB-FREQ-SURF] No vibration_speed/windows_spectra – skipping.")
        return

    df = pd.DataFrame(records)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["speed_kmh", "peak_hz", "participant_id"])

    if df.empty:
        print("[VIB-FREQ-SURF] No valid peak frequency records – skipping.")
        return

    # ---- 2) Determine each participant's DOMINANT surface ----
    surf_by_route = results.get("surface_by_route", [])
    if not surf_by_route:
        print("[VIB-FREQ-SURF] No surface_by_route in results – cannot split by surface.")
        return

    surf_df = pd.DataFrame(surf_by_route)

    grouped = (
        surf_df.groupby(["participant_id", "surface"], as_index=False)["distance_km"]
        .sum()
    )

    idx = grouped.groupby("participant_id")["distance_km"].idxmax()
    dominant_surface = grouped.loc[idx, ["participant_id", "surface"]]

    pid_to_surface = dict(
        zip(dominant_surface["participant_id"], dominant_surface["surface"])
    )

    df["surface"] = df["participant_id"].map(pid_to_surface).fillna("unknown")

    # ---- 3) Choose surfaces to plot (top 3 by distance) ----
    surface_breakdown = results.get("surface_breakdown", {})
    if surface_breakdown:
        surfaces = [
            k for k, _ in sorted(
                surface_breakdown.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
        ]
    else:
        surfaces = df["surface"].value_counts().head(3).index.tolist()

    # ---- 4) Colour palette (consistent with previous plots) ----
    base_colors = {
        "natural": "#636EFA",
        "asphalt": "#636EFA",
        "unpaved": "#636EFA",
        "paved": "#636EFA",
        "unknown": "#636EFA",
    }

    # ---- 5) Create subplots ----
    fig = make_subplots(
        rows=1,
        cols=len(surfaces),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=[s.replace("_", " ").title() for s in surfaces],
        horizontal_spacing=0.06,
    )

    # Marker size scaling (peak amplitude if available)
    if "peak_amp" in df.columns:
        amp = df["peak_amp"].astype(float)
        amp_norm = (amp - amp.min()) / (amp.max() - amp.min() + 1e-9)
        size_px = 6 + 18 * amp_norm  # 6–24 px
    else:
        size_px = 10

    for i, s in enumerate(surfaces, start=1):
        dfi = df[df["surface"] == s]
        if dfi.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=dfi["speed_kmh"],
                y=dfi["peak_hz"],
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=size_px.loc[dfi.index] if hasattr(size_px, "loc") else size_px,
                    color=base_colors.get(s, "#999999"),
                    opacity=0.55,
                    line=dict(width=0),
                ),
                hovertemplate=(
                    f"Surface: {s}"
                    "<br>Speed: %{x:.1f} km/h"
                    "<br>Dominant frequency: %{y:.1f} Hz"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=i,
        )

    # ---- 6) Apply YOUR standard layout (NO per-axis titles) ----
    apply_layout(
        fig,
        "Dominant vibration frequency vs cycling speed (split by surface type)",
        "",   # no per-axis x-title
        "",   # no per-axis y-title
        cfg,
    )

    # ---- 7) Single global axis labels ----
    fig.add_annotation(
        text="Speed (km/h)",
        x=0.5,
        y=-0.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=cfg.get("font_size", 18)),
    )

    fig.add_annotation(
        text="Dominant vibration frequency (Hz)",
        x=-0.08,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=dict(size=cfg.get("font_size", 18)),
    )

    save_figure(fig, "vibration_peak_freq_vs_speed_by_surface", cfg)
    print("[VIB-FREQ-SURF] Saved vibration_peak_freq_vs_speed_by_surface.")



def plot_surface_breakdown(results: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    surf = results.get("surface_breakdown", {})
    if not surf:
        print("[SURFACE] No surface breakdown in results.")
        return

    labels = list(surf.keys())
    dist_m = np.array(list(surf.values()), dtype=float)
    dist_km = dist_m / 1000.0
    total_km = dist_km.sum()
    if total_km <= 0:
        print("[SURFACE] Total distance is zero.")
        return

    perc = 100.0 * dist_km / total_km

    text_labels = [f"{d:.1f} km<br>{p:.1f}%" for d, p in zip(dist_km, perc)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=dist_km,
        text=text_labels,
        textposition="outside",
        hovertemplate="Surface: %{x}<br>Distance: %{y:.1f} km<br>Share: %{text}<extra></extra>",
        name="Distance (km)",
    ))
    apply_layout(fig, "Road-surface breakdown (distance)", "Surface type", "Distance (km)", cfg)
    save_figure(fig, "surface_breakdown", cfg)



# ---------------------------
# QUESTIONNAIRE PLOTS (PRE)
# ---------------------------

def load_pre_questionnaire_from_config(cfg: Dict[str, Any]) -> pd.DataFrame | None:
    pre_path = cfg.get("pre_survey_csv", None)
    if not pre_path:
        print("[SURVEY] pre_survey_csv not set in config.json – skipping questionnaire plots.")
        return None
    pre_path = Path(pre_path)
    if not pre_path.exists():
        print(f"[SURVEY] Pre-questionnaire CSV not found: {pre_path}")
        return None
    try:
        df = pd.read_csv(pre_path)
        print(f"[SURVEY] Loaded pre-questionnaire from {pre_path}")
        return df
    except Exception as e:
        print(f"[SURVEY] Failed to read pre-questionnaire: {e}")
        return None


def load_questionnaire(path: str) -> pd.DataFrame:
    """
    Simple questionnaire loader for the plotting script.
    Just reads a CSV with UTF-8 encoding.
    """
    return pd.read_csv(path)



def find_column(df: pd.DataFrame, prefix: str) -> str | None:
    """
    Return the first column whose name starts with the given prefix.
    Prints a message and returns None if not found.
    """
    for c in df.columns:
        if str(c).strip().startswith(prefix):
            return c
    print(f"[SURVEY] Column starting with '{prefix}' not found.")
    return None



def plot_gender_pie(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    col = "What is your gender?"
    if col not in pre_df.columns:
        print(f"[SURVEY] Column not found for gender: {col}")
        return

    s = pre_df[col].dropna().astype(str)
    if s.empty:
        print("[SURVEY] No gender data.")
        return

    vc = s.value_counts()
    labels = vc.index.tolist()
    values = vc.values.tolist()

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.0,
        textinfo="label+value+percent",
        hovertemplate="%{label}<br>%{value} participants<br>%{percent}<extra></extra>",
    )])
    apply_layout(fig, "Participant gender distribution", "", "", cfg)
    save_figure(fig, "survey_gender_pie", cfg)

def plot_age_hist(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    col = "What is your age?"
    if col not in pre_df.columns:
        print(f"[SURVEY] Column not found for age: {col}")
        return

    s = pd.to_numeric(pre_df[col], errors="coerce").dropna()
    if s.empty:
        print("[SURVEY] No age data.")
        return

    # 5-year bins
    min_age = int(s.min())
    max_age = int(s.max())
    start = min_age - (min_age % 5)
    end = max_age + (5 - max_age % 5) if max_age % 5 != 0 else max_age
    bins = list(range(start, end + 5, 5))

    # Bin ages and count participants per bin
    cats = pd.cut(s, bins=bins, right=False)  # [x, y)
    vc = cats.value_counts().sort_index()

    bin_labels = [f"{int(interval.left)}–{int(interval.right - 1)}" for interval in vc.index]
    counts = vc.values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_labels,
        y=counts,
        text=[str(c) for c in counts],
        textposition="outside",
        hovertemplate="Age group: %{x}<br>Participants: %{y}<extra></extra>",
        name="Age",
    ))
    apply_layout(fig, "Participant age distribution", "Age group (years)", "Number of participants", cfg)
    save_figure(fig, "survey_age_hist", cfg)

def plot_km_last_12_months(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    prefix = "About how many kilometers (miles) did you cycle in the last 12 months?"
    col = find_column(pre_df, prefix)
    if not col:
        return

    s_raw = pre_df[col].dropna()

    # Try numeric first
    s_num = pd.to_numeric(s_raw, errors="coerce")
    if s_num.notna().sum() >= 0.5 * len(s_raw):
        s = s_num.dropna()
        if s.empty:
            print("[SURVEY] No numeric data for km/year.")
            return

        max_km = int(np.ceil(s.max() / 1000.0) * 1000)
        bins = list(range(0, max_km + 1000, 1000))
        cats = pd.cut(s, bins=bins, right=False)
        vc = cats.value_counts().sort_index()

        labels = [f"{int(i.left)}–{int(i.right - 1)}" for i in vc.index]
        counts = vc.values
    else:
        # Treat as categorical ranges, but sort them by the lower bound of the range
        vc = s_raw.astype(str).value_counts()

        def sort_key(label: str) -> int:
            # find first number in the label, e.g. "1 - 1,000 km (1 - 621 mi)"
            m = re.search(r"\d[\d,]*", label)
            if m:
                return int(m.group(0).replace(",", ""))
            return 10**9  # push weird labels to the end

        labels = sorted(vc.index.tolist(), key=sort_key)
        counts = [vc[l] for l in labels]

    total = sum(counts)
    texts = [f"{c} ({100.0 * c / total:.0f}%)" for c in counts]


    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        text=texts,
        textposition="outside",
        hovertemplate="Km/year group: %{x}<br>Participants: %{y}<extra></extra>",
        name="Km/year",
    ))
    apply_layout(fig, "Distance cycled in last 12 months", "Km/year (groups)", "Number of participants", cfg)
    save_figure(fig, "survey_km_last_12_months", cfg)

def plot_cycling_frequency(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    """
    Plot: 'On average, how often have you cycle in the last 12 months?'
    Robust to small changes and non-breaking spaces in the header.
    """
    col = None
    target_snippet = "on average, how often have you cycle"
    for c in pre_df.columns:
        c_norm = str(c).replace("\xa0", " ").lower()
        if target_snippet in c_norm:
            col = c
            break

    if not col:
        print("[SURVEY] Frequency column not found (snippet search).")
        return

    s = pre_df[col].dropna().astype(str)
    if s.empty:
        print("[SURVEY] No frequency data.")
        return

    vc = s.value_counts()

    # --- custom logical order for the categories ---
    desired_order_norm = [
        "once a month to once a week",
        "1 to 3 days a week",
        "4 to 6 days a week",
        "every day or almost every day",
    ]
    order_map = {name: i for i, name in enumerate(desired_order_norm)}

    def sort_key(label: str) -> int:
        norm = label.strip().lower()
        return order_map.get(norm, len(desired_order_norm) + 1)

    labels = sorted(vc.index.tolist(), key=sort_key)
    counts = [vc[l] for l in labels]

    # FIX: counts is a list here → use built-in sum()
    total = sum(counts)
    texts = [f"{c} ({100.0 * c / total:.0f}%)" for c in counts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        text=texts,
        textposition="outside",
        hovertemplate="Frequency: %{x}<br>Participants: %{y}<extra></extra>",
        name="Frequency",
    ))
    apply_layout(
        fig,
        "Cycling frequency in last 12 months",
        "Frequency category",
        "Number of participants",
        cfg,
    )
    save_figure(fig, "survey_cycling_frequency", cfg)




def plot_height(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    prefix = "What is your height?"
    col = find_column(pre_df, prefix)
    if not col:
        return

    s = pd.to_numeric(pre_df[col], errors="coerce").dropna()
    if s.empty:
        print("[SURVEY] No height data.")
        return

    h_min = float(s.min())
    h_max = float(s.max())
    h_mean = float(s.mean())

    fig = go.Figure()

    # --- main box plot ---
    fig.add_trace(
        go.Box(
            y=s,
            name="Height",
            boxmean=True,
            hovertemplate="Height: %{y} cm<extra></extra>",
        )
    )

    # Horizontal reference lines
    for value, label, colour in [
        (h_mean, f"Mean: {h_mean:.0f} cm", "blue"),
        (h_min, f"Min: {h_min:.0f} cm", "green"),
        (h_max, f"Max: {h_max:.0f} cm", "red"),
    ]:
        fig.add_shape(
            type="line",
            x0=-0.3, x1=0.3,   # width of the boxplot area
            y0=value, y1=value,
            line=dict(color=colour, width=2, dash="dash")
        )
        fig.add_annotation(
            x=0.35,
            y=value,
            text=label,
            showarrow=False,
            xanchor="left",
            font=dict(size=12, color=colour)
        )

    apply_layout(
        fig,
        "Participant height distribution",
        "",
        "Height (cm)",
        cfg,
    )

    save_figure(fig, "survey_height", cfg)




def plot_years_cycling(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    prefix = "For how many years have you been cycling regularly?"
    col = find_column(pre_df, prefix)
    if not col:
        return

    s = pd.to_numeric(pre_df[col], errors="coerce").dropna()
    if s.empty:
        print("[SURVEY] No 'years cycling' data.")
        return

    max_y = int(np.ceil(s.max() / 5.0) * 5)
    bins = list(range(0, max_y + 5, 5))
    cats = pd.cut(s, bins=bins, right=False)
    vc = cats.value_counts().sort_index()
    labels = [f"{int(i.left)}–{int(i.right - 1)}" for i in vc.index]
    counts = vc.values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        text=[str(c) for c in counts],
        textposition="outside",
        hovertemplate="Years group: %{x}<br>Participants: %{y}<extra></extra>",
        name="Years cycling",
    ))
    apply_layout(fig, "Years of regular cycling", "Years (groups)", "Number of participants", cfg)
    save_figure(fig, "survey_years_cycling", cfg)

def plot_cycling_types(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    prefix = "What types of cycling do you usually do?"
    col = find_column(pre_df, prefix)
    if not col:
        return

    s = pre_df[col].dropna().astype(str)
    if s.empty:
        print("[SURVEY] No 'types of cycling' data.")
        return

    all_types: List[str] = []
    for val in s:
        parts = re.split(r"[;,]", val)
        for p in parts:
            p = p.strip()
            if p:
                all_types.append(p)

    if not all_types:
        print("[SURVEY] No parsed cycling-type entries.")
        return

    vc = pd.Series(all_types).value_counts()
    labels = vc.index.tolist()
    counts = vc.values
    total = counts.sum()
    texts = [f"{c} ({100.0*c/total:.0f}%)" for c in counts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        text=texts,
        textposition="outside",
        hovertemplate="Type: %{x}<br>Mentions: %{y}<extra></extra>",
        name="Cycling type",
    ))
    apply_layout(fig, "Types of cycling usually done", "Type", "Number of mentions", cfg)
    save_figure(fig, "survey_cycling_types", cfg)

def plot_bike_cost(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    prefix = "Approximately how much did your current primary bike cost"
    col = find_column(pre_df, prefix)
    if not col:
        return

    s = pre_df[col].dropna().astype(str)
    if s.empty:
        print("[SURVEY] No bike-cost data.")
        return

    vc = s.value_counts()

    # --- sort price ranges from lowest to highest ---
    def sort_key(label: str) -> float:
        lab = label.lower().strip()

        # 1. Handle "Less than €1,000"
        if "less than" in lab:
            m = re.search(r"\d[\d,.]*", lab)
            if m:
                # slightly below the value to ensure correct sorting
                return float(m.group(0).replace(",", "").replace("€", "")) - 0.01
            return 0

        # 2. Handle "More than €X"
        if "more than" in lab:
            m = re.search(r"\d[\d,.]*", lab)
            if m:
                return float(m.group(0).replace(",", "").replace("€", "")) + 1e6
            return 1e9

        # 3. Handle normal ranges like "€1,000 – €2,500"
        m = re.search(r"\d[\d,.]*", lab)
        if m:
            return float(m.group(0).replace(",", "").replace("€", ""))

        return 1e9


    labels = sorted(vc.index.tolist(), key=sort_key)
    counts = [vc[l] for l in labels]

    total = sum(counts)
    texts = [f"{c} ({100.0 * c / total:.0f}%)" for c in counts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        text=texts,
        textposition="outside",
        hovertemplate="Cost range: %{x}<br>Participants: %{y}<extra></extra>",
        name="Bike cost",
    ))
    apply_layout(
        fig,
        "Cost of current primary bike",
        "Price range (with VAT)",
        "Number of participants",
        cfg
    )
    save_figure(fig, "survey_bike_cost", cfg)

def plot_bike_weight(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    """
    Plot the distribution of:
    'Approximately how much does your current primary bicycle weigh (in kilograms)?'
    Treats the answer as categorical ranges (e.g. '7 – 8 kg', '8 – 9 kg').
    """
    prefix = "Approximately how much does your current primary bicycle weigh"
    col = find_column(pre_df, prefix)
    if not col:
        return

    s = pre_df[col].dropna().astype(str)
    if s.empty:
        print("[SURVEY] No bike-weight data.")
        return

    vc = s.value_counts()

    # --- sort ranges from lighter to heavier ---
    def sort_key(label: str) -> float:
        # normalise and grab first number (handles '7 – 8 kg', 'More than 10 kg', etc.)
        m = re.search(r"\d+(\.\d+)?", label)
        if m:
            return float(m.group(0))
        # push anything weird to the end
        return 1e9

    labels = sorted(vc.index.tolist(), key=sort_key)
    counts = [vc[l] for l in labels]

    total = sum(counts)
    texts = [f"{c} ({100.0 * c / total:.0f}%)" for c in counts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        text=texts,
        textposition="outside",
        hovertemplate="Weight range: %{x}<br>Participants: %{y}<extra></extra>",
        name="Bike weight",
    ))
    apply_layout(
        fig,
        "Weight of current primary bike",
        "Weight range (kg)",
        "Number of participants",
        cfg
    )
    save_figure(fig, "survey_bike_weight", cfg)


# ---------------------------
# DIVERGING STACKED BAR (LIKERT)
# ---------------------------

LIKERT_ORDER = [
    "Strongly not important",
    "Somewhat not important",
    "Neutral",
    "Somewhat important",
    "Strongly important",
]

NEGATIVE_CATS = ["Strongly not important", "Somewhat not important"]
POSITIVE_CATS = ["Somewhat important", "Strongly important"]
NEUTRAL_CAT = "Neutral"


def normalise_likert(value: Any) -> str | None:
    if pd.isna(value):
        return None
    s = str(value)
    # Clean up unicode NBSP etc.
    s = s.replace("\xa0", " ")
    s = s.strip()

    lower = s.lower()
    if "strongly not important" in lower:
        return "Strongly not important"
    if "somewhat not important" in lower:
        return "Somewhat not important"
    if "neutral" in lower:
        return "Neutral"
    if "somewhat important" in lower and "not" not in lower:
        return "Somewhat important"
    if "strongly important" in lower and "not" not in lower:
        return "Strongly important"

    # Fallback: keep original string
    return s


def plot_importance_diverging(pre_df: pd.DataFrame, cfg: Dict[str, Any]) -> None:
    """
    Diverging stacked bar chart for:

    'Before buying, how important do you consider the following when choosing a frame?'

    - Each bar = 100% of respondents for one attribute.
    - Negative categories (not important) are plotted to the left,
      positive categories (important) to the right, Neutral around the centre.
    - Axis runs internally from -100 to +100, but tick labels are mirrored
      so it reads 100–50–0–50–100, similar to Microsoft Forms.
    - Percentages are printed inside each coloured segment (absolute value).
    """

    # 1. collect all relevant columns
    likert_cols: List[str] = [
        c for c in pre_df.columns
        if c.startswith(
            "Before buying, how important do you consider the following when choosing a frame?"
        )
    ]
    if not likert_cols:
        print("[SURVEY] No 'Before buying...' Likert columns found – skipping diverging chart.")
        return

    # 2. build table: index = item (short label), columns = ordered Likert categories
    items: List[str] = []
    data = {cat: [] for cat in LIKERT_ORDER}

    for col in likert_cols:
        # Short label after the last dot
        short = col.split(".")[-1].strip() if "." in col else col

        s = pre_df[col].dropna().map(normalise_likert).dropna()
        if s.empty:
            items.append(short)
            for cat in LIKERT_ORDER:
                data[cat].append(0.0)
            continue

        vc = s.value_counts()
        total = float(vc.sum())
        items.append(short)
        for cat in LIKERT_ORDER:
            pct = 100.0 * float(vc.get(cat, 0)) / total if total > 0 else 0.0
            data[cat].append(pct)

    df_likert = pd.DataFrame(data, index=items)

    # 3. build diverging bars: negatives left, positives right
    fig = go.Figure()

    for cat in LIKERT_ORDER:
        pct_vals = df_likert[cat].values.astype(float)

        # signed for plotting (left/right)
        if cat in NEGATIVE_CATS:
            signed = -pct_vals
        else:
            signed = pct_vals

        # labels: absolute percentage, hide if <1% to avoid clutter
        labels = [f"{abs(v):.0f}%" if abs(v) >= 1.0 else "" for v in pct_vals]

        fig.add_trace(
            go.Bar(
                x=signed,
                y=df_likert.index,
                name=cat,
                orientation="h",
                marker_color=None,   # <-- use Plotly default colours
                text=labels,
                texttemplate="%{text}",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate="%{y}<br>%{customdata:.1f}% of respondents",
                customdata=np.abs(pct_vals),
            )
        )

    # Use Plotly's default qualitative palette
    fig.update_layout(colorway=px.colors.qualitative.Plotly)

    # 4. apply your global layout (template, size, font)
    apply_layout(
        fig,
        "Importance of bike attributes before buying",
        "Share of respondents",
        "",
        cfg,
    )

    # relative stacking around 0
    fig.update_layout(barmode="relative")

    # X axis: internal -100..100, labels mirrored to 100–50–0–50–100
    fig.update_xaxes(
        range=[-100, 100],
        tickvals=[-100, -50, 0, 50, 100],
        ticktext=["100", "50", "0", "50", "100"],
    )

    # vertical centre line at 0 (i.e., 0 on mirrored scale)
    fig.add_vline(x=0, line_dash="dash", line_color="black")

    # Attributes from top to bottom
    fig.update_yaxes(autorange="reversed")

    # Legend at bottom, horizontal
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=120),
    )

    save_figure(fig, "survey_importance_diverging", cfg)



# ----------------------------------
# Questionnaire helpers
# ----------------------------------

def load_post_questionnaire_from_config(cfg: dict) -> pd.DataFrame:
    """
    Load the *post* questionnaire CSV specified in config.json
    under the key 'post_survey_csv'.
    """
    post_path = cfg.get("post_survey_csv", None)
    if not post_path:
        print("[SURVEY] No 'post_survey_csv' entry in config.json")
        return pd.DataFrame()

    p = Path(post_path)
    if not p.exists():
        print(f"[SURVEY] Post-questionnaire CSV not found at {p}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[SURVEY] Could not read post-questionnaire CSV: {e}")
        return pd.DataFrame()

    print(f"[SURVEY] Loaded post-questionnaire from {p}")
    return df


def bin_0_100_to_likert(series: pd.Series) -> pd.Series:
    """
    Bin a 0–100 slider into 5 categories:
    Very low / Low / Neutral / High / Very high
    """
    x = pd.to_numeric(series, errors="coerce")
    cats = pd.cut(
        x,
        bins=[-0.1, 20, 40, 60, 80, 100.1],
        labels=["Very low", "Low", "Neutral", "High", "Very high"]
    )
    return cats

def plot_post_handling_diverging(post_df: pd.DataFrame, cfg: dict):
    """
    Violin-style view of handling / stability / stiffness questions
    in the POST questionnaire (0–100 sliders).
    Shows the full score distribution per item.
    """
    # all 0–100 sliders
    cols_0_100 = [c for c in post_df.columns if c.startswith("Rate from 0 to 100")]
    if not cols_0_100:
        print("[SURVEY] No 0–100 columns found in post questionnaire.")
        return

    # Pick *handling-related* sliders via keywords
    handling_keywords = ["steering", "stability", "responsiveness", "percived stiffness"]
    handling_cols = [
        c for c in cols_0_100
        if any(k in c.lower() for k in handling_keywords)
    ]
    if not handling_cols:
        print("[SURVEY] No handling-related columns found for violin plot.")
        return

    # Long format
    df_long = (
        post_df[handling_cols]
        .melt(var_name="question", value_name="score")
    )
    df_long["score"] = pd.to_numeric(df_long["score"], errors="coerce")
    df_long = df_long.dropna(subset=["score"])

    # Shorter labels: drop the "Rate from 0 to 100" prefix, clean spaces / punctuation
    def shorten(q: str) -> str:
        q = q.replace("Rate from 0 to 100", "")
        return q.strip(" :-_")

    df_long["question_short"] = df_long["question"].astype(str).map(shorten)

    # Add N per item (for hover)
    df_long["n_responses"] = (
        df_long.groupby("question_short")["score"]
        .transform("count")
    )

    # Horizontal violins: x = score, y = item
    fig = px.violin(
        df_long,
        x="score",
        y="question_short",
        orientation="h",
        box=False,          # inner boxplot (median + IQR)
        points=False,      # hide individual dots; set to "all" if you want them
        hover_data={"n_responses": True},
    )

    # Global layout (font, size, template)
    apply_layout(
        fig,
        "Handling & stability ratings (0–100 sliders)",
        "Score (0–100)",
        "Statement",
        cfg,
    )

    # Keep 0–100 scale with a small margin so violins are not clipped
    fig.update_xaxes(range=[-5, 105])
    fig.update_yaxes(automargin=True)
    fig.update_traces(meanline_visible=True)

    # No legend needed (single colour)
    fig.update_layout(showlegend=False)

    save_figure(fig, "post_handling_diverging", cfg)



def plot_post_comfort_vibration_box(post_df: pd.DataFrame, cfg: dict):
    """
    Violin-style view of comfort & vibration-related questions (0–100).
    Uses global layout settings from config via apply_layout().
    """
    # All 0–100 sliders
    cols_0_100 = [c for c in post_df.columns if c.startswith("Rate from 0 to 100")]
    comfort_cols = [
        c for c in cols_0_100
        if ("comfort" in c.lower()) or ("vibration" in c.lower())
    ]
    if not comfort_cols:
        print("[SURVEY] No comfort/vibration columns found in post questionnaire.")
        return

    # Long format
    df_long = (
        post_df[comfort_cols]
        .melt(var_name="question", value_name="score")
    )
    df_long["score"] = pd.to_numeric(df_long["score"], errors="coerce")
    df_long = df_long.dropna(subset=["score"])

    # Shorten labels for axis
    df_long["question_short"] = (
        df_long["question"]
        .str.replace("Rate from 0 to 100", "", regex=False)
        .str.strip(" :-_")
    )

    # Add N per item (for hover, so you still “see” how many people)
    df_long["n_responses"] = (
        df_long.groupby("question_short")["score"]
        .transform("count")
    )

    # Horizontal violins: x = score, y = item
    fig = px.violin(
        df_long,
        x="score",
        y="question_short",
        orientation="h",
        box=False,           # inner boxplot (like the example image)
        points=False,       # hide individual dots; set to "all" if you want them
        hover_data={"n_responses": True},
    )

    apply_layout(
        fig,
        "Comfort & vibration (0–100 sliders)",
        "Score (0–100)",
        "Statement",
        cfg,
    )

    # Fix score range and avoid cropped labels
    fig.update_xaxes(range=[-5, 105])
    fig.update_yaxes(automargin=True)
    fig.update_traces(meanline_visible=True)

    save_figure(fig, "post_comfort_vibration_box", cfg)




def plot_post_perception_adoption_scatter(post_df: pd.DataFrame, cfg: dict):
    """
    Scatter plot:
      x = innovation,
      y = sustainability,
      colour = 'Would you consider buying?',
      size = overall satisfaction.
    """
    # find needed columns by substring (robust to spacing / typos)
    def find_col(substr: str) -> str | None:
        matches = [c for c in post_df.columns if substr.lower() in c.lower()]
        return matches[0] if matches else None

    col_innov = find_col("How innovative do you perceive")
    col_sust = find_col("How sustainable do you consider")
    col_will = find_col("How willing would you be to consider")
    col_overall = find_col("Overall riding satisfaction")

    required = {
        "innovation": col_innov,
        "sustainability": col_sust,
        "willingness": col_will,
        "overall": col_overall,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        print(f"[SURVEY] Missing columns for perception scatter: {missing}")
        return

    df = pd.DataFrame({
        "innovation": pd.to_numeric(post_df[col_innov], errors="coerce"),
        "sustainability": pd.to_numeric(post_df[col_sust], errors="coerce"),
        "consider_buying": pd.to_numeric(post_df[col_will], errors="coerce"),
        "overall_satisfaction": pd.to_numeric(post_df[col_overall], errors="coerce"),
    })

    df = df.dropna(subset=["innovation", "sustainability"])

    fig = px.scatter(
        df,
        x="innovation",
        y="sustainability",
        color="consider_buying",
        size="overall_satisfaction",
        hover_data=["overall_satisfaction", "consider_buying"],
    )

    apply_layout(
        fig,
        "Perception space: innovation vs sustainability",
        "Innovation (0–100)",
        "Sustainability vs carbon (0–100)",
        cfg,
    )

    save_figure(fig, "post_perception_adoption_scatter", cfg)


def plot_post_overall_vs_usual(post_df: pd.DataFrame, cfg: dict):
    """
    Simple bar chart: how the prototype compares to the participant's usual frame.
    """
    matches = [
        c for c in post_df.columns
        if "Overall performance compared to your usual frame" in c
    ]
    if not matches:
        print("[SURVEY] Column 'Overall performance compared to your usual frame' not found.")
        return

    col = matches[0]
    counts = (
        post_df[col]
        .astype(str)
        .str.strip()
        .value_counts()
        .reset_index()
    )
    counts.columns = ["category", "count"]

    fig = px.bar(
        counts,
        x="category",
        y="count",
        text="count",
    )

    apply_layout(
        fig,
        "Overall performance compared to riders' usual frame",
        "Category",
        "Number of participants",
        cfg,
    )

    fig.update_traces(textposition="outside")

    save_figure(fig, "post_overall_vs_usual", cfg)


def plot_post_overall_vs_usual(post_df: pd.DataFrame, cfg: dict):
    """
    Simple bar chart: how the prototype compares to the participant's usual frame.
    """
    matches = [
        c for c in post_df.columns
        if "Overall performance compared to your usual frame" in c
    ]
    if not matches:
        print("[SURVEY] Column 'Overall performance compared to your usual frame' not found.")
        return

    col = matches[0]
    counts = (
        post_df[col]
        .astype(str)
        .str.strip()
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )

    # Optional: enforce logical order
    cat_order = [
        "Much worse",
        "Worse",
        "About the same",
        "Better",
        "Much better",
    ]
    counts["category"] = pd.Categorical(counts["category"], cat_order)
    counts = counts.sort_values("category")

    fig = px.bar(
        counts,
        x="category",
        y="count",
        text="count",
    )

    # Use global config-driven layout
    apply_layout(
        fig,
        "Overall performance compared to riders' usual frame",
        "Category",
        "Number of participants",
        cfg,
    )

    fig.update_traces(textposition="outside")

    save_figure(fig, "post_overall_vs_usual", cfg)


def plot_post_compared_to_carbon(post_df: pd.DataFrame, cfg: dict):
    """
    Violin plot of 'Compared to a carbon fibre frame' (0–100).
    50 = same as carbon; >50 = better, <50 = worse.
    """
    matches = [
        c for c in post_df.columns
        if "Compared to a carbon fibre frame" in c
    ]
    if not matches:
        print("[SURVEY] Column 'Compared to a carbon fibre frame' not found.")
        return

    col = matches[0]
    s = pd.to_numeric(post_df[col], errors="coerce").dropna()

    if s.empty:
        print("[SURVEY] No numeric data for 'Compared to a carbon fibre frame'.")
        return

    df = pd.DataFrame({"score": s})

    # Horizontal violin
    fig = px.violin(
        df,
        x="score",
        y=None,             # single violin
        orientation="h",
        box=False,           # show median + IQR
        points=False,       # no individual dots
    )

    # Apply global layout (template, font, margins)
    apply_layout(
        fig,
        "Perceived performance compared to carbon frame",
        "Rating (0–100, 50 = same as carbon)",
        "",
        cfg,
    )

    # Axis scaling + margin so violin is not clipped
    fig.update_xaxes(range=[-5, 105])
    fig.update_yaxes(showticklabels=False)  # nothing on y-axis for single item
    fig.update_traces(meanline_visible=True)

    # Add vertical 50-line (same as carbon)
    fig.add_vline(
        x=50,
        line_dash="dash",
        line_width=2,
        line_color="black",
    )

    save_figure(fig, "post_compared_to_carbon", cfg)



def plot_post_willingness_to_pay(post_df: pd.DataFrame, cfg: dict):
    """
    Distribution of willingness to pay for a complete bicycle (Euro),
    shown as a violin plot.
    """
    matches = [
        c for c in post_df.columns
        if "How much would you be willing to pay for a complete bicycle" in c
    ]
    if not matches:
        print("[SURVEY] Column 'How much would you be willing to pay for a complete bicycle' not found.")
        return

    col = matches[0]
    s = pd.to_numeric(post_df[col], errors="coerce").dropna()

    if s.empty:
        print("[SURVEY] No numeric data for willingness-to-pay.")
        return

    df = pd.DataFrame({"willingness_eur": s})

    # Horizontal violin, no box, no points
    fig = px.violin(
        df,
        x="willingness_eur",
        y=None,
        orientation="h",
        box=False,
        points=False,
    )

    # Dotted mean line inside the violin
    fig.update_traces(meanline_visible=True, width=0.2)  # width < 1.0 makes it less “fat”

    # Global layout
    apply_layout(
        fig,
        "Willingness to pay for a complete natural-fibre bicycle",
        "Price (€)",
        "",
        cfg,
    )

    # Nice, fixed x-axis: from 0 to a rounded-up max
    xmax = float(s.max())
    xmax_extended = xmax + 3000   # extend by 1000 euros
    fig.update_xaxes(range=[0, xmax_extended])


    # No y tick labels for a single violin
    fig.update_yaxes(showticklabels=False)

    save_figure(fig, "post_willingness_to_pay", cfg)

def plot_post_split_violin_by_q21(post_df: pd.DataFrame, cfg: dict) -> None:
    """
    Split horizontal violin plots (Yes / No) showing:
      - distribution (split violins)
      - mean (solid), median (dot), 95% CI of mean (thin with caps)
      - ONLY the mean value as text, shifted in Y so it sits outside the coloured half
      - CI endpoint values (lo/hi) placed at the ends of the CI line

    All internal summary lines are ONE neutral colour.
    No boxes, no points, no extra labels.
    """

    # ---- Q21: Yes / No ---------------------------------------------
    q21_matches = [
        c for c in post_df.columns
        if "notice any difference in the frame’s behaviour" in c.lower()
        or "notice any difference in the frame's behaviour" in c.lower()
    ]
    if not q21_matches:
        print("[SURVEY] Q21 not found.")
        return
    q21_col = q21_matches[0]

    def map_yes_no(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s.startswith("y"):
            return "Yes"
        if s.startswith("n"):
            return "No"
        return np.nan

    group = post_df[q21_col].apply(map_yes_no)

    # ---- target questions ------------------------------------------
    targets = [
        ("Sustainability vs carbon", "how sustainable do you consider"),
        ("Willingness to use", "how willing would you be"),
        ("Overall riding satisfaction", "overall riding satisfaction"),
        ("Overall vs carbon frame", "compared to a carbon fibre frame"),
        ("Confidence riding natural-fibre frame", "how confident would you be riding"),
    ]

    records = []
    for label, key in targets:
        matches = [c for c in post_df.columns if key.lower() in c.lower()]
        if not matches:
            print(f"[SURVEY] Column for '{label}' not found.")
            continue

        col = matches[0]
        records.append(
            pd.DataFrame({
                "question": label,
                "score": pd.to_numeric(post_df[col], errors="coerce"),
                "group": group,
            }).dropna(subset=["score", "group"])
        )

    if not records:
        print("[SURVEY] No valid data.")
        return

    df_long = pd.concat(records, ignore_index=True)
    question_order = [t[0] for t in targets]

    # ---- Bootstrap 95% CI for the mean ------------------------------
    def mean_ci_bootstrap(x: np.ndarray, n_boot: int = 1500, alpha: float = 0.05, seed: int = 7):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan, np.nan
        if x.size == 1:
            v = float(x[0])
            return v, v

        rng = np.random.default_rng(seed)
        boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
        lo = float(np.quantile(boots, alpha / 2))
        hi = float(np.quantile(boots, 1 - alpha / 2))
        return lo, hi

    # ---- Build split violins ---------------------------------------
    fig = go.Figure()

    for grp, side in [("No", "negative"), ("Yes", "positive")]:
        dfg = df_long[df_long["group"] == grp]
        if dfg.empty:
            continue

        fig.add_trace(
            go.Violin(
                y=dfg["question"],
                x=dfg["score"],
                orientation="h",
                side=side,
                name=grp,
                legendgroup=grp,
                scalegroup="all",
                box_visible=False,
                points=False,
                spanmode="hard",
                meanline_visible=False,  # we draw our own summary lines
            )
        )

    # ---- Styling (one colour for all internal lines) ----------------
    LINE_COLOR = "rgba(60, 60, 60, 1)"  # neutral dark grey

    CI_WIDTH = 2
    MEAN_WIDTH = 3
    MEDIAN_WIDTH = 3

    # ---- Labels -----------------------------------------------------
    # mean label outside each coloured half
    YSHIFT_MEAN = 30
    XSHIFT_MEAN = 6

    # CI endpoint values at bar ends (smaller)
    CI_VALUE_FONT = dict(size=9, color=LINE_COLOR)
    CI_VALUE_YSHIFT = 35  # push CI values slightly away from the line

    DRAW_CAPS = True

    for grp in ["No", "Yes"]:
        dfg = df_long[df_long["group"] == grp]
        if dfg.empty:
            continue

        for q in question_order:
            vals = dfg.loc[dfg["question"] == q, "score"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            mean_v = float(np.mean(vals))
            med_v = float(np.median(vals))
            ci_lo, ci_hi = mean_ci_bootstrap(vals, n_boot=1500, alpha=0.05, seed=7)

            # 95% CI line (thin)
            if np.isfinite(ci_lo) and np.isfinite(ci_hi):
                fig.add_trace(
                    go.Scatter(
                        x=[ci_lo, ci_hi],
                        y=[q, q],
                        mode="lines",
                        showlegend=False,
                        hoverinfo="skip",
                        line=dict(width=CI_WIDTH, color=LINE_COLOR),
                    )
                )

                # Optional minimal "caps" using a "|" character (neutral colour)
                if DRAW_CAPS:
                    for xcap in (ci_lo, ci_hi):
                        fig.add_annotation(
                            x=xcap,
                            y=q,
                            text="|",
                            showarrow=False,
                            font=dict(size=14, color=LINE_COLOR),
                            yshift=6 if grp == "Yes" else -6,
                        )

                # CI endpoint values at the ends of the CI bar
                fig.add_annotation(
                    x=ci_lo,
                    y=q,
                    text=f"{ci_lo:.1f}",
                    showarrow=False,
                    xanchor="right",
                    yanchor="middle",
                    yshift=CI_VALUE_YSHIFT if grp == "Yes" else -CI_VALUE_YSHIFT,
                    font=CI_VALUE_FONT,
                )
                fig.add_annotation(
                    x=ci_hi,
                    y=q,
                    text=f"{ci_hi:.1f}",
                    showarrow=False,
                    xanchor="left",
                    yanchor="middle",
                    yshift=CI_VALUE_YSHIFT if grp == "Yes" else -CI_VALUE_YSHIFT,
                    font=CI_VALUE_FONT,
                )

            # mean marker (solid)
            fig.add_trace(
                go.Scatter(
                    x=[mean_v, mean_v],
                    y=[q, q],
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(width=MEAN_WIDTH, color=LINE_COLOR),
                )
            )

            # median marker (dotted)
            fig.add_trace(
                go.Scatter(
                    x=[med_v, med_v],
                    y=[q, q],
                    mode="lines",
                    showlegend=False,
                    hoverinfo="skip",
                    line=dict(width=MEDIAN_WIDTH, dash="dot", color=LINE_COLOR),
                )
            )

            # ONLY mean label, outside in Y (for each group)
            fig.add_annotation(
                x=mean_v,
                y=q,
                text=f"μ {mean_v:.1f}",
                showarrow=False,
                xshift=XSHIFT_MEAN if grp == "Yes" else -XSHIFT_MEAN,
                yshift=YSHIFT_MEAN if grp == "Yes" else -YSHIFT_MEAN,
                xanchor="left" if grp == "Yes" else "right",
                yanchor="middle",
                font=dict(size=11),
            )

    # ---- layout ----------------------------------------------------
    apply_layout(
        fig,
        "Sustainability, willingness & satisfaction (split by perceived fibre difference)",
        "Score (0–100)",
        "Question",
        cfg,
    )

    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(categoryorder="array", categoryarray=question_order)

    save_figure(fig, "post_split_violin_by_q21", cfg)






def plot_q21_by_cycling_type_splitbar(pre_df: pd.DataFrame,
                                      post_df: pd.DataFrame,
                                      cfg: dict) -> None:
    """
    Split column (stacked bar) chart showing, for each cycling type from the PRE
    questionnaire, the percentage of participants who answered Yes / No to Q21:

        Q21 – 'Did you notice any difference in the frame’s behaviour related to
        the amount of natural fibre used in this version?'

    PRE:  'What types of cycling do you usually do?'  (multi-select)
    """

    # ---------- 1) PRE: cycling types ------------------------------
    prefix_types = "What types of cycling do you usually do?"
    col_types = find_column(pre_df, prefix_types)
    if not col_types:
        print("[SURVEY] Column for cycling types not found in PRE questionnaire.")
        return

    # ---------- 2) POST: Q21 Yes / No ------------------------------
    q21_matches = [
        c for c in post_df.columns
        if "notice any difference in the frame’s behaviour" in c.lower()
        or "notice any difference in the frame's behaviour" in c.lower()
    ]
    if not q21_matches:
        print("[SURVEY] Q21 ('Did you notice any difference in the frame…') not found in POST.")
        return
    q21_col = q21_matches[0]

    # Align on common respondents (index)
    common_idx = pre_df.index.intersection(post_df.index)
    if common_idx.empty:
        print("[SURVEY] No overlapping respondents between PRE and POST data.")
        return

    s_types = pre_df.loc[common_idx, col_types]
    s_q21_raw = post_df.loc[common_idx, q21_col]

    # Map Yes/No → clean labels
    def map_yes_no(val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip().lower()
        if s.startswith("y"):
            return "Yes"
        if s.startswith("n"):
            return "No"
        return np.nan


    groups = s_q21_raw.apply(map_yes_no)

    # ---------- 3) Build long format: one row per (participant, cycling type) ----
    records: list[dict[str, Any]] = []
    for idx in common_idx:
        types_str = s_types.get(idx, np.nan)
        grp = groups.get(idx, np.nan)

        if pd.isna(types_str) or pd.isna(grp):
            continue

        # Split multi-select entries (same style as plot_cycling_types)
        parts = re.split(r"[;,]", str(types_str))
        for p in parts:
            p = p.strip()
            if p:
                records.append({
                    "cycling_type": p,
                    "group": grp,   # "Yes" / "No"
                })

    if not records:
        print("[SURVEY] No valid (cycling_type, Q21) combinations found.")
        return

    df_long = pd.DataFrame(records)

    # ---------- 4) Aggregate to percentages per cycling type --------
    counts = (
        df_long
        .groupby(["cycling_type", "group"], observed=False)
        .size()
        .reset_index(name="count")
    )

    # total per cycling type
    totals = counts.groupby("cycling_type")["count"].transform("sum")
    counts["percent"] = counts["count"] / totals * 100.0

    # Nice ordering of cycling types
    type_order = sorted(counts["cycling_type"].unique().tolist())

    # ---------- 5) Split column chart (stacked bar) -----------------
    fig = px.bar(
        counts,
        x="cycling_type",
        y="percent",
        color="group",
        category_orders={"cycling_type": type_order, "group": ["No", "Yes"]},
        barmode="stack",   # split column (stacked to 100%)
        text=counts["percent"].map(lambda v: f"{v:.0f}%"),
    )

    fig.update_traces(textposition="inside")

    apply_layout(
        fig,
        "Perceived fibre-related difference by cycling type (Q21)",
        "Cycling type",
        "Percentage of participants",
        cfg,
    )

    fig.update_yaxes(range=[0, 100], ticksuffix="%")

    save_figure(fig, "post_q21_by_cycling_type_splitbar", cfg)


def plot_correlation_matrix(results: Dict[str, Any],
                            pre_df: pd.DataFrame,
                            post_df: pd.DataFrame,
                            cfg: dict) -> None:
    """
    Produces two heatmaps:
      1. Correlation heatmap: rider characteristics × post-ride ratings
      2. Significance heatmap: p-value thresholds (ns, p<.05, p<.01, p<.001)
    """

    # ---------------------------
    # 1. Vibration metrics (per participant)
    # ---------------------------
    vib_windows = results.get("vibration_speed", {}).get("windows_rms", [])
    if not vib_windows:
        vib_windows = results.get("vibration_vs_speed", [])

    df_vib = pd.DataFrame(vib_windows)
    if df_vib.empty or "participant_id" not in df_vib.columns:
        df_vib_agg = pd.DataFrame(columns=["participant_id", "avg_rms", "avg_speed"])
    else:
        df_vib = df_vib.replace([np.inf, -np.inf], np.nan)
        df_vib = df_vib.dropna(subset=["participant_id", "rms_m_s2", "speed_kmh"])
        if df_vib.empty:
            df_vib_agg = pd.DataFrame(columns=["participant_id", "avg_rms", "avg_speed"])
        else:
            df_vib_agg = (
                df_vib.groupby("participant_id", observed=False)[["rms_m_s2", "speed_kmh"]]
                .mean()
                .reset_index()
                .rename(columns={"rms_m_s2": "avg_rms", "speed_kmh": "avg_speed"})
            )

    if not df_vib_agg.empty:
        df_vib_agg["participant_id"] = df_vib_agg["participant_id"].astype(str)

    # ---------------------------
    # 2. POST questionnaire numeric fields
    # ---------------------------
    post = post_df.reset_index(drop=True).copy()
    post["participant_id"] = post.index.astype(str)

    numeric_cols: Dict[str, pd.Series] = {}
    for col in post.columns:
        if col == "participant_id":
            continue
        s_num = pd.to_numeric(post[col], errors="coerce")
        if s_num.notna().sum() >= 2:
            numeric_cols[col] = s_num

    if not numeric_cols:
        print("[CORR] No numeric columns in POST questionnaire.")
        return

    df_post_num = pd.DataFrame(numeric_cols)
    df_post_num["participant_id"] = post["participant_id"]

    # ---- Short labels for Y-axis ----
    def shorten_post_label(q: str) -> str:
        q_low = q.lower()
        mapping = {
            "how much would you be willing to pay": "Willingness to pay",
            "willing would you be to consider": "Willingness to adopt",
            "overall riding satisfaction": "Overall satisfaction",
            "compared to a carbon fibre frame": "Compared to carbon frame",
            "how sustainable": "Sustainability vs carbon",
            "how innovative": "Innovation",
            "how much do you trust": "Long-ride trust",
            "how confident would you be riding a bike frame": "Confidence riding NF",
            "how sturdy or solid": "Frame sturdiness",
            "how light did the frame": "Frame lightness",
            "power transfer": "Power transfer",
            "aerodynamic": "Aero feel",
            "responsiveness when accelerating": "Accel responsiveness",
            "stability in corners": "Corner stability",
            "stability during straight riding": "Straight-line stability",
            "stability under braking": "Braking stability",
            "steering responsiveness": "Steering responsiveness",
            "comfort of the ride on rough road": "Comfort rough",
            "comfort on smooth road surfaces": "Comfort smooth",
            "vibration dampening": "Vibration feel",
            "weight balance distribution": "Weight balance",
            "effort required to ride this bike": "Required effort",
            "estimate in kilograms": "Weight estimate",
            "total distance ridden": "Distance (km)",
        }
        for key, val in mapping.items():
            if key in q_low:
                return val
        return q.strip()[:40]

    df_post_num = df_post_num.rename(
        {col: shorten_post_label(col) for col in df_post_num.columns if col != "participant_id"}
    )

    # ---------------------------
    # 3. PRE questionnaire predictors
    # ---------------------------
    demo_df = None
    if pre_df is not None and not pre_df.empty:
        pre = pre_df.reset_index(drop=True).copy()
        pre["participant_id"] = pre.index.astype(str)

        demo_cols: Dict[str, pd.Series] = {"participant_id": pre["participant_id"]}

        # Age
        age_col = "What is your age?"
        if age_col in pre.columns:
            demo_cols["Age (years)"] = pd.to_numeric(pre[age_col], errors="coerce")

        # Height
        h_col = find_column(pre, "What is your height?")
        if h_col:
            demo_cols["Height (cm)"] = pd.to_numeric(pre[h_col], errors="coerce")

        # Km last 12 months — categorical → numeric
        km_prefix = "About how many kilometers"
        km_col = find_column(pre, km_prefix)
        if km_col:
            s_raw = pre[km_col].astype(str)

            def km_label_to_value(label: str) -> float:
                if "prefer not" in label.lower():
                    return np.nan
                m = re.search(r"\d[\d,]*", label)
                return float(m.group(0).replace(",", "")) if m else np.nan

            demo_cols["Km last 12 months"] = s_raw.apply(km_label_to_value)

        # Years regular cycling
        yrs_prefix = "For how many years have you been cycling"
        yrs_col = find_column(pre, yrs_prefix)
        if yrs_col:
            demo_cols["Years regular cycling"] = pd.to_numeric(pre[yrs_col], errors="coerce")

        # Gender
        g_col = "What is your gender?"
        if g_col in pre.columns:
            g = pre[g_col].astype(str).str.lower()
            demo_cols["Gender (code)"] = g.map(
                {"male": 0.0, "man": 0.0, "female": 1.0, "woman": 1.0}
            )

        demo_df = pd.DataFrame(demo_cols)

    # ---------------------------
    # 4. Merge POST + vibration + PRE
    # ---------------------------
    merged = df_post_num.copy()
    if not df_vib_agg.empty:
        merged = merged.merge(df_vib_agg, on="participant_id", how="left")
    if demo_df is not None:
        merged = merged.merge(demo_df, on="participant_id", how="left")

    merged = merged.drop(columns=["participant_id"], errors="ignore")

    merged_num = merged.select_dtypes(include=[float, int])
    merged_num = merged_num.loc[:, merged_num.notna().sum() >= 2]

    # Remove any id-like columns
    merged_num = merged_num[[c for c in merged_num.columns if "id" not in c.lower()]]

    df_corr_all = merged_num.corr()

    # ---------------------------
    # 5. Predictors vs Outcomes
    # ---------------------------
    predictor_candidates = [
        "avg_rms", "avg_speed", "Distance (km)",
        "Age (years)", "Height (cm)",
        "Km last 12 months", "Years regular cycling",
        "Gender (code)",
    ]
    predictor_cols = [c for c in predictor_candidates if c in df_corr_all.columns]
    outcome_cols = [c for c in df_corr_all.columns if c not in predictor_cols]

    if not predictor_cols or not outcome_cols:
        print("[CORR] Not enough variables for correlation matrix.")
        return

    corr_block = df_corr_all.loc[outcome_cols, predictor_cols]

    # ---------------------------
    # 6. MAIN HEATMAP — correlations
    # ---------------------------
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_block.values,
            x=predictor_cols,
            y=outcome_cols,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_block.values, 2),
            texttemplate="%{text}",
        )
    )

    apply_layout(
        fig,
        "Correlation: rider characteristics × post-ride ratings",
        "Rider / ride characteristics",
        "Post-ride questions",
        cfg,
    )

    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(tickfont=dict(size=10))

    save_figure(fig, "correlation_matrix", cfg)

    # --------------------------------------------------
    # 7. SIGNIFICANCE MATRIX — p-value thresholds
    # --------------------------------------------------
    p_matrix = np.zeros_like(corr_block.values, dtype=float)

    for i, out_var in enumerate(outcome_cols):
        for j, pred_var in enumerate(predictor_cols):
            x = merged_num[pred_var].dropna()
            y = merged_num[out_var].dropna()
            idx = x.index.intersection(y.index)

            if len(idx) < 3:
                p_matrix[i, j] = np.nan
                continue

            _, p = pearsonr(x.loc[idx], y.loc[idx])
            p_matrix[i, j] = p

    # p → label
    def p_to_label(p):
        if np.isnan(p):
            return ""
        if p < 0.001:
            return "p<.001"
        elif p < 0.01:
            return "p<.01"
        elif p < 0.05:
            return "p<.05"
        else:
            return "ns"

    label_matrix = np.vectorize(p_to_label)(p_matrix)

    # p → level 0–3
    def p_to_level(p):
        if np.isnan(p):
            return 0
        if p < 0.001:
            return 3
        elif p < 0.01:
            return 2
        elif p < 0.05:
            return 1
        else:
            return 0

    color_levels = np.vectorize(p_to_level)(p_matrix)

    colorscale = [
        [0.00, "white"],    # ns
        [0.33, "#ffcccc"],  # p<.05
        [0.66, "#ff6666"],  # p<.01
        [1.00, "#b30000"],  # p<.001
    ]

    fig_sig = go.Figure(
        data=go.Heatmap(
            z=color_levels,
            x=predictor_cols,
            y=outcome_cols,
            zmin=0,
            zmax=3,
            colorscale=colorscale,
            text=label_matrix,
            texttemplate="%{text}",
            showscale=False,   # 🔥 hide misleading numeric colourbar
        )
    )

    apply_layout(
        fig_sig,
        "Significance matrix (p-value thresholds)",
        "Rider / ride characteristics",
        "Post-ride questions",
        cfg,
    )

    fig_sig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig_sig.update_yaxes(tickfont=dict(size=10))

    save_figure(fig_sig, "correlation_significance_matrix", cfg)

    print("[CORR] Saved → correlation_matrix and correlation_significance_matrix")





def plot_questionnaire_pre(cfg: Dict[str, Any]) -> None:
    """
    Load the pre-questionnaire CSV and create all 'before riding' plots:
    - gender pie
    - age distribution
    - km last 12 months
    - cycling frequency
    - height
    - years of regular cycling
    - types of cycling
    - bike cost
    - bike weight
    - importance of frame attributes (diverging Likert)
    """
    pre_path = cfg.get("pre_survey_csv", None)
    if not pre_path:
        print("[SURVEY] No 'pre_survey_csv' entry in config.json – skipping pre-questionnaire plots.")
        return

    if not Path(pre_path).exists():
        print(f"[SURVEY] Pre-questionnaire file not found: {pre_path}")
        return

    pre_df = load_questionnaire(pre_path)
    if pre_df.empty:
        print("[SURVEY] Pre-questionnaire is empty – no plots generated.")
        return

    print(f"[SURVEY] Loaded pre-questionnaire from {pre_path}")

    # 🌈 Here are all the plots for the PRE questionnaire
    plot_gender_pie(pre_df, cfg)
    plot_age_hist(pre_df, cfg)
    plot_km_last_12_months(pre_df, cfg)
    plot_cycling_frequency(pre_df, cfg)
    plot_height(pre_df, cfg)
    plot_years_cycling(pre_df, cfg)
    plot_cycling_types(pre_df, cfg)
    plot_bike_cost(pre_df, cfg)
    plot_bike_weight(pre_df, cfg)
    plot_importance_diverging(pre_df, cfg)

def plot_questionnaire_post(results: Dict[str, Any], cfg: dict):
    """
    High-level wrapper: generate all post-questionnaire plots.
    """

    # Load post questionnaire
    post_df = load_post_questionnaire_from_config(cfg)
    if post_df.empty:
        return

    # --- load pre questionnaire too (for some split plots) ---
    pre_path = cfg.get("pre_survey_csv", None)
    if pre_path and Path(pre_path).exists():
        pre_df = load_questionnaire(pre_path)
    else:
        pre_df = pd.DataFrame()   # fallback empty DF

    # --- Existing post-questionnaire plots ---
    plot_post_handling_diverging(post_df, cfg)
    plot_post_comfort_vibration_box(post_df, cfg)
    plot_post_perception_adoption_scatter(post_df, cfg)
    plot_post_overall_vs_usual(post_df, cfg)
    plot_post_compared_to_carbon(post_df, cfg)
    plot_post_willingness_to_pay(post_df, cfg)
    plot_post_split_violin_by_q21(post_df, cfg)

    # --- NEW: correlation heatmap (POST × vibration) ---
    plot_correlation_matrix(results, pre_df, post_df, cfg)

    # --- NEW: Q21 × cycling-type horizontal split violins ---
    if not pre_df.empty:
        plot_q21_by_cycling_type_splitbar(pre_df, post_df, cfg)
    else:
        print("[SURVEY] Skipping cycling-type × Q21 violin (pre data missing).")



    

    




# ---------------------------
# Sampling-rate auto-detect (optional, from CSV)
# ---------------------------

def estimate_fs_from_first_csv(cfg: Dict[str, Any]) -> float | None:
    csv_dir = Path(cfg.get("csv_dir", "data/csv"))
    if not csv_dir.exists():
        print(f"[AUTO_FS] csv_dir does not exist: {csv_dir}")
        return None

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"[AUTO_FS] No CSV files in {csv_dir}")
        return None

    try:
        df = pd.read_csv(csv_files[0])
    except Exception as e:
        print(f"[AUTO_FS] Failed to read {csv_files[0]}: {e}")
        return None

    # Prefer 'seconds_elapsed' if present, otherwise 'timestamp'
    if "seconds_elapsed" in df.columns:
        t = pd.to_numeric(df["seconds_elapsed"], errors="coerce").dropna()
    elif "timestamp" in df.columns:
        t = pd.to_datetime(df["timestamp"], errors="coerce")
        t = (t - t.iloc[0]).dt.total_seconds()
    else:
        print("[AUTO_FS] No usable time column (seconds_elapsed or timestamp).")
        return None

    t = t.sort_values()
    dt = np.diff(t.to_numpy())
    dt = dt[dt > 0]
    if dt.size == 0:
        print("[AUTO_FS] Could not estimate dt.")
        return None

    fs_est = 1.0 / np.median(dt)
    print(f"[AUTO_FS] Estimated sampling rate from {csv_files[0].name}: {fs_est:.2f} Hz")
    return fs_est

def estimate_sampling_rate_from_first_csv(csv_dir: str) -> float | None:
    csv_paths = sorted(Path(csv_dir).glob("*.csv"))
    if not csv_paths:
        print(f"[AUTO_FS] No CSV files found in {csv_dir}")
        return None

    first_csv = csv_paths[0]
    df = pd.read_csv(first_csv)

    if "seconds_elapsed" not in df.columns:
        print(f"[AUTO_FS] 'seconds_elapsed' not found in {first_csv.name}")
        return None

    t = pd.to_numeric(df["seconds_elapsed"], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        print(f"[AUTO_FS] Could not estimate sample rate from {first_csv.name}")
        return None

    fs_est = 1.0 / float(np.nanmedian(dt))
    print(f"[AUTO_FS] Estimated sampling rate from {first_csv.name}: {fs_est:.2f} Hz")
    return fs_est

# ---------------------------
# Main
# ---------------------------

def main(
    results_path: str = "outputs/analysis_results.json",
    hz_if_known: float | None = None,
    auto_fs: bool = False,
):
    # load config
    cfg = load_config("config.json")

    # --- AUTO / FALLBACK FOR SAMPLING RATE ---
    # If user asked for auto OR didn't provide fs, try to estimate from CSV
    if auto_fs or hz_if_known is None:
        csv_dir = cfg.get("csv_dir", "data/csv")

        # try to estimate from first CSV
        est_fs = estimate_sampling_rate_from_first_csv(csv_dir)
        if est_fs is not None:
            hz_if_known = est_fs
            print(f"[AUTO_FS] Using estimated sampling rate: {hz_if_known:.2f} Hz")
        elif hz_if_known is None:
            # final fallback
            hz_if_known = 100.0
            print("[AUTO_FS] Could not estimate sampling rate; falling back to 100 Hz.")

    # from here on, hz_if_known is NEVER None
    # load results etc.
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)


    print("[PLOT] Plotting vibration / surface / questionnaire figures...")

    # This one already takes hz_if_known for x-axis in Hz (if it uses it)
    plot_vibration_spectrum(cfg)



    plot_vibration_rms(results, cfg)
    plot_vibration_vs_speed(results, cfg)

    plot_vibration_peak_freq_vs_speed_by_surface(results, cfg)


    # RMS vs speed (already uses JSON data)
    plot_vibration_rms_vs_speed_binned(cfg, results_path=results_path, bin_width_kmh=5.0)
    plot_vibration_peak_freq_vs_speed_by_surface(results, cfg)

    # (If you have this function)
    plot_vibration_peak_freq_vs_speed(results, cfg)

    plot_surface_breakdown(results, cfg)
    plot_questionnaire_pre(cfg)
    plot_questionnaire_post(results, cfg)


    print("[PLOT] Completed.")






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot bike data (vibration, surface, questionnaire) using Plotly.")
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/analysis_results.json",
        help="Path to analysis_results.json",
    )
    parser.add_argument(
        "--fs_hz",
        type=float,
        default=None,
        help="Sampling rate (Hz) for vibration spectrum axis (if known).",
    )
    parser.add_argument(
        "--auto_fs",
        action="store_true",
        help="Automatically estimate sampling rate from first CSV in csv_dir.",
    )
    args = parser.parse_args()

    main(
        results_path=args.results,
        hz_if_known=args.fs_hz,
        auto_fs=args.auto_fs,
    )

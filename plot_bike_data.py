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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

    if ps["save_figures"]:
        fig.write_html(str(html_path))
        print(f"[SAVE] Saved HTML → {html_path}")
        try:
            # requires 'kaleido' installed
            fig.write_image(str(png_path))
            print(f"[SAVE] Saved PNG  → {png_path}")
        except Exception as e:
            print(f"[WARN] Could not save PNG ({e}). HTML is still saved.")


# ---------------------------
# Load analysis results
# ---------------------------

def load_results(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# VIBRATION PLOTS
# ---------------------------

def plot_vibration_spectrum(results: Dict[str, Any],
                            cfg: Dict[str, Any],
                            hz_if_known: float | None = None) -> None:
    vib = results.get("vibration", {})
    av = vib.get("averaged", None)
    if not av:
        print("[VIB] No averaged spectrum found in results.")
        return

    bins = np.array(av.get("bins_cyc_per_sample", []), dtype=float)
    amp = np.array(av.get("amplitude", []), dtype=float)
    if bins.size == 0 or amp.size == 0:
        print("[VIB] Empty spectrum.")
        return

    # Default: cycles/sample
    x_vals = bins
    x_label = "Frequency (cycles/sample)"

    # If we know the sampling rate, convert to Hz and cut everything below 5 Hz
    if hz_if_known is not None and hz_if_known > 0:
        x_hz = bins * hz_if_known
        mask = x_hz >= 5.0  # keep only components ≥ 5 Hz
        if not mask.any():
            print("[VIB] No spectral components above 5 Hz – nothing to plot.")
            return
        x_vals = x_hz[mask]
        amp = amp[mask]
        x_label = "Frequency (Hz)"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=amp,
        mode="lines",
        name="Average spectrum",
    ))
    apply_layout(fig, "Average vibration spectrum", x_label, "Amplitude (index-based)", cfg)
    save_figure(fig, "vibration_spectrum", cfg)



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


# ---------------------------
# SURFACE BREAKDOWN PLOT
# ---------------------------

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
        # Treat as categorical ranges
        vc = s_raw.astype(str).value_counts()
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
        hovertemplate="Frequency: %{x}<br>Participants: %{y}<extra></extra>",
        name="Frequency",
    ))
    apply_layout(fig,
                 "Cycling frequency in last 12 months",
                 "Frequency category",
                 "Number of participants",
                 cfg)
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

    min_h = int(s.min())
    max_h = int(s.max())
    start = min_h - (min_h % 5)
    end = max_h + (5 - max_h % 5) if max_h % 5 != 0 else max_h
    bins = list(range(start, end + 5, 5))

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
        hovertemplate="Height group: %{x} cm<br>Participants: %{y}<extra></extra>",
        name="Height",
    ))
    apply_layout(fig, "Participant height distribution", "Height group (cm)", "Number of participants", cfg)
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
        hovertemplate="Cost range: %{x}<br>Participants: %{y}<extra></extra>",
        name="Bike cost",
    ))
    apply_layout(fig, "Cost of current primary bike", "Price range (with VAT)", "Number of participants", cfg)
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
        hovertemplate="Weight range: %{x}<br>Participants: %{y}<extra></extra>",
        name="Bike weight",
    ))
    apply_layout(fig,
                 "Weight of current primary bike",
                 "Weight range (kg)",
                 "Number of participants",
                 cfg)
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
    Diverging stacked bar chart for the block:

    "Before buying, how important do you consider the following when choosing a frame?..."

    Shows percentage of responses in each Likert category.
    Negative categories ("not important") are plotted to the left (negative values),
    positive categories ("important") to the right, and "Neutral" in the centre.
    """
    likert_cols: List[str] = [
        c for c in pre_df.columns
        if c.startswith("Before buying, how important do you consider the following when choosing a frame?")
    ]
    if not likert_cols:
        print("[SURVEY] No 'Before buying...' Likert columns found – skipping diverging chart.")
        return

    # Build a table: index = item (short label), columns = ordered Likert categories
    items: List[str] = []
    data = {cat: [] for cat in LIKERT_ORDER}

    for col in likert_cols:
        # Short label: part after the last dot
        if "." in col:
            short = col.split(".")[-1].strip()
        else:
            short = col

        s = pre_df[col].dropna().map(normalise_likert)
        s = s.dropna()
        if s.empty:
            # no answers -> zeros for all categories
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

    # Build diverging stacked bars:
    #   - negative values for "not important" categories
    #   - positive for "important"
    #   - neutral kept around zero
    fig = go.Figure()

    colour_map = {
        "Strongly not important": "#B2182B",   # dark red
        "Somewhat not important": "#EF8A62",   # light red
        "Neutral": "#CCCCCC",                  # grey
        "Somewhat important": "#67A9CF",       # light blue
        "Strongly important": "#2166AC",       # dark blue
    }

    for cat in LIKERT_ORDER:
        vals = df_likert[cat].values.astype(float)

        # left side = negative, right side = positive
        if cat in NEGATIVE_CATS:
            vals_plot = -vals
        else:
            vals_plot = vals

        fig.add_trace(go.Bar(
            x=vals_plot,
            y=df_likert.index,
            name=cat,
            orientation="h",
            marker_color=colour_map.get(cat, None),
            hovertemplate="%{y}<br>%{x:.1f}%",
        ))

    fig.update_layout(
        barmode="relative",
    )
    fig.update_xaxes(
        # You can hide ticks if you really want no numbers:
        # showticklabels=False,
        title_text="Percentage of respondents (negative = not important, positive = important)",
        range=[-100, 100],
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black",
    )
    # Attributes top to bottom
    fig.update_yaxes(autorange="reversed")

    apply_layout(fig, "Importance of frame attributes before buying (diverging Likert)", "", "", cfg)
    save_figure(fig, "survey_importance_diverging", cfg)



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


# ---------------------------
# Main
# ---------------------------

def main(results_path: str,
         hz_if_known: float | None = None,
         auto_fs: bool = False) -> None:

    cfg = load_config("config.json")
    ps = get_plot_settings(cfg)
    ps["out_dir"].mkdir(parents=True, exist_ok=True)

    if not Path(results_path).exists():
        print(f"[ERR] Results file not found: {results_path}")
        return

    results = load_results(results_path)

    # Decide sampling rate for the spectrum x-axis
    if auto_fs and hz_if_known is None:
        hz_if_known = estimate_fs_from_first_csv(cfg)

    print("[PLOT] Plotting vibration / surface / questionnaire figures...")
    plot_vibration_spectrum(results, cfg, hz_if_known=hz_if_known)
    plot_vibration_rms(results, cfg)
    plot_surface_breakdown(results, cfg)
    plot_questionnaire_pre(cfg)
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

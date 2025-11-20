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
    Split violin plots for key 0–100 questions, split by Q21 (Yes / No).
    Horizontal version.
    """

    # ---- Q21: Yes / No ---------------------------------------------
    q21_matches = [
        c for c in post_df.columns
        if "notice any difference in the frame’s behaviour" in c.lower()
        or "notice any difference in the frame's behaviour" in c.lower()
    ]
    if not q21_matches:
        print("[SURVEY] Q21 ('Did you notice any difference in the frame…') not found.")
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

    # ---- target questions (0–100 sliders) --------------------------
    targets = [
        ("Sustainability vs carbon", "how sustainable do you consider a natural or mineral fibre frame"),
        ("Willingness to use", "how willing would you be to consider a natural or mineral fibre composite frame"),
        ("Overall riding satisfaction", "overall riding satisfaction"),
        ("Overall vs carbon frame", "compared to a carbon fibre frame"),
        ("Confidence riding natural-fibre frame", "how confident would you be riding a bike frame made from natural fibres regularly"),
    ]

    records = []
    for label, key in targets:
        matches = [c for c in post_df.columns if key.lower() in c.lower()]
        if not matches:
            print(f"[SURVEY] Column for '{label}' not found (key='{key}').")
            continue
        col = matches[0]

        values = pd.to_numeric(post_df[col], errors="coerce")
        df_tmp = pd.DataFrame({
            "question": label,
            "score": values,
            "group": group,
        })
        df_tmp = df_tmp.dropna(subset=["score", "group"])
        records.append(df_tmp)

    if not records:
        print("[SURVEY] No 0–100 columns found for split violin plot.")
        return

    df_long = pd.concat(records, ignore_index=True)

    # Fixed question order
    question_order = [lbl for (lbl, _) in targets]

    # ---- Horizontal split violins using go.Violin -------------------
    fig = go.Figure()

    for grp, side in [("No", "negative"), ("Yes", "positive")]:
        df_g = df_long[df_long["group"] == grp]
        if df_g.empty:
            continue

        fig.add_trace(
            go.Violin(
                y=df_g["question"],      # ← question on Y (vertical categories)
                x=df_g["score"],         # ← score on X (horizontal)
                orientation="h",
                legendgroup=grp,
                scalegroup="all",
                name=grp,
                side=side,
                box_visible=False,
                meanline_visible=True,   # dotted mean line like your other violins
                points=False,            # no individual dots unless you want them
                spanmode="hard",
            )
        )

    # ---- global layout ----------------------------------------------
    apply_layout(
        fig,
        "Sustainability, willingness & satisfaction (split by perceived fibre difference)",
        "Score (0–100)",
        "Question",
        cfg,
    )

    # Fixed score range
    fig.update_xaxes(range=[0, 100])

    # Order questions on Y axis (not X!)
    fig.update_yaxes(categoryorder="array", categoryarray=question_order)

    save_figure(fig, "post_split_violin_by_q21", cfg)





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

def plot_questionnaire_post(cfg: dict):
    """
    High-level wrapper: generate all post-questionnaire plots.
    """
    post_df = load_post_questionnaire_from_config(cfg)
    if post_df.empty:
        return

    # 1) Handling & stability as diverging Likert
    plot_post_handling_diverging(post_df, cfg)

    # 2) Comfort & vibration box / violin
    plot_post_comfort_vibration_box(post_df, cfg)

    # 3) Perception vs adoption scatter
    plot_post_perception_adoption_scatter(post_df, cfg)

    # 4) Overall performance vs usual frame
    plot_post_overall_vs_usual(post_df, cfg)

    # 5) Comparison to carbon
    plot_post_compared_to_carbon(post_df, cfg)

    # 6) Willingness to pay
    plot_post_willingness_to_pay(post_df, cfg)

    plot_post_split_violin_by_q21(post_df, cfg)




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
    plot_questionnaire_post(cfg)
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

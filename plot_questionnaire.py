import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


# ---------------------------------------------------------
# Load questionnaire CSV safely
# ---------------------------------------------------------
def load_csv(path):
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


# ---------------------------------------------------------
# Clean numeric column (bike price, willingness, etc.)
# ---------------------------------------------------------
def clean_numeric(series):
    return (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.extract(r"(\d+\.?\d*)")[0]
        .astype(float)
    )


# ---------------------------------------------------------
# Gender cleaning
# ---------------------------------------------------------
def clean_gender(series):
    mapping = {
        "male": "Male",
        "m": "Male",
        "man": "Male",
        "female": "Female",
        "f": "Female",
        "woman": "Female",
    }
    s = series.astype(str).str.lower().str.strip()
    return s.map(mapping).fillna("Other")


# ---------------------------------------------------------
# PLOT 1: Cost of bike (pre)
# ---------------------------------------------------------
def plot_bike_cost(pre_df, out_dir="outputs"):
    colname = "What is the approximate price of your bike?"
    if colname not in pre_df.columns:
        print("Column not found in pre questionnaire:", colname)
        return

    cost = clean_numeric(pre_df[colname])
    cost.to_csv(Path(out_dir) / "cleaned_bike_cost.csv", index=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Participant {i+1}" for i in range(len(cost))],
        y=cost,
        marker_color="#1f77b4"
    ))

    fig.update_layout(
        title="Cost of Participant-Owned Bikes (Pre-questionnaire)",
        xaxis_title="Participant",
        yaxis_title="Bike Cost (€)",
        template="plotly_white",
        height=600
    )

    fig.write_html(Path(out_dir) / "plot_bike_cost.html")
    fig.write_image(Path(out_dir) / "plot_bike_cost.png")
    print("[SAVE] plot_bike_cost saved.")

# ---------------------------------------------------------
# PLOT 2: Willingness to buy natural-fibre frame (post)
# ---------------------------------------------------------
def plot_willingness(post_df, out_dir="outputs"):
    colname = "On a scale from 0 to 100, how willing would you be to consider a natural or mineral fiber composite frame for your own use?"
    if colname not in post_df.columns:
        print("Column not in post questionnaire:", colname)
        return

    will = clean_numeric(post_df[colname])
    will.to_csv(Path(out_dir) / "cleaned_willingness.csv", index=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Participant {i+1}" for i in range(len(will))],
        y=will,
        marker_color="#2ca02c"
    ))

    fig.update_layout(
        title="Willingness to Use Natural/ Mineral Fibre Frame (0–100)",
        xaxis_title="Participant",
        yaxis_title="Willingness (0–100)",
        template="plotly_white",
        height=600
    )

    fig.write_html(Path(out_dir) / "plot_willingness.html")
    fig.write_image(Path(out_dir) / "plot_willingness.png")
    print("[SAVE] plot_willingness saved.")


# ---------------------------------------------------------
# PLOT 3: Gender pie chart
# ---------------------------------------------------------
def plot_gender(pre_df, out_dir="outputs"):
    colname = "Gender"
    if colname not in pre_df.columns:
        print("Gender column not found.")
        return

    gender = clean_gender(pre_df[colname])
    gender.to_csv(Path(out_dir) / "cleaned_gender.csv", index=False)

    counts = gender.value_counts()

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.3
    ))

    fig.update_layout(
        title="Gender Distribution",
        template="plotly_white",
        height=600
    )

    fig.write_html(Path(out_dir) / "plot_gender.html")
    fig.write_image(Path(out_dir) / "plot_gender.png")
    print("[SAVE] plot_gender saved.")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    pre_path = "D:/Files/TUe/Tesi/Data/questionnaires/pre_questionnaire.csv"
    post_path = "D:/Files/TUe/Tesi/Data/questionnaires/post_questionnaire.csv"
    out_dir = "outputs"
    Path(out_dir).mkdir(exist_ok=True)

    pre_df = load_csv(pre_path)
    post_df = load_csv(post_path)

    plot_bike_cost(pre_df, out_dir)
    plot_willingness(post_df, out_dir)
    plot_gender(pre_df, out_dir)

    print("\n[DONE] All questionnaire plots generated.\n")


if __name__ == "__main__":
    main()

"""
EDA_Solar.py
============
EDA for Solar Power Generation
- Scatter Plot (auto-select best polynomial degree 0-2)
- Correlation Heatmap (best of Pearson/Spearman)
- Print summary
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics       import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model  import LinearRegression
from sklearn.pipeline      import make_pipeline

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "axes.facecolor"   : "#F8F9FA",
    "figure.facecolor" : "white",
    "axes.grid"        : True,
    "grid.color"       : "#E0E0E0",
    "grid.linewidth"   : 0.6,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "sans-serif",
    "font.sans-serif"  : ["Tahoma", "Arial", "DejaVu Sans"],
})

SOLAR = "#F0A500"
BLUE  = "#4C9BE8"
RED   = "#E8564C"
GREEN = "#5CB85C"

PLANTS = {
    1: "Plant_1_Joined.csv",
    2: "Plant_2_Joined.csv",
}

# ==============================================================================
# LOAD
# ==============================================================================

def load_plant(plant_num):
    df = pd.read_csv(os.path.join(DATASET_DIR, PLANTS[plant_num]))
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

    df["HOUR"]      = df["DATE_TIME"].dt.hour
    df["MINUTE"]    = df["DATE_TIME"].dt.minute
    df["DAY"]       = df["DATE_TIME"].dt.day
    df["MONTH"]     = df["DATE_TIME"].dt.month
    df["DATE"]      = df["DATE_TIME"].dt.date
    df["TIME_MIN"]  = df["HOUR"] * 60 + df["MINUTE"]
    df["TIME_SLOT"] = (df["TIME_MIN"] / 15).astype(int)

    df_day = df[df["IRRADIATION"] > 0].copy()
    return df, df_day

# ==============================================================================
# HELPER — best polynomial degree 0-2
# ==============================================================================

def best_poly_trend(x_vals, y_vals, x_line):
    results = {}

    for deg in [0, 1, 2]:
        try:
            if deg == 0:
                y_pred  = np.full_like(y_vals, y_vals.mean(), dtype=float)
                y_line_ = np.full_like(x_line, y_vals.mean(), dtype=float)
            else:
                pipe    = make_pipeline(PolynomialFeatures(deg),
                                        LinearRegression())
                pipe.fit(x_vals.reshape(-1, 1), y_vals)
                y_pred  = pipe.predict(x_vals.reshape(-1, 1))
                y_line_ = pipe.predict(x_line.reshape(-1, 1))

            r2 = r2_score(y_vals, y_pred)
            results[deg] = {"r2": r2, "y_line": y_line_}
        except Exception:
            pass

    # select lowest degree unless improvement > 5%
    best_degree = 0
    for deg in [0, 1, 2]:
        if deg not in results:
            continue
        if deg == 0:
            best_degree = 0
            continue
        r2_current  = results[deg]["r2"]
        r2_previous = results[deg - 1]["r2"]
        if r2_current - r2_previous > 0.05:
            best_degree = deg

    best_r2     = results[best_degree]["r2"]
    best_y_line = results[best_degree]["y_line"]

    return best_degree, best_r2, best_y_line

# ==============================================================================
# FIGURE 1 — Scatter Plots
# ==============================================================================

def plot_scatter(df_day, plant_num):

    sample = df_day.sample(min(3000, len(df_day)), random_state=42)

    SCATTER_PAIRS = [
        ("IRRADIATION",         "AC_POWER",           "Irradiation vs AC Power",         SOLAR),
        ("MODULE_TEMPERATURE",  "AC_POWER",            "Module Temp vs AC Power",         RED),
        ("AMBIENT_TEMPERATURE", "AC_POWER",            "Ambient Temp vs AC Power",        GREEN),
        ("MODULE_TEMPERATURE",  "IRRADIATION",         "Module Temp vs Irradiation",      SOLAR),
        ("TIME_SLOT",           "AC_POWER",            "Time (15-min) vs AC Power",       BLUE),
        ("AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",  "Ambient Temp vs Module Temp",     RED),
        ("IRRADIATION",         "DAILY_YIELD",         "Irradiation vs Daily Yield",      GREEN),
        ("TIME_SLOT",           "IRRADIATION",         "Time (15-min) vs Irradiation",    SOLAR),
        ("AMBIENT_TEMPERATURE", "DAILY_YIELD",         "Ambient Temp vs Daily Yield",     BLUE),
    ]

    # global axis limits
    axis_limits = {}
    all_cols = set()
    for x_col, y_col, _, _ in SCATTER_PAIRS:
        all_cols.add(x_col); all_cols.add(y_col)
    for col in all_cols:
        axis_limits[col] = (df_day[col].quantile(0.01),
                            df_day[col].quantile(0.99))

    ncols = 3
    nrows = int(np.ceil(len(SCATTER_PAIRS) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
    fig.suptitle(f"Plant {plant_num} — Scatter Plots  "
                 f"(auto best trend deg 0-2, threshold +5%)",
                 fontsize=15, fontweight="bold", y=1.01)

    degree_summary = []

    for ax, (x_col, y_col, title, color) in zip(axes.flat, SCATTER_PAIRS):
        x_vals = sample[x_col].values.astype(float)
        y_vals = sample[y_col].values.astype(float)

        ax.scatter(x_vals, y_vals, c=color, alpha=0.35, s=8, rasterized=True)

        x_line                    = np.linspace(x_vals.min(), x_vals.max(), 300)
        best_deg, best_r2, y_line = best_poly_trend(x_vals, y_vals, x_line)

        ax.plot(x_line, y_line, color=RED, linewidth=2,
                linestyle="--", label=f"deg={best_deg}  R2={best_r2:.3f}")
        ax.legend(fontsize=8)

        if x_col == "TIME_SLOT":
            tick_slots  = sorted(df_day["TIME_SLOT"].unique())[::4]
            tick_labels = [f"{(s*15)//60:02d}:{(s*15)%60:02d}"
                           for s in tick_slots]
            ax.set_xticks(tick_slots)
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=7)
            ax.set_xlabel("Time (HH:MM)", fontsize=8)
        else:
            ax.set_xlim(axis_limits[x_col])
            ax.set_xlabel(x_col.replace("_", " "), fontsize=8)

        ax.set_ylim(axis_limits[y_col])
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(y_col.replace("_", " "), fontsize=8)

        degree_summary.append({
            "pair"       : f"{x_col} -> {y_col}",
            "best_degree": best_deg,
            "R2"         : round(best_r2, 4),
        })

    for ax in list(axes.flat)[len(SCATTER_PAIRS):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"EDA_P{plant_num}_Scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Saved] EDA_P{plant_num}_Scatter.png")

    return pd.DataFrame(degree_summary)

# ==============================================================================
# FIGURE 2 — Correlation Heatmap (best of Pearson/Spearman)
# ==============================================================================

def plot_heatmap(df_day, plant_num):

    CORR_COLS = [
        "AC_POWER", "DC_POWER", "DAILY_YIELD",
        "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
        "IRRADIATION", "HOUR",
    ]

    pearson  = df_day[CORR_COLS].corr(method="pearson")
    spearman = df_day[CORR_COLS].corr(method="spearman")

    best_corr  = pearson.copy()
    method_map = pd.DataFrame("P", index=pearson.index, columns=pearson.columns)
    for r in CORR_COLS:
        for c in CORR_COLS:
            if abs(spearman.loc[r, c]) > abs(pearson.loc[r, c]):
                best_corr.loc[r, c]  = spearman.loc[r, c]
                method_map.loc[r, c] = "S"

    mask = np.triu(np.ones_like(best_corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.suptitle(f"Plant {plant_num} — Correlation Heatmap\n"
                 f"(P=Pearson, S=Spearman — higher |r| selected)",
                 fontsize=14, fontweight="bold")

    sns.heatmap(
        best_corr, mask=mask, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=False, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )

    n = len(CORR_COLS)
    for i in range(n):
        for j in range(n):
            if i > j:
                val    = best_corr.iloc[i, j]
                method = method_map.iloc[i, j]
                color  = "white" if abs(val) > 0.5 else "black"
                ax.text(j + 0.5, i + 0.38, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")
                ax.text(j + 0.5, i + 0.65, f"({method})",
                        ha="center", va="center",
                        fontsize=7, color=color)

    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"EDA_P{plant_num}_Heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Saved] EDA_P{plant_num}_Heatmap.png")

    return best_corr, method_map

# ==============================================================================
# PRINT SUMMARY
# ==============================================================================

def print_summary(plant_num, df_day, degree_df, best_corr, method_map):
    print(f"\n{'='*65}")
    print(f"  Plant {plant_num} — EDA Summary")
    print(f"{'='*65}")
    print(f"  Rows (daytime) : {len(df_day):,}")

    print(f"\n-- Best Polynomial Degree per Scatter Pair --")
    print(degree_df.to_string(index=False))

    print(f"\n-- Best Correlation with AC_POWER -----------")
    rows = []
    for col in best_corr.columns:
        if col == "AC_POWER":
            continue
        val    = best_corr.loc[col, "AC_POWER"]
        method = method_map.loc[col, "AC_POWER"]
        rows.append({"Feature"    : col,
                     "Correlation": round(float(val), 4),
                     "Method"     : method})
    ac_df = (pd.DataFrame(rows)
               .sort_values("Correlation", key=abs, ascending=False))
    print(ac_df.to_string(index=False))

# ==============================================================================
# MAIN
# ==============================================================================

for plant_num in [1, 2]:
    print(f"\n{'='*55}")
    print(f"  Processing Plant {plant_num}")
    print(f"{'='*55}")
    df, df_day = load_plant(plant_num)
    print(f"  Rows: {len(df):,} | Daytime: {len(df_day):,}")

    degree_df       = plot_scatter(df_day, plant_num)
    best_corr, mmap = plot_heatmap(df_day, plant_num)
    print_summary(plant_num, df_day, degree_df, best_corr, mmap)

print("\n  Done -- 4 PNG files saved to output/")
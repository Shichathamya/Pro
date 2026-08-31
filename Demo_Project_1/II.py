"""
II.py
============
EDA for Solar Power Generation
- Scatter Plot (vs AC Power only)
- Correlation Heatmap (Pearson/Spearman + Mutual Information)
- Print summary
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

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
    df_day          = df[df["IRRADIATION"] > 0].copy()
    return df, df_day

# ==============================================================================
# FIGURE 1 — Scatter Plots vs AC Power
# ==============================================================================

def plot_scatter(df_day, plant_num):

    sample = df_day.sample(min(3000, len(df_day)), random_state=42)

    SCATTER_PAIRS = [
        ("IRRADIATION",         "AC_POWER", "Irradiation vs AC Power",     SOLAR),
        ("MODULE_TEMPERATURE",  "AC_POWER", "Module Temp vs AC Power",      RED),
        ("AMBIENT_TEMPERATURE", "AC_POWER", "Ambient Temp vs AC Power",     GREEN),
        ("HOUR",                "AC_POWER", "Hour vs AC Power",             BLUE),
        ("TIME_SLOT",           "AC_POWER", "Time (15-min) vs AC Power",    BLUE),
        ("DAILY_YIELD",         "AC_POWER", "Daily Yield vs AC Power",      SOLAR),
    ]

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
    fig.suptitle(f"Plant {plant_num} — Scatter Plots vs AC Power",
                 fontsize=15, fontweight="bold", y=1.01)

    for ax, (x_col, y_col, title, color) in zip(axes.flat, SCATTER_PAIRS):
        ax.scatter(sample[x_col], sample[y_col],
                   c=color, alpha=0.35, s=8, rasterized=True)

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
        ax.set_ylabel("AC Power (kW)", fontsize=8)

    for ax in list(axes.flat)[len(SCATTER_PAIRS):]:
        ax.set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"EDA_P{plant_num}_Scatter.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Saved] EDA_P{plant_num}_Scatter.png")

# ==============================================================================
# FIGURE 2 — Correlation Heatmap
# ==============================================================================

def plot_heatmap(df_day, plant_num):

    CORR_COLS = [
        "AC_POWER", "DC_POWER",
        "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
        "IRRADIATION", "HOUR",
    ]

    pearson  = df_day[CORR_COLS].corr(method="pearson")
    spearman = df_day[CORR_COLS].corr(method="spearman")

    # best of Pearson/Spearman
    best_corr  = pearson.copy()
    method_map = pd.DataFrame("P", index=pearson.index, columns=pearson.columns)
    for r in CORR_COLS:
        for c in CORR_COLS:
            if abs(spearman.loc[r, c]) > abs(pearson.loc[r, c]):
                best_corr.loc[r, c]  = spearman.loc[r, c]
                method_map.loc[r, c] = "S"

    # Mutual Information
    X_mi = df_day[CORR_COLS].dropna()
    mi_rows = {}
    for target in CORR_COLS:
        features  = [c for c in CORR_COLS if c != target]
        mi_scores = mutual_info_regression(
            X_mi[features], X_mi[target], random_state=42
        )
        mi_norm = mi_scores / (mi_scores.max() + 1e-9)
        mi_rows[target] = dict(zip(features, mi_norm))

    mi_df  = pd.DataFrame(mi_rows).fillna(0)
    mi_sym = (mi_df + mi_df.T) / 2
    mi_sym = mi_sym.reindex(index=CORR_COLS, columns=CORR_COLS).fillna(0)

    mask    = np.triu(np.ones_like(best_corr, dtype=bool))
    mask_mi = np.triu(np.ones_like(mi_sym,   dtype=bool))
    n       = len(CORR_COLS)

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(f"Plant {plant_num} — Correlation Heatmap",
                 fontsize=14, fontweight="bold")

    # Left: Pearson/Spearman
    sns.heatmap(
        best_corr, mask=mask, ax=axes[0],
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=False, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    for i in range(n):
        for j in range(n):
            if i > j:
                val    = best_corr.iloc[i, j]
                method = method_map.iloc[i, j]
                color  = "white" if abs(val) > 0.5 else "black"
                axes[0].text(j + 0.5, i + 0.38, f"{val:.2f}",
                             ha="center", va="center",
                             fontsize=9, color=color, fontweight="bold")
                axes[0].text(j + 0.5, i + 0.65, f"({method})",
                             ha="center", va="center",
                             fontsize=7, color=color)
    axes[0].set_title("Pearson / Spearman  (higher |r| selected)",
                      fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45, labelsize=9)
    axes[0].tick_params(axis="y", rotation=0,  labelsize=9)

    # Right: Mutual Information
    sns.heatmap(
        mi_sym, mask=mask_mi, ax=axes[1],
        cmap="YlOrRd", vmin=0, vmax=1,
        annot=False, linewidths=0.5, cbar_kws={"shrink": 0.8},
    )
    for i in range(n):
        for j in range(n):
            if i > j:
                val   = mi_sym.iloc[i, j]
                color = "white" if val > 0.6 else "black"
                axes[1].text(j + 0.5, i + 0.5, f"{val:.2f}",
                             ha="center", va="center",
                             fontsize=9, color=color, fontweight="bold")
    axes[1].set_title("Mutual Information\n(captures non-linear + bell curve)",
                      fontsize=12, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45, labelsize=9)
    axes[1].tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"EDA_P{plant_num}_Heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [Saved] EDA_P{plant_num}_Heatmap.png")

    return best_corr, method_map

# ==============================================================================
# PRINT SUMMARY
# ==============================================================================

def print_summary(plant_num, df_day, best_corr, method_map):
    print(f"\n{'='*65}")
    print(f"  Plant {plant_num} — EDA Summary")
    print(f"{'='*65}")
    print(f"  Rows (daytime) : {len(df_day):,}")

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

    plot_scatter(df_day, plant_num)
    best_corr, mmap = plot_heatmap(df_day, plant_num)
    print_summary(plant_num, df_day, best_corr, mmap)

print("\n  Done -- 4 PNG files saved to output/")
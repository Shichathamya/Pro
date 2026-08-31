"""
IIII.py
====================
ทดสอบ Performance ของ Model RF_AC_POWER_Plant1
โดยใช้ข้อมูล Plant 1 และ Plant 2
แสดงผล Performance Ratio ต่อ Inverter
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOLAR = "#F0A500"
BLUE  = "#4C9BE8"
RED   = "#E8564C"
GREEN = "#5CB85C"
GRAY  = "#AAAAAA"

THRESHOLD = 0.85

# ==============================================================================
# 1. LOAD MODEL
# ==============================================================================

model        = joblib.load(os.path.join(MODEL_DIR, "RF_AC_POWER_Plant1.pkl"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "RF_AC_POWER_Plant1_features.pkl"))

print(f"  Model    : RF_AC_POWER_Plant1.pkl")
print(f"  Features : {feature_cols}")

# ==============================================================================
# 2. LOAD & PREDICT FUNCTION
# ==============================================================================

def load_and_predict(plant_num):
    df = pd.read_csv(os.path.join(DATASET_DIR,
                                  f"Plant_{plant_num}_Joined.csv"))
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    df["HOUR"]      = df["DATE_TIME"].dt.hour
    df["DATE"]      = df["DATE_TIME"].dt.date

    df = df[df["IRRADIATION"] > 0].copy()
    df = df.dropna(subset=feature_cols + ["AC_POWER"])

    df["AC_PREDICTED"]      = model.predict(df[feature_cols])
    df["ERROR"]             = df["AC_POWER"] - df["AC_PREDICTED"]
    df["ABS_ERROR"]         = df["ERROR"].abs()
    df["PERFORMANCE_RATIO"] = (
        df["AC_POWER"] / df["AC_PREDICTED"].replace(0, np.nan)
    ).clip(0, 2)

    print(f"\n  Plant {plant_num}: {len(df):,} rows | "
          f"Inverters: {df['SOURCE_KEY'].nunique()}")
    return df

# ==============================================================================
# 3. INVERTER SUMMARY FUNCTION
# ==============================================================================

def inverter_summary(df):
    inv = df.groupby("SOURCE_KEY").agg(
        AC_ACTUAL    = ("AC_POWER",           "mean"),
        AC_PREDICTED = ("AC_PREDICTED",        "mean"),
        PERF_RATIO   = ("PERFORMANCE_RATIO",   "mean"),
        ABS_ERROR    = ("ABS_ERROR",           "mean"),
        TOTAL_ACTUAL = ("AC_POWER",            "sum"),
    ).reset_index().sort_values("PERF_RATIO", ascending=True)

    inv["STATUS"] = inv["PERF_RATIO"].apply(
        lambda x: "⚠ Under" if x < THRESHOLD else "✓ Normal"
    )
    return inv

# ==============================================================================
# 4. PLOT FUNCTION
# ==============================================================================

def plot_performance(df, inv, plant_num):

    colors_inv = [RED if r < THRESHOLD else GREEN
                  for r in inv["PERF_RATIO"]]
    n = len(inv)

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Plant {plant_num} — Performance Test  "
                 f"(Model: RF_AC_POWER_Plant1)",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── 1. Performance Ratio per Inverter ────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    bars = ax.barh(range(n), inv["PERF_RATIO"],
                   color=colors_inv, edgecolor="white", height=0.6)
    ax.axvline(THRESHOLD, color=RED, linestyle="--",
               linewidth=2, label=f"Threshold = {THRESHOLD}")
    ax.axvline(1.0, color=GRAY, linestyle=":", linewidth=1.5,
               label="Perfect = 1.0")
    for bar, val, status in zip(bars, inv["PERF_RATIO"], inv["STATUS"]):
        ax.text(val + 0.005,
                bar.get_y() + bar.get_height()/2,
                f"{val:.3f}  {status}", va="center", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([k[:10] for k in inv["SOURCE_KEY"]], fontsize=7)
    ax.set_title("Performance Ratio per Inverter  (Actual / Predicted)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Performance Ratio")
    ax.legend(fontsize=9)

    # ── 2. Actual vs Predicted per Inverter (bar) ────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    x = range(n)
    ax.barh([i - 0.2 for i in x], inv["AC_ACTUAL"],
            height=0.4, color=BLUE, edgecolor="white",
            alpha=0.85, label="Actual")
    ax.barh([i + 0.2 for i in x], inv["AC_PREDICTED"],
            height=0.4, color=GRAY, edgecolor="white",
            alpha=0.85, label="Predicted")
    ax.set_yticks(list(x))
    ax.set_yticklabels([k[:10] for k in inv["SOURCE_KEY"]], fontsize=7)
    ax.set_title("Actual vs Predicted per Inverter",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean AC Power (kW)")
    ax.legend(fontsize=9)

    # ── 3. ABS Error per Inverter ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.barh(range(n), inv["ABS_ERROR"],
            color=colors_inv, edgecolor="white", height=0.6)
    ax.axvline(inv["ABS_ERROR"].mean(), color=SOLAR, linestyle="--",
               linewidth=2, label=f"Mean = {inv['ABS_ERROR'].mean():.1f}")
    ax.set_yticks(range(n))
    ax.set_yticklabels([k[:10] for k in inv["SOURCE_KEY"]], fontsize=7)
    ax.set_title("Mean Absolute Error per Inverter",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("MAE (kW)")
    ax.legend(fontsize=9)

    # ── 4. Error Distribution Boxplot ────────────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    inv_keys  = inv["SOURCE_KEY"].tolist()
    error_box = [df[df["SOURCE_KEY"] == k]["ERROR"].values
                 for k in inv_keys]
    bp = ax.boxplot(error_box, patch_artist=True, showfliers=False,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, c in zip(bp["boxes"], colors_inv):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.axhline(0, color=RED, linestyle="--", linewidth=1.5)
    ax.set_xticks(range(1, len(inv_keys) + 1))
    ax.set_xticklabels([k[:10] for k in inv_keys],
                       rotation=45, fontsize=6)
    ax.set_title("Error Distribution per Inverter  (Actual − Predicted)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Error (kW)")

    # ── 5. Scatter Actual vs Predicted (all data) ─────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    sample = df.sample(min(3000, len(df)), random_state=42)
    ax.scatter(sample["AC_POWER"], sample["AC_PREDICTED"],
               alpha=0.3, s=6, color=BLUE, rasterized=True)
    max_v = max(df["AC_POWER"].max(), df["AC_PREDICTED"].max())
    ax.plot([0, max_v], [0, max_v], color=RED,
            linewidth=1.5, linestyle="--", label="Perfect")
    ax.set_title("Actual vs Predicted (all inverters)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual AC Power (kW)")
    ax.set_ylabel("Predicted AC Power (kW)")
    ax.legend(fontsize=9)

    # ── 6. Performance Ratio Over Time ───────────────────────────────────────
    ax = fig.add_subplot(gs[3, 1])
    daily = df.groupby("DATE").agg(
        PERF_RATIO = ("PERFORMANCE_RATIO", "mean")
    ).reset_index()
    colors_day = [RED if r < THRESHOLD else GREEN
                  for r in daily["PERF_RATIO"]]
    ax.bar(range(len(daily)), daily["PERF_RATIO"],
           color=colors_day, edgecolor="white", alpha=0.85)
    ax.axhline(THRESHOLD, color=RED, linestyle="--",
               linewidth=2, label=f"Threshold = {THRESHOLD}")
    ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.5)
    ax.set_xticks(range(0, len(daily), 5))
    ax.set_xticklabels([str(d) for d in daily["DATE"].iloc[::5]],
                       rotation=30, fontsize=7)
    ax.set_title("Daily Performance Ratio Over Time",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Performance Ratio")
    ax.legend(fontsize=9)

    fig.savefig(os.path.join(OUTPUT_DIR,
                             f"TestPerf_P{plant_num}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] TestPerf_P{plant_num}.png")

# ==============================================================================
# 5. PRINT TABLE
# ==============================================================================

def print_summary(inv, plant_num):
    print(f"\n{'='*65}")
    print(f"  Plant {plant_num} — Inverter Performance Summary")
    print(f"{'='*65}")
    print(inv[["SOURCE_KEY", "AC_ACTUAL", "AC_PREDICTED",
               "PERF_RATIO", "ABS_ERROR", "STATUS"]].to_string(index=False))
    under = inv[inv["PERF_RATIO"] < THRESHOLD]
    print(f"\n  Underperforming: {len(under)}/{len(inv)} inverters")
    print(f"  Avg Performance Ratio: {inv['PERF_RATIO'].mean():.4f}")
    print(f"  Avg ABS Error        : {inv['ABS_ERROR'].mean():.2f} kW")

# ==============================================================================
# 6. EXPORT CSV
# ==============================================================================

def export_csv(inv, plant_num):
    path = os.path.join(OUTPUT_DIR,
                        f"TestPerf_P{plant_num}_inverter.csv")
    inv.to_csv(path, index=False)
    print(f"  [Saved] TestPerf_P{plant_num}_inverter.csv")

# ==============================================================================
# MAIN
# ==============================================================================

print("\n" + "#"*60)
print("#  TEST PERFORMANCE — RF_AC_POWER_Plant1")
print("#"*60)

for plant_num in [1, 2]:
    df  = load_and_predict(plant_num)
    inv = inverter_summary(df)
    print_summary(inv, plant_num)
    plot_performance(df, inv, plant_num)
    export_csv(inv, plant_num)

print("\n  Done.")
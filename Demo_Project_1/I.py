"""
Inverter_Sensor_Analysis.py
============================
1. แสดงค่าต่อ Inverter แต่ละ Plant
2. ค่า Sensor ในช่วงเวลา แต่ละ Plant
3. AC Power ต่อ Inverter ตาม Time Series
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "axes.facecolor"   : "#F8F9FA",
    "figure.facecolor" : "white",
    "axes.grid"        : True,
    "grid.color"       : "#E0E0E0",
    "grid.linewidth"   : 0.6,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "font.family"      : "DejaVu Sans",
})

SOLAR   = "#F0A500"
BLUE    = "#4C9BE8"
RED     = "#E8564C"
GREEN   = "#5CB85C"
GRAY    = "#AAAAAA"
PALETTE = [plt.cm.tab20(i) for i in range(20)]

# ==============================================================================
# CONFIG
# ==============================================================================

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")

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
    df["DATE"]      = df["DATE_TIME"].dt.date
    df["HOUR"]      = df["DATE_TIME"].dt.hour
    df_day          = df[df["IRRADIATION"] > 0].copy()
    return df, df_day

# ==============================================================================
# FIGURE 1 — Inverter Performance (target: AC_POWER)
# ==============================================================================

def plot_inverter(df_day, plant_num):

    inv = df_day.groupby("SOURCE_KEY").agg(
        AC_mean  = ("AC_POWER",   "mean"),
        AC_max   = ("AC_POWER",   "max"),
        AC_total = ("AC_POWER",   "sum"),
        DC_mean  = ("DC_POWER",   "mean"),
        YIELD    = ("DAILY_YIELD","mean"),
    ).reset_index().sort_values("AC_total", ascending=True)

    mean_ac = inv["AC_mean"].mean()
    colors  = [RED if v < mean_ac * 0.9 else GREEN for v in inv["AC_mean"]]
    n       = len(inv)

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f"Plant {plant_num} — Inverter Performance  (Target: AC Power)",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # 1a. Total AC Power per Inverter
    ax = fig.add_subplot(gs[0, :])
    bars = ax.barh(range(n), inv["AC_total"],
                   color=colors, edgecolor="white", height=0.6)
    ax.axvline(inv["AC_total"].mean(), color=SOLAR, linestyle="--",
               linewidth=2, label=f"Mean = {inv['AC_total'].mean():,.0f}")
    for bar, val in zip(bars, inv["AC_total"]):
        ax.text(val + inv["AC_total"].max() * 0.005,
                bar.get_y() + bar.get_height()/2,
                f"{val:,.0f}", va="center", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([k[:10] for k in inv["SOURCE_KEY"]], fontsize=7)
    ax.set_title("Total AC Power per Inverter  (red = underperforming)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Total AC Power (kW)")
    ax.legend(fontsize=9)

    # 1b. Mean AC Power per Inverter
    ax = fig.add_subplot(gs[1, 0])
    ax.barh(range(n), inv["AC_mean"],
            color=colors, edgecolor="white", height=0.6)
    ax.axvline(mean_ac, color=SOLAR, linestyle="--", linewidth=2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([k[:10] for k in inv["SOURCE_KEY"]], fontsize=7)
    ax.set_title("Mean AC Power per Inverter", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean AC Power (kW)")

    # 1c. Max AC Power per Inverter
    ax = fig.add_subplot(gs[1, 1])
    ax.barh(range(n), inv["AC_max"],
            color=BLUE, edgecolor="white", height=0.6, alpha=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([k[:10] for k in inv["SOURCE_KEY"]], fontsize=7)
    ax.set_title("Max AC Power per Inverter", fontsize=12, fontweight="bold")
    ax.set_xlabel("Max AC Power (kW)")

    # 1d. DC vs AC Mean per Inverter
    ax = fig.add_subplot(gs[2, 0])
    x = range(n)
    ax.barh([i - 0.2 for i in x], inv["DC_mean"],
            height=0.4, color=SOLAR, edgecolor="white", alpha=0.85, label="DC Mean")
    ax.barh([i + 0.2 for i in x], inv["AC_mean"],
            height=0.4, color=BLUE,  edgecolor="white", alpha=0.85, label="AC Mean")
    ax.set_yticks(list(x))
    ax.set_yticklabels([k[:10] for k in inv["SOURCE_KEY"]], fontsize=7)
    ax.set_title("DC vs AC Mean per Inverter", fontsize=12, fontweight="bold")
    ax.set_xlabel("Power (kW)")
    ax.legend(fontsize=9)

    # 1e. Mean Daily Yield per Inverter
    ax = fig.add_subplot(gs[2, 1])
    ax.barh(range(n), inv["YIELD"],
            color=GREEN, edgecolor="white", height=0.6, alpha=0.8)
    ax.axvline(inv["YIELD"].mean(), color=RED, linestyle="--",
               linewidth=2, label=f"Mean = {inv['YIELD'].mean():,.0f}")
    ax.set_yticks(range(n))
    ax.set_yticklabels([k[:10] for k in inv["SOURCE_KEY"]], fontsize=7)
    ax.set_title("Mean Daily Yield per Inverter", fontsize=12, fontweight="bold")
    ax.set_xlabel("Daily Yield (kWh)")
    ax.legend(fontsize=9)

    fig.savefig(os.path.join(OUTPUT_DIR, f"P{plant_num}_Inverter.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] P{plant_num}_Inverter.png")

# ==============================================================================
# FIGURE 2 — Sensor Over Time (target: AC_POWER)
# ==============================================================================

def plot_sensor_time(df, df_day, plant_num):

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(f"Plant {plant_num} — Sensor Values Over Time",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)

    ts = df.sort_values("DATE_TIME").iloc[::4]

    # 2a. AC Power over time
    ax = fig.add_subplot(gs[0, :])
    ax.plot(ts["DATE_TIME"], ts["AC_POWER"],
            color=BLUE, linewidth=0.7, alpha=0.8)
    ax.set_title("AC Power Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("AC Power (kW)")
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # 2b. DC Power over time
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ts["DATE_TIME"], ts["DC_POWER"],
            color=SOLAR, linewidth=0.7, alpha=0.8)
    ax.set_title("DC Power Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("DC Power (kW)")
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # 2c. Irradiation over time
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ts["DATE_TIME"], ts["IRRADIATION"],
            color=GREEN, linewidth=0.7, alpha=0.8)
    ax.set_title("Irradiation Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Irradiation (W/m²)")
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # 2d. Ambient Temperature over time
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(ts["DATE_TIME"], ts["AMBIENT_TEMPERATURE"],
            color=RED, linewidth=0.7, alpha=0.8)
    ax.set_title("Ambient Temperature Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ambient Temp (°C)")
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # 2e. Module Temperature over time
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(ts["DATE_TIME"], ts["MODULE_TEMPERATURE"],
            color="#E87C4C", linewidth=0.7, alpha=0.8)
    ax.set_title("Module Temperature Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Module Temp (°C)")
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # 2f. Mean AC Power per Hour
    ax = fig.add_subplot(gs[3, 0])
    hourly_ac = df_day.groupby("HOUR")["AC_POWER"].mean()
    all_hours  = pd.Series(0.0, index=range(24))
    all_hours.update(hourly_ac)
    ax.bar(all_hours.index, all_hours.values,
           color=BLUE, edgecolor="white", alpha=0.85)
    ax.set_title("Mean AC Power by Hour", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean AC Power (kW)")
    ax.set_xticks(range(24))

    # 2g. Mean Irradiation per Hour
    ax = fig.add_subplot(gs[3, 1])
    hourly_irr = df_day.groupby("HOUR")["IRRADIATION"].mean()
    all_hours  = pd.Series(0.0, index=range(24))
    all_hours.update(hourly_irr)
    ax.plot(all_hours.index, all_hours.values,
            marker="o", color=GREEN, linewidth=2, markersize=5)
    ax.fill_between(all_hours.index, all_hours.values,
                    alpha=0.2, color=GREEN)
    ax.set_title("Mean Irradiation by Hour", fontsize=12, fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Irradiation (W/m²)")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24), fontsize=8)

    fig.savefig(os.path.join(OUTPUT_DIR, f"P{plant_num}_Sensor_Time.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] P{plant_num}_Sensor_Time.png")

# ==============================================================================
# FIGURE 3 — AC Power per Inverter Time Series
# ==============================================================================

def plot_inverter_timeseries(df_day, plant_num):

    inv_keys = sorted(df_day["SOURCE_KEY"].unique())
    n_inv    = len(inv_keys)

    # hourly mean AC ต่อ inverter
    hourly = (
        df_day.groupby(["DATE_TIME", "SOURCE_KEY"])["AC_POWER"]
        .mean().reset_index()
        .sort_values("DATE_TIME")
    )

    # ── Figure 3a: ทุก inverter ในกราฟเดียว ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 8))
    for i, key in enumerate(inv_keys):
        sub = hourly[hourly["SOURCE_KEY"] == key]
        ax.plot(sub["DATE_TIME"], sub["AC_POWER"],
                color=PALETTE[i % len(PALETTE)],
                linewidth=0.8, alpha=0.7, label=key[:10])
    ax.set_title(f"Plant {plant_num} — AC Power per Inverter Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("AC Power (kW)")
    ax.set_xlabel("Date Time")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.legend(fontsize=6, ncol=4, loc="upper right",
              bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"P{plant_num}_AC_Inverter_All.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] P{plant_num}_AC_Inverter_All.png")

    # ── Figure 3b: แยกต่อ inverter (subplots) ────────────────────────────────
    ncols   = 4
    nrows   = int(np.ceil(n_inv / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(22, nrows * 3.5),
                             sharey=False)
    fig.suptitle(f"Plant {plant_num} — AC Power Each Inverter Over Time",
                 fontsize=14, fontweight="bold", y=1.01)

    axes_flat = axes.flat
    for i, key in enumerate(inv_keys):
        ax  = next(axes_flat)
        sub = hourly[hourly["SOURCE_KEY"] == key].iloc[::2]
        ax.plot(sub["DATE_TIME"], sub["AC_POWER"],
                color=PALETTE[i % len(PALETTE)], linewidth=0.8)
        ax.fill_between(sub["DATE_TIME"], sub["AC_POWER"],
                        alpha=0.15, color=PALETTE[i % len(PALETTE)])
        ax.set_title(key[:12], fontsize=8, fontweight="bold")
        ax.set_ylabel("AC (kW)", fontsize=7)
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.tick_params(axis="y", labelsize=6)

    # ซ่อน subplot ที่เหลือ
    for ax in axes_flat:
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"P{plant_num}_AC_Inverter_Each.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] P{plant_num}_AC_Inverter_Each.png")

# ==============================================================================
# MAIN
# ==============================================================================

for plant_num in [1, 2]:
    print(f"\n{'='*50}")
    print(f"  Processing Plant {plant_num}")
    print(f"{'='*50}")
    df, df_day = load_plant(plant_num)
    print(f"  Rows: {len(df):,} | Inverters: {df['SOURCE_KEY'].nunique()}")
    plot_inverter(df_day, plant_num)
    plot_sensor_time(df, df_day, plant_num)
    plot_inverter_timeseries(df_day, plant_num)

print("\n  Done — 6 PNG files saved to output/")
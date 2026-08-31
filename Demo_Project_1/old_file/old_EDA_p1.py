"""
EDA_Plant1.py
=============
Exploratory Data Analysis for Plant 1 Solar Power Generation
Target: DC_POWER
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
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

PALETTE = [plt.cm.tab10(i) for i in range(10)]
SOLAR   = "#F0A500"
BLUE    = "#4C9BE8"
RED     = "#E8564C"
GREEN   = "#5CB85C"

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================

df = pd.read_csv(os.path.join(DATASET_DIR, "Plant_1_Joined.csv"))
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

df["DATE"]  = df["DATE_TIME"].dt.date
df["HOUR"]  = df["DATE_TIME"].dt.hour
df["MONTH"] = df["DATE_TIME"].dt.month
df["DAY"]   = df["DATE_TIME"].dt.day

df_day = df[df["IRRADIATION"] > 0].copy()

print(f"  Total rows     : {len(df):,}")
print(f"  Daytime rows   : {len(df_day):,}")
print(f"  Inverters      : {df['SOURCE_KEY'].nunique()}")
print(f"  Date range     : {df['DATE_TIME'].min()} → {df['DATE_TIME'].max()}")
print(f"  Missing values : {df.isna().sum().to_dict()}")

# ==============================================================================
# FIGURE 1 — Overview (target: DC_POWER)
# ==============================================================================

fig1 = plt.figure(figsize=(20, 16))
fig1.suptitle("Plant 1 — Solar Power Generation Overview (Target: DC Power)",
              fontsize=16, fontweight="bold", y=0.98)
gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.35)

# 1a. Daily Total DC Power
ax = fig1.add_subplot(gs1[0, :2])
daily_dc = df.groupby("DATE")["DC_POWER"].sum().reset_index()
ax.bar(range(len(daily_dc)), daily_dc["DC_POWER"],
       color=SOLAR, edgecolor="white", alpha=0.85)
ax.set_title("Daily Total DC Power Output", fontsize=12, fontweight="bold")
ax.set_xlabel("Day"); ax.set_ylabel("DC Power (kW)")
ax.set_xticks(range(0, len(daily_dc), 5))
ax.set_xticklabels([str(d) for d in daily_dc["DATE"].iloc[::5]],
                   rotation=30, fontsize=7)

# 1b. AC Power vs DC Power scatter
ax = fig1.add_subplot(gs1[0, 2])
sample = df_day.sample(min(3000, len(df_day)), random_state=42)
ax.scatter(sample["AC_POWER"], sample["DC_POWER"],
           alpha=0.3, s=5, color=BLUE, rasterized=True)
ax.set_title("AC Power vs DC Power", fontsize=12, fontweight="bold")
ax.set_xlabel("AC Power (kW)"); ax.set_ylabel("DC Power (kW)")

# 1c. Irradiation vs DC Power scatter
ax = fig1.add_subplot(gs1[1, 0])
ax.scatter(sample["IRRADIATION"], sample["DC_POWER"],
           alpha=0.3, s=5, color=SOLAR, rasterized=True)
ax.set_title("Irradiation vs DC Power", fontsize=12, fontweight="bold")
ax.set_xlabel("Irradiation (W/m²)"); ax.set_ylabel("DC Power (kW)")

# 1d. Module Temp vs DC Power scatter
ax = fig1.add_subplot(gs1[1, 1])
ax.scatter(sample["MODULE_TEMPERATURE"], sample["DC_POWER"],
           alpha=0.3, s=5, color=RED, rasterized=True)
ax.set_title("Module Temperature vs DC Power", fontsize=12, fontweight="bold")
ax.set_xlabel("Module Temp (°C)"); ax.set_ylabel("DC Power (kW)")

# 1e. Correlation Heatmap
ax = fig1.add_subplot(gs1[1, 2])
corr_cols = ["DC_POWER", "AC_POWER", "DAILY_YIELD",
             "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]
corr = df_day[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 7}, linewidths=0.4,
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Heatmap", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.tick_params(axis="y", rotation=0,  labelsize=7)

# 1f. Hourly Average DC Power
ax = fig1.add_subplot(gs1[2, 0])
hourly_dc = df_day.groupby("HOUR")["DC_POWER"].mean()
ax.bar(hourly_dc.index, hourly_dc.values,
       color=SOLAR, edgecolor="white", alpha=0.85)
ax.set_title("Average DC Power by Hour", fontsize=12, fontweight="bold")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Avg DC Power (kW)")

# 1g. Hourly Average Irradiation
ax = fig1.add_subplot(gs1[2, 1])
hourly_irr = df_day.groupby("HOUR")["IRRADIATION"].mean()
ax.plot(hourly_irr.index, hourly_irr.values,
        marker="o", color=SOLAR, linewidth=2)
ax.fill_between(hourly_irr.index, hourly_irr.values,
                alpha=0.2, color=SOLAR)
ax.set_title("Average Irradiation by Hour", fontsize=12, fontweight="bold")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Irradiation (W/m²)")

# 1h. Temperature comparison
ax = fig1.add_subplot(gs1[2, 2])
hourly_amb = df_day.groupby("HOUR")["AMBIENT_TEMPERATURE"].mean()
hourly_mod = df_day.groupby("HOUR")["MODULE_TEMPERATURE"].mean()
ax.plot(hourly_amb.index, hourly_amb.values,
        marker="o", color=BLUE, linewidth=2, label="Ambient")
ax.plot(hourly_mod.index, hourly_mod.values,
        marker="s", color=RED,  linewidth=2, label="Module")
ax.set_title("Avg Temperature by Hour", fontsize=12, fontweight="bold")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=9)

fig1.savefig(os.path.join(OUTPUT_DIR, "EDA_P1_01_Overview.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print("  [Saved] EDA_P1_01_Overview.png")

# ==============================================================================
# FIGURE 2 — Inverter Performance
# ==============================================================================

fig2 = plt.figure(figsize=(20, 14))
fig2.suptitle("Plant 1 — Inverter Performance Analysis",
              fontsize=16, fontweight="bold", y=0.98)
gs2 = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.35)

# 2a. Total DC Power per Inverter
ax = fig2.add_subplot(gs2[0, :2])
inv_total = (df_day.groupby("SOURCE_KEY")["DC_POWER"]
             .sum().sort_values(ascending=True))
colors_bar = [RED if v < inv_total.mean() * 0.9 else GREEN
              for v in inv_total.values]
bars = ax.barh(range(len(inv_total)), inv_total.values,
               color=colors_bar, edgecolor="white", height=0.6)
ax.axvline(inv_total.mean(), color=SOLAR, linestyle="--",
           linewidth=2, label=f"Mean = {inv_total.mean():.0f}")
ax.set_yticks(range(len(inv_total)))
ax.set_yticklabels([k[:8] for k in inv_total.index], fontsize=7)
ax.set_title("Total DC Power per Inverter (red = underperforming)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Total DC Power (kW)")
ax.legend(fontsize=9)

# 2b. Daily Yield distribution per Inverter (boxplot)
ax = fig2.add_subplot(gs2[0, 2])
inv_keys = df_day["SOURCE_KEY"].unique()[:8]
data_box = [df_day[df_day["SOURCE_KEY"] == k]["DAILY_YIELD"].values
            for k in inv_keys]
bp = ax.boxplot(data_box, patch_artist=True, showfliers=False,
                medianprops={"color": "white", "linewidth": 2})
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color); patch.set_alpha(0.8)
ax.set_xticklabels([k[:6] for k in inv_keys], rotation=45, fontsize=7)
ax.set_title("Daily Yield Distribution (top 8 inverters)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Daily Yield (kWh)")

# 2c. DC Power heatmap: Hour × Day
ax = fig2.add_subplot(gs2[1, :2])
pivot = df.groupby(["DAY", "HOUR"])["DC_POWER"].mean().unstack(fill_value=0)
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Day of Month")
ax.set_xticks(range(24)); ax.set_xticklabels(range(24), fontsize=7)
ax.set_yticks(range(len(pivot))); ax.set_yticklabels(pivot.index, fontsize=7)
plt.colorbar(im, ax=ax, label="Avg DC Power (kW)")
ax.set_title("DC Power Heatmap (Hour × Day)", fontsize=12, fontweight="bold")

# 2d. Daily Irradiation vs Daily Yield
ax = fig2.add_subplot(gs2[1, 2])
daily_yield  = df.groupby("DATE")["DAILY_YIELD"].max().reset_index()
daily_irr    = df.groupby("DATE")["IRRADIATION"].sum().reset_index()
merged_daily = pd.merge(daily_yield, daily_irr, on="DATE")
ax.scatter(merged_daily["IRRADIATION"], merged_daily["DAILY_YIELD"],
           color=SOLAR, alpha=0.7, s=40, edgecolors="white")
z = np.polyfit(merged_daily["IRRADIATION"], merged_daily["DAILY_YIELD"], 1)
x_line = np.linspace(merged_daily["IRRADIATION"].min(),
                     merged_daily["IRRADIATION"].max(), 100)
ax.plot(x_line, np.poly1d(z)(x_line), color=RED, linewidth=2, label="Trend")
ax.set_title("Daily Irradiation vs Daily Yield", fontsize=12, fontweight="bold")
ax.set_xlabel("Total Irradiation"); ax.set_ylabel("Daily Yield (kWh)")
ax.legend(fontsize=9)

fig2.savefig(os.path.join(OUTPUT_DIR, "EDA_P1_02_Inverter.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig2)
print("  [Saved] EDA_P1_02_Inverter.png")

# ==============================================================================
# FIGURE 3 — Time Series
# ==============================================================================

fig3, axes = plt.subplots(4, 1, figsize=(20, 18), sharex=False)
fig3.suptitle("Plant 1 — Time Series Analysis",
              fontsize=16, fontweight="bold", y=0.98)

metrics = [
    ("DC_POWER",            "DC Power (kW)",         SOLAR),
    ("IRRADIATION",         "Irradiation (W/m²)",    BLUE),
    ("AMBIENT_TEMPERATURE", "Ambient Temp (°C)",      RED),
    ("MODULE_TEMPERATURE",  "Module Temp (°C)",       GREEN),
]

ts = df.sort_values("DATE_TIME")
sample_ts = ts.iloc[::4]

for ax, (col, ylabel, color) in zip(axes, metrics):
    ax.plot(sample_ts["DATE_TIME"], sample_ts[col],
            color=color, linewidth=0.6, alpha=0.8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=7)

axes[-1].set_xlabel("Date Time")
plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_DIR, "EDA_P1_03_TimeSeries.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig3)
print("  [Saved] EDA_P1_03_TimeSeries.png")

# ==============================================================================
# FIGURE 4 — Mean DC Power, Temperature, Irradiation per Hour (Line Chart)
# ==============================================================================

hourly_mean = df_day.groupby("HOUR").agg(
    DC_POWER_mean     = ("DC_POWER",            "mean"),
    AMBIENT_TEMP_mean = ("AMBIENT_TEMPERATURE",  "mean"),
    MODULE_TEMP_mean  = ("MODULE_TEMPERATURE",   "mean"),
    IRRADIATION_mean  = ("IRRADIATION",          "mean"),
).reset_index()

all_hours   = pd.DataFrame({"HOUR": range(24)})
hourly_mean = all_hours.merge(hourly_mean, on="HOUR", how="left").fillna(0)
hours       = hourly_mean["HOUR"]

fig4, axes4 = plt.subplots(3, 1, figsize=(16, 14), sharex=False)  # ← เปลี่ยน
fig4.suptitle("Plant 1 — Mean Values per Hour of Day",
              fontsize=16, fontweight="bold", y=0.98)

hours = hourly_mean["HOUR"]

# 4a. Mean DC Power per Hour
ax = axes4[0]
ax.plot(hours, hourly_mean["DC_POWER_mean"],
        marker="o", color=SOLAR, linewidth=2.5, markersize=6)
ax.fill_between(hours, hourly_mean["DC_POWER_mean"],
                alpha=0.2, color=SOLAR)
for x, y in zip(hours, hourly_mean["DC_POWER_mean"]):
    if y > 0:
        ax.text(x, y + 30, f"{y:.0f}", ha="center", fontsize=6, color=SOLAR)
ax.set_title("Mean DC Power per Hour", fontsize=12, fontweight="bold")
ax.set_ylabel("DC Power (kW)")
ax.set_xlim(-0.5, 23.5)
ax.set_xticks(range(24))
ax.set_xticklabels(range(24), fontsize=8)

# 4b. Mean Temperature per Hour
ax = axes4[1]
ax.plot(hours, hourly_mean["AMBIENT_TEMP_mean"],
        marker="o", color=BLUE, linewidth=2.5, markersize=6, label="Ambient Temp")
ax.plot(hours, hourly_mean["MODULE_TEMP_mean"],
        marker="s", color=RED,  linewidth=2.5, markersize=6, label="Module Temp")
ax.fill_between(hours, hourly_mean["AMBIENT_TEMP_mean"],
                hourly_mean["MODULE_TEMP_mean"],
                alpha=0.1, color=GREEN, label="Gap")
for x, y in zip(hours, hourly_mean["AMBIENT_TEMP_mean"]):
    if y > 0:
        ax.text(x, y - 1.5, f"{y:.1f}", ha="center", fontsize=6, color=BLUE)
for x, y in zip(hours, hourly_mean["MODULE_TEMP_mean"]):
    if y > 0:
        ax.text(x, y + 0.5, f"{y:.1f}", ha="center", fontsize=6, color=RED)
ax.set_title("Mean Temperature per Hour", fontsize=12, fontweight="bold")
ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=9)
ax.set_xlim(-0.5, 23.5)
ax.set_xticks(range(24))
ax.set_xticklabels(range(24), fontsize=8)

# 4c. Mean Irradiation per Hour
ax = axes4[2]
ax.plot(hours, hourly_mean["IRRADIATION_mean"],
        marker="o", color=GREEN, linewidth=2.5, markersize=6)
ax.fill_between(hours, hourly_mean["IRRADIATION_mean"],
                alpha=0.2, color=GREEN)
for x, y in zip(hours, hourly_mean["IRRADIATION_mean"]):
    if y > 0:
        ax.text(x, y + 0.005, f"{y:.3f}", ha="center", fontsize=6, color=GREEN)
ax.set_title("Mean Irradiation per Hour", fontsize=12, fontweight="bold")
ax.set_ylabel("Irradiation (W/m²)")
ax.set_xlabel("Hour of Day")
ax.set_xlim(-0.5, 23.5)
ax.set_xticks(range(24))
ax.set_xticklabels(range(24), fontsize=8)

plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, "EDA_P1_04_MeanPerHour.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig4)
print("  [Saved] EDA_P1_04_MeanPerHour.png")

print("\n  Done — 4 PNG files saved to output/")
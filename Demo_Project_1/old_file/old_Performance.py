"""
Performance_Forecast.py
=======================
1. Performance Monitoring  — วัดประสิทธิภาพแผง Solar
2. Power Forecasting       — ทำนายกำลังผลิต DC_POWER
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# CONFIG — แก้ตรงนี้จุดเดียว
# ==============================================================================

INPUT_FILE  = "Plant_2_Joined.csv"   # ← เปลี่ยน data ที่เข้า
THRESHOLD   = 0.85                    # Performance Ratio threshold
TRAIN_DAYS  = 30                      # จำนวนวันที่ใช้เป็น train สำหรับ forecast

MODEL_FILE  = "RF_DC_POWER_Plant1.pkl"
FEAT_FILE   = "RF_DC_POWER_Plant1_features.pkl"

# ==============================================================================

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SOLAR = "#F0A500"
BLUE  = "#4C9BE8"
RED   = "#E8564C"
GREEN = "#5CB85C"
GRAY  = "#AAAAAA"

# prefix output ตามชื่อไฟล์ที่เข้า
PREFIX = os.path.splitext(INPUT_FILE)[0]   # e.g. "Plant_2_Joined"

# ==============================================================================
# 1. LOAD DATA & MODEL
# ==============================================================================

df = pd.read_csv(os.path.join(DATASET_DIR, INPUT_FILE))
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
df["HOUR"]      = df["DATE_TIME"].dt.hour
df["MINUTE"]    = df["DATE_TIME"].dt.minute
df["DAY"]       = df["DATE_TIME"].dt.day
df["MONTH"]     = df["DATE_TIME"].dt.month
df["DATE"]      = df["DATE_TIME"].dt.date

df_day = df[df["IRRADIATION"] > 0].copy()
df_day = df_day.dropna(subset=["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"])

model        = joblib.load(os.path.join(MODEL_DIR, MODEL_FILE))
feature_cols = joblib.load(os.path.join(MODEL_DIR, FEAT_FILE))

print(f"\n  Input file  : {INPUT_FILE}")
print(f"  Model file  : {MODEL_FILE}  (fixed)")
print(f"  Rows (day)  : {len(df_day):,}")
print(f"  Inverters   : {df_day['SOURCE_KEY'].nunique()}")

# ทำนายทั้ง dataset
df_day = df_day.copy()
df_day["DC_POWER_PREDICTED"] = model.predict(df_day[feature_cols])
df_day["ERROR"]              = df_day["DC_POWER"] - df_day["DC_POWER_PREDICTED"]
df_day["ABS_ERROR"]          = df_day["ERROR"].abs()

# ==============================================================================
# 2. PERFORMANCE MONITORING
# ==============================================================================

df_day["PERFORMANCE_RATIO"] = (
    df_day["DC_POWER"] / df_day["DC_POWER_PREDICTED"].replace(0, np.nan)
).clip(0, 2)

inv_perf = df_day.groupby("SOURCE_KEY").agg(
    DC_ACTUAL    = ("DC_POWER",           "mean"),
    DC_PREDICTED = ("DC_POWER_PREDICTED", "mean"),
    PERF_RATIO   = ("PERFORMANCE_RATIO",  "mean"),
    ABS_ERROR    = ("ABS_ERROR",          "mean"),
).reset_index().sort_values("PERF_RATIO", ascending=True)

inv_perf["STATUS"] = inv_perf["PERF_RATIO"].apply(
    lambda x: "⚠ Underperforming" if x < THRESHOLD else "✓ Normal"
)

daily_perf = df_day.groupby("DATE").agg(
    DC_ACTUAL    = ("DC_POWER",           "sum"),
    DC_PREDICTED = ("DC_POWER_PREDICTED", "sum"),
    PERF_RATIO   = ("PERFORMANCE_RATIO",  "mean"),
).reset_index()

print("\n" + "="*65)
print(f"  PERFORMANCE MONITORING — {INPUT_FILE}")
print("="*65)
print(inv_perf[["SOURCE_KEY", "DC_ACTUAL", "DC_PREDICTED",
                "PERF_RATIO", "ABS_ERROR", "STATUS"]].to_string(index=False))

underperform = inv_perf[inv_perf["PERF_RATIO"] < THRESHOLD]
print(f"\n  Underperforming: {len(underperform)}/{len(inv_perf)} inverters")

# ==============================================================================
# 3. POWER FORECASTING
# ==============================================================================

dates       = sorted(df_day["DATE"].unique())
train_dates = dates[:TRAIN_DAYS]
test_dates  = dates[TRAIN_DAYS:]

df_test_fc = df_day[df_day["DATE"].isin(test_dates)]

print(f"\n  Forecast train : {len(train_dates)} days")
print(f"  Forecast test  : {len(test_dates)} days  {test_dates}")

hourly_fc = df_test_fc.groupby(["DATE", "HOUR"]).agg(
    ACTUAL    = ("DC_POWER",           "mean"),
    PREDICTED = ("DC_POWER_PREDICTED", "mean"),
).reset_index()
hourly_fc["DATETIME"] = (
    pd.to_datetime(hourly_fc["DATE"].astype(str))
    + pd.to_timedelta(hourly_fc["HOUR"], unit="h")
)

# ==============================================================================
# 4. FIGURE 1 — Performance Monitoring
# ==============================================================================

colors_inv = [RED if r < THRESHOLD else GREEN for r in inv_perf["PERF_RATIO"]]

fig1 = plt.figure(figsize=(20, 18))
fig1.suptitle(f"Performance Monitoring — {INPUT_FILE}  (Model: {MODEL_FILE})",
              fontsize=14, fontweight="bold", y=0.98)
gs1 = gridspec.GridSpec(3, 2, figure=fig1, hspace=0.5, wspace=0.35)

# 1a. Performance Ratio per Inverter
ax = fig1.add_subplot(gs1[0, :])
bars = ax.barh(range(len(inv_perf)), inv_perf["PERF_RATIO"],
               color=colors_inv, edgecolor="white", height=0.6)
ax.axvline(THRESHOLD, color=RED, linestyle="--",
           linewidth=2, label=f"Threshold = {THRESHOLD}")
ax.axvline(1.0, color=GRAY, linestyle=":", linewidth=1.5, label="Perfect = 1.0")
for bar, val, status in zip(bars, inv_perf["PERF_RATIO"], inv_perf["STATUS"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}  {status}", va="center", fontsize=7)
ax.set_yticks(range(len(inv_perf)))
ax.set_yticklabels([k[:8] for k in inv_perf["SOURCE_KEY"]], fontsize=7)
ax.set_title("Performance Ratio per Inverter  (Actual / Predicted)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Performance Ratio")
ax.legend(fontsize=9)

# 1b. Actual vs Predicted per Inverter
ax = fig1.add_subplot(gs1[1, 0])
x = range(len(inv_perf))
ax.bar([i - 0.2 for i in x], inv_perf["DC_ACTUAL"],
       width=0.4, color=BLUE, edgecolor="white", alpha=0.85, label="Actual")
ax.bar([i + 0.2 for i in x], inv_perf["DC_PREDICTED"],
       width=0.4, color=GRAY, edgecolor="white", alpha=0.85, label="Predicted")
ax.set_xticks(list(x))
ax.set_xticklabels([k[:6] for k in inv_perf["SOURCE_KEY"]],
                   rotation=45, fontsize=6)
ax.set_title("Avg DC Power: Actual vs Predicted per Inverter",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Avg DC Power (kW)")
ax.legend(fontsize=9)

# 1c. Daily Performance Ratio
ax = fig1.add_subplot(gs1[1, 1])
colors_day = [RED if r < THRESHOLD else GREEN for r in daily_perf["PERF_RATIO"]]
ax.bar(range(len(daily_perf)), daily_perf["PERF_RATIO"],
       color=colors_day, edgecolor="white", alpha=0.85)
ax.axhline(THRESHOLD, color=RED, linestyle="--",
           linewidth=2, label=f"Threshold = {THRESHOLD}")
ax.axhline(1.0, color=GRAY, linestyle=":", linewidth=1.5)
ax.set_xticks(range(0, len(daily_perf), 5))
ax.set_xticklabels([str(d) for d in daily_perf["DATE"].iloc[::5]],
                   rotation=30, fontsize=7)
ax.set_title("Daily Performance Ratio", fontsize=11, fontweight="bold")
ax.set_ylabel("Performance Ratio")
ax.legend(fontsize=9)

# 1d. Error Distribution per Inverter
ax = fig1.add_subplot(gs1[2, :])
inv_keys  = inv_perf["SOURCE_KEY"].tolist()
error_box = [df_day[df_day["SOURCE_KEY"] == k]["ERROR"].values for k in inv_keys]
bp = ax.boxplot(error_box, patch_artist=True, showfliers=False,
                medianprops={"color": "white", "linewidth": 2})
for patch, c in zip(bp["boxes"], colors_inv):
    patch.set_facecolor(c); patch.set_alpha(0.8)
ax.axhline(0, color=RED, linestyle="--", linewidth=1.5)
ax.set_xticks(range(1, len(inv_keys) + 1))
ax.set_xticklabels([k[:8] for k in inv_keys], rotation=45, fontsize=6)
ax.set_title("Prediction Error Distribution per Inverter",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Error (kW)")

fig1.savefig(os.path.join(OUTPUT_DIR, f"{PREFIX}_Performance.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\n  [Saved] {PREFIX}_Performance.png")

# ==============================================================================
# 5. FIGURE 2 — Power Forecasting
# ==============================================================================

fig2 = plt.figure(figsize=(20, 16))
fig2.suptitle(f"Power Forecasting — {INPUT_FILE}  "
              f"(Day {TRAIN_DAYS+1}–{len(dates)}  |  Model: {MODEL_FILE})",
              fontsize=13, fontweight="bold", y=0.98)
gs2 = gridspec.GridSpec(3, 2, figure=fig2, hspace=0.5, wspace=0.35)

# 2a. Actual vs Forecast time series
ax = fig2.add_subplot(gs2[0, :])
ax.plot(hourly_fc["DATETIME"], hourly_fc["ACTUAL"],
        color=BLUE, linewidth=1.5, label="Actual DC Power")
ax.plot(hourly_fc["DATETIME"], hourly_fc["PREDICTED"],
        color=RED, linewidth=1.5, linestyle="--", label="Forecasted DC Power")
ax.fill_between(hourly_fc["DATETIME"],
                hourly_fc["ACTUAL"], hourly_fc["PREDICTED"],
                alpha=0.15, color=RED, label="Error Gap")
for d in test_dates[1:]:
    ax.axvline(pd.to_datetime(str(d)), color=GRAY, linestyle=":", linewidth=1)
ax.set_title("Actual vs Forecasted DC Power (Hourly)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Date Time"); ax.set_ylabel("DC Power (kW)")
ax.legend(fontsize=9)
ax.tick_params(axis="x", rotation=30)

# 2b. Scatter per day
ax = fig2.add_subplot(gs2[1, 0])
for i, d in enumerate(test_dates):
    sub = hourly_fc[hourly_fc["DATE"] == d]
    ax.scatter(sub["ACTUAL"], sub["PREDICTED"],
               alpha=0.6, s=30, color=plt.cm.tab10(i), label=str(d))
max_v = max(hourly_fc["ACTUAL"].max(), hourly_fc["PREDICTED"].max())
ax.plot([0, max_v], [0, max_v], color=RED, linestyle="--", linewidth=1.5)
ax.set_title("Actual vs Predicted per Day", fontsize=11, fontweight="bold")
ax.set_xlabel("Actual DC Power (kW)")
ax.set_ylabel("Predicted DC Power (kW)")
ax.legend(fontsize=8)

# 2c. Forecast error per hour
ax = fig2.add_subplot(gs2[1, 1])
hourly_fc["FC_ERROR"] = hourly_fc["ACTUAL"] - hourly_fc["PREDICTED"]
hourly_err = hourly_fc.groupby("HOUR")["FC_ERROR"].mean()
ax.bar(hourly_err.index, hourly_err.values,
       color=[RED if v < 0 else GREEN for v in hourly_err.values],
       edgecolor="white", alpha=0.85)
ax.axhline(0, color=RED, linestyle="--", linewidth=1.5)
ax.set_title("Mean Forecast Error by Hour", fontsize=11, fontweight="bold")
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Error (kW)")
ax.set_xticks(range(24))

# 2d. Daily summary
ax = fig2.add_subplot(gs2[2, :])
daily_fc = hourly_fc.groupby("DATE").agg(
    ACTUAL    = ("ACTUAL",    "sum"),
    PREDICTED = ("PREDICTED", "sum"),
).reset_index()
daily_fc["MAPE"] = (
    (daily_fc["ACTUAL"] - daily_fc["PREDICTED"]).abs()
    / daily_fc["ACTUAL"].replace(0, np.nan) * 100
).round(2)

x_day = range(len(daily_fc))
ax.bar([i - 0.2 for i in x_day], daily_fc["ACTUAL"],
       width=0.4, color=BLUE, edgecolor="white", alpha=0.85, label="Actual")
ax.bar([i + 0.2 for i in x_day], daily_fc["PREDICTED"],
       width=0.4, color=RED,  edgecolor="white", alpha=0.85, label="Forecasted")
for i, (actual, pred, mape) in enumerate(zip(
        daily_fc["ACTUAL"], daily_fc["PREDICTED"], daily_fc["MAPE"])):
    ax.text(i, max(actual, pred) + 200,
            f"MAPE\n{mape:.1f}%", ha="center", fontsize=8, color=RED)
ax.set_xticks(list(x_day))
ax.set_xticklabels([str(d) for d in daily_fc["DATE"]], fontsize=9)
ax.set_title("Daily Total DC Power: Actual vs Forecasted",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Total DC Power (kW)")
ax.legend(fontsize=9)

fig2.savefig(os.path.join(OUTPUT_DIR, f"{PREFIX}_Forecast.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  [Saved] {PREFIX}_Forecast.png")

# ==============================================================================
# 6. EXPORT CSV
# ==============================================================================

inv_perf.to_csv(os.path.join(OUTPUT_DIR,   f"{PREFIX}_inverter_performance.csv"), index=False)
daily_perf.to_csv(os.path.join(OUTPUT_DIR, f"{PREFIX}_daily_performance.csv"),    index=False)
daily_fc.to_csv(os.path.join(OUTPUT_DIR,   f"{PREFIX}_forecast_summary.csv"),     index=False)

print(f"  [Saved] {PREFIX}_inverter_performance.csv")
print(f"  [Saved] {PREFIX}_daily_performance.csv")
print(f"  [Saved] {PREFIX}_forecast_summary.csv")
print("\n  Done.")
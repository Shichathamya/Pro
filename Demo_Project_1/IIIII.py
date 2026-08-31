"""
IIIII.py
=======================
Performance Monitoring & Power Forecasting
- Performance Ratio per Inverter (AC_actual / AC_predicted)
- Flag inverters below 85% threshold -> maintenance needed
- Forecast future AC Power
- Compare all models
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import timedelta
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# CONFIG
# ==============================================================================
INPUT_FILE     = "Plant_1_Joined.csv"
FEATURE_FILE   = "AC_Plant1_features.pkl"
PERF_THRESHOLD = 0.85
FORECAST_DAYS  = 7

MODEL_FILES = {
    "Linear Regression" : "Linear_Regression_AC_Plant1.pkl",
    "Polynomial deg=2"  : "Polynomial_deg2_AC_Plant1.pkl",
    "Polynomial deg=3"  : "Polynomial_deg3_AC_Plant1.pkl",
    "Decision Tree"     : "Decision_Tree_AC_Plant1.pkl",
    "Random Forest"     : "Random_Forest_AC_Plant1.pkl",
}
BEST_MODEL = "Random Forest"

PREFIX = os.path.splitext(INPUT_FILE)[0]
SOLAR  = "#F0A500"
BLUE   = "#4C9BE8"
RED    = "#E8564C"
GREEN  = "#5CB85C"
GRAY   = "#AAAAAA"
PURPLE = "#9B59B6"
ORANGE = "#E67E22"

MODEL_COLORS = {
    "Linear Regression" : GREEN,
    "Polynomial deg=2"  : PURPLE,
    "Polynomial deg=3"  : ORANGE,
    "Decision Tree"     : SOLAR,
    "Random Forest"     : BLUE,
}

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

PALETTE = [plt.cm.tab20(i) for i in range(20)]

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print(f"\n{'='*65}")
print(f"  Performance Monitoring + Forecast")
print(f"  Input : {INPUT_FILE}")
print(f"{'='*65}")

df = pd.read_csv(os.path.join(DATASET_DIR, INPUT_FILE))
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
df["HOUR"]      = df["DATE_TIME"].dt.hour
df["DATE"]      = df["DATE_TIME"].dt.date

FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, FEATURE_FILE))
print(f"  Features    : {FEATURE_COLS}")
print(f"  Rows        : {len(df):,}")

df_day      = df[df["IRRADIATION"] > 0].copy()
X           = df_day[FEATURE_COLS]
n_inverters = df_day["SOURCE_KEY"].nunique()
print(f"  Inverters   : {n_inverters}")

# ==============================================================================
# 2. LOAD ALL MODELS & PREDICT
# ==============================================================================
print(f"\n{'='*65}")
print(f"  MODEL COMPARISON")
print(f"{'='*65}")

models        = {}
predictions   = {}
model_metrics = []
tscv          = TimeSeriesSplit(n_splits=5)

for name, fname in MODEL_FILES.items():
    fpath = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [SKIP] {fname} not found")
        continue

    model  = joblib.load(fpath)
    y_pred = np.clip(model.predict(X), 0, None)
    y_true = df_day["AC_POWER"].values

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(
                y_true[y_true > 0], y_pred[y_true > 0]) * 100
    cv_r2 = cross_val_score(model, X, df_day["AC_POWER"],
                             cv=tscv, scoring="r2", n_jobs=-1).mean()

    models[name]      = model
    predictions[name] = y_pred
    model_metrics.append({
        "Model" : name,
        "MAE"   : round(mae,   2),
        "RMSE"  : round(rmse,  2),
        "R2"    : round(r2,    4),
        "CV_R2" : round(cv_r2, 4),
        "MAPE_%" : round(mape, 2),
    })
    print(f"  [{name:<20}]  R2={r2:.4f}  MAE={mae:.1f}  "
          f"RMSE={rmse:.1f}  MAPE={mape:.1f}%  CV_R2={cv_r2:.4f}")

metrics_df = (pd.DataFrame(model_metrics)
              .sort_values("R2", ascending=False)
              .reset_index(drop=True))

# ==============================================================================
# 3. FIGURE 1 — Model Comparison bar chart
# ==============================================================================
fig1, axes = plt.subplots(1, 4, figsize=(22, 6))
fig1.suptitle("Model Comparison — Performance Metrics",
              fontsize=14, fontweight="bold")

for ax, (col, title, higher) in zip(axes, [
        ("R2",     "R2 (higher = better)",    True),
        ("CV_R2",  "CV R2 TimeSeriesSplit",   True),
        ("MAE",    "MAE (lower = better)",    False),
        ("RMSE",   "RMSE (lower = better)",   False),
]):
    sdf    = metrics_df.sort_values(col, ascending=not higher)
    colors = [MODEL_COLORS.get(m, GRAY) for m in sdf["Model"]]
    bars   = ax.barh(sdf["Model"], sdf[col],
                     color=colors, edgecolor="white", height=0.5, alpha=0.85)
    for bar, val in zip(bars, sdf[col]):
        fmt = f"{val:.4f}" if col in ["R2","CV_R2"] else f"{val:.1f}"
        ax.text(val + abs(val)*0.01,
                bar.get_y() + bar.get_height()/2,
                fmt, va="center", fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_DIR, f"PM_{PREFIX}_ModelComparison.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\n  [Saved] PM_{PREFIX}_ModelComparison.png")

# ==============================================================================
# 4. FIGURE 2 — Actual vs Predicted scatter per model
# ==============================================================================
sample_idx = np.random.RandomState(42).choice(
                 len(df_day), min(2000, len(df_day)), replace=False)
y_true_arr = df_day["AC_POWER"].values
ncols      = len(models)

fig2, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5), sharey=True)
fig2.suptitle("Actual vs Predicted AC Power — All Models",
              fontsize=13, fontweight="bold")

ax_list = axes if ncols > 1 else [axes]
for ax, (name, y_pred) in zip(ax_list, predictions.items()):
    r2  = metrics_df[metrics_df["Model"]==name]["R2"].values[0]
    mae = metrics_df[metrics_df["Model"]==name]["MAE"].values[0]
    ax.scatter(y_true_arr[sample_idx], y_pred[sample_idx],
               color=MODEL_COLORS.get(name, GRAY),
               alpha=0.25, s=5, rasterized=True)
    max_v = max(y_true_arr.max(), y_pred.max())
    ax.plot([0, max_v], [0, max_v], color=RED, linewidth=1.5, linestyle="--")
    ax.set_title(f"{name}\nR2={r2:.4f}  MAE={mae:.1f}",
                 fontsize=8, fontweight="bold")
    ax.set_xlabel("Actual (kW)", fontsize=8)
    ax.set_ylabel("Predicted (kW)", fontsize=8)

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, f"PM_{PREFIX}_ActualVsPred.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  [Saved] PM_{PREFIX}_ActualVsPred.png")

# ==============================================================================
# 5. FIGURE 3 — Metrics Table PNG
# ==============================================================================
show_cols = ["Model","R2","CV_R2","MAE","RMSE","MAPE_%"]
fig_t, ax_t = plt.subplots(figsize=(14, 3.5))
ax_t.axis("off")
tbl = ax_t.table(cellText  = metrics_df[show_cols].values,
                 colLabels = show_cols,
                 cellLoc   = "center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.8)
for j in range(len(show_cols)):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(metrics_df)+1):
    bg = "#D5F5E3" if i==1 else "#F0F4F8" if i%2==0 else "white"
    for j in range(len(show_cols)):
        tbl[i, j].set_facecolor(bg)
ax_t.set_title("Model Comparison Table  (sorted by R2)",
               fontsize=11, fontweight="bold", pad=14)
plt.tight_layout()
fig_t.savefig(os.path.join(OUTPUT_DIR, f"PM_{PREFIX}_ModelTable.png"),
              dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_t)
print(f"  [Saved] PM_{PREFIX}_ModelTable.png")

# ==============================================================================
# 6. PERFORMANCE RATIO — Best Model
# ==============================================================================
print(f"\n{'='*65}")
print(f"  PERFORMANCE MONITORING  (model: {BEST_MODEL})")
print(f"{'='*65}")

df_day                 = df_day.copy()
df_day["AC_PREDICTED"] = np.clip(predictions[BEST_MODEL], 0, None)

perf_daily = (df_day.groupby(["DATE","SOURCE_KEY"])
              .agg(AC_ACTUAL    = ("AC_POWER",    "sum"),
                   AC_PREDICTED = ("AC_PREDICTED","sum"))
              .reset_index())

perf_daily["PERF_RATIO"] = (
    perf_daily["AC_ACTUAL"] /
    perf_daily["AC_PREDICTED"].clip(lower=1e-6)
).clip(upper=2.0)

inv_summary = (perf_daily.groupby("SOURCE_KEY")
               .agg(PERF_MEAN = ("PERF_RATIO","mean"),
                    PERF_MIN  = ("PERF_RATIO","min"),
                    PERF_STD  = ("PERF_RATIO","std"),
                    AC_ACTUAL = ("AC_ACTUAL", "sum"))
               .reset_index()
               .sort_values("PERF_MEAN"))

inv_summary["NEEDS_MAINTENANCE"] = inv_summary["PERF_MEAN"] < PERF_THRESHOLD
n_maintain = inv_summary["NEEDS_MAINTENANCE"].sum()
n_total    = len(inv_summary)

print(f"\n  Maintenance needed (<{PERF_THRESHOLD*100:.0f}%): {n_maintain}/{n_total}")
print(f"\n  {'SOURCE_KEY':<22} {'Perf%':>7} {'Min%':>7}  Status")
print(f"  {'-'*58}")
for _, row in inv_summary.sort_values("PERF_MEAN").iterrows():
    status = "!! MAINTENANCE !!" if row["NEEDS_MAINTENANCE"] else "OK"
    print(f"  {row['SOURCE_KEY']:<22} "
          f"{row['PERF_MEAN']*100:>6.1f}%  "
          f"{row['PERF_MIN']*100:>6.1f}%    {status}")

# ==============================================================================
# 7. FIGURE 4 — Performance Ratio per Inverter
# ==============================================================================
fig4, ax = plt.subplots(figsize=(16, 7))
fig4.suptitle(f"Performance Ratio per Inverter  "
              f"(model: {BEST_MODEL}, threshold={PERF_THRESHOLD*100:.0f}%)",
              fontsize=13, fontweight="bold")

colors = [RED if m else GREEN for m in inv_summary["NEEDS_MAINTENANCE"]]
bars   = ax.barh(inv_summary["SOURCE_KEY"],
                 inv_summary["PERF_MEAN"]*100,
                 color=colors, edgecolor="white", height=0.6, alpha=0.85)
ax.axvline(PERF_THRESHOLD*100, color=RED, linewidth=2, linestyle="--")
for bar, val in zip(bars, inv_summary["PERF_MEAN"]):
    ax.text(val*100+0.5, bar.get_y()+bar.get_height()/2,
            f"{val*100:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Performance Ratio (%)")
ax.set_xlim(0, 120)
ax.legend(handles=[Patch(facecolor=GREEN, label="OK"),
                   Patch(facecolor=RED,   label="Needs Maintenance")],
          fontsize=9)

plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_DIR, f"PM_{PREFIX}_Performance.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig4)
print(f"\n  [Saved] PM_{PREFIX}_Performance.png")

# ==============================================================================
# 8. FIGURE 5 — Performance Over Time
# ==============================================================================
inv_keys = sorted(inv_summary["SOURCE_KEY"].unique())

fig5, ax = plt.subplots(figsize=(20, 7))
fig5.suptitle("Performance Ratio Over Time — All Inverters",
              fontsize=13, fontweight="bold")

for i, key in enumerate(inv_keys):
    sub = perf_daily[perf_daily["SOURCE_KEY"]==key].sort_values("DATE")
    ax.plot(sub["DATE"], sub["PERF_RATIO"]*100,
            color=PALETTE[i%len(PALETTE)],
            linewidth=0.9, alpha=0.7, label=key[:10])

ax.axhline(PERF_THRESHOLD*100, color=RED, linewidth=2,
           linestyle="--", label=f"Threshold {PERF_THRESHOLD*100:.0f}%")
ax.set_ylabel("Performance Ratio (%)")
ax.set_xlabel("Date")
ax.tick_params(axis="x", rotation=30, labelsize=7)
ax.legend(fontsize=6, ncol=4, loc="lower right")

plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_DIR, f"PM_{PREFIX}_PerfOverTime.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig5)
print(f"  [Saved] PM_{PREFIX}_PerfOverTime.png")

# ==============================================================================
# 9. FORECAST
# ==============================================================================
print(f"\n{'='*65}")
print(f"  FORECAST  (next {FORECAST_DAYS} days)")
print(f"{'='*65}")

last_date = pd.Timestamp(df_day["DATE"].max())
ref_day   = last_date - timedelta(days=6)
ref_data  = df_day[df_day["DATE_TIME"].dt.date >= ref_day.date()].copy()
ref_data["TIME_SLOT"] = (ref_data["HOUR"]*60 +
                          ref_data["DATE_TIME"].dt.minute) // 15

ref_dates = sorted(ref_data["DATE"].unique())[-FORECAST_DAYS:]

forecast_rows = []
for day_offset in range(1, FORECAST_DAYS+1):
    future_date = last_date + timedelta(days=day_offset)
    ref_date    = ref_dates[(day_offset-1) % len(ref_dates)]
    pattern     = (ref_data[ref_data["DATE"]==ref_date]
                   .groupby("TIME_SLOT")[FEATURE_COLS]
                   .mean().reset_index())

    for _, row in pattern.iterrows():
        feat    = row[FEATURE_COLS].values.reshape(1, -1)
        ac_pred = max(0, models[BEST_MODEL].predict(feat)[0])
        slot    = int(row["TIME_SLOT"])
        forecast_rows.append({
            "DATE"       : future_date.date(),
            "TIME_SLOT"  : slot,
            "TIME_LABEL" : f"{(slot*15)//60:02d}:{(slot*15)%60:02d}",
            "AC_FORECAST": ac_pred * n_inverters,   # คูณจำนวน inverter
        })

forecast_df = pd.DataFrame(forecast_rows)

daily_fc = (forecast_df.groupby("DATE")["AC_FORECAST"]
            .sum().reset_index()
            .rename(columns={"AC_FORECAST":"TOTAL_AC_FORECAST"}))
daily_fc["TOTAL_AC_FORECAST_kWh"] = daily_fc["TOTAL_AC_FORECAST"] * 0.25

print(f"\n  {'Date':<14} {'Total kW-slots':>16}  {'kWh':>10}")
print(f"  {'-'*44}")
for _, row in daily_fc.iterrows():
    print(f"  {str(row['DATE']):<14} "
          f"{row['TOTAL_AC_FORECAST']:>16,.1f}  "
          f"{row['TOTAL_AC_FORECAST_kWh']:>10,.1f}")

# ==============================================================================
# 10. FIGURE 6 — Forecast Chart
# ==============================================================================
hist_daily = (df_day[df_day["DATE_TIME"].dt.date >=
                     (last_date - timedelta(days=30)).date()]
              .groupby("DATE")["AC_POWER"].sum().reset_index())
hist_daily["kWh"] = hist_daily["AC_POWER"] * 0.25

fig6, axes = plt.subplots(2, 1, figsize=(18, 12))
fig6.suptitle(f"AC Power Forecast — next {FORECAST_DAYS} days  "
              f"(model: {BEST_MODEL})",
              fontsize=13, fontweight="bold")

# Top: daily bar
ax = axes[0]
ax.bar(range(len(hist_daily)), hist_daily["kWh"],
       color=BLUE, alpha=0.7, label="Actual (last 30 days)")
ax.bar(range(len(hist_daily), len(hist_daily)+len(daily_fc)),
       daily_fc["TOTAL_AC_FORECAST_kWh"],
       color=SOLAR, alpha=0.85, label="Forecast")
ax.axvline(len(hist_daily)-0.5, color=RED,
           linewidth=2, linestyle="--", label="Today")

all_labels = list(hist_daily["DATE"]) + list(daily_fc["DATE"])
ax.set_xticks(range(len(all_labels)))
ax.set_xticklabels([str(d) for d in all_labels], rotation=45, fontsize=7)
ax.set_ylabel("Total AC Power (kWh)")
ax.set_title("Daily Total AC Power — Actual vs Forecast",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)

# Bottom: intraday — เรียงต่อเนื่องตามเวลา ไม่ซ้อนกัน
ax2       = axes[1]
fc_palette = [plt.cm.Set1(i) for i in range(FORECAST_DAYS)]

x_offset  = 0
x_ticks   = []
x_labels  = []

for i, (date, grp) in enumerate(forecast_df.groupby("DATE")):
    grp_sorted = grp.sort_values("TIME_SLOT").reset_index(drop=True)
    n_slots    = len(grp_sorted)
    x_vals     = range(x_offset, x_offset + n_slots)

    ax2.plot(x_vals, grp_sorted["AC_FORECAST"],
             color=fc_palette[i % len(fc_palette)],
             linewidth=2, alpha=0.85, label=str(date))

    # เพิ่ม tick ที่ต้นวัน
    x_ticks.append(x_offset)
    x_labels.append(str(date))

    # เส้นแบ่งวัน
    ax2.axvline(x_offset, color=GRAY, linewidth=0.8,
                linestyle="--", alpha=0.5)

    x_offset += n_slots

ax2.set_xticks(x_ticks)
ax2.set_xticklabels(x_labels, rotation=45, fontsize=8)
ax2.set_xlabel("Date")
ax2.set_ylabel("Predicted AC Power (kW)")
ax2.set_title("Intraday AC Power Forecast per Day (sequential)",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=8, ncol=4, loc="upper right")

plt.tight_layout()
fig6.savefig(os.path.join(OUTPUT_DIR, f"PM_{PREFIX}_Forecast.png"),
             dpi=150, bbox_inches="tight")
plt.close(fig6)
print(f"\n  [Saved] PM_{PREFIX}_Forecast.png")

# ==============================================================================
# 11. EXPORT CSV
# ==============================================================================
metrics_df.to_csv(os.path.join(OUTPUT_DIR,
                               f"PM_{PREFIX}_model_comparison.csv"),
                  index=False)
print(f"  [Saved] PM_{PREFIX}_model_comparison.csv")

inv_summary["PERF_MEAN_%"] = (inv_summary["PERF_MEAN"]*100).round(2)
inv_summary["PERF_MIN_%"]  = (inv_summary["PERF_MIN"] *100).round(2)
inv_summary.to_csv(os.path.join(OUTPUT_DIR,
                                f"PM_{PREFIX}_inverter_summary.csv"),
                   index=False)
print(f"  [Saved] PM_{PREFIX}_inverter_summary.csv")

daily_fc.to_csv(os.path.join(OUTPUT_DIR,
                              f"PM_{PREFIX}_forecast.csv"), index=False)
print(f"  [Saved] PM_{PREFIX}_forecast.csv")

print(f"\n{'='*65}")
print(f"  Done.")
print(f"  Best model         : {metrics_df.iloc[0]['Model']}  "
      f"(R2={metrics_df.iloc[0]['R2']})")
print(f"  Maintenance needed : {n_maintain}/{n_total} inverters")
print(f"  Forecast           : {FORECAST_DAYS} days ahead")
print(f"{'='*65}")
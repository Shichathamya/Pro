"""
ML_Regression_Plant1.py
========================
Regression ML for Plant 1 Solar Power Generation
Target  : DC_POWER
Model   : Random Forest
"""

import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

SOLAR = "#F0A500"
BLUE  = "#4C9BE8"
RED   = "#E8564C"

# ==============================================================================
# 1. LOAD & PREPARE
# ==============================================================================

df = pd.read_csv(os.path.join(DATASET_DIR, "Plant_1_Joined.csv"))
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

df["HOUR"]   = df["DATE_TIME"].dt.hour
df["MINUTE"] = df["DATE_TIME"].dt.minute
df["DAY"]    = df["DATE_TIME"].dt.day
df["MONTH"]  = df["DATE_TIME"].dt.month

df = df[df["IRRADIATION"] > 0].copy()
df = df.dropna(subset=["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"])

FEATURE_COLS = [
    "IRRADIATION", "MODULE_TEMPERATURE", "AMBIENT_TEMPERATURE",
    "HOUR", "MINUTE", "DAY", "MONTH",
]
TARGET = "DC_POWER"

X = df[FEATURE_COLS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Train : {X_train.shape}")
print(f"  Test  : {X_test.shape}")

# ==============================================================================
# 2. TRAIN — Random Forest
# ==============================================================================

print("\n" + "="*60)
print("  TRAINING — Random Forest")
print("="*60)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
cv_r2 = cross_val_score(model, X_train, y_train,
                        cv=5, scoring="r2", n_jobs=-1).mean()

cv_mae_scores = cross_val_score(model, X_train, y_train,
                                cv=5, scoring="neg_mean_absolute_error",
                                n_jobs=-1)
eve     = (-cv_mae_scores).mean()
eve_std = (-cv_mae_scores).std()

print(f"  R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}  "
      f"CV R²={cv_r2:.4f}  EVE={eve:.2f} ±{eve_std:.2f}")

results_df = pd.DataFrame([{
    "Model"         : "Random Forest",
    "MAE"           : round(mae,     4),
    "RMSE"          : round(rmse,    4),
    "R²"            : round(r2,      4),
    "CV R² (5-fold)": round(cv_r2,  4),
    "EVE (CV MAE)"  : round(eve,     4),
    "EVE Std"       : round(eve_std, 4),
}])

# ==============================================================================
# 3. SAVE MODEL
# ==============================================================================

model_path = os.path.join(MODEL_DIR, "RF_DC_POWER_Plant1.pkl")
joblib.dump(model, model_path)
print(f"\n  [Saved Model] → {model_path}")

# บันทึก feature columns เพื่อใช้ตอน load
feature_path = os.path.join(MODEL_DIR, "RF_DC_POWER_Plant1_features.pkl")
joblib.dump(FEATURE_COLS, feature_path)
print(f"  [Saved Features] → {feature_path}")

# ==============================================================================
# 4. PRINT & EXPORT RESULTS
# ==============================================================================

print("\n\n" + "="*60)
print("  REGRESSION RESULTS")
print("="*60)
print(results_df.to_string(index=False))

results_df.to_csv(os.path.join(OUTPUT_DIR, "ML_P1_regression_results.csv"), index=False)
print(f"\n  [Saved] ML_P1_regression_results.csv")

# ==============================================================================
# 5. VISUALIZATIONS
# ==============================================================================

fig = plt.figure(figsize=(18, 16))
fig.suptitle("Plant 1 — DC Power Regression (Random Forest)",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

sample_idx = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
y_test_arr = y_test.values
residuals  = y_test_arr - y_pred

# 5a. Actual vs Predicted
ax = fig.add_subplot(gs[0, :])
ax.scatter(y_test_arr[sample_idx], y_pred[sample_idx],
           alpha=0.3, s=6, color=BLUE, rasterized=True)
max_val = max(y_test_arr.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], color=RED, linewidth=1.5,
        linestyle="--", label="Perfect")
ax.set_title(f"Actual vs Predicted  (R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f})",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Actual DC Power (kW)")
ax.set_ylabel("Predicted DC Power (kW)")
ax.legend(fontsize=9)

# 5b. Residual Plot
ax = fig.add_subplot(gs[1, 0])
ax.scatter(y_pred[sample_idx], residuals[sample_idx],
           alpha=0.3, s=6, color=SOLAR, rasterized=True)
ax.axhline(0, color=RED, linewidth=1.5, linestyle="--")
ax.set_title("Residuals", fontsize=12, fontweight="bold")
ax.set_xlabel("Predicted DC Power (kW)")
ax.set_ylabel("Residual (Actual − Predicted)")

# 5c. Residual Distribution
ax = fig.add_subplot(gs[1, 1])
ax.hist(residuals, bins=50, color=BLUE, edgecolor="white", alpha=0.8)
ax.axvline(0, color=RED, linewidth=1.5, linestyle="--")
ax.set_title("Residual Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Residual"); ax.set_ylabel("Count")

# 5d. Feature Importance
ax = fig.add_subplot(gs[2, :])
importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)
importances = importances.sort_values(ascending=True)
bars = ax.barh(importances.index, importances.values,
               color=BLUE, edgecolor="white", height=0.6)
for bar, val in zip(bars, importances.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_title("Feature Importance", fontsize=12, fontweight="bold")
ax.set_xlabel("Importance Score")

fig.savefig(os.path.join(OUTPUT_DIR, "ML_P1_regression.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  [Saved] ML_P1_regression.png")

# ── Table PNG ─────────────────────────────────────────────────────────────────
fig_t, ax_t = plt.subplots(figsize=(12, 2.5))
ax_t.axis("off")
tbl = ax_t.table(
    cellText  = results_df.values,
    colLabels = results_df.columns,
    cellLoc   = "center",
    loc       = "center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
for j in range(len(results_df.columns)):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for j in range(len(results_df.columns)):
    tbl[1, j].set_facecolor("#D5F5E3")
ax_t.set_title("Plant 1 — DC Power Regression Results (Random Forest)",
               fontsize=12, fontweight="bold", pad=14)
plt.tight_layout()
fig_t.savefig(os.path.join(OUTPUT_DIR, "ML_P1_regression_table.png"),
              dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_t)
print(f"  [Saved] ML_P1_regression_table.png")

print("\n  Done.")
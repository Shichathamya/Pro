"""
ML_Regression_Plant1.py
========================
Regression ML for Plant 1 Solar Power Generation
Target  : AC_POWER
Models  : Linear Regression, Decision Tree, Random Forest
Features: IRRADIATION, MODULE_TEMPERATURE, AMBIENT_TEMPERATURE, HOUR
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
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
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
GREEN = "#5CB85C"

# ==============================================================================
# 1. LOAD & PREPARE
# ==============================================================================

df = pd.read_csv(os.path.join(DATASET_DIR, "Plant_1_Joined.csv"))
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
df["HOUR"]      = df["DATE_TIME"].dt.hour

df = df[df["IRRADIATION"] > 0].copy()
df = df.dropna(subset=["IRRADIATION", "MODULE_TEMPERATURE",
                        "AMBIENT_TEMPERATURE", "AC_POWER"])

FEATURE_COLS = [
    "IRRADIATION",
    "MODULE_TEMPERATURE",
    "AMBIENT_TEMPERATURE",
    "HOUR",
]
TARGET = "AC_POWER"

X = df[FEATURE_COLS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"  Train : {X_train.shape}")
print(f"  Test  : {X_test.shape}")

# ==============================================================================
# 2. TRAIN — Multiple Models
# ==============================================================================

print("\n" + "="*60)
print("  TRAINING — Multiple Models  (target: AC_POWER)")
print("="*60)

models = {
    "Linear Regression" : LinearRegression(),
    "Decision Tree"     : DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest"     : RandomForestRegressor(n_estimators=100,
                                                random_state=42, n_jobs=-1),
}

results = []
preds   = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    preds[name] = y_pred

    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
    r2    = r2_score(y_test, y_pred)
    cv_r2 = cross_val_score(model, X_train, y_train,
                             cv=5, scoring="r2", n_jobs=-1).mean()
    cv_mae  = cross_val_score(model, X_train, y_train,
                               cv=5,
                               scoring="neg_mean_absolute_error",
                               n_jobs=-1)
    eve     = (-cv_mae).mean()
    eve_std = (-cv_mae).std()

    results.append({
        "Model"          : name,
        "MAE"            : round(mae,     4),
        "RMSE"           : round(rmse,    4),
        "R²"             : round(r2,      4),
        "CV R² (5-fold)" : round(cv_r2,  4),
        "EVE (CV MAE)"   : round(eve,     4),
        "EVE Std"        : round(eve_std, 4),
    })
    print(f"  [{name:<20}]  R²={r2:.4f}  MAE={mae:.2f}  "
          f"RMSE={rmse:.2f}  EVE={eve:.2f} ±{eve_std:.2f}")

results_df = (pd.DataFrame(results)
              .sort_values("R²", ascending=False)
              .reset_index(drop=True))

# ==============================================================================
# 3. SAVE MODEL — all model
# ==============================================================================

for name, model in models.items():
    filename = name.replace(" ", "_")
    model_path = os.path.join(MODEL_DIR, f"{filename}_AC_Plant1.pkl")
    joblib.dump(model, model_path)
    print(f"  [Saved] {filename}_AC_Plant1.pkl")

joblib.dump(FEATURE_COLS,
            os.path.join(MODEL_DIR, "AC_Plant1_features.pkl"))
print(f"  [Saved] AC_Plant1_features.pkl")

# ==============================================================================
# 4. EXPORT CSV
# ==============================================================================

results_df.to_csv(os.path.join(OUTPUT_DIR, "ML_P1_AC_results.csv"), index=False)
print(f"  [Saved] ML_P1_AC_results.csv")

# ==============================================================================
# 5. VISUALIZATIONS
# ==============================================================================

COLORS = {
    "Linear Regression" : GREEN,
    "Decision Tree"     : SOLAR,
    "Random Forest"     : BLUE,
}

sample_idx = np.random.choice(len(y_test), min(2000, len(y_test)), replace=False)
y_test_arr = y_test.values

fig = plt.figure(figsize=(20, 20))
fig.suptitle("Plant 1 — AC Power Regression Comparison",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

# 5a–5c. Actual vs Predicted per model
for col_idx, (name, y_pred) in enumerate(preds.items()):
    ax  = fig.add_subplot(gs[0, col_idx])
    r2  = results_df[results_df["Model"] == name]["R²"].values[0]
    mae = results_df[results_df["Model"] == name]["MAE"].values[0]
    ax.scatter(y_test_arr[sample_idx], y_pred[sample_idx],
               alpha=0.3, s=6, color=COLORS[name], rasterized=True)
    max_v = max(y_test_arr.max(), y_pred.max())
    ax.plot([0, max_v], [0, max_v], color=RED,
            linewidth=1.5, linestyle="--")
    ax.set_title(f"{name}\nR²={r2:.4f}  MAE={mae:.2f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Actual AC Power (kW)", fontsize=8)
    ax.set_ylabel("Predicted AC Power (kW)", fontsize=8)

# 5d–5f. Residuals per model
for col_idx, (name, y_pred) in enumerate(preds.items()):
    ax        = fig.add_subplot(gs[1, col_idx])
    residuals = y_test_arr - y_pred
    ax.scatter(y_pred[sample_idx], residuals[sample_idx],
               alpha=0.3, s=6, color=COLORS[name], rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.5, linestyle="--")
    ax.set_title(f"{name} — Residuals", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted (kW)", fontsize=8)
    ax.set_ylabel("Residual", fontsize=8)

# 5g. Feature Importance — Decision Tree
ax = fig.add_subplot(gs[2, 0])
imp_dt = pd.Series(models["Decision Tree"].feature_importances_,
                   index=FEATURE_COLS).sort_values(ascending=True)
bars = ax.barh(imp_dt.index, imp_dt.values,
               color=SOLAR, edgecolor="white", height=0.5)
for bar, val in zip(bars, imp_dt.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_title("Decision Tree — Feature Importance",
             fontsize=10, fontweight="bold")
ax.set_xlabel("Importance Score")

# 5h. Feature Importance — Random Forest
ax = fig.add_subplot(gs[2, 1])
imp_rf = pd.Series(models["Random Forest"].feature_importances_,
                   index=FEATURE_COLS).sort_values(ascending=True)
bars = ax.barh(imp_rf.index, imp_rf.values,
               color=BLUE, edgecolor="white", height=0.5)
for bar, val in zip(bars, imp_rf.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_title("Random Forest — Feature Importance",
             fontsize=10, fontweight="bold")
ax.set_xlabel("Importance Score")

# 5i. R² Comparison bar
ax = fig.add_subplot(gs[2, 2])
bar_colors = [COLORS[m] for m in results_df["Model"]]
bars = ax.bar(results_df["Model"], results_df["R²"],
              color=bar_colors, edgecolor="white", alpha=0.85)
for bar, val in zip(bars, results_df["R²"]):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.005, f"{val:.4f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title("R² Comparison", fontsize=10, fontweight="bold")
ax.set_ylabel("R²")
ax.set_ylim(0, 1.1)
ax.tick_params(axis="x", rotation=15, labelsize=8)

fig.savefig(os.path.join(OUTPUT_DIR, "ML_P1_AC_regression.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  [Saved] ML_P1_AC_regression.png")

# ── Table PNG ─────────────────────────────────────────────────────────────────
show_cols = ["Model","MAE","RMSE","R²","CV R² (5-fold)","EVE (CV MAE)","EVE Std"]
fig_t, ax_t = plt.subplots(figsize=(16, 4))
ax_t.axis("off")
tbl = ax_t.table(
    cellText  = results_df[show_cols].values,
    colLabels = show_cols,
    cellLoc   = "center",
    loc       = "center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.8)
for j in range(len(show_cols)):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(results_df) + 1):
    for j in range(len(show_cols)):
        tbl[i, j].set_facecolor(
            "#D5F5E3" if i == 1 else
            "#F0F4F8" if i % 2 == 0 else "white"
        )
ax_t.set_title("Plant 1 — AC Power Regression Results",
               fontsize=12, fontweight="bold", pad=14)
plt.tight_layout()
fig_t.savefig(os.path.join(OUTPUT_DIR, "ML_P1_AC_table.png"),
              dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig_t)
print(f"  [Saved] ML_P1_AC_table.png")

print("\n  Done.")
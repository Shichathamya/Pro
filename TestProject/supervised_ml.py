"""
supervised_ml.py
================
Supervised ML for Power Plant Predictive Maintenance
Tasks:
  - Classification : failure (0/1)
  - Regression     : RUL_days
Models:
  - Decision Tree
  - Logistic Regression / Linear Regression
  - Numeric Prediction  (SVR / SVC)
  - Ensemble            (Random Forest, Gradient Boosting, XGBoost)
  - Neural Network      (MLP)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection     import train_test_split, cross_val_score
from sklearn.preprocessing       import StandardScaler
from sklearn.tree                import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model        import LogisticRegression, LinearRegression, Ridge
from sklearn.svm                 import SVC, SVR
from sklearn.ensemble            import (RandomForestClassifier, RandomForestRegressor,
                                         GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.neural_network      import MLPClassifier, MLPRegressor
from sklearn.metrics             import (accuracy_score, f1_score, roc_auc_score,
                                         precision_score, recall_score,
                                         mean_absolute_error, mean_squared_error, r2_score)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [INFO] xgboost not installed — skipping XGBoost models")

# ==============================================================================
# 1. LOAD & PREPARE DATA
# ==============================================================================

# df = pd.read_csv("output/powerplant_maintenance.csv")

# FEATURE_COLS = [
#     "operating_hours", "load_pct", "days_since_maintenance", "maintenance_count",
#     "temperature_C", "pressure_bar", "vibration_mms", "rotation_speed_rpm",
#     "voltage_V", "current_A", "oil_temp_C", "power_output_MW", "efficiency_pct",
# ]

# X = df[FEATURE_COLS]
# y_cls = df["failure"]        # Classification target
# y_reg = df["RUL_days"]       # Regression target

# # Train/Test split
# X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
#     X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
# )

# # Scale (needed for LR, SVM, MLP)
# scaler  = StandardScaler()
# Xs_train = scaler.fit_transform(X_train)
# Xs_test  = scaler.transform(X_test)

train_df = pd.read_csv("DataSet/supervised_dataset.csv")
test_df  = pd.read_csv("DataSet/supervised_dataset_test.csv")

FEATURE_COLS = [
    "operating_hours", "load_pct", "days_since_maintenance", "maintenance_count",
    "temperature_C", "pressure_bar", "vibration_mms", "rotation_speed_rpm",
    "voltage_V", "current_A", "oil_temp_C", "power_output_MW", "efficiency_pct",
]

X_train  = train_df[FEATURE_COLS]
X_test   = test_df[FEATURE_COLS]
yc_train = train_df["failure"]
yc_test  = test_df["failure"]
yr_train = train_df["RUL_days"]
yr_test  = test_df["RUL_days"]

# Scale
scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

# ==============================================================================
# 2. MODEL DEFINITIONS
# ==============================================================================

clf_models = {
    "Decision Tree":          DecisionTreeClassifier(max_depth=8, random_state=42),
    "Logistic Regression":    LogisticRegression(max_iter=500, random_state=42),
    "SVC (Numeric Pred.)":    SVC(probability=True, random_state=42),
    "Random Forest":          RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":      GradientBoostingClassifier(n_estimators=100, random_state=42),
    "MLP Neural Network":     MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                                            random_state=42),
}
if HAS_XGB:
    clf_models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42,
                                           eval_metric="logloss", verbosity=0)

reg_models = {
    "Decision Tree":          DecisionTreeRegressor(max_depth=8, random_state=42),
    "Linear Regression":      LinearRegression(),
    "Ridge Regression":       Ridge(alpha=1.0),
    "SVR (Numeric Pred.)":    SVR(kernel="rbf"),
    "Random Forest":          RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":      GradientBoostingRegressor(n_estimators=100, random_state=42),
    "MLP Neural Network":     MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300,
                                           random_state=42),
}
if HAS_XGB:
    reg_models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

# ==============================================================================
# 3. TRAIN & EVALUATE
# ==============================================================================

NEEDS_SCALE = {"Logistic Regression", "SVC (Numeric Pred.)", "MLP Neural Network",
               "Linear Regression", "Ridge Regression", "SVR (Numeric Pred.)"}

def run_classification(models):
    rows = []
    for name, model in models.items():
        Xtr = Xs_train if name in NEEDS_SCALE else X_train
        Xte = Xs_test  if name in NEEDS_SCALE else X_test

        model.fit(Xtr, yc_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else y_pred

        cv_auc = cross_val_score(model, Xtr, yc_train, cv=5,
                                 scoring="roc_auc", n_jobs=-1).mean()
        rows.append({
            "Model"         : name,
            "Accuracy"      : accuracy_score(yc_test, y_pred),
            "Precision"     : precision_score(yc_test, y_pred, zero_division=0),
            "Recall"        : recall_score(yc_test, y_pred, zero_division=0),
            "F1-Score"      : f1_score(yc_test, y_pred, zero_division=0),
            "ROC-AUC"       : roc_auc_score(yc_test, y_prob),
            "CV AUC (5-fold)": cv_auc,
        })
        print(f"  [CLF] {name:<25} AUC={rows[-1]['ROC-AUC']:.4f}  F1={rows[-1]['F1-Score']:.4f}")
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def run_regression(models):
    rows = []
    for name, model in models.items():
        Xtr = Xs_train if name in NEEDS_SCALE else X_train
        Xte = Xs_test  if name in NEEDS_SCALE else X_test

        model.fit(Xtr, yr_train)
        y_pred = model.predict(Xte)

        cv_r2 = cross_val_score(model, Xtr, yr_train, cv=5,
                                scoring="r2", n_jobs=-1).mean()
        mae  = mean_absolute_error(yr_test, y_pred)
        rmse = np.sqrt(mean_squared_error(yr_test, y_pred))
        r2   = r2_score(yr_test, y_pred)

        rows.append({
            "Model"         : name,
            "MAE"           : mae,
            "RMSE"          : rmse,
            "R²"            : r2,
            "CV R² (5-fold)": cv_r2,
        })
        print(f"  [REG] {name:<25} R²={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    return pd.DataFrame(rows).sort_values("R²", ascending=False).reset_index(drop=True)


# ==============================================================================
# 4. RUN
# ==============================================================================

print("\n" + "="*65)
print("  CLASSIFICATION  (target: failure)")
print("="*65)
clf_results = run_classification(clf_models)

print("\n" + "="*65)
print("  REGRESSION  (target: RUL_days)")
print("="*65)
reg_results = run_regression(reg_models)

# ==============================================================================
# 5. PRINT COMPARISON TABLES
# ==============================================================================

def fmt_table(df, pct_cols=None, round_cols=None):
    df = df.copy()
    if pct_cols:
        for c in pct_cols:
            df[c] = df[c].map(lambda x: f"{x:.4f}")
    if round_cols:
        for c in round_cols:
            df[c] = df[c].map(lambda x: f"{x:.2f}")
    return df.to_string(index=True)

print("\n\n" + "="*65)
print("  CLASSIFICATION RESULTS  (sorted by ROC-AUC ↓)")
print("="*65)
print(fmt_table(clf_results,
                pct_cols=["Accuracy","Precision","Recall","F1-Score","ROC-AUC","CV AUC (5-fold)"]))

print("\n\n" + "="*65)
print("  REGRESSION RESULTS  (sorted by R² ↓)")
print("="*65)
print(fmt_table(reg_results, round_cols=["MAE","RMSE"],
                pct_cols=["R²","CV R² (5-fold)"]))

# ==============================================================================
# 6. EXPORT TO CSV
# ==============================================================================

clf_results.to_csv("output/clf_results.csv", index=False)
reg_results.to_csv("output/reg_results.csv", index=False)

print("  [Saved] output/clf_results.csv")
print("  [Saved] output/reg_results.csv")
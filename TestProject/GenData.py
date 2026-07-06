"""
generate_dataset.py
===================
Generates a Power Plant Predictive Maintenance dataset (~10,000 rows)
suitable for:
  - Supervised   : Classification (failure) + Regression (RUL)
  - Unsupervised : Clustering / Anomaly Detection
"""

import os
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "DataSet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 67
N    = 10_000
rng  = np.random.default_rng(SEED)

# ==============================================================================
# 1. BASE OPERATING CONDITIONS
# ==============================================================================

operating_hours = rng.uniform(0, 8_760, N)          # 0–1 year in hours
load            = rng.uniform(40, 100, N)            # % load
age_factor      = operating_hours / 8_760            # 0–1 normalised wear

# ==============================================================================
# 2. SENSOR FEATURES  (degradation increases with age_factor)
# ==============================================================================

temperature    = (rng.normal(520, 15, N)
                  + age_factor * 40                  # rises as machine ages
                  + (load - 70) * 0.3)               # load effect

pressure       = (rng.normal(160, 8, N)
                  + age_factor * 10
                  + (load - 70) * 0.15)

vibration      = (rng.exponential(1.5, N)
                  + age_factor * 4                   # key degradation signal
                  + rng.normal(0, 0.3, N))
vibration      = np.clip(vibration, 0.1, 20)

rotation_speed = (rng.normal(3_000, 50, N)
                  - age_factor * 80
                  + rng.normal(0, 20, N))

voltage        = (rng.normal(11_000, 200, N)
                  - age_factor * 300
                  + rng.normal(0, 80, N))

current        = (rng.normal(500, 30, N)
                  + age_factor * 60
                  + (load - 70) * 2)

oil_temp       = (rng.normal(65, 5, N)
                  + age_factor * 15
                  + rng.normal(0, 2, N))

power_output   = (load * rng.normal(1.8, 0.05, N)   # ~MW per % load
                  - age_factor * 20
                  + rng.normal(0, 5, N))

efficiency     = (rng.normal(92, 2, N)
                  - age_factor * 8
                  - (vibration * 0.3)
                  + rng.normal(0, 0.5, N))
efficiency     = np.clip(efficiency, 60, 99)

# ==============================================================================
# 3. MAINTENANCE HISTORY
# ==============================================================================

days_since_maintenance = rng.uniform(0, 365, N)
maintenance_count      = (operating_hours / 2_000).astype(int)

# ==============================================================================
# 4. ANOMALY INJECTION  (~5% of rows = pre-failure events)
# ==============================================================================

anomaly_idx = rng.choice(N, size=int(N * 0.05), replace=False)
temperature   [anomaly_idx] += rng.uniform(30, 80,  len(anomaly_idx))
vibration     [anomaly_idx] += rng.uniform(5,  15,  len(anomaly_idx))
oil_temp      [anomaly_idx] += rng.uniform(10, 30,  len(anomaly_idx))
pressure      [anomaly_idx] -= rng.uniform(10, 30,  len(anomaly_idx))
efficiency    [anomaly_idx] -= rng.uniform(5,  20,  len(anomaly_idx))
efficiency     = np.clip(efficiency, 60, 99)

# ==============================================================================
# 5. TARGET VARIABLES
# ==============================================================================

# ── Risk score (0–1) used to derive both targets ─────────────────────────────
risk_score = (
    0.30 * age_factor
    + 0.25 * (vibration      / vibration.max())
    + 0.20 * ((temperature   - 520) / 100).clip(0, 1)
    + 0.15 * ((oil_temp      - 65)  / 40 ).clip(0, 1)
    + 0.10 * (days_since_maintenance / 365)
)
risk_score = np.clip(risk_score + rng.normal(0, 0.03, N), 0, 1)

# mark anomaly rows as high risk
risk_score[anomaly_idx] = np.clip(risk_score[anomaly_idx] + 0.35, 0, 1)

# ── Classification target : failure (0 / 1) ───────────────────────────────────
failure_prob = 1 / (1 + np.exp(-10 * (risk_score - 0.65)))  # sigmoid
failure      = (rng.uniform(0, 1, N) < failure_prob).astype(int)

# ── Regression target : RUL in days ──────────────────────────────────────────
RUL = np.round(
    (1 - risk_score) * 365
    + rng.normal(0, 10, N)
).clip(0, 365).astype(int)

RUL[failure == 1] = rng.integers(0, 15, size=failure.sum())  # near-failure → low RUL

# ==============================================================================
# 6. ASSEMBLE DATAFRAME
# ==============================================================================

df = pd.DataFrame({
    # identifiers
    "machine_id"            : [f"GEN_{i:04d}" for i in rng.integers(1, 21, N)],
    "timestamp"             : pd.date_range("2020-01-01", periods=N, freq="h"),

    # operational
    "operating_hours"       : operating_hours.round(1),
    "load_pct"              : load.round(2),
    "days_since_maintenance": days_since_maintenance.round(1),
    "maintenance_count"     : maintenance_count,

    # sensors
    "temperature_C"         : temperature.round(2),
    "pressure_bar"          : pressure.round(2),
    "vibration_mms"         : vibration.round(3),
    "rotation_speed_rpm"    : rotation_speed.round(1),
    "voltage_V"             : voltage.round(1),
    "current_A"             : current.round(2),
    "oil_temp_C"            : oil_temp.round(2),
    "power_output_MW"       : power_output.round(2),
    "efficiency_pct"        : efficiency.round(2),

    # derived / targets
    "risk_score"            : risk_score.round(4),
    "RUL_days"              : RUL,                   # Regression target
    "failure"               : failure,               # Classification target
})

# ==============================================================================
# 7. EXPORT
# ==============================================================================

# Full dataset
full_path = os.path.join(OUTPUT_DIR, "powerplant_maintenance.csv")
df.to_csv(full_path, index=False)
print(f"  [Full]           → {full_path}  {df.shape}")

# Supervised  : drop risk_score (leakage), keep RUL + failure as targets
supervised = df.drop(columns=["risk_score"])
sup_path   = os.path.join(OUTPUT_DIR, "supervised_dataset.csv")
supervised.to_csv(sup_path, index=False)
print(f"  [Supervised]     → {sup_path}  {supervised.shape}")

# Unsupervised : drop targets, keep sensors + operational only
unsupervised = df.drop(columns=["risk_score", "RUL_days", "failure"])
uns_path     = os.path.join(OUTPUT_DIR, "unsupervised_dataset.csv")
unsupervised.to_csv(uns_path, index=False)
print(f"  [Unsupervised]   → {uns_path}  {unsupervised.shape}")

# Summary
print("\n── Class balance ──────────────────────────")
print(df["failure"].value_counts().to_string())
print(f"\n── RUL stats ──────────────────────────────")
print(df["RUL_days"].describe().round(1).to_string())
print(f"\n── Sensor stats ───────────────────────────")
print(df[["temperature_C","vibration_mms","efficiency_pct"]].describe().round(2).to_string())
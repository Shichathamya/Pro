"""
generate_dataset.py
===================
Generates Power Plant Predictive Maintenance datasets
  Train : 10,000 rows  (SEED=67)
  Test  :  1,000 rows  (SEED=99)
"""

import os
import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "DataSet")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# GENERATOR FUNCTION
# ==============================================================================

def generate(N, SEED):

    rng = np.random.default_rng(SEED)

    # ── Base ──────────────────────────────────────────────────────────────────
    operating_hours        = rng.uniform(0, 8_760, N)
    load                   = rng.uniform(40, 100, N)
    age_factor             = operating_hours / 8_760

    # ── Sensors ───────────────────────────────────────────────────────────────
    temperature    = rng.normal(520, 15, N) + age_factor * 40 + (load - 70) * 0.3
    pressure       = rng.normal(160,  8, N) + age_factor * 10 + (load - 70) * 0.15
    vibration      = np.clip(rng.exponential(1.5, N) + age_factor * 4
                             + rng.normal(0, 0.3, N), 0.1, 20)
    rotation_speed = rng.normal(3_000, 50, N) - age_factor * 80 + rng.normal(0, 20, N)
    voltage        = rng.normal(11_000, 200, N) - age_factor * 300 + rng.normal(0, 80, N)
    current        = rng.normal(500, 30, N) + age_factor * 60 + (load - 70) * 2
    oil_temp       = rng.normal(65, 5, N) + age_factor * 15 + rng.normal(0, 2, N)
    power_output   = load * rng.normal(1.8, 0.05, N) - age_factor * 20 + rng.normal(0, 5, N)
    efficiency     = np.clip(rng.normal(92, 2, N) - age_factor * 8
                             - vibration * 0.3 + rng.normal(0, 0.5, N), 60, 99)

    # ── Maintenance ───────────────────────────────────────────────────────────
    days_since_maintenance = rng.uniform(0, 365, N)
    maintenance_count      = (operating_hours / 2_000).astype(int)

    # ── Anomaly Injection (5%) ────────────────────────────────────────────────
    anomaly_idx = rng.choice(N, size=int(N * 0.05), replace=False)
    temperature [anomaly_idx] += rng.uniform(30, 80, len(anomaly_idx))
    vibration   [anomaly_idx] += rng.uniform(5,  15, len(anomaly_idx))
    oil_temp    [anomaly_idx] += rng.uniform(10, 30, len(anomaly_idx))
    pressure    [anomaly_idx] -= rng.uniform(10, 30, len(anomaly_idx))
    efficiency  [anomaly_idx] -= rng.uniform(5,  20, len(anomaly_idx))
    efficiency   = np.clip(efficiency, 60, 99)

    # ── Targets ───────────────────────────────────────────────────────────────
    risk_score = np.clip(
        0.30 * age_factor
        + 0.25 * (vibration / vibration.max())
        + 0.20 * ((temperature - 520) / 100).clip(0, 1)
        + 0.15 * ((oil_temp - 65) / 40).clip(0, 1)
        + 0.10 * (days_since_maintenance / 365)
        + rng.normal(0, 0.03, N), 0, 1
    )
    risk_score[anomaly_idx] = np.clip(risk_score[anomaly_idx] + 0.35, 0, 1)

    failure_prob = 1 / (1 + np.exp(-10 * (risk_score - 0.65)))
    failure      = (rng.uniform(0, 1, N) < failure_prob).astype(int)

    RUL = np.round((1 - risk_score) * 365 + rng.normal(0, 10, N)).clip(0, 365).astype(int)
    RUL[failure == 1] = rng.integers(0, 15, size=failure.sum())

    # ── Assemble ──────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "machine_id"            : [f"GEN_{i:04d}" for i in rng.integers(1, 21, N)],
        "timestamp"             : pd.date_range("2020-01-01", periods=N, freq="h"),
        "operating_hours"       : operating_hours.round(1),
        "load_pct"              : load.round(2),
        "days_since_maintenance": days_since_maintenance.round(1),
        "maintenance_count"     : maintenance_count,
        "temperature_C"         : temperature.round(2),
        "pressure_bar"          : pressure.round(2),
        "vibration_mms"         : vibration.round(3),
        "rotation_speed_rpm"    : rotation_speed.round(1),
        "voltage_V"             : voltage.round(1),
        "current_A"             : current.round(2),
        "oil_temp_C"            : oil_temp.round(2),
        "power_output_MW"       : power_output.round(2),
        "efficiency_pct"        : efficiency.round(2),
        "risk_score"            : risk_score.round(4),
        "RUL_days"              : RUL,
        "failure"               : failure,
    })

    return df


# ==============================================================================
# EXPORT FUNCTION
# ==============================================================================

def export(df, suffix=""):
    tag = f"_{suffix}" if suffix else ""

    # Full
    path = os.path.join(OUTPUT_DIR, f"powerplant_maintenance{tag}.csv")
    df.to_csv(path, index=False)
    print(f"  [Full]          → {path}  {df.shape}")

    # Supervised (drop risk_score)
    sup  = df.drop(columns=["risk_score"])
    path = os.path.join(OUTPUT_DIR, f"supervised_dataset{tag}.csv")
    sup.to_csv(path, index=False)
    print(f"  [Supervised]    → {path}  {sup.shape}")

    # Unsupervised (drop risk_score, RUL_days, failure)
    uns  = df.drop(columns=["risk_score", "RUL_days", "failure"])
    path = os.path.join(OUTPUT_DIR, f"unsupervised_dataset{tag}.csv")
    uns.to_csv(path, index=False)
    print(f"  [Unsupervised]  → {path}  {uns.shape}")

    print(f"  failure: {df['failure'].value_counts().to_dict()}")


# ==============================================================================
# MAIN
# ==============================================================================

print("\n" + "="*55)
print("  GEN 1 — TRAIN  (N=10,000  SEED=67)")
print("="*55)
df_train = generate(N=10_000, SEED=67)
export(df_train, suffix="")

print("\n" + "="*55)
print("  GEN 2 — TEST   (N=1,000   SEED=99)")
print("="*55)
df_test = generate(N=1_000, SEED=99)
export(df_test, suffix="test")

print("\n  Done — 6 CSV files saved to DataSet/")
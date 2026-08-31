"""
00_Data_Quality.py
==================
Data Quality Process for Solar Power Plant Dataset
- completeness  : missing values, missing timestamps
- consistency   : value range, negative values, outliers
- validity      : data types, date format
- uniqueness    : duplicate rows
- summary report
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    "font.family"      : "sans-serif",
    "font.sans-serif"  : ["Tahoma", "Arial", "DejaVu Sans"],
})

BLUE  = "#4C9BE8"
RED   = "#E8564C"
GREEN = "#5CB85C"
SOLAR = "#F0A500"
GRAY  = "#AAAAAA"

FILES = {
    "Plant_1_Generation_Data" : {
        "file"      : "Plant_1_Generation_Data.csv",
        "type"      : "generation",
        "plant"     : 1,
        "freq"      : "15min",
        "per_inv"   : True,
    },
    "Plant_2_Generation_Data" : {
        "file"      : "Plant_2_Generation_Data.csv",
        "type"      : "generation",
        "plant"     : 2,
        "freq"      : "15min",
        "per_inv"   : True,
    },
    "Plant_1_Weather_Sensor"  : {
        "file"      : "Plant_1_Weather_Sensor_Data.csv",
        "type"      : "sensor",
        "plant"     : 1,
        "freq"      : "15min",
        "per_inv"   : False,
    },
    "Plant_2_Weather_Sensor"  : {
        "file"      : "Plant_2_Weather_Sensor_Data.csv",
        "type"      : "sensor",
        "plant"     : 2,
        "freq"      : "15min",
        "per_inv"   : False,
    },
}

GENERATION_COLS = ["DC_POWER", "AC_POWER", "DAILY_YIELD", "TOTAL_YIELD"]
SENSOR_COLS     = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]

VALID_RANGES = {
    "DC_POWER"            : (0,    20000),
    "AC_POWER"            : (0,    20000),
    "DAILY_YIELD"         : (0,    100000),
    "TOTAL_YIELD"         : (0,    1e9),
    "AMBIENT_TEMPERATURE" : (-10,  60),
    "MODULE_TEMPERATURE"  : (-10,  100),
    "IRRADIATION"         : (0,    1500),
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def load_file(meta: dict) -> pd.DataFrame:
    fpath = os.path.join(DATASET_DIR, meta["file"])
    df    = pd.read_csv(fpath)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"],
                                      format="mixed", dayfirst=True)
    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    return df


def check_completeness(df: pd.DataFrame, meta: dict) -> dict:
    result = {}

    # NaN per column
    nan_counts = df.isna().sum()
    nan_pct    = (nan_counts / len(df) * 100).round(2)
    result["nan_counts"] = nan_counts
    result["nan_pct"]    = nan_pct
    result["total_rows"] = len(df)

    # Missing timestamps
    full_timeline = pd.date_range(
        start = df["DATE_TIME"].min(),
        end   = df["DATE_TIME"].max(),
        freq  = meta["freq"]
    )
    result["expected_rows"] = len(full_timeline)

    if meta["per_inv"]:
        inv_keys  = df["SOURCE_KEY"].unique()
        n_inv     = len(inv_keys)
        expected  = len(full_timeline) * n_inv
        result["expected_total"] = expected
        result["n_inverters"]    = n_inv

        inv_counts = df.groupby("SOURCE_KEY")["DATE_TIME"].count()
        missing_inv = {
            k: int(len(full_timeline) - v)
            for k, v in inv_counts.items()
            if v < len(full_timeline)
        }
        result["missing_inv"] = missing_inv
    else:
        missing_ts = set(full_timeline) - set(df["DATE_TIME"])
        result["missing_timestamps"] = len(missing_ts)
        result["expected_total"]     = len(full_timeline)

    result["actual_total"]   = len(df)
    result["missing_rows"]   = result["expected_total"] - len(df)
    result["missing_pct"]    = round(result["missing_rows"] /
                                     result["expected_total"] * 100, 2)
    return result


def check_consistency(df: pd.DataFrame, meta: dict) -> dict:
    result     = {}
    value_cols = GENERATION_COLS if meta["type"] == "generation" else SENSOR_COLS

    out_of_range  = {}
    negative      = {}
    outlier_upper = {}

    for col in value_cols:
        if col not in df.columns:
            continue
        lo, hi = VALID_RANGES.get(col, (None, None))

        if lo is not None:
            neg = (df[col] < lo).sum()
            negative[col] = int(neg)

        if hi is not None:
            oor = ((df[col] < lo) | (df[col] > hi)).sum()
            out_of_range[col] = int(oor)

        # IQR outlier
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr     = q3 - q1
        upper   = q3 + 3 * iqr
        outlier_upper[col] = int((df[col] > upper).sum())

    result["out_of_range"]  = out_of_range
    result["negative"]      = negative
    result["outlier_upper"] = outlier_upper

    # DC > AC (generation only)
    if meta["type"] == "generation":
        dc_lt_ac = (df["DC_POWER"] < df["AC_POWER"]).sum()
        result["dc_lt_ac"] = int(dc_lt_ac)

    return result


def check_uniqueness(df: pd.DataFrame, meta: dict) -> dict:
    result = {}

    if meta["per_inv"]:
        dup = df.duplicated(subset=["DATE_TIME","SOURCE_KEY"]).sum()
    else:
        dup = df.duplicated(subset=["DATE_TIME"]).sum()

    result["duplicates"] = int(dup)
    return result


def check_validity(df: pd.DataFrame) -> dict:
    result = {}
    result["dtypes"]       = df.dtypes.to_dict()
    result["date_nulls"]   = int(df["DATE_TIME"].isna().sum())
    result["date_min"]     = str(df["DATE_TIME"].min())
    result["date_max"]     = str(df["DATE_TIME"].max())
    return result

# ==============================================================================
# PRINT REPORT
# ==============================================================================

def print_report(name: str, df: pd.DataFrame, meta: dict,
                 comp: dict, cons: dict, uniq: dict, valid: dict):

    print(f"\n{'='*70}")
    print(f"  DATA QUALITY REPORT — {name}")
    print(f"{'='*70}")

    # Basic
    print(f"\n  [Basic]")
    print(f"  File         : {meta['file']}")
    print(f"  Rows         : {comp['total_rows']:,}")
    print(f"  Date range   : {valid['date_min']}  ->  {valid['date_max']}")
    if meta["per_inv"]:
        print(f"  Inverters    : {comp['n_inverters']}")

    # Completeness
    print(f"\n  [Completeness]")
    print(f"  Expected rows: {comp['expected_total']:,}")
    print(f"  Actual rows  : {comp['actual_total']:,}")
    print(f"  Missing rows : {comp['missing_rows']:,}  ({comp['missing_pct']}%)")

    if comp["nan_counts"].sum() > 0:
        print(f"\n  NaN per column:")
        for col, cnt in comp["nan_counts"].items():
            if cnt > 0:
                print(f"    {col:<28} {cnt:>8,}  ({comp['nan_pct'][col]:.2f}%)")
    else:
        print(f"  NaN          : none")

    if meta["per_inv"] and comp.get("missing_inv"):
        print(f"\n  Missing timestamps per Inverter:")
        for k, v in comp["missing_inv"].items():
            print(f"    {k:<28} missing {v:>6,} rows")

    # Consistency
    print(f"\n  [Consistency]")
    for col, cnt in cons["out_of_range"].items():
        status = "OK" if cnt == 0 else f"!! {cnt:,} rows"
        print(f"  Out of range  {col:<28} {status}")

    for col, cnt in cons["negative"].items():
        status = "OK" if cnt == 0 else f"!! {cnt:,} rows"
        print(f"  Negative      {col:<28} {status}")

    for col, cnt in cons["outlier_upper"].items():
        status = "OK" if cnt == 0 else f"!! {cnt:,} rows"
        print(f"  Outlier(IQR)  {col:<28} {status}")

    if "dc_lt_ac" in cons:
        status = "OK" if cons["dc_lt_ac"] == 0 else f"!! {cons['dc_lt_ac']:,} rows"
        print(f"  DC < AC                              {status}")

    # Uniqueness
    print(f"\n  [Uniqueness]")
    dup = uniq["duplicates"]
    print(f"  Duplicates   : {'none' if dup == 0 else f'!! {dup:,} rows'}")

    # Validity
    print(f"\n  [Validity]")
    print(f"  Date nulls   : {valid['date_nulls']}")

# ==============================================================================
# FIGURE — Data Quality Summary Dashboard
# ==============================================================================

def plot_quality_dashboard(all_results: dict):

    names = list(all_results.keys())
    n     = len(names)

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle("Data Quality Dashboard — All Files",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # ── 1. Missing rows % per file ────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    miss_pct = [all_results[n]["comp"]["missing_pct"] for n in names]
    colors   = [RED if p > 10 else SOLAR if p > 0 else GREEN
                for p in miss_pct]
    bars = ax.bar(names, miss_pct, color=colors, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, miss_pct):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.5, f"{val:.1f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Missing Rows % per File", fontsize=12, fontweight="bold")
    ax.set_ylabel("Missing %")
    ax.set_ylim(0, max(miss_pct) * 1.2 + 5)
    ax.axhline(10, color=RED, linestyle="--", linewidth=1.5,
               label="10% threshold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", labelsize=9)

    # ── 2. NaN count per file ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    nan_total = [all_results[n]["comp"]["nan_counts"].sum() for n in names]
    colors    = [RED if v > 0 else GREEN for v in nan_total]
    bars = ax.bar(names, nan_total, color=colors, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, nan_total):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + max(nan_total)*0.01,
                f"{int(val):,}", ha="center", fontsize=8)
    ax.set_title("Total NaN Values per File", fontsize=11, fontweight="bold")
    ax.set_ylabel("NaN Count")
    ax.tick_params(axis="x", labelsize=8)

    # ── 3. Duplicates per file ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    dups   = [all_results[n]["uniq"]["duplicates"] for n in names]
    colors = [RED if v > 0 else GREEN for v in dups]
    bars = ax.bar(names, dups, color=colors, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, dups):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.1, f"{int(val):,}",
                ha="center", fontsize=8)
    ax.set_title("Duplicate Rows per File", fontsize=11, fontweight="bold")
    ax.set_ylabel("Duplicate Count")
    ax.tick_params(axis="x", labelsize=8)

    # ── 4. Out of range per file ──────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    oor_total = [sum(all_results[n]["cons"]["out_of_range"].values())
                 for n in names]
    colors    = [RED if v > 0 else GREEN for v in oor_total]
    bars = ax.bar(names, oor_total, color=colors, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, oor_total):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.1, f"{int(val):,}",
                ha="center", fontsize=8)
    ax.set_title("Out of Range Values per File", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelsize=8)

    # ── 5. Outliers per file ──────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    out_total = [sum(all_results[n]["cons"]["outlier_upper"].values())
                 for n in names]
    colors    = [SOLAR if v > 0 else GREEN for v in out_total]
    bars = ax.bar(names, out_total, color=colors, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, out_total):
        ax.text(bar.get_x() + bar.get_width()/2,
                val + 0.1, f"{int(val):,}",
                ha="center", fontsize=8)
    ax.set_title("Outliers (IQR x3) per File", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "DQ_Dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [Saved] DQ_Dashboard.png")

# ==============================================================================
# EXPORT SUMMARY CSV
# ==============================================================================

def export_summary(all_results: dict):
    rows = []
    for name, res in all_results.items():
        row = {
            "File"            : name,
            "Total_Rows"      : res["comp"]["total_rows"],
            "Expected_Rows"   : res["comp"]["expected_total"],
            "Missing_Rows"    : res["comp"]["missing_rows"],
            "Missing_%"       : res["comp"]["missing_pct"],
            "NaN_Total"       : int(res["comp"]["nan_counts"].sum()),
            "Duplicates"      : res["uniq"]["duplicates"],
            "OutOfRange_Total": sum(res["cons"]["out_of_range"].values()),
            "Outlier_Total"   : sum(res["cons"]["outlier_upper"].values()),
            "Date_Min"        : res["valid"]["date_min"],
            "Date_Max"        : res["valid"]["date_max"],
        }
        if "dc_lt_ac" in res["cons"]:
            row["DC_lt_AC"] = res["cons"]["dc_lt_ac"]
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out        = os.path.join(OUTPUT_DIR, "DQ_Summary.csv")
    summary_df.to_csv(out, index=False)
    print(f"  [Saved] DQ_Summary.csv")
    return summary_df

# ==============================================================================
# MAIN
# ==============================================================================

print(f"\n{'#'*70}")
print(f"#  DATA QUALITY PROCESS")
print(f"{'#'*70}")

all_results = {}

for name, meta in FILES.items():
    fpath = os.path.join(DATASET_DIR, meta["file"])
    if not os.path.exists(fpath):
        print(f"\n  [SKIP] {meta['file']} not found")
        continue

    df    = load_file(meta)
    comp  = check_completeness(df, meta)
    cons  = check_consistency(df, meta)
    uniq  = check_uniqueness(df, meta)
    valid = check_validity(df)

    print_report(name, df, meta, comp, cons, uniq, valid)

    all_results[name] = {
        "comp" : comp,
        "cons" : cons,
        "uniq" : uniq,
        "valid": valid,
    }

# Dashboard + CSV
plot_quality_dashboard(all_results)
summary_df = export_summary(all_results)

print(f"\n{'='*70}")
print(f"  OVERALL SUMMARY")
print(f"{'='*70}")
print(summary_df.to_string(index=False))

print(f"\n{'#'*70}")
print(f"#  Done.")
print(f"{'#'*70}")
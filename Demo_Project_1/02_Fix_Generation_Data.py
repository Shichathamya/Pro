"""
02_Fix_Generation_Data.py
======================
Fix missing timestamps in Generation Data
- เติม timestamp ให้ครบทุก 15 นาที
- PLANT_ID คงค่าเดิม
- SOURCE_KEY ครบทุก inverter
- DC_POWER, AC_POWER:
    หาย 1 ช่วง  → interpolate (ค่าระหว่างก่อน-หลัง)
    หายต่อเนื่อง → mean ของ inverter นั้น ช่วงเวลาเดียวกัน
    AC=0 ทั้งที่ IRRADIATION > 0.3 → mean ของ inverter นั้น ช่วงเวลาเดียวกัน
- DAILY_YIELD  : ใช้ค่าก่อนหาย / ขึ้นวันใหม่ = 0
- TOTAL_YIELD  : ใช้ค่าก่อนหาย
- บันทึกเป็นไฟล์ใหม่
"""

import os
import pandas as pd
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")

INPUT_FILES = {
    1: "Plant_1_Generation_Data.csv",
    2: "Plant_2_Generation_Data.csv",
}

INTERP_COLS           = ["DC_POWER", "AC_POWER"]
IRRADIATION_THRESHOLD = 0.3

# ==============================================================================
# HELPER — คำนวณ time-of-day mean ของ inverter นั้นเอง
# ใช้เฉพาะแถวปกติ (AC > 0 และ IRRADIATION > threshold)
# ==============================================================================

def calc_tod_mean(df_inv: pd.DataFrame, col: str) -> dict:
    """
    คำนวณ mean ต่อ TIME_OF_DAY จากข้อมูลปกติของ inverter นั้น
    ปกติ = AC_POWER > 0 และ IRRADIATION > threshold (กลางวัน)
    กลางคืน = IRRADIATION <= threshold → mean จากทุกแถว
    """
    df_inv = df_inv.copy()
    df_inv["TIME_OF_DAY"] = df_inv["DATE_TIME"].dt.strftime("%H:%M:%S")

    # แถวปกติกลางวัน
    normal = df_inv[
        (df_inv["AC_POWER"]    > 0) &
        (df_inv["IRRADIATION"] > IRRADIATION_THRESHOLD)
    ] if "IRRADIATION" in df_inv.columns else df_inv[df_inv[col] > 0]

    # mean กลางวัน
    tod_day = normal.groupby("TIME_OF_DAY")[col].mean().to_dict()

    # mean ทั้งหมด (รวมกลางคืน)
    tod_all = df_inv.groupby("TIME_OF_DAY")[col].mean().to_dict()

    # รวม: ถ้า time slot นั้นมีข้อมูลปกติ → ใช้ normal mean
    #       ถ้าไม่มี (กลางคืน) → ใช้ all mean
    tod_mean = {t: tod_day.get(t, tod_all.get(t, 0.0))
                for t in tod_all}
    return tod_mean

# ==============================================================================
# HELPER — เติม DC_POWER / AC_POWER
# ==============================================================================

def fill_power_col(df_inv: pd.DataFrame,
                   col: str,
                   tod_mean: dict) -> pd.Series:
    series = df_inv[col].copy()
    tod    = df_inv["DATE_TIME"].dt.strftime("%H:%M:%S")

    is_nan     = series.isna()
    nan_groups = (is_nan != is_nan.shift()).cumsum()

    for gid in nan_groups[is_nan].unique():
        group_pos = nan_groups[nan_groups == gid].index.tolist()

        if len(group_pos) == 1:
            # หาย 1 ช่วง → interpolate
            pos  = series.index.get_loc(group_pos[0])
            prev = series.iloc[pos - 1] if pos > 0 else np.nan
            rest = series.iloc[pos + 1:]
            nxt  = rest.dropna().iloc[0] if len(rest.dropna()) > 0 else np.nan

            if not np.isnan(prev) and not np.isnan(nxt):
                series.iloc[pos] = (prev + nxt) / 2.0
            elif not np.isnan(prev):
                series.iloc[pos] = prev
            elif not np.isnan(nxt):
                series.iloc[pos] = nxt
            else:
                series.iloc[pos] = 0.0

        else:
            # หายต่อเนื่อง → mean ของ inverter นั้น ช่วงเวลาเดียวกัน
            for pos_idx in group_pos:
                t               = tod[pos_idx]
                series[pos_idx] = tod_mean.get(t, 0.0)

    return series

# ==============================================================================
# HELPER — Fix AC=0 ทั้งที่ IRRADIATION > threshold
# ==============================================================================

def fix_zero_ac_dc(df_inv: pd.DataFrame,
                   tod_mean_dc: dict,
                   tod_mean_ac: dict) -> tuple:
    """
    ถ้า IRRADIATION > threshold แต่ AC_POWER = 0
    → แทนด้วย mean ของ inverter นั้น ช่วงเวลาเดียวกัน
    """
    df_inv = df_inv.copy()
    df_inv["TIME_OF_DAY"] = df_inv["DATE_TIME"].dt.strftime("%H:%M:%S")

    mask  = ((df_inv["IRRADIATION"] > IRRADIATION_THRESHOLD) &
              (df_inv["AC_POWER"]   == 0))
    n_fix = mask.sum()

    if n_fix > 0:
        df_inv.loc[mask, "AC_POWER"] = (df_inv.loc[mask, "TIME_OF_DAY"]
                                        .map(tod_mean_ac))
        df_inv.loc[mask, "DC_POWER"] = (df_inv.loc[mask, "TIME_OF_DAY"]
                                        .map(tod_mean_dc))

    df_inv = df_inv.drop(columns=["TIME_OF_DAY"])
    return df_inv, int(n_fix)

# ==============================================================================
# HELPER — เติม DAILY_YIELD
# ==============================================================================

def fill_daily_yield(df_inv: pd.DataFrame) -> pd.Series:
    series = df_inv["DAILY_YIELD"].copy()
    dates  = df_inv["DATE_TIME"].dt.date
    series_ffill = series.ffill()

    for i in series[series.isna()].index:
        pos       = series.index.get_loc(i)
        curr_date = dates[i]

        if pos == 0:
            series.iloc[pos] = 0.0
        else:
            prev_date  = dates.iloc[pos - 1]
            prev_value = series_ffill.iloc[pos - 1]

            if curr_date != prev_date:
                series.iloc[pos] = 0.0
            else:
                series.iloc[pos] = prev_value if not np.isnan(prev_value) else 0.0

    return series

# ==============================================================================
# HELPER — เติม TOTAL_YIELD
# ==============================================================================

def fill_total_yield(df_inv: pd.DataFrame) -> pd.Series:
    series = df_inv["TOTAL_YIELD"].copy()

    for i in series[series.isna()].index:
        pos = series.index.get_loc(i)
        if pos == 0:
            series.iloc[pos] = 0.0
        else:
            prev = series.iloc[pos - 1]
            series.iloc[pos] = prev if not np.isnan(prev) else 0.0

    return series

# ==============================================================================
# MAIN
# ==============================================================================

for plant_num, fname in INPUT_FILES.items():
    fpath = os.path.join(DATASET_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [SKIP] {fname} not found")
        continue

    print(f"\n{'='*60}")
    print(f"  Plant {plant_num} — {fname}")
    print(f"{'='*60}")

    df = pd.read_csv(fpath)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"],
                                      format="mixed", dayfirst=True)
    df = df.sort_values(["SOURCE_KEY","DATE_TIME"]).reset_index(drop=True)

    plant_id = df["PLANT_ID"].iloc[0]
    inv_keys = sorted(df["SOURCE_KEY"].unique())
    n_inv    = len(inv_keys)

    print(f"  PLANT_ID   : {plant_id}")
    print(f"  Inverters  : {n_inv}")
    print(f"  Original   : {len(df):,} rows")

    # สร้าง full timeline
    full_timeline    = pd.date_range(
        start = df["DATE_TIME"].min(),
        end   = df["DATE_TIME"].max(),
        freq  = "15min"
    )
    expected_per_inv = len(full_timeline)

    df["TIME_OF_DAY"]   = df["DATE_TIME"].dt.strftime("%H:%M:%S")
    df["IRRADIATION"]   = np.nan   # ยังไม่มีใน generation file
                                   # จะ join หลัง fix

    all_inv_fixed = []
    total_zero_fix = 0

    for key in inv_keys:
        df_inv = df[df["SOURCE_KEY"] == key].copy()

        # สร้าง full timeline ต่อ inverter
        full_df = pd.DataFrame({"DATE_TIME": full_timeline})
        merged  = pd.merge(full_df,
                           df_inv.drop(columns=["TIME_OF_DAY",
                                                 "IRRADIATION"],
                                       errors="ignore"),
                           on="DATE_TIME", how="left")
        merged["PLANT_ID"]   = plant_id
        merged["SOURCE_KEY"] = key
        merged = merged.sort_values("DATE_TIME").reset_index(drop=True)
        merged["TIME_OF_DAY"] = merged["DATE_TIME"].dt.strftime("%H:%M:%S")

        n_missing = merged["AC_POWER"].isna().sum()

        # ── DC_POWER, AC_POWER ─────────────────────────────────────────
        for col in INTERP_COLS:
            # คำนวณ tod_mean จาก inverter นั้นเอง (ก่อน merge)
            tod_mean    = calc_tod_mean(df_inv, col)
            merged[col] = fill_power_col(merged, col, tod_mean)

        # ── DAILY_YIELD ────────────────────────────────────────────────
        merged["DAILY_YIELD"] = fill_daily_yield(merged)

        # ── TOTAL_YIELD ────────────────────────────────────────────────
        merged["TOTAL_YIELD"] = fill_total_yield(merged)

        # ── Fix AC=0 ทั้งที่ควรมีค่า ──────────────────────────────────
        # ใช้ Joined data ถ้ามี ไม่งั้นข้ามขั้นตอนนี้ไปก่อน
        joined_path = os.path.join(DATASET_DIR,
                                   f"Plant_{plant_num}_Joined.csv")
        if os.path.exists(joined_path):
            joined = pd.read_csv(joined_path,
                                 usecols=["DATE_TIME","SOURCE_KEY",
                                          "IRRADIATION"])
            joined["DATE_TIME"] = pd.to_datetime(joined["DATE_TIME"])
            joined_inv = joined[joined["SOURCE_KEY"] == key]

            merged = pd.merge(merged, joined_inv[["DATE_TIME","IRRADIATION"]],
                              on="DATE_TIME", how="left")

            # คำนวณ tod_mean จาก inverter นั้น ช่วงปกติ
            normal = df_inv[(df_inv["AC_POWER"] > 0)]
            tod_ac = normal.groupby("TIME_OF_DAY")["AC_POWER"].mean().to_dict()
            tod_dc = normal.groupby("TIME_OF_DAY")["DC_POWER"].mean().to_dict()

            merged, n_fix = fix_zero_ac_dc(merged, tod_dc, tod_ac)
            total_zero_fix += n_fix

            merged = merged.drop(columns=["IRRADIATION"], errors="ignore")
        else:
            n_fix = 0

        # ลบ TIME_OF_DAY
        merged = merged.drop(columns=["TIME_OF_DAY"], errors="ignore")

        # จัดเรียง columns
        merged = merged[["DATE_TIME","PLANT_ID","SOURCE_KEY",
                          "DC_POWER","AC_POWER",
                          "DAILY_YIELD","TOTAL_YIELD"]]

        print(f"  [{key[:14]}]  "
              f"missing={n_missing:>5,}  "
              f"zero_fix={n_fix:>3}")
        all_inv_fixed.append(merged)

    # รวมทุก inverter
    result = (pd.concat(all_inv_fixed, ignore_index=True)
              .sort_values(["DATE_TIME","SOURCE_KEY"])
              .reset_index(drop=True))

    print(f"\n  After fix  : {len(result):,} rows")
    print(f"  NaN remain : {result.isna().sum().sum()}")
    print(f"  Zero AC fix: {total_zero_fix} rows")
    print(f"  Expected   : {expected_per_inv * n_inv:,} rows")

    print(f"\n  -- Row count per Inverter --")
    for k in inv_keys:
        cnt = result[result["SOURCE_KEY"]==k].shape[0]
        ok  = "OK" if cnt == expected_per_inv else "!!"
        print(f"    {ok} {k:<25} {cnt:>6,} rows")

    # บันทึก
    out_name = fname.replace(".csv", "_Fixed.csv")
    out_path = os.path.join(DATASET_DIR, out_name)
    result.to_csv(out_path, index=False)
    print(f"\n  [Saved] {out_name}")

print(f"\n  Done.")
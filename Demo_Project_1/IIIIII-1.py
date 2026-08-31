"""
IIIIII-1s.py
===================
Fix missing timestamps in Weather Sensor Data
- เติม timestamp ให้ครบทุก 15 นาที
- SOURCE_KEY, PLANT_ID คงค่าเดิม
- ค่าที่หาย → ใช้ mean ของ timestamp เดียวกัน (HH:MM) จากทุกวัน
- บันทึกเป็นไฟล์ใหม่
"""

import os
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")

INPUT_FILES = {
    1: "Plant_1_Weather_Sensor_Data.csv",
    2: "Plant_2_Weather_Sensor_Data.csv",
}

VALUE_COLS = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]

# ==============================================================================

for plant_num, fname in INPUT_FILES.items():
    fpath = os.path.join(DATASET_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  [SKIP] {fname} not found")
        continue

    print(f"\n{'='*55}")
    print(f"  Plant {plant_num} — {fname}")
    print(f"{'='*55}")

    df = pd.read_csv(fpath)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"],
                                      format="mixed", dayfirst=True)
    df = df.sort_values("DATE_TIME").reset_index(drop=True)

    plant_id   = df["PLANT_ID"].iloc[0]
    source_key = df["SOURCE_KEY"].iloc[0]

    print(f"  PLANT_ID   : {plant_id}")
    print(f"  SOURCE_KEY : {source_key}")
    print(f"  Original   : {len(df):,} rows")
    print(f"  Date range : {df['DATE_TIME'].min()} -> {df['DATE_TIME'].max()}")

    # ==================================================================
    # คำนวณ mean ต่อ TIME_OF_DAY (HH:MM:SS)
    # ใช้เฉพาะแถวที่มีข้อมูลจริง (IRRADIATION > 0 หรือทุกแถว)
    # ==================================================================
    df["TIME_OF_DAY"] = df["DATE_TIME"].dt.strftime("%H:%M:%S")

    time_mean = (df.groupby("TIME_OF_DAY")[VALUE_COLS]
                 .mean()
                 .round(6)
                 .reset_index())

    print(f"  Time slots : {len(time_mean)} unique HH:MM:SS")

    # สร้าง full timeline ทุก 15 นาที
    full_timeline = pd.date_range(
        start = df["DATE_TIME"].min(),
        end   = df["DATE_TIME"].max(),
        freq  = "15min"
    )
    print(f"  Expected   : {len(full_timeline):,} rows")
    print(f"  Missing    : {len(full_timeline) - len(df):,} rows")

    # สร้าง full DataFrame
    full_df = pd.DataFrame({"DATE_TIME": full_timeline})
    full_df["TIME_OF_DAY"] = full_df["DATE_TIME"].dt.strftime("%H:%M:%S")

    # merge กับข้อมูลจริง
    merged = pd.merge(full_df, df.drop(columns=["TIME_OF_DAY"]),
                      on="DATE_TIME", how="left")

    # เติม PLANT_ID และ SOURCE_KEY
    merged["PLANT_ID"]   = plant_id
    merged["SOURCE_KEY"] = source_key

    # นับแถวที่หาย
    missing_mask = merged[VALUE_COLS[0]].isna()
    n_missing    = missing_mask.sum()
    print(f"  Filling    : {n_missing:,} missing rows with time-of-day mean")

    # merge กับ time_mean เพื่อเติมค่าที่หาย
    merged = pd.merge(merged, time_mean,
                      on="TIME_OF_DAY", how="left",
                      suffixes=("", "_MEAN"))

    for col in VALUE_COLS:
        mean_col        = f"{col}_MEAN"
        merged[col]     = merged[col].fillna(merged[mean_col])
        merged.drop(columns=[mean_col], inplace=True)

    # ถ้ายังมี NaN (time slot ที่ไม่มีข้อมูลเลย) → ใช้ 0
    remaining_nan = merged[VALUE_COLS].isna().sum().sum()
    if remaining_nan > 0:
        print(f"  Warning    : {remaining_nan} values still NaN → fill with 0")
        merged[VALUE_COLS] = merged[VALUE_COLS].fillna(0)

    # จัดเรียง columns
    merged = merged[["DATE_TIME","PLANT_ID","SOURCE_KEY"] + VALUE_COLS]
    merged = merged.sort_values("DATE_TIME").reset_index(drop=True)

    print(f"  After fill : {len(merged):,} rows")
    print(f"  NaN remain : {merged.isna().sum().sum()}")

    # ==================================================================
    # แสดงตัวอย่างแถวที่ถูกเติม
    # ==================================================================
    print(f"\n  -- Sample: filled rows (first 5) --")
    filled_sample = merged[missing_mask.reindex(merged.index, fill_value=False)].head(5)
    if len(filled_sample) > 0:
        print(filled_sample.to_string(index=False))

    print(f"\n  -- Time-of-day mean (sample 5 slots) --")
    print(time_mean.head(5).to_string(index=False))

    # บันทึกไฟล์ใหม่
    out_name = fname.replace(".csv", "_Fixed.csv")
    out_path = os.path.join(DATASET_DIR, out_name)
    merged.to_csv(out_path, index=False)
    print(f"\n  [Saved] {out_name}")

print(f"\n  Done.")
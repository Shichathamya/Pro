import pandas as pd
import os

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")

for plant_num in [1, 2]:
    df = pd.read_csv(os.path.join(DATASET_DIR,
                     f"Plant_{plant_num}_Generation_Data.csv"))
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"],
                                      format="mixed", dayfirst=True)

    print(f"\n{'='*55}")
    print(f"  Plant {plant_num} — Missing Value Check")
    print(f"{'='*55}")
    print(f"  Total rows    : {len(df):,}")
    print(f"  Missing values:\n{df.isna().sum().to_string()}")

    print(f"\n  -- AC_POWER = 0 per Inverter --")
    zero = (df.groupby("SOURCE_KEY")["AC_POWER"]
              .apply(lambda x: (x == 0).sum())
              .sort_values(ascending=False))
    print(zero.to_string())

    print(f"\n  -- Date range per Inverter --")
    date_range = df.groupby("SOURCE_KEY")["DATE_TIME"].agg(["min","max","count"])
    print(date_range.to_string())
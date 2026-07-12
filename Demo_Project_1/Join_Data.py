"""
Join_Data.py
============
Join Generation + Weather Sensor data for Plant 1 & Plant 2
"""

import pandas as pd
import os

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
os.makedirs(DATASET_DIR, exist_ok=True)


def join_plant(plant_num):
    gen    = pd.read_csv(os.path.join(DATASET_DIR, f"Plant_{plant_num}_Generation_Data.csv"))
    sensor = pd.read_csv(os.path.join(DATASET_DIR, f"Plant_{plant_num}_Weather_Sensor_Data.csv"))

    gen["DATE_TIME"]    = pd.to_datetime(gen["DATE_TIME"],    format="mixed", dayfirst=True)
    sensor["DATE_TIME"] = pd.to_datetime(sensor["DATE_TIME"], format="mixed", dayfirst=True)    

    sensor_clean = sensor[["DATE_TIME", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    joined = pd.merge(gen, sensor_clean, on="DATE_TIME", how="left")

    out_path = os.path.join(DATASET_DIR, f"Plant_{plant_num}_Joined.csv")
    joined.to_csv(out_path, index=False)
    print(f"  [Plant {plant_num}] → {joined.shape}  saved: {out_path}")
    return joined


print("\n" + "="*55)
print("  JOIN GENERATION + SENSOR DATA")
print("="*55 + "\n")

df1 = join_plant(1)
df2 = join_plant(2)

combined = pd.concat([df1, df2], ignore_index=True)
combined_path = os.path.join(DATASET_DIR, "All_Plants_Joined.csv")
combined.to_csv(combined_path, index=False)
print(f"  [Combined] → {combined.shape}  saved: {combined_path}")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")

df = pd.read_csv(os.path.join(DATASET_DIR, "Plant_1_Joined.csv"))
df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
df["DATE"]      = df["DATE_TIME"].dt.date

df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

day   = df[df["DATE_TIME"].dt.date == pd.Timestamp("2020-06-07").date()]
key   = day["SOURCE_KEY"].iloc[0]
sub   = day[day["SOURCE_KEY"] == key].sort_values("DATE_TIME")

plt.figure(figsize=(14, 5))
plt.plot(sub["DATE_TIME"], sub["AC_POWER"], color="#4C9BE8", linewidth=1.5)
plt.title(f"AC Power — {key} — 2020-06-07")
plt.xlabel("Time")
plt.ylabel("AC Power (kW)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("zoom_day.png", dpi=150)
plt.close()
print("Saved zoom_day.png")
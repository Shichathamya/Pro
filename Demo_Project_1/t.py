"""
Data_Table.py
=============
แสดงตัวอย่างข้อมูลจาก Plant_1_Joined.csv เป็นตาราง PNG
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# CONFIG — แก้ตรงนี้
# ==============================================================================
INPUT_FILE = "Plant_1_Weather_Sensor_Data.csv"   # ← เปลี่ยนไฟล์ได้
# Plant_1_Weather_Sensor_Data.csv,Plant_1_Generation_Data.csv
N_ROWS     = 20                      # ← จำนวนแถวที่แสดง

# ==============================================================================

df      = pd.read_csv(os.path.join(DATASET_DIR, INPUT_FILE))
df_show = df.head(N_ROWS).round(4)

# ปรับขนาด figure ตามจำนวน col และ row
n_cols = len(df_show.columns)
n_rows = len(df_show)

fig_w  = max(n_cols * 1.6, 14)
fig_h  = max(n_rows * 0.45 + 1.5, 5)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

tbl = ax.table(
    cellText  = df_show.values,
    colLabels = df_show.columns,
    cellLoc   = "center",
    loc       = "center",
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.5)

# Header style
for j in range(n_cols):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Row alternating color
for i in range(1, n_rows + 1):
    for j in range(n_cols):
        tbl[i, j].set_facecolor("#F0F4F8" if i % 2 == 0 else "white")

ax.set_title(f"{INPUT_FILE}  (showing {N_ROWS} rows × {n_cols} cols)",
             fontsize=12, fontweight="bold", pad=16)

out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(INPUT_FILE)[0]}_table.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  [Saved] {out_path}")
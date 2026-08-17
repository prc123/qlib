"""
Dump industry data into qlib binary format as a static feature.

1. Fetch industry mapping from Tushare
2. Assign numeric ID (0~N) to each industry
3. For each stock, write a constant-valued industry.day.bin file

Usage: conda run -n qlib python dump_industry.py
"""

import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

QLIB_DIR = Path(r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd")

# ---- Step 1: Fetch industry mapping from Tushare ----
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

import tushare as ts
pro = ts.pro_api("a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202")

df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
df["instrument"] = df["ts_code"].apply(lambda x: f"{x.split('.')[1].upper()}{x.split('.')[0]}")
df = df.dropna(subset=["industry"])
df["industry_id"] = df["industry"].astype("category").cat.codes  # 0~N numeric
print(f"Fetched {len(df)} stocks, {df['industry'].nunique()} industries")

# Save mapping
industry_map = dict(zip(df["industry_id"], df["industry"]))
print("Industry mapping:", industry_map)

# ---- Step 2: Read calendar for date index ----
cal = pd.read_csv(QLIB_DIR / "calendars" / "day.txt")
cal_dates = cal.iloc[:, 0].tolist()
print(f"Calendar: {len(cal_dates)} days")

# ---- Step 3: Write industry.day.bin for each stock ----
features_dir = QLIB_DIR / "features"
inst_dir = QLIB_DIR / "instruments" / "all.txt"
valid_stocks = set()
if inst_dir.exists():
    all_df = pd.read_csv(inst_dir, sep="\t", header=None, names=["symbol", "start", "end"])
    valid_stocks = set(all_df["symbol"].astype(str).str.upper())

count = 0
for _, row in df.iterrows():
    inst = row["instrument"]
    if inst not in valid_stocks:
        continue
    ind_id = row["industry_id"]
    stock_dir = features_dir / inst.lower()
    stock_dir.mkdir(parents=True, exist_ok=True)

    # Build constant array: date_index (timestamps) + industry_id repeated
    date_index = np.array([pd.Timestamp(d).value / 1e9 for d in cal_dates], dtype="<f")
    values = np.full(len(cal_dates), ind_id, dtype="<f")
    data = np.hstack([date_index, values]).astype("<f")

    bin_path = stock_dir / "industry.day.bin"
    data.tofile(str(bin_path))
    count += 1

print(f"\nDone! Dumped industry.day.bin for {count} stocks to {features_dir}")

# Save industry_id -> name mapping
mapping_df = pd.DataFrame({"industry_id": list(industry_map.keys()),
                           "industry_name": list(industry_map.values())})
mapping_path = QLIB_DIR / "industry_mapping.csv"
mapping_df.to_csv(mapping_path, index=False)
print(f"Industry mapping saved to {mapping_path}")

"""Download A-share industry mapping and save as industry.csv.
Usage: conda run -n qlib python _get_industry.py
"""
import os
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

import tushare as ts
import pandas as pd

pro = ts.pro_api("a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202")

# Method 1: stock_basic has built-in 申万一级行业
df = pro.stock_basic(exchange="", list_status="L",
                     fields="ts_code,name,industry")
df["instrument"] = df["ts_code"].apply(lambda x: f"{x.split('.')[1].upper()}{x.split('.')[0]}")
df = df[["instrument", "name", "industry"]]
df.to_csv("industry.csv", index=False)
print(f"Saved {len(df)} stocks to industry.csv")
print(f"\nIndustry distribution (top 15):")
print(df["industry"].value_counts().head(15).to_string())
print(f"\nSample:")
print(df.head(10).to_string())

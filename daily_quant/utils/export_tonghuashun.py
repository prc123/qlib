"""
导出 Top K 预测股票到同花顺自选股格式 (Table.txt)。

同花顺 → 自选股 → 导入 → 选择导出的 Table.txt

Usage
-----
    python export_tonghuashun.py
    python export_tonghuashun.py --date 2026-06-24 --topk 5
"""

import os
import re
import argparse
from pathlib import Path

import pandas as pd

CUR_DIR = Path(__file__).resolve().parent
OUT_DIR = CUR_DIR / "tonghuashun"

TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"


def _get_name_map() -> dict[str, str]:
    """Query Tushare for {instrument: name}."""
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)
    import tushare as ts
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name")
    name_map = {}
    for _, row in df.iterrows():
        parts = row["ts_code"].split(".")
        sym = f"{parts[1].upper()}{parts[0]}"
        name_map[sym] = row["name"]
    return name_map


def find_latest_pred_file():
    """Find the newest pred_score_*.csv file."""
    pattern = re.compile(r"pred_score_(\d{4}-\d{2}-\d{2})\.csv$")
    files = []
    for d in [CUR_DIR, CUR_DIR / "predictions"]:
        if not d.is_dir():
            continue
        for f in d.rglob("pred_score_*.csv"):
            m = pattern.match(f.name)
            if m:
                files.append((m.group(1), f))
    if not files:
        raise FileNotFoundError("未找到 pred_score_*.csv，请先运行 predict.py")
    files.sort(key=lambda x: x[0], reverse=True)
    return files[0]


def find_pred_file_for_date(date_str: str):
    """Find pred_score file for a specific date."""
    pattern = re.compile(r"pred_score_(\d{4}-\d{2}-\d{2})\.csv$")
    for d in [CUR_DIR, CUR_DIR / "predictions"]:
        if not d.is_dir():
            continue
        for f in d.rglob("pred_score_*.csv"):
            m = pattern.match(f.name)
            if m and m.group(1) == date_str:
                return m.group(1), f
    raise FileNotFoundError(f"未找到 pred_score_{date_str}.csv")


def main():
    parser = argparse.ArgumentParser(description="导出 Top K 股票到同花顺 Table.txt")
    parser.add_argument("--date", default=None, help="预测日期 (YYYY-MM-DD)，默认最新")
    parser.add_argument("--topk", type=int, default=10, help="导出前 K 只 (默认 10)")
    args = parser.parse_args()

    if args.date:
        date_str, path = find_pred_file_for_date(args.date)
    else:
        date_str, path = find_latest_pred_file()

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index("instrument")
    top = df["score"].nlargest(args.topk)

    # Get stock names from Tushare
    print("查询股票名称...")
    name_map = _get_name_map()

    # Build 同花顺 Table.txt format (GBK, tab-separated, \\r\\n)
    lines = ["代码\t\t    名称\t\t"]
    for inst in top.index:
        name = name_map.get(inst, "")
        lines.append(f"{inst}\t{name}\t")

    out_name = f"top{args.topk}_{date_str}.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    with open(out_path, "w", encoding="gbk") as f:
        f.write("\r\n".join(lines) + "\r\n")

    print(f"日期: {date_str}  |  Top {args.topk}")
    for i, (inst, score) in enumerate(top.items(), 1):
        name = name_map.get(inst, "")
        print(f"  {i:>2}. {inst}  {name}  {score:.4f}")
    print(f"\n已导出: {out_path}")
    print(f"导入: 同花顺 → 自选股 → 导入 → 选择 {out_name}")


if __name__ == "__main__":
    main()

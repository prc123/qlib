"""
基于 TopkDropoutStrategy 生成买卖信号。

从 daily_quant/ 目录中自动发现所有 pred_score_*.csv，按日期排序后逐一处理。
通过 Tushare 获取股票名称。

Usage
-----
    $ python daily_quant/trade_signals.py

Output
------
    trade_signals.csv  — 每日调仓信号（含股票名称）
"""

import os
import re
import pandas as pd
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent

TOPK = 10       # 持仓数量
N_DROP = 5      # 每日淘汰最差 N 只

TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"


def build_name_map() -> dict[str, str]:
    """Query Tushare stock_basic for all stock names, return {instrument: name}."""
    # Clear proxy so Tushare works without 127.0.0.1:7897 proxy
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)

    import tushare as ts
    pro = ts.pro_api(TUSHARE_TOKEN)
    df = pro.stock_basic(exchange="", list_status="L",
                         fields="ts_code,name")
    if df is None or df.empty:
        return {}
    name_map: dict[str, str] = {}
    for _, row in df.iterrows():
        ts_code = row["ts_code"]
        parts = ts_code.split(".")
        qlib_sym = f"{parts[1].upper()}{parts[0]}"
        name_map[qlib_sym] = row["name"]
    return name_map


def discover_pred_files() -> dict[str, Path]:
    """Scan CUR_DIR for ``pred_score_YYYY-MM-DD.csv``, return {date_str: path}."""
    pattern = re.compile(r"pred_score_(\d{4}-\d{2}-\d{2})\.csv$")
    result: dict[str, Path] = {}
    for f in sorted(CUR_DIR.glob("pred_score_*.csv")):
        m = pattern.match(f.name)
        if m:
            result[m.group(1)] = f
    return result


def load_scores(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index("instrument")
    return df["score"]


def run():
    pred_files = discover_pred_files()
    if len(pred_files) < 1:
        print("未找到 pred_score_*.csv 文件")
        return
    print(f"发现 {len(pred_files)} 个预测文件: {', '.join(sorted(pred_files.keys()))}")

    # 收集所有出现过的 instrument，查询名称
    all_instruments: set[str] = set()
    for path in pred_files.values():
        df = pd.read_csv(path)
        all_instruments.update(df.iloc[:, 0].astype(str).str.strip().tolist())

    print(f"查询 {len(all_instruments)} 只股票名称...")
    name_map = build_name_map()
    print(f"获取到 {len(name_map)} 只股票名称")

    dates = sorted(pred_files.keys())
    holdings: set[str] = set()
    records: list[dict] = []

    for pred_date, trade_date in zip(dates, dates[1:] + [None]):
        scores = load_scores(pred_files[pred_date]).sort_values(ascending=False)

        if not holdings:
            buy_list = list(scores.index[:TOPK])
            sell_list = []
        else:
            held_scores = scores.reindex(list(holdings)).sort_values()
            n_sell = min(N_DROP, len(held_scores))
            sell_list = list(held_scores.index[:n_sell])

            keep = holdings - set(sell_list)
            available = scores[~scores.index.isin(holdings)]
            n_buy = TOPK - len(keep)
            buy_list = list(available.index[:n_buy])
            holdings = keep

        holdings |= set(buy_list)
        holdings -= set(sell_list)

        next_trade_date = trade_date or (pd.Timestamp(pred_date) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

        for sym in buy_list:
            records.append({
                "date": pred_date,
                "trade_date": next_trade_date,
                "action": "BUY",
                "instrument": sym,
                "name": name_map.get(sym, ""),
                "score": round(float(scores[sym]), 6),
            })

        for sym in sell_list:
            records.append({
                "date": pred_date,
                "trade_date": next_trade_date,
                "action": "SELL",
                "instrument": sym,
                "name": name_map.get(sym, ""),
                "score": round(float(scores[sym]), 6),
            })

        for sym in sorted(holdings - set(buy_list)):
            records.append({
                "date": pred_date,
                "trade_date": next_trade_date,
                "action": "HOLD",
                "instrument": sym,
                "name": name_map.get(sym, ""),
                "score": round(float(scores[sym]), 6),
            })

    result = pd.DataFrame(records)
    out_path = CUR_DIR / "trade_signals.csv"
    result.to_csv(out_path, index=False)

    print(f"\n参数: topk={TOPK}, n_drop={N_DROP}")
    for date_key in dates:
        sub = result[result["date"] == date_key]
        buys = sub[sub["action"] == "BUY"]
        sells = sub[sub["action"] == "SELL"]
        holds = sub[sub["action"] == "HOLD"]
        next_td = sub.iloc[0]["trade_date"]
        print(f"\n=== {date_key} 预测 → {next_td} 交易 ===")
        if len(sells):
            items = [f"{r['instrument']}({r['name']})" for _, r in sells.iterrows()]
            print(f"  卖出 ({len(sells)}): {', '.join(items)}")
        items = [f"{r['instrument']}({r['name']})" for _, r in buys.iterrows()]
        print(f"  买入 ({len(buys)}): {', '.join(items)}")
        if len(holds):
            items = [f"{r['instrument']}({r['name']})" for _, r in holds.iterrows()]
            print(f"  持有 ({len(holds)}): {', '.join(items)}")
        print(f"  持仓数: {len(buys) + len(holds)}")

    print(f"\n信号已保存至: {out_path}")
    return result


if __name__ == "__main__":
    run()

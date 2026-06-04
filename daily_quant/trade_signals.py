"""
基于 TopkDropoutStrategy 生成买卖信号。

Usage
-----
    $ python daily_quant/trade_signals.py

Input
-----
    pred_score_2026-06-04.csv, pred_score_2026-06-05.csv

Output
------
    trade_signals.csv  — 每日调仓信号
"""

import pandas as pd
from pathlib import Path

CUR_DIR = Path(__file__).resolve().parent

TOPK = 10       # 持仓数量
N_DROP = 5      # 每日淘汰最差 N 只

# 预测文件：日期 -> 文件路径
PRED_FILES = {
    "2026-06-04": CUR_DIR / "pred_score_2026-06-04.csv",  # 6/4 预测 -> 6/5 交易
    "2026-06-05": CUR_DIR / "pred_score_2026-06-05.csv",  # 6/5 预测 -> 6/6 交易
}


def load_scores(date_str: str) -> pd.DataFrame:
    path = PRED_FILES[date_str]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.set_index("instrument")
    return df["score"]


def run():
    dates = sorted(PRED_FILES.keys())
    holdings: set[str] = set()
    records: list[dict] = []

    for pred_date, trade_date in zip(dates, dates[1:] + [None]):
        scores = load_scores(pred_date).sort_values(ascending=False)

        if not holdings:
            # Day 1: 初始建仓，买入 topk
            buy_list = list(scores.index[:TOPK])
            sell_list = []
        else:
            # 后续交易日：ranking-based dropout
            # 当前持仓按分数排序（分数低的 = 排名靠后 = 需要卖）
            held_scores = scores.reindex(list(holdings)).sort_values()
            n_sell = min(N_DROP, len(held_scores))
            sell_list = list(held_scores.index[:n_sell])

            # 从 top 中选新股补足 topk
            keep = holdings - set(sell_list)
            available = scores[~scores.index.isin(holdings)]
            n_buy = TOPK - len(keep)
            buy_list = list(available.index[:n_buy])
            holdings = keep

        holdings |= set(buy_list)
        holdings -= set(sell_list)

        next_trade_date = trade_date or (pd.Timestamp(pred_date) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

        # 买入
        for sym in buy_list:
            records.append({
                "date": pred_date,
                "trade_date": next_trade_date,
                "action": "BUY",
                "instrument": sym,
                "score": round(float(scores[sym]), 6),
            })

        # 卖出
        for sym in sell_list:
            records.append({
                "date": pred_date,
                "trade_date": next_trade_date,
                "action": "SELL",
                "instrument": sym,
                "score": round(float(scores[sym]), 6),
            })

        # 持有
        for sym in sorted(holdings - set(buy_list)):
            records.append({
                "date": pred_date,
                "trade_date": next_trade_date,
                "action": "HOLD",
                "instrument": sym,
                "score": round(float(scores[sym]), 6),
            })

    result = pd.DataFrame(records)
    out_path = CUR_DIR / "trade_signals.csv"
    result.to_csv(out_path, index=False)

    # 打印摘要
    print(f"参数: topk={TOPK}, n_drop={N_DROP}")
    for date_key in dates:
        sub = result[result["date"] == date_key]
        buys = sub[sub["action"] == "BUY"]
        sells = sub[sub["action"] == "SELL"]
        holds = sub[sub["action"] == "HOLD"]
        print(f"\n=== {date_key} 预测 → {sub.iloc[0]['trade_date']} 交易 ===")
        if len(sells):
            print(f"  卖出 ({len(sells)}): {', '.join(sells['instrument'].tolist())}")
        print(f"  买入 ({len(buys)}): {', '.join(buys['instrument'].tolist())}")
        if len(holds):
            print(f"  持有 ({len(holds)}): {', '.join(holds['instrument'].tolist())}")
        print(f"  持仓数: {len(buys) + len(holds)}")

    print(f"\n信号已保存至: {out_path}")
    return result


if __name__ == "__main__":
    run()

"""Analyze last 60 trading days from latest backtest."""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

exp_dir = Path("mlruns/948584480955148621")
recs = sorted(exp_dir.glob("*/artifacts/portfolio_analysis/report_normal_1day.pkl"),
              key=lambda p: p.stat().st_mtime, reverse=True)
report = pd.read_pickle(recs[0])

daily = report[["return", "bench", "cost"]].copy()
daily["excess"] = daily["return"] - daily["bench"]
daily["excess_cost"] = daily["excess"] - daily["cost"]

last60 = daily.tail(60).copy()
last60["cum_strat"] = (last60["return"] + 1).cumprod()
last60["cum_bench"] = (last60["bench"] + 1).cumprod()
last60["cum_excess"] = (last60["excess"] + 1).cumprod()
last60["dd"] = last60["cum_strat"] / last60["cum_strat"].cummax() - 1

r, b, e = last60["return"], last60["bench"], last60["excess"]
ann = 238

print(f"区间: {last60.index[0].strftime('%Y-%m-%d')} → {last60.index[-1].strftime('%Y-%m-%d')}  ({len(last60)}天)")
print()
print(f"{'指标':<16} {'策略':>10} {'沪深300':>10}")
print(f"{'-'*38}")
print(f"{'日均收益':<16} {r.mean()*100:>9.3f}% {b.mean()*100:>9.3f}%")
print(f"{'日波动率':<16} {r.std()*100:>9.3f}% {b.std()*100:>9.3f}%")
print(f"{'年化收益':<16} {r.mean()*ann*100:>9.2f}% {b.mean()*ann*100:>9.2f}%")
print(f"{'夏普比率':<16} {r.mean()/r.std()*ann**0.5:>9.2f} {b.mean()/b.std()*ann**0.5:>9.2f}")
print(f"{'最大回撤':<16} {last60['dd'].min()*100:>9.2f}%")
print(f"{'胜率(>0)':<16} {(r>0).mean()*100:>9.1f}%")
print(f"{'超额日均':<16} {e.mean()*100:>9.3f}%")
print(f"{'跑赢天数':<16} {(e>0).mean()*100:>9.1f}%")
print(f"{'累计超额':<16} {last60['cum_excess'].iloc[-1]-1:>9.2%}")
print(f"{'信息比率':<16} {e.mean()/e.std()*ann**0.5:>9.2f}")

# Group by week/biweek
print(f"\n{'='*55}")
print(f"{'周':<8} {'天数':>4} {'策略':>8} {'基准':>8} {'超额':>8} {'跑赢':>6}")
print(f"{'-'*45}")
last60["week"] = last60.index.isocalendar().week.astype(str)
for w, grp in last60.groupby("week"):
    print(f"  W{w:<6} {len(grp):>3}  {grp['return'].mean()*100:>+7.3f}% {grp['bench'].mean()*100:>+7.3f}% {grp['excess'].mean()*100:>+7.3f}% {(grp['excess']>0).mean():>5.0%}")

# Full table
print(f"\n{'='*55}")
print(f"{'Date':<12} {'策略%':>7} {'基准%':>7} {'超额%':>7} {'累计':>7} {'DD%':>6} {'W':>3}")
print(f"{'-'*55}")
for idx, row in last60.iterrows():
    w = "Y" if row["excess"] > 0 else "N"
    print(f"{idx.strftime('%m-%d'):<12} {row['return']*100:>+6.2f} {row['bench']*100:>+6.2f} {row['excess']*100:>+6.2f} {row['cum_strat']:>6.3f} {row['dd']*100:>+5.1f} {w:>3}")

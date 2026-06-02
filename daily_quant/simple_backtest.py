"""
Simple factor backtesting — no GRU, no training, just raw factor signals.

Usage
-----
    python simple_backtest.py                        # 默认：反转因子
    python simple_backtest.py --factor momentum      # 动量因子
    python simple_backtest.py --factor value         # 价值因子（低PE）
    python simple_backtest.py --factor reversal5     # 5日反转
    python simple_backtest.py --factor turnover      # 低换手率
    python simple_backtest.py --list                  # 列出所有可用因子
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import pandas as pd
import numpy as np
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data import D
from daily_quant.ops.date_ops import BoardLimit
from qlib.model.base import Model


# ============================================================================
# Simple factor models — each returns a score per (date, instrument) pair.
# Higher score = better expected return → TopkDropoutStrategy buys it.
# ============================================================================

class SimpleFactorModel(Model):
    """Base class for factor-based models.  Override ``_score()``."""

    def __init__(self, factor_name="unknown"):
        self.factor_name = factor_name
        self.fitted = True  # no training needed

    def _score(self, instruments, start, end, freq) -> pd.Series:
        """Return a Series indexed by (datetime, instrument) with factor values."""
        raise NotImplementedError

    def predict(self, dataset, segment="test"):
        """qlib's TopkDropoutStrategy calls model.predict(dataset)."""
        dl = dataset.prepare(segment, col_set=["feature", "label"])
        if dl.empty:
            raise ValueError("Empty test segment.")
        # Extract date range and instruments from the TSDataSampler index
        idx = dl.get_index()
        start = idx[0][0]
        end = idx[-1][0]
        instruments = sorted(idx.get_level_values("instrument").unique().tolist())
        return self._score(instruments, start, end, "day")

    def fit(self, *args, **kwargs):
        pass  # no training


class MomentumModel(SimpleFactorModel):
    """Buy recent winners: yesterday's return (Ref($close, -1)/$close - 1)."""
    def __init__(self):
        super().__init__("momentum")

    def _score(self, instruments, start, end, freq):
        return D.features(instruments, ["Ref($close, -1)/$close - 1"], start, end, freq).iloc[:, 0].rename("score")


class ReversalModel(SimpleFactorModel):
    """Buy recent losers: negative of yesterday's return (mean reversion)."""
    def __init__(self):
        super().__init__("reversal1")

    def _score(self, instruments, start, end, freq):
        s = D.features(instruments, ["Ref($close, -1)/$close - 1"], start, end, freq).iloc[:, 0]
        return -s.rename("score")


class Reversal5Model(SimpleFactorModel):
    """Buy 5-day losers: negative 5-day return."""
    def __init__(self):
        super().__init__("reversal5")

    def _score(self, instruments, start, end, freq):
        s = D.features(instruments, ["Ref($close, -5)/$close - 1"], start, end, freq).iloc[:, 0]
        return -s.rename("score")


class ValueModel(SimpleFactorModel):
    """Buy low PE stocks."""
    def __init__(self):
        super().__init__("value_pe")

    def _score(self, instruments, start, end, freq):
        pe = D.features(instruments, ["$pe_ttm"], start, end, freq).iloc[:, 0]
        return (-pe).rename("score")  # negative PE: lower PE -> higher score


class LowTurnoverModel(SimpleFactorModel):
    """Buy low turnover stocks (less speculative)."""
    def __init__(self):
        super().__init__("low_turnover")

    def _score(self, instruments, start, end, freq):
        t = D.features(instruments, ["$turnover_rate"], start, end, freq).iloc[:, 0]
        return (-t).rename("score")


class SizeModel(SimpleFactorModel):
    """Buy small cap stocks."""
    def __init__(self):
        super().__init__("small_cap")

    def _score(self, instruments, start, end, freq):
        mv = D.features(instruments, ["$total_mv"], start, end, freq).iloc[:, 0]
        return (-np.log(mv + 1)).rename("score")  # smaller market cap -> higher score


# ============================================================================
# Factor registry
# ============================================================================

FACTORS = {
    "reversal1":    (ReversalModel,    "1-day reversal (buy yesterday's losers)"),
    "momentum":     (MomentumModel,    "1-day momentum (buy yesterday's winners)"),
    "reversal5":    (Reversal5Model,   "5-day reversal (buy 5-day losers)"),
    "value":        (ValueModel,       "Low PE (buy cheap stocks)"),
    "turnover":     (LowTurnoverModel, "Low turnover (buy neglected stocks)"),
    "size":         (SizeModel,        "Small cap (buy small stocks)"),
}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Simple factor backtest")
    parser.add_argument("--factor", default="reversal1", choices=list(FACTORS.keys()),
                        help="Which factor to backtest")
    parser.add_argument("--instruments", default="mid_cap", help="Stock pool")
    parser.add_argument("--start", default="2025-07-01", help="Backtest start date")
    parser.add_argument("--end", default="2026-05-27", help="Backtest end date")
    parser.add_argument("--topk", type=int, default=30, help="How many stocks to hold")
    parser.add_argument("--n_drop", type=int, default=5, help="Minimum drop before replacing")
    parser.add_argument("--account", type=int, default=10_000_000, help="Initial capital")
    parser.add_argument("--list", action="store_true", help="List all factors and exit")
    args = parser.parse_args()

    if args.list:
        print("Available factors:")
        for name, (cls, desc) in FACTORS.items():
            print(f"  {name:12s}  {desc}")
        return

    model_cls, desc = FACTORS[args.factor]
    print(f"Factor: {args.factor} — {desc}")
    print(f"Instruments: {args.instruments}, TopK={args.topk}, N_Drop={args.n_drop}")
    print(f"Period: {args.start} ~ {args.end}")

    qlib.init(provider_uri=r"C:\Users\pp\.qlib\qlib_data\cn_data_10y", region=REG_CN,
              custom_ops=[BoardLimit])

    model = model_cls()

    # Minimal dataset — we only need the test segment for the date range
    from qlib.data.dataset import TSDatasetH
    dataset = TSDatasetH(
        handler={
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": args.start,
                "end_time": args.end,
                "fit_start_time": args.start,
                "fit_end_time": args.end,
                "instruments": args.instruments,
            },
        },
        segments={"test": (args.start, args.end)},
        step_len=1,  # only need 1 step for factor models
    )

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"model": model, "dataset": dataset, "topk": args.topk, "n_drop": args.n_drop},
        },
        "backtest": {
            "start_time": args.start,
            "end_time": args.end,
            "account": args.account,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "deal_price": "open", "open_cost": 0.0005,
                "close_cost": 0.0015, "min_cost": 5,
                "limit_threshold": (
                    "Greater($change, BoardLimit($close) * 0.98)",
                    "Less($change, -BoardLimit($close) * 0.98)",
                ),
            },
        },
    }

    exp_name = f"simple_{args.factor}_{args.instruments}"
    with R.start(experiment_name=exp_name):
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        rid = recorder.id

    # Show results
    recorder = R.get_recorder(recorder_id=rid, experiment_name=exp_name)
    report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    print("\n=== 基准 (沪深300) ===")
    print(report["bench"].describe())
    print(f"  年化: {report['bench'].mean()*238:.2%}")
    print(f"  夏普: {report['bench'].mean()/report['bench'].std()*238**0.5:.2f}")

    print("\n=== 策略 ===")
    print(report["return"].describe())
    print(f"  年化: {report['return'].mean()*238:.2%}")
    print(f"  夏普: {report['return'].mean()/report['return'].std()*238**0.5:.2f}")

    print("\n=== 超额收益 ===")
    excess = report["return"] - report["bench"]
    print(excess.describe())
    win_rate = (excess > 0).mean()
    print(f"  年化超额: {excess.mean()*238:.2%}")
    print(f"  跑赢胜率: {win_rate:.1%}")
    print(f"  最大回撤: {float(analysis.loc['excess_return_with_cost', 'max_drawdown']):.2%}")
    print(f"  信息比率: {float(excess.mean()/excess.std()*238**0.5):.2f}")


if __name__ == "__main__":
    main()

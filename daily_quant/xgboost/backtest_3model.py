"""
Backtest a heterogeneous ensemble: average predictions of multiple single models
(e.g. XGBoost + LightGBM + CatBoost).

Loads each model from its experiment name, averages their predictions, and runs
the standard TopK backtest.

Usage
-----
    python daily_quant/xgboost/backtest_3model.py --models XGB_Single,LGB_Single,CAT_Single
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.evaluate import risk_analysis
from qlib.data.dataset import DatasetH

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, dataset, segment="test"):
        preds = [m.predict(dataset, segment=segment) for m in self.models]
        return sum(preds) / len(preds)


def load_model(exp_name):
    recs = R.list_recorders(experiment_name=exp_name)
    recs = [r for r in recs.values() if "trained_model" in r.list_artifacts()]
    if not recs:
        raise ValueError(f"No trained model in '{exp_name}'")
    recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
    recorder = R.get_recorder(recorder_id=recs[0].id, experiment_name=exp_name)
    return recorder.load_object("trained_model")


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous model ensemble backtest")
    parser.add_argument("--models", default="XGB_Single,LGB_Single,CAT_Single", help="Comma-separated experiment names")
    parser.add_argument("--instruments", default="stock")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\etf_data")
    parser.add_argument("--backtest_start", default="2025-07-01")
    parser.add_argument("--backtest_end", default="2026-06-29")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n_drop", type=int, default=3)
    parser.add_argument("--benchmark", default="SH510050")
    parser.add_argument("--freq", default="day", choices=["day", "week"])
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=1)

    exp_names = args.models.split(",")
    models = [load_model(n) for n in exp_names]
    print(f"Loaded {len(models)} models: {exp_names}")

    ensemble = EnsembleModel(models)

    ds = DatasetH(
        handler={
            "class": "Alpha158ETF",
            "module_path": "daily_quant.handler.alpha158_etf",
            "kwargs": {
                "start_time": "2022-01-01",
                "end_time": args.backtest_end,
                "fit_start_time": "2022-01-01",
                "fit_end_time": "2024-12-31",
                "instruments": args.instruments,
                "use_alpha_factors": False,
                "label_type": "sharpe",
                "include_prem_disc": False,
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": [
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                ],
            },
        },
        segments={"test": (args.backtest_start, args.backtest_end)},
    )

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": args.freq, "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "BoardLimitTopkDropoutStrategy",
            "module_path": "daily_quant.strategies",
            "kwargs": {
                "model": ensemble, "dataset": ds,
                "topk": args.topk, "n_drop": args.n_drop,
                "check_limit": True,
            },
        },
        "backtest": {
            "start_time": args.backtest_start,
            "end_time": args.backtest_end,
            "account": 100_000_000,
            "benchmark": args.benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "deal_price": "open",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "limit_threshold": None,
            },
        },
    }

    with R.start(experiment_name="backtest_3model"):
        sr_recorder = R.get_recorder()
        ba_rid = sr_recorder.id
        sr = SignalRecord(ensemble, ds, sr_recorder)
        sr.generate()
        par = PortAnaRecord(sr_recorder, port_analysis_config, args.freq)
        par.generate()

    sr_recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_3model")
    report = sr_recorder.load_object(f"portfolio_analysis/report_normal_1{args.freq}.pkl")

    print("\n" + "=" * 60)
    print("=== 基准 ===")
    print(risk_analysis(report["bench"], freq=args.freq))
    print("\n=== 策略收益 ===")
    print(risk_analysis(report["return"], freq=args.freq))
    print("\n=== 超额收益（未扣费） ===")
    excess = report["return"] - report["bench"]
    print(risk_analysis(excess, freq=args.freq))
    print("\n=== 超额收益（扣费） ===")
    net = report["return"] - report["cost"]
    net_excess = net - report["bench"]
    print(risk_analysis(net_excess, freq=args.freq))

    ann = 238 if args.freq == "day" else 50
    print("\n=== 汇总 ===")
    print(f"  策略年化: {(report['return'].mean() * ann):.2%}")
    print(f"  超额扣费年化: {(net_excess.mean() * ann):.2%}  IR={net_excess.mean()/net_excess.std()*ann**0.5:.2f}")


if __name__ == "__main__":
    main()

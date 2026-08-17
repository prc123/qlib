"""
Backtest with k-fold ensemble: average predictions of K models.

Loads K models trained by train_kfold.py (experiments ``<prefix>_fold0..K``),
averages their predictions, and runs the standard backtest.

Usage
-----
    python daily_quant/xgboost/backtest_kfold.py --instruments stock --n_folds 3
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
from qlib.data import D

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


class EnsembleModel:
    """Average predictions of multiple models."""

    def __init__(self, models):
        self.models = models

    def predict(self, dataset, segment="test"):
        preds = [m.predict(dataset, segment=segment) for m in self.models]
        return sum(preds) / len(preds)


def main():
    parser = argparse.ArgumentParser(description="K-fold ensemble backtest")
    parser.add_argument("--instruments", default="stock")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\etf_data")
    parser.add_argument("--exp_prefix", default="XGB_KFold")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=None, help="Backtest a single fold only (0-indexed); default = ensemble of all folds")
    parser.add_argument("--backtest_exp", default="backtest_kfold")

    parser.add_argument("--backtest_start", default="2025-07-01")
    parser.add_argument("--backtest_end", default="2026-06-29")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n_drop", type=int, default=3)
    parser.add_argument("--account", type=int, default=100_000_000)
    parser.add_argument("--benchmark", default="SH510050")
    parser.add_argument("--freq", default="day", choices=["day", "week"])
    parser.add_argument("--no_limit", action="store_true")
    parser.add_argument("--deal_price", default="open", choices=["open", "close"])
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=4)
    print(f"qlib initialized, data: {args.qlib_data_dir}")

    # Load K models
    if args.fold is not None:
        fold_list = [args.fold]
    else:
        fold_list = list(range(args.n_folds))

    models = []
    tags = None
    for k in fold_list:
        exp_name = f"{args.exp_prefix}_fold{k}"
        recs = R.list_recorders(experiment_name=exp_name)
        recs = [r for r in recs.values() if "trained_model" in r.list_artifacts()]
        if not recs:
            raise ValueError(f"No model in '{exp_name}'. Run train_kfold.py first.")
        recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
        rid = recs[0].id
        recorder = R.get_recorder(recorder_id=rid, experiment_name=exp_name)
        model = recorder.load_object("trained_model")
        models.append(model)
        if tags is None:
            tags = recorder.client.get_run(rid).data.tags
        print(f"  fold {k}: model={rid[:16]}")

    handler_class = tags.get("handler_class", "Alpha158ETF")
    label_type = tags.get("label_type", "sharpe")
    share_only = tags.get("share_only", "True") == "True"
    is_base = handler_class == "Alpha158Base"
    print(f"Ensemble of {len(models)} models, handler={handler_class}, "
          f"label={label_type}, share_only={share_only}")

    ensemble = EnsembleModel(models)

    handler_kwargs = {
        "start_time": "2022-01-01",
        "end_time": args.backtest_end,
        "fit_start_time": "2022-01-01",
        "fit_end_time": "2024-12-31",
        "instruments": args.instruments,
        "label_type": label_type,
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
    }
    if not is_base:
        handler_kwargs.update({
            "use_alpha_factors": False,
            "include_prem_disc": not share_only,
        })

    ds = DatasetH(
        handler={
            "class": handler_class,
            "module_path": "daily_quant.handler.alpha158_etf",
            "kwargs": handler_kwargs,
        },
        segments={
            "test": (args.backtest_start, args.backtest_end),
        },
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
                "check_limit": not args.no_limit,
            },
        },
        "backtest": {
            "start_time": args.backtest_start,
            "end_time": args.backtest_end,
            "account": args.account,
            "benchmark": args.benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "deal_price": args.deal_price,
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "limit_threshold": None,
            },
        },
    }

    with R.start(experiment_name=args.backtest_exp):
        sr_recorder = R.get_recorder()
        ba_rid = sr_recorder.id
        sr = SignalRecord(ensemble, ds, sr_recorder)
        sr.generate()
        par = PortAnaRecord(sr_recorder, port_analysis_config, args.freq)
        par.generate()
        print(f"Backtest done, recorder_id: {ba_rid}")

    sr_recorder = R.get_recorder(recorder_id=ba_rid, experiment_name=args.backtest_exp)
    report = sr_recorder.load_object(f"portfolio_analysis/report_normal_1{args.freq}.pkl")

    print("\n" + "=" * 60)
    print("=== 基准 ===")
    print(risk_analysis(report["bench"], freq=args.freq))
    print("\n=== 策略收益 ===")
    print(risk_analysis(report["return"], freq=args.freq))
    print("\n=== 超额收益（未扣费） ===")
    excess = report["return"] - report["bench"]
    print(risk_analysis(excess, freq=args.freq))

    print("\n=== 汇总 ===")
    ann_factor = 238 if args.freq == "day" else 50
    strat_ret = report["return"]
    bench_ret = report["bench"]
    cum_strat = (strat_ret + 1).cumprod().iloc[-1] - 1
    dd = (strat_ret + 1).cumprod() / (strat_ret + 1).cumprod().cummax() - 1
    print(f"  策略: 年化={strat_ret.mean() * ann_factor:.2%}  "
          f"累计={cum_strat:.2%}  最大回撤={dd.min():.2%}")
    print(f"  基准: 年化={bench_ret.mean() * ann_factor:.2%}")
    print(f"  超额: 年化={excess.mean() * ann_factor:.2%}  "
          f"IR={excess.mean() / excess.std() * ann_factor ** 0.5:.2f}")


if __name__ == "__main__":
    main()

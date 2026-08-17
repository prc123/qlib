"""
CatBoost full workflow with flattened time-series features.

Key difference from GRU: CatBoost can't process (T, F) sequences directly.
FlattenedTSDatasetH converts each time window into statistical features
(mean, std, last, trend, min, max per raw feature), giving the tree model
access to temporal patterns.

Usage
-----
    python train_catboost.py
    python train_catboost.py --instruments csi300 --step_len 60 --topk 50
    python train_catboost.py --skip_train --model_id <id>
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.model.catboost_model import CatBoostModel
from qlib.contrib.evaluate import risk_analysis

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear
from daily_quant.flatten_ts_dataset import FlattenedTSDatasetH

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear]


def build_dataset(instruments, start, end, fit_start, fit_end, step_len, flatten_mode):
    """Build FlattenedTSDatasetH with Alpha158Date handler."""
    data_handler_config = {
        "start_time": start,
        "end_time": end,
        "fit_start_time": fit_start,
        "fit_end_time": fit_end,
        "instruments": instruments,
    }
    ds = FlattenedTSDatasetH(
        handler={
            "class": "Alpha158Date",
            "module_path": "daily_quant.handler.alpha158_date",
            "kwargs": data_handler_config,
        },
        segments={
            "train": (fit_start, fit_end),
            "valid": ("2025-01-01", "2025-06-30"),
            "test": ("2025-07-01", end),
        },
        step_len=step_len,
        flatten_mode=flatten_mode,
    )
    return ds, data_handler_config


def train_model(dataset, args, exp_name):
    """Train CatBoost and save to MLflow."""
    model_config = {
        "class": "CatBoostModel",
        "kwargs": {
            "loss": "RMSE",
            "learning_rate": args.lr,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "bootstrap_type": "Poisson",
            "thread_count": args.threads,
        },
    }
    model = CatBoostModel(**model_config["kwargs"])

    dataset_config = {
        "class": "FlattenedTSDatasetH",
        "kwargs": {
            "handler": {
                "class": "Alpha158Date",
                "module_path": "daily_quant.handler.alpha158_date",
            },
            "segments": {
                "train": ("2022-01-01", "2024-12-31"),
                "valid": ("2025-01-01", "2025-06-30"),
                "test": ("2025-07-01", args.backtest_end),
            },
            "step_len": args.step_len,
            "flatten_mode": args.flatten_mode,
        },
    }

    print(f"Experiment: {exp_name}")
    print(f"Params: lr={args.lr}, depth={args.max_depth}, rounds={args.num_boost_round}, "
          f"step_len={args.step_len}, mode={args.flatten_mode}")

    with R.start(experiment_name=exp_name):
        R.log_params(**flatten_dict({"model": model_config, "dataset": dataset_config}))
        model.fit(
            dataset,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stop,
            verbose_eval=args.verbose_eval,
        )
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        R.set_tags(
            status="completed",
            model_type="CatBoost_TS",
            instruments=args.instruments,
            step_len=str(args.step_len),
            train_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        print(f"Training done, recorder_id: {rid}")
    return rid


def run_backtest(model, dataset, args, exp_name, model_id):
    """Run backtest with TopkDropoutStrategy."""
    backtest_exp = "backtest_catboost_ts"
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
            "start_time": args.backtest_start,
            "end_time": args.backtest_end,
            "account": args.account,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "deal_price": "open",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "limit_threshold": 0.095,
            },
        },
    }

    with R.start(experiment_name=backtest_exp):
        recorder_train = R.get_recorder(recorder_id=model_id, experiment_name=exp_name)
        trained_model = recorder_train.load_object("trained_model")
        sr_recorder = R.get_recorder()
        ba_rid = sr_recorder.id
        sr = SignalRecord(trained_model, dataset, sr_recorder)
        sr.generate()
        par = PortAnaRecord(sr_recorder, port_analysis_config, "day")
        par.generate()
        print(f"Backtest done, recorder_id: {ba_rid}")
    return ba_rid, backtest_exp


def evaluate(ba_rid, backtest_exp):
    """Print full evaluation metrics."""
    recorder = R.get_recorder(recorder_id=ba_rid, experiment_name=backtest_exp)
    report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    print("\n" + "=" * 60)
    print("=== 基准收益 (沪深300) ===")
    print(risk_analysis(report["bench"], freq="1day"))

    print("\n=== 超额收益分析 ===")
    print(analysis)

    print("\n=== 策略 vs HS300 汇总 ===")
    strat_ret = report["return"]
    bench_ret = report["bench"]
    excess = strat_ret - bench_ret

    cum_strat = (strat_ret + 1).cumprod().iloc[-1] - 1
    cum_bench = (bench_ret + 1).cumprod().iloc[-1] - 1

    summary = pd.DataFrame(
        {
            "策略": {
                "累计收益": f"{cum_strat:.2%}",
                "年化收益": f"{strat_ret.mean() * 238:.2%}",
                "波动率": f"{strat_ret.std():.2%}",
                "夏普": f"{strat_ret.mean() / strat_ret.std() * 238 ** 0.5:.2f}",
                "最大回撤": f"{((strat_ret + 1).cumprod() / (strat_ret + 1).cumprod().cummax() - 1).min():.2%}",
                "胜率": f"{(strat_ret > 0).mean():.1%}",
                "跑赢胜率": f"{(excess > 0).mean():.1%}",
            },
            "HS300": {
                "累计收益": f"{cum_bench:.2%}",
                "年化收益": f"{bench_ret.mean() * 238:.2%}",
                "波动率": f"{bench_ret.std():.2%}",
                "夏普": f"{bench_ret.mean() / bench_ret.std() * 238 ** 0.5:.2f}",
                "最大回撤": f"{((bench_ret + 1).cumprod() / (bench_ret + 1).cumprod().cummax() - 1).min():.2%}",
                "胜率": f"{(bench_ret > 0).mean():.1%}",
                "跑赢胜率": "-",
            },
        }
    ).T
    print(summary.to_string())

    print(f"\n年化超额(未扣费): {excess.mean() * 238:.2%}")
    print(f"年化超额(扣费):   {(excess - report.get('cost', pd.Series(0, index=excess.index))).mean() * 238:.2%}")
    print(f"信息比率:         {excess.mean() / excess.std() * 238 ** 0.5:.2f}")
    print(f"跑赢胜率:         {(excess > 0).mean():.1%}")


def main():
    parser = argparse.ArgumentParser(description="CatBoost full workflow with time-series features")
    parser.add_argument("--instruments", default="mid_cap", help="Stock pool")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_10y")
    parser.add_argument("--exp_name", default=None, help="Experiment name")

    # Time-series params
    parser.add_argument("--step_len", type=int, default=20, help="Time window in trading days")
    parser.add_argument("--flatten_mode", default="stats",
                        choices=["stats", "full"], help="stats=6 stats per feat; full=all time steps")

    # CatBoost params
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--num_boost_round", type=int, default=2000)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--verbose_eval", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)

    # Backtest params
    parser.add_argument("--backtest_start", default="2025-07-01")
    parser.add_argument("--backtest_end", default="2026-05-27")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n_drop", type=int, default=5)
    parser.add_argument("--account", type=int, default=10_000_000)

    # Control
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--model_id", default=None, help="Model recorder ID for --skip_train")
    args = parser.parse_args()

    # --- Init qlib ---
    qlib.init(
        provider_uri=args.qlib_data_dir,
        region=REG_CN,
        custom_ops=_CUSTOM_OPS,
    )
    print(f"qlib initialized, data: {args.qlib_data_dir}")

    # --- Build dataset ---
    dataset, handler_cfg = build_dataset(
        instruments=args.instruments,
        start="2022-01-01",
        end=args.backtest_end,
        fit_start="2022-01-01",
        fit_end="2024-12-31",
        step_len=args.step_len,
        flatten_mode=args.flatten_mode,
    )

    train_df = dataset.prepare("train", col_set=["feature", "label"], data_key="learn")
    valid_df = dataset.prepare("valid", col_set=["feature", "label"], data_key="learn")
    n_feats = len([c for c in train_df.columns if isinstance(c, tuple) and c[0] == "feature"])
    print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
    print(f"Raw features: 170, Flat features: {n_feats} (step_len={args.step_len}, mode={args.flatten_mode})")

    # --- Train ---
    exp_name = args.exp_name or f"CatBoost_TS_{args.instruments}_{args.step_len}d"
    if args.skip_train:
        if not args.model_id:
            recs_dict = R.list_recorders(experiment_name=exp_name)
            recs = [r for r in recs_dict.values() if "trained_model" in r.list_artifacts()]
            if not recs:
                raise ValueError(f"No trained model found in '{exp_name}'")
            recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
            model_id = recs[0].id
        else:
            model_id = args.model_id
        print(f"Using existing model: {model_id}")
        recorder = R.get_recorder(recorder_id=model_id, experiment_name=exp_name)
        model = recorder.load_object("trained_model")
    else:
        model_id = train_model(dataset, args, exp_name)
        recorder = R.get_recorder(recorder_id=model_id, experiment_name=exp_name)
        model = recorder.load_object("trained_model")

    # --- Backtest ---
    ba_rid, backtest_exp = run_backtest(model, dataset, args, exp_name, model_id)

    # --- Evaluate ---
    evaluate(ba_rid, backtest_exp)


if __name__ == "__main__":
    main()

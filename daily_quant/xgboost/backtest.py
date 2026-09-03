"""
Backtest evaluation for XGBoost model (DatasetH + Alpha158Date).

Parameters referenced from:
    examples/benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml

Usage
-----
    python backtest_xgboost.py
    python backtest_xgboost.py --model_id <id> --topk 50
    python backtest_xgboost.py --instruments csi300 --exp_name XGB_Alpha158_csi300
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


def main():
    parser = argparse.ArgumentParser(description="Backtest XGBoost model (DatasetH + Alpha158Date)")

    parser.add_argument("--model_id", default=None, help="Model recorder ID")
    parser.add_argument("--exp_name", default=None, help="Training experiment name")
    parser.add_argument("--backtest_exp", default="backtest_xgboost", help="Backtest experiment name")

    parser.add_argument("--instruments", default="csi500", help="Stock pool")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd")

    parser.add_argument("--backtest_start", default="2025-07-01")
    parser.add_argument("--backtest_end", default="2026-06-29")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n_drop", type=int, default=5)
    parser.add_argument("--account", type=int, default=100_000_000)
    parser.add_argument("--benchmark", default="SH000300")
    parser.add_argument("--no_limit", action="store_true")
    parser.add_argument("--deal_price", default="open", choices=["open", "close"])
    parser.add_argument("--freq", default="day", choices=["day", "week"], help="Rebalance frequency")

    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=1)
    print(f"qlib initialized, data: {args.qlib_data_dir}")

    exp_name = args.exp_name or f"XGB_Alpha158_{args.instruments}"
    print(f"Experiment: {exp_name}")

    # --- Load model ---
    recs_dict = R.list_recorders(experiment_name=exp_name)
    recs = [r for r in recs_dict.values() if "trained_model" in r.list_artifacts()]
    if not recs:
        raise ValueError(f"No trained model found in '{exp_name}'. Run train_xgboost.py first.")
    recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
    model_id = args.model_id or recs[0].id
    print(f"Model: {model_id}")

    recorder = R.get_recorder(recorder_id=model_id, experiment_name=exp_name)
    model = recorder.load_object("trained_model")

    # Read tags to match training config
    client_tags = recorder.client.get_run(model_id).data.tags
    use_alpha = client_tags.get("use_alpha_factors", "False") == "True"
    handler_class = client_tags.get("handler_class", "Alpha158Date")
    is_multi = handler_class == "Alpha158DateMultiHorizon"
    share_only = client_tags.get("share_only", "False") == "True"
    label_type = client_tags.get("label_type", "return")
    handler_module = "daily_quant.handler.alpha158_etf" if handler_class == "Alpha158ETF" else "daily_quant.handler.alpha158_date"
    print(f"alpha={use_alpha}, multi_horizon={is_multi}, share_only={share_only}, label_type={label_type}, handler={handler_class}")

    if is_multi:
        learn_processors = [{"class": "DropnaLabel"}]
    else:
        learn_processors = [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]

    handler_extra = {}
    if handler_class == "Alpha158ETF":
        handler_extra["include_prem_disc"] = not share_only
    if handler_class in ("Alpha158ETF", "Alpha158Date"):
        handler_extra["label_type"] = label_type

    ds = DatasetH(
        handler={
            "class": handler_class,
            "module_path": handler_module,
            "kwargs": {
                "start_time": "2022-01-01",
                "end_time": args.backtest_end,
                "fit_start_time": "2022-01-01",
                "fit_end_time": "2024-12-31",
                "instruments": args.instruments,
                "use_alpha_factors": use_alpha,
                **handler_extra,
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": learn_processors,
            },
        },
        segments={
            "train": ("2022-01-01", "2024-12-31"),
            "valid": ("2025-01-01", "2025-06-30"),
            "test": (args.backtest_start, args.backtest_end),
        },
    )
    print(f"Backtest period: {args.backtest_start} -> {args.backtest_end}")

    # --- Quick diagnosis ---
    print("\n" + "=" * 60)
    print("=== 诊断：首日预测 ===")
    pred_all = model.predict(ds)
    first_day = pred_all.index.get_level_values("datetime").min()
    pred_day = pred_all.loc[pred_all.index.get_level_values("datetime") == first_day]
    print(f"日期: {first_day.strftime('%Y-%m-%d')}, 预测数: {len(pred_day)}")
    print(f"Score: min={pred_day.min():.4f}, max={pred_day.max():.4f}, "
          f"mean={pred_day.mean():.4f}, std={pred_day.std():.4f}")
    print(f"Score > 0: {int((pred_day > 0).sum())}, NaN: {int(pred_day.isna().sum())}")

    print(f"\nTop 10:")
    for (dt, inst), val in pred_day.nlargest(10).items():
        print(f"  {inst}  {val:.6f}")

    top5 = [inst for (dt, inst) in pred_day.nlargest(5).index]
    opens = D.features(top5, ["$open", "$close"],
                       start_time=first_day.strftime("%Y-%m-%d"),
                       end_time=first_day.strftime("%Y-%m-%d"))
    print(f"\nTop5 $open/$close on {first_day.strftime('%Y-%m-%d')}:")
    for inst in top5:
        try:
            row = opens.loc[(inst, slice(None)), :]
            if len(row) > 0:
                r = row.iloc[0]
                print(f"  {inst}  open={float(r['$open']):.3f}  close={float(r['$close']):.3f}")
            else:
                print(f"  {inst}  NO DATA")
        except Exception:
            print(f"  {inst}  ERROR")

    # --- Backtest ---
    print("\n" + "=" * 60)
    print("=== 回测 ===")
    limit_info = "DISABLED" if args.no_limit else "BoardLimit (5/10/20/30%)"
    print(f"limit checks: {limit_info}")

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
                "model": model, "dataset": ds,
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
        sr = SignalRecord(model, ds, sr_recorder)
        sr.generate()
        par = PortAnaRecord(sr_recorder, port_analysis_config, args.freq)
        par.generate()
        print(f"Backtest done, recorder_id: {ba_rid}")

    # --- Results ---
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
    cum_bench = (bench_ret + 1).cumprod().iloc[-1] - 1
    dd = (strat_ret + 1).cumprod() / (strat_ret + 1).cumprod().cummax() - 1

    print(f"  策略: 年化={strat_ret.mean() * ann_factor:.2%}  "
          f"累计={cum_strat:.2%}  "
          f"夏普={strat_ret.mean() / strat_ret.std() * ann_factor ** 0.5:.2f}  "
          f"最大回撤={dd.min():.2%}")
    print(f"  基准: 年化={bench_ret.mean() * ann_factor:.2%}  "
          f"累计={cum_bench:.2%}")
    print(f"  跑赢天数: {(excess > 0).mean():.1%}")
    print(f"  策略胜率: {(strat_ret > 0).mean():.1%}")
    print(f"  信息比率: {excess.mean() / excess.std() * ann_factor ** 0.5:.2f}")


if __name__ == "__main__":
    main()

"""
Evaluate model ranking quality using IC (Information Coefficient).

IC = cross-sectional rank correlation between predicted scores and actual
forward returns.  This is the standard metric in quant finance — RMSE is
irrelevant for a TopK stock selection strategy.

Usage
-----
    python daily_quant/xgboost/eval_ic.py --instruments stock
    python daily_quant/xgboost/eval_ic.py --exp_name XGB_Alpha158_stock
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.data.dataset import DatasetH
from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


def evaluate_model(exp_name, instruments, qlib_data_dir, use_alpha=False, period="valid"):
    """Calculate IC metrics for a trained model."""
    qlib.init(provider_uri=qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=4)

    recs_dict = R.list_recorders(experiment_name=exp_name)
    recs = [r for r in recs_dict.values() if "trained_model" in r.list_artifacts()]
    if not recs:
        raise ValueError(f"No trained model in '{exp_name}'")
    recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
    rid = recs[0].id
    model = R.get_recorder(recorder_id=rid, experiment_name=exp_name).load_object("trained_model")

    client_tags = R.get_recorder(recorder_id=rid, experiment_name=exp_name).client.get_run(rid).data.tags
    handler_class = client_tags.get("handler_class", "Alpha158Date")

    if period == "valid":
        start, end = "2025-01-01", "2025-06-30"
    else:
        start, end = "2025-07-01", "2026-06-29"

    ds = DatasetH(
        handler={
            "class": handler_class,
            "module_path": "daily_quant.handler.alpha158_date",
            "kwargs": {
                "start_time": "2022-01-01", "end_time": end,
                "fit_start_time": "2022-01-01", "fit_end_time": "2024-12-31",
                "instruments": instruments,
                "use_alpha_factors": use_alpha,
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": [{"class": "DropnaLabel"}],
            },
        },
        segments={"test": (start, end)},
    )

    pred = model.predict(ds)
    df = pred.reset_index()
    df.columns = ["date", "instrument", "score"]

    # Also get actual labels for the same period
    label_ds = DatasetH(
        handler={
            "class": handler_class,
            "module_path": "daily_quant.handler.alpha158_date",
            "kwargs": {
                "start_time": start, "end_time": end,
                "fit_start_time": start, "fit_end_time": end,
                "instruments": instruments,
                "use_alpha_factors": use_alpha,
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": [{"class": "DropnaLabel"}],
            },
        },
        segments={"label": (start, end)},
    )
    label_df = label_ds.prepare("label", col_set="label", data_key="learn")
    label_vals = label_df.values
    if label_vals.ndim == 2:
        label_vals = label_vals.ravel()

    pred_index = label_df.index
    pred_series = pd.Series(label_vals, index=pred_index)

    # Merge predictions and labels
    label_s = pred_series.reset_index()
    label_s.columns = ["date", "instrument", "actual"]
    merged = df.merge(label_s, on=["date", "instrument"])

    # Daily IC
    daily_ic = []
    daily_rank_ic = []
    for date, group in merged.groupby("date"):
        if len(group) < 10:
            continue
        ic, _ = pearsonr(group["score"], group["actual"])
        ric, _ = spearmanr(group["score"], group["actual"])
        daily_ic.append({"date": date, "ic": ic, "rank_ic": ric})

    ic_df = pd.DataFrame(daily_ic)
    mean_ic = ic_df["ic"].mean()
    std_ic = ic_df["ic"].std()
    icir = mean_ic / std_ic if std_ic > 0 else 0
    mean_rank_ic = ic_df["rank_ic"].mean()
    std_rank_ic = ic_df["rank_ic"].std()
    rank_icir = mean_rank_ic / std_rank_ic if std_rank_ic > 0 else 0

    # TopK cumulative return
    topk_returns = []
    for date, group in merged.groupby("date"):
        top10 = group.nlargest(10, "score")
        topk_returns.append(top10["actual"].mean())
    cum_topk = np.cumprod(1 + np.array(topk_returns))[-1] - 1

    # Benchmark: equal-weight all stocks
    all_returns = merged.groupby("date")["actual"].mean().values
    cum_all = np.cumprod(1 + all_returns)[-1] - 1

    print(f"\n{'='*60}")
    print(f"Model: {exp_name} (id={rid[:16]}...)")
    print(f"Period: {start} → {end}, {len(ic_df)} trading days")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Value':>12}")
    print(f"{'-'*32}")
    print(f"{'Mean IC (Pearson)':<20} {mean_ic:>12.4f}")
    print(f"{'Std IC':<20} {std_ic:>12.4f}")
    print(f"{'ICIR (Mean/Std)':<20} {icir:>12.4f}")
    print(f"{'Mean Rank IC':<20} {mean_rank_ic:>12.4f}")
    print(f"{'Std Rank IC':<20} {std_rank_ic:>12.4f}")
    print(f"{'Rank ICIR':<20} {rank_icir:>12.4f}")
    print(f"{'IC > 0 ratio':<20} {(ic_df['ic'] > 0).mean():>12.1%}")
    print(f"{'IC > 0.02 ratio':<20} {(ic_df['ic'] > 0.02).mean():>12.1%}")
    print(f"{'Cum Top10 return':<20} {cum_topk:>12.2%}")
    print(f"{'Cum Equal-weight':<20} {cum_all:>12.2%}")
    print(f"{'Top10 - Bench':<20} {cum_topk - cum_all:>12.2%}")

    return {"icir": icir, "rank_icir": rank_icir, "mean_ic": mean_ic,
            "cum_top10": cum_topk, "cum_bench": cum_all}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="XGB_Alpha158_stock")
    parser.add_argument("--instruments", default="stock")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\etf_data")
    parser.add_argument("--period", default="valid", choices=["valid", "test"])
    args = parser.parse_args()
    evaluate_model(args.exp_name, args.instruments, args.qlib_data_dir, period=args.period)


if __name__ == "__main__":
    main()

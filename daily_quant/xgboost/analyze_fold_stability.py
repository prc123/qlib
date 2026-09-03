"""
Analyze k-fold model stability: how much do the K fold models disagree?

Loads the K fold models, predicts each on the SAME test segment, and computes
the cross-fold standard deviation of the prediction scores for every
(date, instrument). A low std means the folds agree (stable), a high std means
the prediction is sensitive to the training window (unstable).

Also reports pairwise correlation (Pearson + Spearman) between fold predictions.

Usage
-----
    python daily_quant/xgboost/analyze_fold_stability.py --exp_prefix XGB_OOS
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
os.environ["NO_PROXY"] = "*"

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.data.dataset import DatasetH

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


def main():
    parser = argparse.ArgumentParser(description="Analyze k-fold prediction stability")
    parser.add_argument("--instruments", default="stock")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\etf_data")
    parser.add_argument("--exp_prefix", default="XGB_OOS")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--start", default="2025-07-01")
    parser.add_argument("--end", default="2026-06-29")
    parser.add_argument("--lookback", default="2022-01-01")
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=1)

    # Load K models
    models = []
    tags = None
    for k in range(args.n_folds):
        exp_name = f"{args.exp_prefix}_fold{k}"
        recs = R.list_recorders(experiment_name=exp_name)
        recs = [r for r in recs.values() if "trained_model" in r.list_artifacts()]
        if not recs:
            raise ValueError(f"No model in '{exp_name}'")
        recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
        recorder = R.get_recorder(recorder_id=recs[0].id, experiment_name=exp_name)
        models.append(recorder.load_object("trained_model"))
        if tags is None:
            tags = recorder.client.get_run(recs[0].id).data.tags
        print(f"  fold {k}: {recs[0].id[:16]}")

    handler_class = tags.get("handler_class", "Alpha158ETF")
    label_type = tags.get("label_type", "sharpe")
    share_only = tags.get("share_only", "True") == "True"
    is_base = handler_class == "Alpha158Base"
    print(f"{args.n_folds} models loaded, handler={handler_class}, label={label_type}")

    handler_kwargs = {
        "start_time": args.lookback,
        "end_time": args.end,
        "fit_start_time": args.lookback,
        "fit_end_time": args.start,
        "instruments": args.instruments,
        "label_type": label_type,
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": [{"class": "DropnaLabel"}],
    }
    if not is_base:
        handler_kwargs.update({
            "use_alpha_factors": False,
            "include_prem_disc": not share_only,
        })

    ds = DatasetH(
        handler={"class": handler_class, "module_path": "daily_quant.handler.alpha158_etf", "kwargs": handler_kwargs},
        segments={"test": (args.start, args.end)},
    )

    # Predict with each fold model
    pred_dfs = []
    for i, m in enumerate(models):
        p = m.predict(ds).reset_index()
        p.columns = ["date", "instrument", f"fold{i}"]
        pred_dfs.append(p)

    merged = pred_dfs[0]
    for p in pred_dfs[1:]:
        merged = merged.merge(p, on=["date", "instrument"], how="inner")

    fold_cols = [f"fold{i}" for i in range(args.n_folds)]
    merged["std"] = merged[fold_cols].std(axis=1)
    merged["mean"] = merged[fold_cols].mean(axis=1)

    print("\n" + "=" * 60)
    print(f"跨 fold 预测值标准差分布 (n={len(merged)} 个 date×instrument 样本)")
    print("=" * 60)
    print(merged["std"].describe().to_string())

    print("\n=== fold 两两相关系数 ===")
    for i in range(args.n_folds):
        for j in range(i + 1, args.n_folds):
            pearson = merged[f"fold{i}"].corr(merged[f"fold{j}"])
            spearman = merged[f"fold{i}"].corr(merged[f"fold{j}"], method="spearman")
            print(f"  fold{i} vs fold{j}: Pearson={pearson:.4f}  Spearman={spearman:.4f}")

    # Per-date stability (mean std over time)
    daily_std = merged.groupby("date")["std"].mean()
    print("\n=== 每日平均 std（前5 / 后5 个交易日）===")
    for d, v in list(daily_std.items())[:5]:
        print(f"  {d}: {v:.4f}")
    print("  ...")
    for d, v in list(daily_std.items())[-5:]:
        print(f"  {d}: {v:.4f}")

    # Rank stability: within each date, Spearman between fold0 and fold1
    rank_corr = []
    for date, g in merged.groupby("date"):
        if len(g) < 20:
            continue
        rank_corr.append(g["fold0"].corr(g["fold1"], method="spearman"))
    rank_corr = pd.Series(rank_corr)
    print(f"\n=== 逐日 Spearman 排名相关 (fold0 vs fold1) ===")
    print(f"  均值={rank_corr.mean():.4f}  中位数={rank_corr.median():.4f}  最小={rank_corr.min():.4f}  最大={rank_corr.max():.4f}")

    # === 相关系数：预测值与真实未来收益的 IC ===
    from scipy.stats import spearmanr, pearsonr
    from qlib.data import D

    inst_file = Path(args.qlib_data_dir) / "instruments" / f"{args.instruments}.txt"
    inst = pd.read_csv(inst_file, sep="\t", header=None, names=["symbol", "start", "end"])
    inst["start"] = inst["start"].astype(str)
    inst["end"] = inst["end"].astype(str)
    active = inst[(inst["end"] >= args.start) & (inst["start"] <= args.end)]["symbol"].tolist()

    ret = D.features(active, ["Ref($close, -5)/$close - 1"],
                     start_time=args.start, end_time=args.end, freq="day")
    ret_df = ret.reset_index()
    ret_df.columns = ["instrument", "date", "fwd_ret"]
    ret_df["date"] = pd.to_datetime(ret_df["date"]).dt.strftime("%Y-%m-%d")

    merged_ic = merged.copy()
    merged_ic["date"] = pd.to_datetime(merged_ic["date"]).dt.strftime("%Y-%m-%d")
    merged_ic = merged_ic.merge(ret_df, on=["date", "instrument"], how="inner")
    merged_ic["ensemble"] = merged_ic[fold_cols].mean(axis=1)

    print("\n" + "=" * 60)
    print("IC 相关系数：预测值 vs 真实 5 日未来收益（逐日截面）")
    print("=" * 60)
    for col in fold_cols + ["ensemble"]:
        daily_ic, daily_ric = [], []
        for date, g in merged_ic.groupby("date"):
            if len(g) < 10:
                continue
            ic, _ = pearsonr(g[col], g["fwd_ret"])
            ric, _ = spearmanr(g[col], g["fwd_ret"])
            daily_ic.append(ic)
            daily_ric.append(ric)
        daily_ic = np.array(daily_ic)
        daily_ric = np.array(daily_ric)
        icir = daily_ic.mean() / daily_ic.std() if daily_ic.std() > 0 else 0
        ricir = daily_ric.mean() / daily_ric.std() if daily_ric.std() > 0 else 0
        tag = "(集成)" if col == "ensemble" else ""
        print(f"  {col:<8} {tag}: Pearson IC={daily_ic.mean():.4f} (ICIR={icir:.2f})   "
              f"Rank IC={daily_ric.mean():.4f} (ICIR={ricir:.2f})")


if __name__ == "__main__":
    main()

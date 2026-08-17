"""
K-fold (walk-forward) training for XGBoost ETF model.

Trains K models using time-series walk-forward split (expanding window),
which respects temporal ordering and avoids look-ahead bias.  Each model
is saved under experiment ``<exp_prefix>_fold{k}`` for later ensembling.

Usage
-----
    python daily_quant/xgboost/train_kfold.py --instruments stock --n_folds 3
    python daily_quant/xgboost/train_kfold.py --n_folds 5 --label_type sharpe
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.utils import flatten_dict
from qlib.workflow import R
from qlib.data.dataset import DatasetH
from qlib.contrib.model.xgboost import XGBModel

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


def compute_folds(train_start, valid_end, n_folds):
    """Walk-forward fold boundaries: expanding train window, next block as valid.

    Returns list of (train_end, valid_start, valid_end) for each fold.
    """
    t0 = pd.Timestamp(train_start)
    t1 = pd.Timestamp(valid_end)
    total_days = (t1 - t0).days
    block_days = total_days // (n_folds + 1)

    folds = []
    for k in range(n_folds):
        train_end = t0 + pd.Timedelta(days=(k + 1) * block_days)
        valid_start = train_end + pd.Timedelta(days=1)
        valid_end_k = t0 + pd.Timedelta(days=(k + 2) * block_days)
        folds.append((train_end, valid_start, valid_end_k))
    return folds


def main():
    parser = argparse.ArgumentParser(description="K-fold XGBoost ETF training")
    parser.add_argument("--instruments", default="stock")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\etf_data")
    parser.add_argument("--exp_prefix", default="XGB_KFold")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--label_type", default="sharpe", choices=["return", "sharpe", "ret_vol", "score"])
    parser.add_argument("--share_only", action="store_true", default=True)
    parser.add_argument("--base_alpha158", action="store_true", help="Use vanilla Alpha158 (no fund/date/ETF factors)")
    parser.add_argument("--model_type", default="xgboost", choices=["xgboost", "lightgbm", "catboost"], help="Model type")

    parser.add_argument("--train_start", default="2022-01-01")
    parser.add_argument("--valid_end", default="2025-06-30")
    parser.add_argument("--test_start", default="2025-07-01")
    parser.add_argument("--test_end", default="2026-06-29")

    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample_bytree", type=float, default=0.85)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--reg_alpha", type=float, default=0.0)
    parser.add_argument("--num_boost_round", type=int, default=1000)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=4)

    if args.base_alpha158:
        handler_class = "Alpha158Base"
        handler_module = "daily_quant.handler.alpha158_etf"
    else:
        handler_class = "Alpha158ETF"
        handler_module = "daily_quant.handler.alpha158_etf"
    learn_processors = [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ]

    folds = compute_folds(args.train_start, args.valid_end, args.n_folds)
    print(f"Fold boundaries (train_end, valid_start, valid_end):")
    for k, (te, vs, ve) in enumerate(folds):
        print(f"  fold {k}: train [{args.train_start}, {te.date()}]  valid [{vs.date()}, {ve.date()}]")

    for k, (train_end, valid_start, valid_end) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {k}/{args.n_folds-1}: train [{args.train_start}, {train_end.date()}], "
              f"valid [{valid_start.date()}, {valid_end.date()}]")

        handler_kwargs = {
            "start_time": args.train_start,
            "end_time": valid_end,
            "fit_start_time": args.train_start,
            "fit_end_time": train_end,
            "instruments": args.instruments,
            "label_type": args.label_type,
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": learn_processors,
        }
        if not args.base_alpha158:
            handler_kwargs.update({
                "use_alpha_factors": False,
                "include_prem_disc": not args.share_only,
            })

        ds = DatasetH(
            handler={
                "class": handler_class,
                "module_path": handler_module,
                "kwargs": handler_kwargs,
            },
            segments={
                "train": (args.train_start, train_end),
                "valid": (valid_start, valid_end),
            },
        )

        if args.model_type == "xgboost":
            model = XGBModel(
                eval_metric="rmse",
                eta=args.lr, max_depth=args.max_depth,
                subsample=args.subsample, colsample_bytree=args.colsample_bytree,
                reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha,
                nthread=args.threads, tree_method="hist",
            )
        elif args.model_type == "lightgbm":
            from qlib.contrib.model.gbdt import LGBModel
            model = LGBModel(
                loss="mse",
                learning_rate=args.lr,
                num_leaves=2 ** args.max_depth - 1,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                lambda_l2=args.reg_lambda,
                num_threads=args.threads,
            )
        elif args.model_type == "catboost":
            from qlib.contrib.model.catboost_model import CatBoostModel
            model = CatBoostModel(
                loss="RMSE",
                learning_rate=args.lr,
                depth=args.max_depth,
                l2_leaf_reg=args.reg_lambda,
                thread_count=args.threads,
            )

        exp_name = f"{args.exp_prefix}_fold{k}"
        with R.start(experiment_name=exp_name):
            R.log_params(**flatten_dict({
                "fold": k, "label_type": args.label_type,
                "share_only": args.share_only,
                "train_end": str(train_end.date()),
                "valid_start": str(valid_start.date()),
                "valid_end": str(valid_end.date()),
            }))
            model.fit(ds, num_boost_round=args.num_boost_round,
                      early_stopping_rounds=args.early_stop, verbose_eval=50)
            R.save_objects(trained_model=model)
            rid = R.get_recorder().id
            R.set_tags(
                status="completed", model_type="XGBoost",
                instruments=args.instruments, handler_class=handler_class,
                label_type=args.label_type, share_only=str(args.share_only),
                fold=str(k), total_feat=str(157 if args.base_alpha158 else (171 if args.share_only else 172)),
            )
            print(f"Fold {k} done, recorder_id: {rid}")

    print(f"\nAll {args.n_folds} folds trained. Ensemble backtest via backtest_kfold.py")


if __name__ == "__main__":
    main()

"""
CatBoost training with DatasetH + Alpha158Date (single-day cross-sectional, 170 features).

Each sample = one stock on one date, no time-series window.
Tree models don't need feature normalization — raw Alpha158 features go directly to CatBoost.

Parameters referenced from:
    examples/benchmarks/CatBoost/workflow_config_catboost_Alpha158.yaml

Usage
-----
    python train_catboost_date.py
    python train_catboost_date.py --instruments csi300 --num_boost_round 3000
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
from qlib.contrib.model.catboost_model import CatBoostModel

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


def main():
    parser = argparse.ArgumentParser(description="CatBoost training with Alpha158Date (170-feat, DatasetH)")

    # Data
    parser.add_argument("--instruments", default="csi500", help="Stock pool: csi500, csi300, mid_cap, all")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd")
    parser.add_argument("--exp_name", default=None, help="Experiment name")

    # Date range
    parser.add_argument("--train_start", default="2022-01-01")
    parser.add_argument("--train_end", default="2024-12-31")
    parser.add_argument("--valid_start", default="2025-01-01")
    parser.add_argument("--valid_end", default="2025-06-30")
    parser.add_argument("--test_start", default="2025-07-01")
    parser.add_argument("--test_end", default="2025-12-31")

    # CatBoost params (from Alpha158 YAML)
    parser.add_argument("--lr", type=float, default=0.0421)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8789)
    parser.add_argument("--num_leaves", type=int, default=100)
    parser.add_argument("--grow_policy", default="Lossguide")
    parser.add_argument("--bootstrap_type", default="Poisson")
    parser.add_argument("--num_boost_round", type=int, default=2000)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--verbose_eval", type=int, default=50)
    parser.add_argument("--threads", type=int, default=20)

    args = parser.parse_args()

    # --- Init ---
    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS)
    print(f"qlib initialized, data: {args.qlib_data_dir}")

    # --- Build DatasetH (single-day, no time window) ---
    ds = DatasetH(
        handler={
            "class": "Alpha158Date",
            "module_path": "daily_quant.handler.alpha158_date",
            "kwargs": {
                "start_time": args.train_start,
                "end_time": args.test_end,
                "fit_start_time": args.train_start,
                "fit_end_time": args.train_end,
                "instruments": args.instruments,
            },
        },
        segments={
            "train": (args.train_start, args.train_end),
            "valid": (args.valid_start, args.valid_end),
            "test": (args.test_start, args.test_end),
        },
    )

    train_df = ds.prepare("train", col_set=["feature", "label"], data_key="learn")
    valid_df = ds.prepare("valid", col_set=["feature", "label"], data_key="learn")
    n_feats = len([c for c in train_df.columns if isinstance(c, tuple) and c[0] == "feature"])
    print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
    print(f"Features: {n_feats} (Alpha158Date: kbar + price + rolling + 6 fund + 7 date/board)")

    # --- Model ---
    model_config = {
        "class": "CatBoostModel",
        "kwargs": {
            "loss": "RMSE",
            "learning_rate": args.lr,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
            "num_leaves": args.num_leaves,
            "grow_policy": args.grow_policy,
            "bootstrap_type": args.bootstrap_type,
            "thread_count": args.threads,
        },
    }

    dataset_config = {
        "class": "DatasetH",
        "kwargs": {
            "handler": {"class": "Alpha158Date", "module_path": "daily_quant.handler.alpha158_date"},
            "segments": {
                "train": (args.train_start, args.train_end),
                "valid": (args.valid_start, args.valid_end),
                "test": (args.test_start, args.test_end),
            },
        },
    }

    model = CatBoostModel(**model_config["kwargs"])

    exp_name = args.exp_name or f"CatBoost_Alpha158_{args.instruments}"
    print(f"\nExperiment: {exp_name}")
    print(f"Model: loss=RMSE, lr={args.lr}, depth={args.max_depth}, leaves={args.num_leaves}")
    print(f"       subsample={args.subsample}, grow={args.grow_policy}, bootstrap={args.bootstrap_type}")
    print(f"       rounds={args.num_boost_round}, early_stop={args.early_stop}")

    with R.start(experiment_name=exp_name):
        R.log_params(**flatten_dict({"model": model_config, "dataset": dataset_config}))
        model.fit(
            ds,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stop,
            verbose_eval=args.verbose_eval,
        )
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        R.set_tags(
            status="completed",
            model_type="CatBoost",
            instruments=args.instruments,
            train_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        print(f"Training done, recorder_id: {rid}")


if __name__ == "__main__":
    main()

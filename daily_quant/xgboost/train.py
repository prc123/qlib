"""
XGBoost training with Alpha158Date handler (170/180 features).

Usage
-----
    python daily_quant/xgboost/train.py --instruments CSI300
    python daily_quant/xgboost/train.py --instruments CSI300 --use_alpha_factors --use_v2
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


def _train_with_ic_early_stop(model, ds, args, train_df, valid_df):
    """Train XGBoost with Rank IC for early stopping (keep MSE loss).

    Computes daily Rank IC on validation set at each eval interval,
    stops when IC stops improving (same as early_stopping_rounds logic).
    """
    import numpy as np
    import xgboost as xgb
    from scipy.stats import spearmanr

    x_train = train_df["feature"].values
    y_train = train_df["label"].values.ravel()
    x_valid = valid_df["feature"].values
    y_valid = valid_df["label"].values.ravel()

    # Compute daily groups for IC calculation
    valid_dates = valid_df.index.get_level_values("datetime")
    unique_dates, valid_groups = np.unique(valid_dates, return_counts=True)

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)

    params = dict(model._params)
    params.update({"objective": "reg:squarederror", "verbosity": 0,
                   "eval_metric": "rmse"})

    best_ic, best_round, stop_rounds = -np.inf, 0, 0
    best_model_bytes = None
    evals_result = {}
    ic_eval_interval = 5
    ic_patience = max(5, args.early_stop // 10)  # fewer checks needed for IC

    for r in range(1, args.num_boost_round + 1):
        model.model = xgb.train(
            params, dtrain, num_boost_round=1,
            xgb_model=model.model, evals=[(dtrain, "train"), (dvalid, "valid")],
            evals_result=evals_result, verbose_eval=False,
        )

        if r % ic_eval_interval == 0 or r == 1:
            pred = model.model.predict(dvalid)
            daily_ric = []
            off = 0
            for gs in valid_groups:
                if gs < 10:
                    off += gs; continue
                ric, _ = spearmanr(pred[off:off + gs], y_valid[off:off + gs])
                daily_ric.append(ric)
                off += gs
            mean_ric = np.mean(daily_ric) if daily_ric else 0

            train_rmse = evals_result["train"]["rmse"][-1]
            valid_rmse = evals_result["valid"]["rmse"][-1]
            print(f"[{r:4d}] train-rmse:{train_rmse:.4f} valid-rmse:{valid_rmse:.4f} valid-rank-ic:{mean_ric:.4f}")

            if mean_ric > best_ic:
                best_ic, best_round, stop_rounds = mean_ric, r, 0
                best_model_bytes = model.model.save_raw()
            else:
                stop_rounds += 1
                if stop_rounds >= ic_patience:
                    print(f"IC early stop @ round {r} (best IC={best_ic:.4f} @ {best_round})")
                    break

    # Restore best IC model
    if best_model_bytes is not None:
        import xgboost as xgb
        model.model = xgb.Booster(model_file=best_model_bytes)
    model.fitted = True
    print(f"Best Rank IC: {best_ic:.4f} @ round {best_round}")


def main():
    parser = argparse.ArgumentParser(description="XGBoost with Alpha158Date")
    parser.add_argument("--instruments", default="CSI300")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--use_alpha_factors", action="store_true")
    parser.add_argument("--use_v2", action="store_true", help="Use Alpha158DateV2 with group norm")
    parser.add_argument("--multi_horizon", action="store_true", help="Multi-horizon labels: T+1, T+3, T+5")
    parser.add_argument("--ranking_loss", action="store_true", help="Use pairwise ranking loss instead of MSE")
    parser.add_argument("--ic_early_stop", action="store_true", help="Use Rank IC for early stopping (keep MSE loss)")
    parser.add_argument("--etf_factors", action="store_true", help="Use Alpha158ETF handler (share + nav factors)")
    parser.add_argument("--label_type", default="return", choices=["return", "sharpe", "ret_vol", "score"], help="Label formulation ('sharpe' works for stocks and ETFs; ret_vol/score require --etf_factors)")
    parser.add_argument("--share_only", action="store_true", help="Only use share factor, drop premium/discount (requires --etf_factors)")
    parser.add_argument("--model_type", default="xgboost", choices=["xgboost", "lightgbm", "catboost"], help="Model type")

    parser.add_argument("--train_start", default="2022-01-01")
    parser.add_argument("--train_end", default="2024-12-31")
    parser.add_argument("--valid_start", default="2025-01-01")
    parser.add_argument("--valid_end", default="2025-06-30")
    parser.add_argument("--test_start", default="2025-07-01")
    parser.add_argument("--test_end", default="2026-05-27")

    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample_bytree", type=float, default=0.85)
    parser.add_argument("--lambda", type=float, default=1.0, dest="reg_lambda")
    parser.add_argument("--alpha", type=float, default=0.0, dest="reg_alpha")
    parser.add_argument("--num_boost_round", type=int, default=1000)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--verbose_eval", type=int, default=50)
    parser.add_argument("--threads", type=int, default=16)

    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=4)
    print(f"qlib initialized")

    TOTAL_FEAT = 180 if args.use_alpha_factors else 170

    if args.multi_horizon:
        handler_class = "Alpha158DateMultiHorizon"
        handler_module = "daily_quant.handler.alpha158_date"
        learn_processors = [{"class": "DropnaLabel"}]
    elif args.ranking_loss:
        handler_class = "Alpha158Date"
        handler_module = "daily_quant.handler.alpha158_date"
        learn_processors = [{"class": "DropnaLabel"}]  # raw returns, ranking loss handles scale
    elif args.etf_factors:
        handler_class = "Alpha158ETF"
        handler_module = "daily_quant.handler.alpha158_etf"
        learn_processors = [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]
    else:
        handler_class = "Alpha158Date"
        handler_module = "daily_quant.handler.alpha158_date"
        learn_processors = [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]

    handler_kwargs = {
        "start_time": args.train_start,
        "end_time": args.test_end,
        "fit_start_time": args.train_start,
        "fit_end_time": args.train_end,
        "instruments": args.instruments,
        "use_alpha_factors": args.use_alpha_factors,
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        "learn_processors": learn_processors,
    }
    if args.label_type != "return" and not args.multi_horizon and not args.ranking_loss:
        handler_kwargs["label_type"] = args.label_type
    if args.etf_factors and args.share_only:
        handler_kwargs["include_prem_disc"] = False

    ds = DatasetH(
        handler={
            "class": handler_class,
            "module_path": handler_module,
            "kwargs": handler_kwargs,
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
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Features: {n_feats}")

    if args.multi_horizon:
        from daily_quant.xgboost.model_multi_horizon import MultiHorizonXGBModel
        model = MultiHorizonXGBModel(
            eval_metric="rmse",
            eta=args.lr, max_depth=args.max_depth,
            subsample=args.subsample, colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha,
            nthread=args.threads, tree_method="hist",
        )
    elif args.ranking_loss:
        from daily_quant.xgboost.model_ranking import RankingXGBModel
        model = RankingXGBModel(
            objective="rank:pairwise",
            eta=args.lr, max_depth=args.max_depth,
            subsample=args.subsample, colsample_bytree=args.colsample_bytree,
            reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha,
            nthread=args.threads, tree_method="hist",
        )
    else:
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

    exp_name = args.exp_name or f"XGB_Alpha158_{args.instruments}"
    if args.multi_horizon:
        exp_name = exp_name.replace("XGB_Alpha158", "XGB_MultiH")
    if args.ranking_loss:
        exp_name = exp_name.replace("XGB_Alpha158", "XGB_Rank")
    if args.ic_early_stop:
        exp_name = exp_name.replace("XGB_Alpha158", "XGB_IC")
    if args.etf_factors:
        exp_name = exp_name.replace("XGB_Alpha158", "XGB_ETF")
    if args.label_type != "return":
        exp_name = exp_name.replace("XGB_ETF", f"XGB_ETF_{args.label_type}")
        exp_name = exp_name.replace("XGB_Alpha158", f"XGB_{args.label_type}")
    if args.share_only:
        exp_name = exp_name.replace("XGB_ETF", "XGB_ETF_share")
    print(f"Experiment: {exp_name}")
    print(f"XGBoost: depth={args.max_depth}, lr={args.lr}, rounds={args.num_boost_round}")

    with R.start(experiment_name=exp_name):
        R.log_params(**flatten_dict({
            "model": {"class": "XGBModel"},
            "handler": handler_class,
            "features": n_feats,
            "alpha_factors": args.use_alpha_factors,
            "ic_early_stop": args.ic_early_stop,
        }))
        if args.ic_early_stop:
            _train_with_ic_early_stop(model, ds, args, train_df, valid_df)
        elif args.ranking_loss:
            model.fit(ds, num_boost_round=args.num_boost_round,
                      early_stopping_rounds=args.early_stop, verbose_eval=args.verbose_eval)
        else:
            model.fit(ds, num_boost_round=args.num_boost_round,
                      early_stopping_rounds=args.early_stop, verbose_eval=args.verbose_eval)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        R.set_tags(
            status="completed", model_type="XGBoost",
            instruments=args.instruments,
            handler_class=handler_class,
            use_alpha_factors=str(args.use_alpha_factors),
            total_feat=str(n_feats),
            label_type=args.label_type,
            share_only=str(args.share_only),
            train_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        print(f"Done, recorder_id: {rid}")


if __name__ == "__main__":
    main()

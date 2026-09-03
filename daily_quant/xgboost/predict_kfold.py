"""
Predict next-period TopK ETFs using k-fold ensemble model.

Loads K fold models (trained up to the latest data), predicts scores for all
ETFs, and outputs the TopK picks as actionable trading signals.

Usage
-----
    python daily_quant/xgboost/predict_kfold.py --topk 10
    python daily_quant/xgboost/predict_kfold.py --exp_prefix XGB_Current --n_folds 3
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
os.environ["NO_PROXY"] = "*"
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.data.dataset import DatasetH

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]

TRACKING_URI = "file:" + str(Path(__file__).resolve().parents[2] / "mlruns_daily")


class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, dataset, segment="test"):
        preds = [m.predict(dataset, segment=segment) for m in self.models]
        return sum(preds) / len(preds)


def get_etf_names():
    """Map qlib symbol -> ETF name via Tushare fund_basic."""
    try:
        import tushare as ts
        pro = ts.pro_api("a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202")
        df = pro.fund_basic(market="E")
        name_map = {}
        for _, row in df.iterrows():
            code, exchange = row["ts_code"].split(".")
            sym = f"{exchange.lower()}{code}".upper()
            name_map[sym] = row["name"]
        return name_map
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser(description="Predict next-period TopK ETFs")
    parser.add_argument("--instruments", default="stock")
    parser.add_argument("--qlib_data_dir",
                        default=str(Path.home() / ".qlib" / "qlib_data" / "etf_data"))
    parser.add_argument("--exp_prefix", default="XGB_KFold")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--lookback", default="2022-01-01", help="Feature start time")
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=1)
    R.set_uri(TRACKING_URI)

    # Load latest data date from calendar
    cal = pd.read_csv(Path(args.qlib_data_dir) / "calendars" / "day.txt")
    latest_date = str(cal.iloc[-1, 0])
    print(f"Latest data date: {latest_date}")

    # Load K models
    models = []
    tags = None
    for k in range(args.n_folds):
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
        print(f"  fold {k}: {rid[:16]}")

    handler_class = tags.get("handler_class", "Alpha158ETF")
    label_type = tags.get("label_type", "sharpe")
    share_only = tags.get("share_only", "True") == "True"
    print(f"Ensemble of {len(models)} models, label={label_type}, share_only={share_only}")

    ensemble = EnsembleModel(models)

    # Build dataset up to latest date (for prediction only)
    ds = DatasetH(
        handler={
            "class": handler_class,
            "module_path": "daily_quant.handler.alpha158_etf",
            "kwargs": {
                "start_time": args.lookback,
                "end_time": latest_date,
                "fit_start_time": args.lookback,
                "fit_end_time": latest_date,
                "instruments": args.instruments,
                "use_alpha_factors": False,
                "label_type": label_type,
                "include_prem_disc": not share_only,
                "infer_processors": [
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": [{"class": "DropnaLabel"}],
            },
        },
        segments={"test": (latest_date, latest_date)},
    )

    # Predict scores on latest date
    pred = ensemble.predict(ds)
    df = pred.reset_index()
    df.columns = ["date", "instrument", "score"]
    df = df.sort_values("score", ascending=False)

    # Get ETF names
    name_map = get_etf_names()

    # 输出到专门文件夹，文件名带日期
    out_dir = Path(__file__).resolve().parent / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pred_{latest_date}.csv"

    # 保存全部 ETF 评分
    out_df = df.copy()
    out_df["rank"] = range(1, len(out_df) + 1)
    out_df["name"] = out_df["instrument"].map(name_map)
    out_df["predict_date"] = latest_date
    out_df = out_df[["predict_date", "rank", "instrument", "name", "score"]]
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 显示全部 ETF 评分
    print(f"\n{'='*60}")
    print(f"预测日期: {latest_date}  全部 {len(df)} 只 ETF 评分")
    print(f"{'='*60}")
    print(f"{'排名':<5} {'代码':<12} {'名称':<32} {'预测分数':>10}")
    print("-" * 60)
    for i, (_, row) in enumerate(df.iterrows(), 1):
        sym = row["instrument"]
        name = name_map.get(sym, "")
        print(f"{i:<5} {sym:<12} {name:<32} {row['score']:>10.4f}")

    print(f"\n已保存全部评分到: {out_path}")


if __name__ == "__main__":
    main()

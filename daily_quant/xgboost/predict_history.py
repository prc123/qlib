"""
Generate historical daily prediction scores, one CSV per trading day.

Loads the K-fold ensemble model (same as predict_kfold.py), predicts the score
for every trading day in [start, end], and saves each day as a separate CSV
under predictions/history/ named pred_YYYY-MM-DD.csv with columns:
    predict_date, rank, instrument, name, score

Usage
-----
    python daily_quant/xgboost/predict_history.py --start 2025-08-14
    python daily_quant/xgboost/predict_history.py --start 2025-08-14 --end 2026-08-13
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
os.environ["NO_PROXY"] = "*"

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.data.dataset import DatasetH

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


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
    parser = argparse.ArgumentParser(description="Generate historical ETF prediction scores")
    parser.add_argument("--instruments", default="stock")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\etf_data")
    parser.add_argument("--exp_prefix", default="XGB_Current")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--start", default=None, help="Start date (default: 1 year before latest)")
    parser.add_argument("--end", default=None, help="End date (default: latest data date)")
    parser.add_argument("--lookback", default="2022-01-01", help="Feature lookback start")
    parser.add_argument("--out_dir", default="history", help="Output subdirectory under predictions/")
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=4)

    cal = pd.read_csv(Path(args.qlib_data_dir) / "calendars" / "day.txt")
    latest_date = str(cal.iloc[-1, 0])
    end_date = args.end or latest_date
    if args.start is None:
        start_date = str((pd.Timestamp(latest_date) - pd.DateOffset(years=1)).date())
    else:
        start_date = args.start
    print(f"Predicting [{start_date}, {end_date}] (latest data: {latest_date})")

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
    is_base = handler_class == "Alpha158Base"
    print(f"Ensemble of {len(models)} models, label={label_type}, share_only={share_only}")

    ensemble = EnsembleModel(models)

    handler_kwargs = {
        "start_time": args.lookback,
        "end_time": end_date,
        "fit_start_time": args.lookback,
        "fit_end_time": start_date,
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
        handler={
            "class": handler_class,
            "module_path": "daily_quant.handler.alpha158_etf",
            "kwargs": handler_kwargs,
        },
        segments={"test": (start_date, end_date)},
    )

    pred = ensemble.predict(ds)
    df = pred.reset_index()
    df.columns = ["date", "instrument", "score"]
    df["date"] = df["date"].astype(str)
    df = df.sort_values(["date", "score"], ascending=[True, False])
    df["rank"] = df.groupby("date")["score"].rank(ascending=False, method="first").astype(int)

    name_map = get_etf_names()

    out_dir = Path(__file__).resolve().parent / "predictions" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for date, group in df.groupby("date"):
        out_df = group.copy()
        out_df["predict_date"] = date
        out_df["name"] = out_df["instrument"].map(name_map)
        out_df = out_df[["predict_date", "rank", "instrument", "name", "score"]]
        out_df.to_csv(out_dir / f"pred_{date}.csv", index=False, encoding="utf-8-sig")

    n_days = df["date"].nunique()
    print(f"\nSaved {n_days} daily CSVs -> {out_dir}")


if __name__ == "__main__":
    main()

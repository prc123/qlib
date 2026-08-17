"""
Daily prediction pipeline:
1. Check if qlib data is up-to-date (last calendar day vs today)
2. If not, pull new data from Tushare and update qlib binary
3. Load the trained model and generate predictions

Usage
-----
    python predict.py
    python predict.py --update         # update data first, then predict
    python predict.py --model_id f6a7bc45e25b481ca295eee21322d259
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import copy

# 确保项目根目录在 sys.path 中（从任意目录运行 predict.py 都能 import daily_quant）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset
from qlib.utils import get_or_create_path
from qlib.contrib.model.pytorch_gru_ts import GRU
from qlib.data.dataset import TSDatasetH

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit
_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


# --- 自定义模型类（与 notebook 中的 GRUWithProgress 完全一致，保证 pickle 能反序列化） ---
class GRUWithProgress(GRU):
    """GRU 模型 + tqdm 进度条"""
    def fit(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")

        wl_train = np.ones(len(dl_train)) if reweighter is None else reweighter.reweight(dl_train)
        wl_valid = np.ones(len(dl_valid)) if reweighter is None else reweighter.reweight(dl_valid)

        train_loader = DataLoader(ConcatDataset(dl_train, wl_train),
                                  batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.n_jobs, drop_last=True)
        valid_loader = DataLoader(ConcatDataset(dl_valid, wl_valid),
                                  batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.n_jobs, drop_last=True)

        save_path = get_or_create_path(save_path)
        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        best_param = copy.deepcopy(self.GRU_model.state_dict())
        evals_result["train"], evals_result["valid"] = [], []
        self.logger.info("training...")
        self.fitted = True

        from tqdm import tqdm
        pbar = tqdm(range(self.n_epochs), desc="Training", unit="epoch")
        for step in pbar:
            self.train_epoch(train_loader)
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            R.log_metrics(train_score=train_score, valid_score=val_score, step=step)

            if val_score > best_score:
                best_score, stop_steps, best_epoch = val_score, 0, step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    pbar.set_description(f"Early stop @ epoch {step}")
                    break

            pbar.set_postfix({"train": f"{train_score:.4f}", "valid": f"{val_score:.4f}", "best": f"{best_score:.4f}"})

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()


def get_latest_calendar_date(qlib_dir: Path) -> pd.Timestamp:
    """Read the last (most recent) date from qlib's trading calendar."""
    cal_path = qlib_dir / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"Calendar not found: {cal_path}")
    cal = pd.read_csv(cal_path)
    return pd.Timestamp(cal.iloc[-1, 0])


def is_data_fresh(qlib_dir: Path) -> bool:
    """Check if the last calendar date is recent enough (consider weekends/holidays)."""
    last_date = get_latest_calendar_date(qlib_dir)
    today = pd.Timestamp.now().normalize()
    # Count business days between last trading date and today.
    # If 0 (weekend / just updated), data is fresh.
    bdays = pd.bdate_range(last_date, today, inclusive="right")
    return len(bdays) == 0


def update_data(qlib_dir: str):
    """Run daily_update.py to fetch new data and rebuild qlib binary."""
    update_script = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "data_collector" / "tushare" / "daily_update.py"
    )
    if not update_script.exists():
        raise FileNotFoundError(f"Update script not found: {update_script}")

    import subprocess
    import os
    # Clear proxy env vars so Tushare API calls go directly (not through
    # a non-running proxy like 127.0.0.1:7897).
    env = os.environ.copy()
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        env.pop(k, None)
    python = sys.executable
    cmd = [
        python, str(update_script),
        "--qlib_data_1d_dir", qlib_dir,
        "--delay", "0.3",
    ]
    print(f"[update] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(update_script.parent), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Data update failed with code {result.returncode}")
    print("[update] Data update completed.")


def refresh_instruments(qlib_dir: Path, instruments: str = None):
    """Extend end dates in instrument files to match the latest calendar date.

    Stock data may be updated but ``all.txt`` and index-specific files
    (mid_cap.txt etc.) can lag behind the calendar. This uses the calendar
    as the ground truth: any stock whose end date matches the previous
    latest trading date gets extended to the current latest.
    """
    instr_dir = qlib_dir / "instruments"
    all_path = instr_dir / "all.txt"
    cal_path = qlib_dir / "calendars" / "day.txt"
    if not all_path.exists() or not cal_path.exists():
        return

    cal = pd.read_csv(cal_path)
    last_cal_date = str(cal.iloc[-1, 0])

    all_df = pd.read_csv(all_path, sep="\t", header=None, names=["symbol", "start", "end"])
    all_max_end = all_df["end"].max()

    all_end_map = dict(zip(all_df["symbol"].astype(str), all_df["end"]))

    # Update all.txt if needed
    if str(all_max_end) < last_cal_date:
        all_changed = 0
        for idx, row in all_df.iterrows():
            if str(row["end"]) == str(all_max_end):
                all_df.at[idx, "end"] = last_cal_date
                all_changed += 1
        if all_changed:
            all_df.to_csv(all_path, sep="\t", header=False, index=False)
            print(f"[refresh] Updated {all_changed} stocks in all.txt")
            all_end_map = dict(zip(all_df["symbol"].astype(str), all_df["end"]))

    # Always update other instrument files (they may lag behind all.txt)
    targets = [f"{instruments}.txt"] if instruments else None
    for f in sorted(instr_dir.glob("*.txt")):
        if f.name == "all.txt":
            continue
        if targets and f.name not in targets:
            continue
        df = pd.read_csv(f, sep="\t", header=None, names=["symbol", "start", "end"])
        changed = 0
        for idx, row in df.iterrows():
            sym = str(row["symbol"])
            if str(row["end"]) != last_cal_date and sym in all_end_map:
                if str(all_end_map[sym]) >= last_cal_date:
                    df.at[idx, "end"] = last_cal_date
                    changed += 1
                elif str(row["end"]) != str(all_end_map[sym]):
                    df.at[idx, "end"] = all_end_map[sym]
                    changed += 1
        if changed:
            df.to_csv(f, sep="\t", header=False, index=False)
            print(f"[refresh] Updated {changed} stocks in {f.name}")


# --- 风险评估（结合预测分数 + qlib 股价数据） ---

def assess_risk(pred_df, instruments, pred_date_str, qlib_path, model_id, exp_name):
    """Compute risk signals from predictions + market data."""
    from qlib.data import D

    scores = pred_df.droplevel("datetime") if "datetime" in pred_df.index.names else pred_df
    if isinstance(scores, pd.DataFrame):
        scores = scores.iloc[:, -1]

    risk = {"date": pred_date_str, "n_stocks": len(scores)}

    # ---- Prediction-based signals ----
    risk["score_mean"] = float(scores.mean())
    risk["score_std"] = float(scores.std())
    risk["score_min"] = float(scores.min())
    risk["score_max"] = float(scores.max())
    risk["neg_ratio"] = float((scores < 0).mean())
    top30 = scores.nlargest(30)
    risk["top30_mean"] = float(top30.mean())

    # ---- Market-based signals (from qlib data) ----
    try:
        inst_list = D.instruments(instruments)
        # Recent N days of market data
        end_dt = pd.Timestamp(pred_date_str)
        start_dt = end_dt - pd.DateOffset(days=120)
        fields = ["$close", "$volume", "$change"]
        mkt = D.features(inst_list, fields,
                         start_time=start_dt.strftime("%Y-%m-%d"),
                         end_time=end_dt.strftime("%Y-%m-%d"))
    except Exception:
        mkt = None

    if mkt is not None and len(mkt) > 0:
        # Last 20 trading days
        all_dates = sorted(mkt.index.get_level_values("datetime").unique())
        recent = all_dates[-20:] if len(all_dates) >= 20 else all_dates

        # Today only
        today_mask = mkt.index.get_level_values("datetime") == pd.Timestamp(pred_date_str)
        today_data = mkt[today_mask] if today_mask.any() else mkt.loc[mkt.index.get_level_values("datetime") == all_dates[-1]]

        # Market breadth: % stocks with positive change today
        if "$change" in today_data.columns:
            chg = today_data["$change"].dropna()
            risk["breadth"] = float((chg > 0).mean()) if len(chg) > 0 else None
        else:
            risk["breadth"] = None

        # Cross-sectional volatility
        if "$change" in today_data.columns:
            risk["cross_vol"] = float(today_data["$change"].dropna().std()) if len(chg) > 0 else None
        else:
            risk["cross_vol"] = None

        # Volume trend: avg volume over last 5 days vs last 20 days
        if "$volume" in mkt.columns:
            vol_by_day = mkt["$volume"].groupby("datetime").mean()
            vol_recent = vol_by_day.iloc[-5:]
            vol_base = vol_by_day.iloc[-20:]
            if len(vol_base) >= 10:
                risk["vol_ratio"] = float(vol_recent.mean() / vol_base.mean())
            else:
                risk["vol_ratio"] = None
        else:
            risk["vol_ratio"] = None

        # Market trend: 5-day vs 20-day average change
        if "$change" in mkt.columns:
            ret_by_day = mkt["$change"].groupby("datetime").mean()
            risk["mkt_ret_5d"] = float(ret_by_day.iloc[-5:].mean()) if len(ret_by_day) >= 5 else None
            risk["mkt_ret_20d"] = float(ret_by_day.iloc[-20:].mean()) if len(ret_by_day) >= 20 else None
        else:
            risk["mkt_ret_5d"] = None
            risk["mkt_ret_20d"] = None

    # ---- Model staleness ----
    try:
        rec = R.get_recorder(recorder_id=model_id, experiment_name=exp_name)
        train_end = rec.info.get("end_time")
        if train_end:
            risk["model_age_days"] = (pd.Timestamp.now() - pd.Timestamp(train_end)).days
    except Exception:
        risk["model_age_days"] = None

    # ---- Data freshness ----
    cal_path = Path(qlib_path) / "calendars" / "day.txt"
    if cal_path.exists():
        cal = pd.read_csv(cal_path)
        last_cal = pd.Timestamp(cal.iloc[-1, 0])
        bdays = pd.bdate_range(last_cal, pd.Timestamp.now().normalize(), inclusive="right")
        risk["data_age_days"] = len(bdays)
    else:
        risk["data_age_days"] = None

    # ---- Decision ----
    warnings = 0
    if risk["score_std"] is not None and risk["score_std"] < 0.2:
        warnings += 1
    if risk["neg_ratio"] is not None and risk["neg_ratio"] > 0.70:
        warnings += 1
    if risk["top30_mean"] is not None and risk["top30_mean"] < 0.5:
        warnings += 1
    if risk.get("breadth") is not None and risk["breadth"] < 0.30:
        warnings += 1
    if risk.get("cross_vol") is not None and risk["cross_vol"] > 0.05:
        warnings += 1
    if risk.get("model_age_days") is not None and risk["model_age_days"] > 180:
        warnings += 1
    if risk.get("data_age_days") is not None and risk["data_age_days"] > 2:
        warnings += 1
    risk["warnings"] = warnings
    risk["level"] = "RED" if warnings >= 5 else ("YELLOW" if warnings >= 3 else "GREEN")

    return risk


def print_risk_report(risk):
    """Print risk assessment results."""
    lvl = risk["level"]
    print(f"\n{'='*60}")
    print(f"  Risk Assessment @ {risk['date']}  Level: {lvl}  ({risk['warnings']} warnings)")
    print(f"{'='*60}")
    print(f"  {'Prediction':<22} {'Market':<22}")
    print(f"  {'  score_mean':<22} {risk['score_mean']:>8.4f}  {'breadth':<22} {risk.get('breadth') or 'N/A':>8}")
    print(f"  {'  score_std':<22} {risk['score_std']:>8.4f}  {'cross_vol':<22} {risk.get('cross_vol') or 'N/A':>8}")
    print(f"  {'  neg_ratio':<22} {risk['neg_ratio']:>8.1%}  {'mkt_ret_5d':<22} {risk.get('mkt_ret_5d') or 'N/A':>8}")
    print(f"  {'  top30_mean':<22} {risk['top30_mean']:>8.4f}  {'vol_ratio':<22} {risk.get('vol_ratio') or 'N/A':>8}")
    print(f"  {'  model_age':<22} {str(risk.get('model_age_days'))+'d' if risk.get('model_age_days') else 'N/A':>8}  {'data_age':<22} {str(risk.get('data_age_days'))+'d' if risk.get('data_age_days') else 'N/A':>8}")
    print(f"  Decision: {'GO' if lvl == 'GREEN' else ('CAUTION' if lvl == 'YELLOW' else 'NO-GO')}")


def predict(
    qlib_dir: str = r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd",
    model_recorder_id: str = None,
    experiment_name: str = "GRU_mid_cap_60d",
    output_csv: str = None,
    skip_update: bool = False,
    instruments: str = "mid_cap_filtered",
):
    """Main prediction pipeline.

    Parameters
    ----------
    qlib_dir : str
        Path to qlib binary data directory.
    model_recorder_id : str
        Recorder ID of the trained model. If None, auto-detects the newest.
    experiment_name : str
        MLflow experiment name where the model was saved.
    output_csv : str
        Path for output CSV. Default: ``pred_score_{date}.csv``
    skip_update : bool
        If True, skip the data freshness check and update step.
    """
    qlib_path = Path(qlib_dir).expanduser().resolve()

    # -------- Step 1: Check data freshness --------
    if not skip_update:
        print(f"[check] Data dir: {qlib_path}")
        last_date = get_latest_calendar_date(qlib_path)
        print(f"[check] Last trading date in data: {last_date.strftime('%Y-%m-%d')}")
        if is_data_fresh(qlib_path):
            print("[check] Data is up-to-date, skip update.")
        else:
            print("[check] Data is stale, updating...")
            update_data(str(qlib_path))
    else:
        print("[check] Skipping data update (use --update to enable).")

    # Sync instrument files so end dates cover the latest trading day
    refresh_instruments(qlib_path, instruments)

    # -------- Step 2: Init qlib and load model --------
    qlib.init(custom_ops=_CUSTOM_OPS, provider_uri=str(qlib_path), region=REG_CN)

    if model_recorder_id is None:
        recs_dict = R.list_recorders(experiment_name=experiment_name)
        recs = list(recs_dict.values())
        if not recs:
            raise ValueError(
                f"No recorders found under experiment '{experiment_name}'. "
                f"Train the model first or specify --model_id."
            )
        # 优先选标记为 production 的模型，其次按时间排序
        prod_recs = [r for r in recs if r.info.get("tags", {}).get("status") == "production"]
        candidates = prod_recs if prod_recs else recs
        candidates.sort(key=lambda r: r.info.get("end_time") or r.info.get("start_time") or "", reverse=True)
        model_recorder_id = candidates[0].id
        if prod_recs:
            print(f"[model] Using production model: {model_recorder_id}")
        else:
            print(f"[model] Auto-detected (newest): {model_recorder_id}")
            print("[model] Tip: tag a good model with `R.set_tags(status='production')`")

    recorder = R.get_recorder(recorder_id=model_recorder_id, experiment_name=experiment_name)
    model = recorder.load_object("trained_model")
    # Show tags if available
    tags = recorder.info.get("tags", {})
    if tags:
        print(f"[model] Tags: {tags}")

    # -------- Step 3: Predict for the specified or latest calendar date --------
    cal = pd.read_csv(qlib_path / "calendars" / "day.txt", header=None, names=["date"])
    cal_dates = cal["date"].tolist()

    if args.pred_date:
        pred_date_str = args.pred_date
        try:
            idx = cal_dates.index(pred_date_str)
            next_trading_day_str = cal_dates[idx + 1] if idx + 1 < len(cal_dates) else pred_date_str
        except ValueError:
            print(f"[predict] WARNING: {pred_date_str} not in calendar, using as-is")
            next_trading_day_str = pred_date_str
            idx = len(cal_dates) - 1
    else:
        last_cal_date = get_latest_calendar_date(qlib_path)
        pred_date_str = last_cal_date.strftime("%Y-%m-%d")
        idx = cal_dates.index(pred_date_str)
        next_bdays = pd.bdate_range(last_cal_date, last_cal_date + pd.DateOffset(days=7), inclusive="right")
        next_trading_day_str = next_bdays[0].strftime("%Y-%m-%d") if len(next_bdays) > 0 else (last_cal_date + pd.DateOffset(days=1)).strftime("%Y-%m-%d")

    # Segment start: go back 240 trading days (~1 year) to ensure batch distribution
    # matches training/backtest conditions and avoids cuDNN batch-dependent variance
    test_start_idx = max(0, idx - 240)
    test_start_str = cal_dates[test_start_idx]
    print(f"[predict] Predicting for: {pred_date_str} → signals apply to: {next_trading_day_str}")

    # -------- Step 4: Build dataset --------
    data_handler_config = {
        "start_time": "2016-01-01",
        "end_time": pred_date_str,
        "fit_start_time": "2016-01-01",
        "fit_end_time": "2024-12-31",
        "instruments": instruments,
        "infer_processors": [
            {"class": "FilterCol", "kwargs": {
                "fields_group": "feature",
                "col_list": [
                    "RESI5", "WVMA5", "RSQR5", "KLEN", "RSQR10", "CORR5", "CORD5", "CORR10",
                    "ROC60", "RESI10", "VSTD5", "RSQR60", "CORR60", "WVMA60", "STD5",
                    "RSQR20", "CORD60", "CORD10", "CORR20", "KLOW",
                ],
            }},
            {"class": "RobustZScoreNorm", "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True,
            }},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
    }

    # TSDatasetH segment end is exclusive, so use next_trading_day as end
    # to ensure pred_date_str is included in the predictions
    # segment start: dynamic ~240 trading days back for consistent batch distribution
    predict_dataset = TSDatasetH(
        handler={
            "class": "Alpha158Date",
            "module_path": "daily_quant.handler.alpha158_date",
            "kwargs": data_handler_config,
        },
        segments={"test": ("2026-05-01", next_trading_day_str)},
        #segments={"test": (test_start_str, next_trading_day_str)},
        step_len=60,
    )

    pred_all = model.predict(predict_dataset)
    # 只取预测日期的结果
    pred_df = pred_all.loc[pred_all.index.get_level_values("datetime") == pd.Timestamp(pred_date_str)]
    print(f"[predict] Generated {len(pred_df)} predictions for {pred_date_str}")

    # -------- Step 5: Risk assessment (combine predictions + market data) --------
    risk = assess_risk(pred_df, instruments, pred_date_str, qlib_path, model_recorder_id, experiment_name)
    print_risk_report(risk)

    # -------- Step 6: Show predictions --------
    print(f"\n========== Predictions (signal for: {next_trading_day_str}) ==========")
    pred_sorted = pred_df.sort_values(ascending=False)
    print(f"\nTop 20 (buy):")
    print(pred_sorted.head(20).to_string())
    print(f"\nBottom 5 (avoid):")
    print(pred_sorted.tail(5).to_string())
    
    # -------- Step 7: Save (organized by prediction date) --------
    out_dir = Path.cwd() / "predictions" / pred_date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_csv is None:
        output_csv = out_dir / f"pred_score_{next_trading_day_str}.csv"
    else:
        output_csv = Path(output_csv)

    # 输出格式：instrument, score, risk_level（含风控标签）
    pred_out = pred_df.reset_index()
    pred_out.columns = ["date", "instrument", "score"]
    pred_out = pred_out.drop(columns=["date"])
    pred_out = pred_out.sort_values("score", ascending=False)
    pred_out["risk_level"] = risk["level"]
    pred_out["risk_warnings"] = risk["warnings"]
    pred_out.to_csv(output_csv, index=False)
    print(f"\n[predict] Full predictions ({len(pred_out)} stocks) saved to {output_csv}")

    # Risk report (summary)
    risk_csv = out_dir / f"risk_{pred_date_str}.csv"
    pd.DataFrame([risk]).to_csv(risk_csv, index=False)
    print(f"[predict] Risk report saved to {risk_csv}")

    # Top 50 精选
    top50_csv = out_dir / f"top50_{pred_date_str}.csv"
    pred_out.head(50).to_csv(top50_csv, index=False)
    print(f"[predict] Top 50 saved to {top50_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily prediction pipeline")
    parser.add_argument("--qlib_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd")
    parser.add_argument("--model_id", default=None, help="Recorder ID, auto-detect if empty")
    parser.add_argument("--experiment", default="GRU_mid_cap_60d", help="Experiment name")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--instruments", default="mid_cap_filtered", help="Stock pool: mid_cap_filtered, csi300, all, small_cap, etc.")
    parser.add_argument("--update", action="store_true", help="Run data update before prediction")
    parser.add_argument("--pred_date", default=None, help="Prediction date (YYYY-MM-DD), default: latest calendar date")
    parser.add_argument("--list", action="store_true", help="List all trained models with tags")
    parser.add_argument("--tag", default=None, help="Set tag on a model: key=value (use with --model_id)")
    parser.add_argument("--production", action="store_true", help="Mark the model as production (shortcut for --tag status=production)")
    args = parser.parse_args()

    if args.list:
        qlib.init(custom_ops=_CUSTOM_OPS, provider_uri=args.qlib_dir, region=REG_CN)
        recs = R.list_recorders(experiment_name=args.experiment)
        if not recs:
            print(f"No recorders found under '{args.experiment}'")
        else:
            print(f"{'ID':40s} {'status':12s} {'tags':30s} {'end_time'}")
            print("-" * 100)
            for rid, rec in sorted(recs.items(), key=lambda x: x[1].info.get("end_time") or ""):
                has_model = "trained_model" in rec.list_artifacts()
                tags = rec.info.get("tags", {})
                tag_str = ", ".join(f"{k}={v}" for k, v in tags.items()) if tags else ""
                end_time = rec.info.get("end_time", "?")
                print(f"{rid:40s} {'model' if has_model else 'no_model':12s} {tag_str:30s} {end_time}")
        sys.exit(0)

    if args.tag and args.model_id:
        qlib.init(custom_ops=_CUSTOM_OPS, provider_uri=args.qlib_dir, region=REG_CN)
        rec = R.get_recorder(recorder_id=args.model_id, experiment_name=args.experiment)
        key, _, value = args.tag.partition("=")
        if key and value:
            rec.set_tags(**{key: value})
            print(f"[tag] Set {key}={value} on {args.model_id}")
        sys.exit(0)

    if args.production and args.model_id:
        qlib.init(custom_ops=_CUSTOM_OPS, provider_uri=args.qlib_dir, region=REG_CN)
        rec = R.get_recorder(recorder_id=args.model_id, experiment_name=args.experiment)
        rec.set_tags(status="production")
        print(f"[tag] Marked {args.model_id} as production")
        sys.exit(0)

    predict(
        qlib_dir=args.qlib_dir,
        model_recorder_id=args.model_id,
        experiment_name=args.experiment,
        output_csv=args.output,
        skip_update=not args.update,
        instruments=args.instruments,
    )

"""
Standalone IC (Information Coefficient) analysis — fast, no backtest needed.

Usage
-----
    python ic_analysis.py --exp_name GRU_all_60d --instruments all
    python ic_analysis.py --exp_name GRU_CSI300_60d --instruments CSI300 --backtest_start 2025-07-01
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import copy
import numpy as np
import pandas as pd
from scipy import stats as st

import torch
from torch.utils.data import DataLoader

import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset
from qlib.contrib.model.pytorch_gru_ts import GRU
from daily_quant.ops.date_ops import (
    DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit,
)


# Must match the class used in train.py (pickle needs it for deserialization)
class GRUWithProgress(GRU):
    def fit(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        handler = dataset.handler
        if hasattr(handler, 'FEATURE_GROUPS'):
            from daily_quant.handler.alpha158_date import Alpha158DateV2
            col_set = Alpha158DateV2.feature_col_set()
        else:
            col_set = ["feature", "label"]
        dl_train = dataset.prepare("train", col_set=col_set, data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=col_set, data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset")
        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")
        wl_train = np.ones(len(dl_train)); wl_valid = np.ones(len(dl_valid))
        train_loader = DataLoader(ConcatDataset(dl_train, wl_train),
                                  batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.n_jobs, drop_last=True)
        valid_loader = DataLoader(ConcatDataset(dl_valid, wl_valid),
                                  batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.n_jobs, drop_last=True)
        save_path = qlib.utils.get_or_create_path(save_path)
        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        best_param = copy.deepcopy(self.GRU_model.state_dict())
        evals_result["train"], evals_result["valid"] = [], []
        self.fitted = True
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.train_optimizer, mode="max", factor=0.5, patience=3)
        for step in range(self.n_epochs):
            self.train_epoch(train_loader)
            _, train_score = self.test_epoch(train_loader)
            _, val_score = self.test_epoch(valid_loader)
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            scheduler.step(val_score)
            if val_score > best_score:
                best_score, stop_steps, best_epoch = val_score, 0, step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    break
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        handler = dataset.handler
        if hasattr(handler, 'FEATURE_GROUPS'):
            from daily_quant.handler.alpha158_date import Alpha158DateV2
            col_set = Alpha158DateV2.feature_col_set()
        else:
            col_set = ["feature", "label"]
        dl_test = dataset.prepare("test", col_set=col_set, data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        self.GRU_model.eval()
        preds = []
        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)
            with torch.no_grad():
                pred = self.GRU_model(feature.float()).detach().cpu().numpy()
            preds.append(pred)
        return pd.Series(np.concatenate(preds), index=dl_test.get_index())

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


def main():
    parser = argparse.ArgumentParser(description="IC Analysis")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--exp_name", default="GRU_all_60d")
    parser.add_argument("--instruments", default="all")
    parser.add_argument("--step_len", type=int, default=60)
    parser.add_argument("--backtest_start", default="2025-07-01")
    parser.add_argument("--backtest_end", default="2026-05-27")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd")
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS)

    # Load model
    recs_dict = R.list_recorders(experiment_name=args.exp_name)
    recs = [r for r in recs_dict.values() if "trained_model" in r.list_artifacts()]
    if not recs:
        raise ValueError(f"No model in '{args.exp_name}'")
    recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
    model_id = args.model_id or recs[0].id

    recorder = R.get_recorder(recorder_id=model_id, experiment_name=args.exp_name)
    model = recorder.load_object("trained_model")
    client_tags = recorder.client.get_run(model_id).data.tags

    handler_class = client_tags.get("handler_class", "Alpha158Date")
    use_alpha = client_tags.get("use_alpha_factors", "False") == "True"
    use_group_norm = client_tags.get("use_group_norm", "False") == "True"
    print(f"Model: {model_id}")
    print(f"Handler: {handler_class}, alpha={use_alpha}, group_norm={use_group_norm}")
    print(f"d_feat: {model.d_feat}")

    # Build dataset
    if use_group_norm:
        data_handler_config = {
            "start_time": "2022-01-01", "end_time": args.backtest_end,
            "fit_start_time": "2022-01-01", "fit_end_time": "2024-12-31",
            "instruments": args.instruments, "use_alpha_factors": use_alpha,
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
        }
    else:
        data_handler_config = {
            "start_time": "2022-01-01", "end_time": args.backtest_end,
            "fit_start_time": "2022-01-01", "fit_end_time": "2024-12-31",
            "instruments": args.instruments, "use_alpha_factors": use_alpha,
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ],
        }

    dataset = TSDatasetH(
        handler={
            "class": handler_class,
            "module_path": "daily_quant.handler.alpha158_date",
            "kwargs": data_handler_config,
        },
        segments={"train": ("2022-01-01", "2024-12-31"),
                   "valid": ("2025-01-01", "2025-06-30"),
                   "test": (args.backtest_start, args.backtest_end)},
        step_len=args.step_len,
    )

    # Predict
    print("\nPredicting...")
    pred_all = model.predict(dataset)
    print(f"Predictions: {len(pred_all)}")

    # Get labels — data_arr has +1 NaN row for padding, trim it
    label_sampler = dataset.prepare("test", col_set=["label"], data_key=DataHandlerLP.DK_I)
    n = len(label_sampler.data_index)
    label_series = pd.Series(
        label_sampler.data_arr[:n, 0],
        index=label_sampler.data_index,
    ).dropna()

    # Swap label index to (datetime, instrument) to match pred
    label_series = label_series.swaplevel()
    common_idx = pred_all.index.intersection(label_series.index)
    pred_aligned = pred_all.loc[common_idx]
    label_aligned = label_series.loc[common_idx]
    print(f"Aligned: {len(pred_aligned)} preds, {len(label_aligned)} labels")

    # IC computation
    print("\n" + "=" * 60)
    print("=== IC 分析 ===")
    ic_series = pred_aligned.groupby("datetime").apply(
        lambda g: st.spearmanr(g, label_aligned.loc[g.index])[0]
        if len(g) > 5 else np.nan
    ).dropna()

    print(f"测试区间: {ic_series.index[0].strftime('%Y-%m-%d')} ~ {ic_series.index[-1].strftime('%Y-%m-%d')}")
    print(f"有效天数: {len(ic_series)}")
    print(f"\n{'指标':<20} {'值':>10}")
    print("-" * 32)
    print(f"{'Mean IC':<20} {ic_series.mean():>10.4f}")
    print(f"{'IC Std':<20} {ic_series.std():>10.4f}")
    print(f"{'ICIR':<20} {ic_series.mean() / ic_series.std():>10.4f}")
    print(f"{'IC > 0':<20} {(ic_series > 0).mean():>10.1%}")
    print(f"{'IC > 0.02':<20} {(ic_series > 0.02).mean():>10.1%}")
    print(f"{'t-stat':<20} {ic_series.mean() / ic_series.std() * np.sqrt(len(ic_series)):>10.2f}")

    # Monthly
    print(f"\n月度 Rank IC:")
    ic_monthly = ic_series.resample("ME").mean()
    for m, ic in ic_monthly.items():
        bar = "+" * int(max(0, ic * 100)) if ic > 0 else "-" * int(abs(ic) * 100)
        print(f"  {m.strftime('%Y-%m')}: {ic:+.4f} {bar}")

    # IC decay
    print(f"\nIC 稳定性:")
    for w in [5, 10, 21, 42, 63]:
        ma = ic_series.rolling(w).mean().dropna()
        print(f"  {w:2d}d MA: min={ma.min():.4f}, max={ma.max():.4f}, frac>0={(ma > 0).mean():.0%}")

    print("\nDone.")


if __name__ == "__main__":
    main()

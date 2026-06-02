"""
Train GRU model with Alpha158Date handler (164 features: 158 Alpha158 + 6 date).

Usage
-----
    python train_date.py
    python train_date.py --instruments csi300 --step_len 60 --n_epochs 50
    python train_date.py --qlib_data_dir C:/path/to/data
"""

import sys
import copy
import argparse
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import qlib
from qlib.constant import REG_CN
from qlib.utils import flatten_dict, get_or_create_path
from qlib.workflow import R
from qlib.data.dataset import TSDatasetH, TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset
from qlib.contrib.model.pytorch_gru_ts import GRU


# ============================================================================
# 1. Custom dataset classes (fixed normalization)
# ============================================================================

class FixedNormalizedTSDataSampler(TSDataSampler):
    """Use nanmean/nanstd to avoid NaN contagion across rows."""

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        process_data = data[:, 0:-1]
        data_mean = np.nanmean(process_data, axis=0)
        data_std = np.nanstd(process_data, axis=0)
        data_std = np.where(data_std < 1e-5, 1.0, data_std)
        normalized = (process_data - data_mean) / data_std
        normalized = np.clip(normalized, -5, 5)
        normalized = np.where(np.isnan(normalized), 0, normalized)
        data[:, 0:-1] = normalized
        return data


class FixedNormalizedTSDatasetH(TSDatasetH):
    def _prepare_seg(self, slc, **kwargs):
        dtype = kwargs.pop("dtype", None)
        if not isinstance(slc, slice):
            slc = slice(*slc)
        flt_col = kwargs.pop("flt_col", None) or self.flt_col
        ext_slice = self._extend_slice(slc, self.cal, self.step_len)
        data = super(TSDatasetH, self)._prepare_seg(ext_slice, **kwargs)
        flt_kwargs = copy.deepcopy(kwargs)
        if flt_col is not None:
            flt_kwargs["col_set"] = flt_col
            flt_data = super(TSDatasetH, self)._prepare_seg(ext_slice, **flt_kwargs)
            assert len(flt_data.columns) == 1
        else:
            flt_data = None
        return FixedNormalizedTSDataSampler(
            data=data, start=slc.start, end=slc.stop,
            step_len=self.step_len, dtype=dtype, flt_data=flt_data,
        )


# ============================================================================
# 2. GRU model with progress bar
# ============================================================================

class GRUWithProgress(GRU):
    """GRU + tqdm + R.log_metrics."""

    def fit(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")

        wl_train = np.ones(len(dl_train)) if reweighter is None else reweighter.reweight(dl_train)
        wl_valid = np.ones(len(dl_valid)) if reweighter is None else reweighter.reweight(dl_valid)

        train_loader = DataLoader(
            ConcatDataset(dl_train, wl_train),
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_jobs, drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, wl_valid),
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_jobs, drop_last=True,
        )

        save_path = get_or_create_path(save_path)
        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        best_param = copy.deepcopy(self.GRU_model.state_dict())
        evals_result["train"], evals_result["valid"] = [], []
        self.logger.info("training...")
        self.fitted = True

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

            pbar.set_postfix({
                "train": f"{train_score:.4f}",
                "valid": f"{val_score:.4f}",
                "best": f"{best_score:.4f}",
            })

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self.use_gpu:
            torch.cuda.empty_cache()


# ============================================================================
# 3. Main training entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GRU model with Alpha158Date")
    parser.add_argument("--instruments", default="mid_cap", help="Stock pool: mid_cap, csi300, all")
    parser.add_argument("--step_len", type=int, default=60, help="Time-series window length (trading days)")
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--early_stop", type=int, default=15)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_10y")
    parser.add_argument("--exp_name", default=None, help="Experiment name (default: GRU_{instruments}_{step_len}d)")
    args = parser.parse_args()

    INSTRUMENTS = args.instruments
    STEP_LEN = args.step_len
    TOTAL_FEAT = 171  # 158 (Alpha158) + 6 (fundamental) + 7 (date+board)

    # --- Register date operators via qlib.init ---
    from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

    qlib.init(
        provider_uri=args.qlib_data_dir,
        region=REG_CN,
        custom_ops=[DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit],
    )
    print(f"qlib initialized, data: {args.qlib_data_dir}")

    # --- Build dataset ---
    data_handler_config = {
        "start_time": "2022-01-01",
        "end_time": "2026-05-28",
        "fit_start_time": "2022-01-01",
        "fit_end_time": "2024-12-31",
        "instruments": INSTRUMENTS,
    }

    dataset_config = {
        "class": "FixedNormalizedTSDatasetH",
        "kwargs": {
            "handler": {
                "class": "Alpha158Date",
                "module_path": "daily_quant.handler.alpha158_date",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2022-01-01", "2024-12-31"),
                "valid": ("2025-01-01", "2025-06-30"),
                "test": ("2025-01-01", "2026-05-27"),
            },
            "step_len": STEP_LEN,
        },
    }

    dataset = FixedNormalizedTSDatasetH(
        handler={
            "class": "Alpha158Date",
            "module_path": "daily_quant.handler.alpha158_date",
            "kwargs": data_handler_config,
        },
        segments=dataset_config["kwargs"]["segments"],
        step_len=STEP_LEN,
    )

    train_ts = dataset.prepare("train", col_set="feature")
    print(f"Training samples: {len(train_ts)}")
    print(f"Sample shape: {train_ts[0].shape}  -> [{STEP_LEN} days, {TOTAL_FEAT} features]")

    # --- Build model ---
    model_config = {
        "class": "GRUWithProgress",
        "kwargs": {
            "d_feat": TOTAL_FEAT,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "early_stop": args.early_stop,
            "loss": "mse",
            "optimizer": "adam",
            "GPU": 0,
            "seed": args.seed,
            "n_jobs": 0,
        },
    }
    model = GRUWithProgress(**model_config["kwargs"])

    exp_name = args.exp_name or f"GRU_{INSTRUMENTS}_{STEP_LEN}d"
    print(f"Experiment: {exp_name}")
    print(f"Model: d_feat={TOTAL_FEAT}, hidden={args.hidden_size}, layers={args.num_layers}")

    with R.start(experiment_name=exp_name):
        R.log_params(**flatten_dict({"model": model_config, "dataset": dataset_config}))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        R.set_tags(
            status="completed",
            model_type="GRU",
            instruments=INSTRUMENTS,
            step_len=str(STEP_LEN),
            train_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        print(f"Training done, recorder_id: {rid}")


if __name__ == "__main__":
    main()

"""
Train GRU model with Alpha158Date handler (170 features: 158 Alpha158 + 6 fundamental + 7 date/board).

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

# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # uncomment if GPU page-file errors

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import qlib
from qlib.constant import REG_CN
from qlib.utils import flatten_dict, get_or_create_path
from qlib.workflow import R
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset
from qlib.contrib.model.pytorch_gru_ts import GRU


# ============================================================================
# 1. GRU model with progress bar
# ============================================================================

class GRUWithProgress(GRU):
    """GRU + tqdm + R.log_metrics."""

    def fit(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        # Detect handler type: V2 uses per-group feature cols, V1 uses "feature"
        handler = dataset.handler
        if hasattr(handler, 'FEATURE_GROUPS'):
            from daily_quant.handler.alpha158_date import Alpha158DateV2
            col_set = Alpha158DateV2.feature_col_set()
        else:
            col_set = ["feature", "label"]

        dl_train = dataset.prepare("train", col_set=col_set, data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=col_set, data_key=DataHandlerLP.DK_L)
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

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.train_optimizer, mode="max", factor=0.5, patience=3,
        )

        pbar = tqdm(range(self.n_epochs), desc="Training", unit="epoch")
        for step in pbar:
            self.train_epoch(train_loader)
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            scheduler.step(val_score)

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
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.GRU_model.load_state_dict(best_param)
        self.GRU_model.rnn.flatten_parameters()
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
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd")
    parser.add_argument("--exp_name", default=None, help="Experiment name (default: GRU_{instruments}_{step_len}d)")
    parser.add_argument("--use_alpha_factors", action="store_true",
                        help="Include moneyflow/margin alpha factors (requires alpha_factors data in qlib)")
    parser.add_argument("--use_group_norm", action="store_true",
                        help="Use Alpha158DateV2 with per-group normalization")
    args = parser.parse_args()

    INSTRUMENTS = args.instruments
    STEP_LEN = args.step_len
    TOTAL_FEAT = 180 if args.use_alpha_factors else 170

    # --- Register date operators via qlib.init ---
    from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

    qlib.init(
        provider_uri=args.qlib_data_dir,
        region=REG_CN,
        custom_ops=[DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit],
    )
    print(f"qlib initialized, data: {args.qlib_data_dir}")

    # --- Handler config ---
    if args.use_group_norm:
        HANDLER_CLASS = "Alpha158DateV2"
        # V2 handles normalization internally; only pass learn_processors
        data_handler_config = {
            "start_time": "2022-01-01",
            "end_time": "2026-05-28",
            "fit_start_time": "2022-01-01",
            "fit_end_time": "2026-05-28",
            "instruments": INSTRUMENTS,
            "use_alpha_factors": args.use_alpha_factors,
            "learn_processors": [
                {"class": "DropnaLabel"},  # 3-class label, no rank norm needed
            ],
        }
        from daily_quant.handler.alpha158_date import Alpha158DateV2 as HandlerClass
        col_set = HandlerClass.feature_col_set(use_alpha_factors=args.use_alpha_factors)
    else:
        HANDLER_CLASS = "Alpha158Date"
        data_handler_config = {
            "start_time": "2022-01-01",
            "end_time": "2026-05-28",
            "fit_start_time": "2022-01-01",
            "fit_end_time": "2026-05-28",
            "instruments": INSTRUMENTS,
            "use_alpha_factors": args.use_alpha_factors,
            "infer_processors": [
                {"class": "RobustZScoreNorm", "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True,
                }},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ],
            "learn_processors": [
                {"class": "DropnaLabel"},  # 3-class label, no rank norm needed
            ],
        }
        col_set = ["feature", "label"]

    dataset_config = {
        "class": "TSDatasetH",
        "kwargs": {
            "handler": {
                "class": HANDLER_CLASS,
                "module_path": "daily_quant.handler.alpha158_date",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2022-01-01", "2024-12-31"),
                "valid": ("2025-01-01", "2026-01-01"),
                "test": ("2026-01-01", "2026-05-27"),
            },
            "step_len": STEP_LEN,
        },
    }

    dataset = TSDatasetH(
        handler={
            "class": HANDLER_CLASS,
            "module_path": "daily_quant.handler.alpha158_date",
            "kwargs": data_handler_config,
        },
        segments=dataset_config["kwargs"]["segments"],
        step_len=STEP_LEN,
    )

    train_ts = dataset.prepare("train", col_set=col_set)
    print(f"Training samples: {len(train_ts)}")
    actual_feat = train_ts[0].shape[-1]  # last dim is feature count (incl label if present)
    print(f"Sample shape: {train_ts[0].shape}  -> [{STEP_LEN} days, {actual_feat} cols] (configured: {TOTAL_FEAT} features)")

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
            "GPU": 0,  # CPU only to avoid Windows page-file errors
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
            use_alpha_factors=str(args.use_alpha_factors),
            use_group_norm=str(args.use_group_norm),
            handler_class=HANDLER_CLASS,
            total_feat=str(TOTAL_FEAT),
            train_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        print(f"Training done, recorder_id: {rid}")


if __name__ == "__main__":
    main()

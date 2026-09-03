"""
Train GRU model on ETF data with Alpha158ETF handler + sharpe label.

Uses TSDatasetH (time-series) with the Alpha158ETF handler (171 features,
sharpe label), then trains a GRU (CPU only).

Usage
-----
    python daily_quant/gru/train_etf.py --step_len 40 --n_epochs 50
"""
import sys
import copy
import argparse
from pathlib import Path

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

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


class GRUWithProgress(GRU):
    def fit(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        col_set = ["feature", "label"]
        dl_train = dataset.prepare("train", col_set=col_set, data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=col_set, data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset")
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
        self.fitted = True

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.train_optimizer, mode="max", factor=0.5, patience=3)

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
            pbar.set_postfix({"train": f"{train_score:.4f}", "valid": f"{val_score:.4f}",
                              "best": f"{best_score:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        self.GRU_model.load_state_dict(best_param)
        self.GRU_model.rnn.flatten_parameters()
        torch.save(best_param, save_path)

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
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


def main():
    parser = argparse.ArgumentParser(description="Train GRU on ETF sharpe label")
    parser.add_argument("--instruments", default="stock")
    parser.add_argument("--step_len", type=int, default=40)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\etf_data")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--label_type", default="sharpe", choices=["return", "sharpe"])
    args = parser.parse_args()

    STEP_LEN = args.step_len
    TOTAL_FEAT = 171  # Alpha158ETF share_only

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=1)
    print(f"qlib initialized, data: {args.qlib_data_dir}")

    data_handler_config = {
        "start_time": "2022-01-01",
        "end_time": "2026-06-29",
        "fit_start_time": "2022-01-01",
        "fit_end_time": "2024-12-31",
        "instruments": args.instruments,
        "use_alpha_factors": False,
        "label_type": args.label_type,
        "include_prem_disc": False,
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
            "class": "Alpha158ETF",
            "module_path": "daily_quant.handler.alpha158_etf",
            "kwargs": data_handler_config,
        },
        segments={
            "train": ("2022-01-01", "2024-12-31"),
            "valid": ("2025-01-01", "2025-06-30"),
            "test": ("2025-07-01", "2026-06-29"),
        },
        step_len=STEP_LEN,
    )

    col_set = ["feature", "label"]
    train_ts = dataset.prepare("train", col_set=col_set)
    print(f"Training samples: {len(train_ts)}")
    print(f"Sample shape: {train_ts[0].shape}  -> [{STEP_LEN} days, {train_ts[0].shape[-1]} cols]")

    model = GRUWithProgress(
        d_feat=TOTAL_FEAT,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        early_stop=args.early_stop,
        loss="mse",
        optimizer="adam",
        GPU=0,
        seed=args.seed,
        n_jobs=0,
    )

    exp_name = args.exp_name or f"GRU_ETF_{args.label_type}_{STEP_LEN}d"
    print(f"Experiment: {exp_name}")
    print(f"Model: d_feat={TOTAL_FEAT}, hidden={args.hidden_size}, layers={args.num_layers}")

    with R.start(experiment_name=exp_name):
        R.log_params(**flatten_dict({
            "model": {"class": "GRU", "d_feat": TOTAL_FEAT, "hidden_size": args.hidden_size},
            "step_len": STEP_LEN, "label_type": args.label_type,
        }))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        R.set_tags(
            status="completed", model_type="GRU", instruments=args.instruments,
            step_len=str(STEP_LEN), handler_class="Alpha158ETF",
            label_type=args.label_type, total_feat=str(TOTAL_FEAT),
            train_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
        )
        print(f"Training done, recorder_id: {rid}")


if __name__ == "__main__":
    main()

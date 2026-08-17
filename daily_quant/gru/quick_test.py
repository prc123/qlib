"""
Quick pipeline test: train simple RNN + backtest — ~5 min total.

Usage
-----
    python quick_test.py
    python quick_test.py --instruments mid_cap_no_sh --topk 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import copy, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset
from qlib.utils import get_or_create_path
from qlib.contrib.model.pytorch_gru_ts import GRU

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit
_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


# --- Simple RNN model (lighter than GRU) ---
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, d_feat=6, hidden_size=32, num_layers=1, dropout=0.0, bidirectional=False):
        super().__init__()
        self.rnn = nn.RNN(d_feat, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class QuickModel(GRU):
    """GRU wrapper but with SimpleRNN inside, short training."""

    def __init__(self, d_feat=6, hidden_size=32, num_layers=1, dropout=0.0, n_epochs=5,
                 lr=0.001, batch_size=256, early_stop=3, loss="mse", optimizer="adam",
                 GPU=0, seed=42, n_jobs=0, **kwargs):
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss_type = loss
        self.optimizer = optimizer
        self._use_gpu = (GPU >= 0 and torch.cuda.is_available())
        self.seed = seed
        self.n_jobs = n_jobs
        self.fitted = False

        if self.seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.GRU_model = SimpleRNN(
            d_feat=d_feat, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
        )
        if self._use_gpu:
            self.GRU_model = self.GRU_model.cuda()

        self.device = "cuda" if self._use_gpu else "cpu"
        self.logger = qlib.log.get_module_logger("QuickModel")
        self.train_epoch = self._make_train_epoch()
        self.test_epoch = self._make_test_epoch()

    def __getstate__(self):
        """Exclude unpicklable local functions from serialization."""
        state = self.__dict__.copy()
        for k in ("train_epoch", "test_epoch", "logger"):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = qlib.log.get_module_logger("QuickModel")
        self.train_epoch = self._make_train_epoch()
        self.test_epoch = self._make_test_epoch()

    def _make_train_epoch(self):
        _model = self.GRU_model
        _device = "cuda" if self._use_gpu else "cpu"
        _optimizer = torch.optim.Adam(_model.parameters(), lr=self.lr)
        _loss_fn = nn.MSELoss()

        def fn(data_loader):
            _model.train()
            for data in data_loader:
                if isinstance(data, (tuple, list)):
                    data = data[0]
                data = torch.from_numpy(np.asarray(data)).float().to(_device)
                x = data[:, :, 0:-1]
                y = data[:, -1, -1]
                _optimizer.zero_grad()
                pred = _model(x)
                loss = _loss_fn(pred, y)
                loss.backward()
                _optimizer.step()

        return fn

    def _make_test_epoch(self):
        _model = self.GRU_model
        _device = "cuda" if self._use_gpu else "cpu"
        _loss_fn = nn.MSELoss()

        def fn(data_loader):
            _model.eval()
            losses, preds = [], []
            with torch.no_grad():
                for data in data_loader:
                    if isinstance(data, (tuple, list)):
                        data = data[0]
                    data = torch.from_numpy(np.asarray(data)).float().to(_device)
                    x = data[:, :, 0:-1]
                    y = data[:, -1, -1]
                    pred = _model(x)
                    losses.append(_loss_fn(pred, y).item())
                    preds.append(pred.cpu().numpy())
            return np.mean(losses), np.mean(np.concatenate(preds))

        return fn

    def fit(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset")

        dl_train.config(fillna_type="ffill+bfill")
        dl_valid.config(fillna_type="ffill+bfill")

        train_loader = DataLoader(
            ConcatDataset(dl_train, np.ones(len(dl_train))),
            batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True,
        )
        valid_loader = DataLoader(
            ConcatDataset(dl_valid, np.ones(len(dl_valid))),
            batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=True,
        )

        save_path = get_or_create_path(save_path)
        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        best_param = copy.deepcopy(self.GRU_model.state_dict())
        self.fitted = True

        from tqdm import tqdm
        pbar = tqdm(range(self.n_epochs), desc="QuickTrain", unit="epoch")
        for step in pbar:
            self.train_epoch(train_loader)
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            if val_score > best_score:
                best_score, stop_steps, best_epoch = val_score, 0, step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    break
            pbar.set_postfix({"train": f"{train_score:.3f}", "val": f"{val_score:.3f}", "best": f"{best_score:.3f}"})

        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)
        if self._use_gpu:
            torch.cuda.empty_cache()


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", default="mid_cap")
    parser.add_argument("--step_len", type=int, default=60)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--qlib_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd")
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_dir, region=REG_CN, custom_ops=_CUSTOM_OPS)

    TOTAL_FEAT = 20   # FilterCol 保留的 20 个 Alpha158 特征

    data_handler_config = {
        "start_time": "2022-01-01", "end_time": "2026-05-27",
        "fit_start_time": "2022-01-01", "fit_end_time": "2024-12-31",
        "instruments": args.instruments,
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
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
    }

    dataset = TSDatasetH(
        handler={"class": "Alpha158Date", "module_path": "daily_quant.handler.alpha158_date",
                 "kwargs": data_handler_config},
        segments={"train": ("2022-01-01", "2024-12-31"), "valid": ("2025-01-01", "2025-06-30"),
                  "test": ("2025-07-01", "2026-05-27")},
        step_len=args.step_len,
    )

    model = QuickModel(d_feat=TOTAL_FEAT, hidden_size=args.hidden_size, n_epochs=args.n_epochs,
                       batch_size=256, early_stop=3)

    EXP = "quick_test"
    print(f"Training {args.n_epochs} epochs...")
    with R.start(experiment_name=EXP):
        model.fit(dataset)
        R.save_objects(trained_model=model)
        rid = R.get_recorder().id
        print(f"Trained: {rid}")

    print("Backtesting...")
    port_analysis_config = {
        "executor": {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                     "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}},
        "strategy": {"class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy.signal_strategy",
                     "kwargs": {"model": model, "dataset": dataset, "topk": args.topk, "n_drop": 3}},
        "backtest": {"start_time": "2025-07-01", "end_time": "2026-05-27", "account": 10_000_000,
                     "benchmark": "SH000300",
                     "exchange_kwargs": {"freq": "day", "deal_price": "open",
                                         "open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 5,
                                         "limit_threshold": 0.095}},
    }

    with R.start(experiment_name="quick_bt"):
        sr = SignalRecord(model, dataset, R.get_recorder())
        sr.generate()
        par = PortAnaRecord(R.get_recorder(), port_analysis_config, "day")
        par.generate()

    # Show results
    recorder = R.get_recorder(experiment_name="quick_bt")
    report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    print(f"\n=== Results ===")
    print(f"Excess (with cost): mean={analysis.loc['excess_return_with_cost','mean']:.4f}")
    print(f"Excess annualized: {analysis.loc['excess_return_with_cost','annualized_return']:.2%}")
    print(f"Max drawdown: {analysis.loc['excess_return_with_cost','max_drawdown']:.2%}")
    print(f"Strategy annualized: {report['return'].mean()*252*100:.1f}%")
    print(f"Benchmark annualized: {report['bench'].mean()*252*100:.1f}%")
    print(f"Win rate: {(report['return'] > report['bench']).mean():.1%}")


if __name__ == "__main__":
    main()

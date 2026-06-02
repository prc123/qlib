"""
Backtest evaluation for Alpha158Date model (170 features).

Usage
-----
    python backtest_eval_date.py
    python backtest_eval_date.py --model_id <id> --topk 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data.dataset import TSDatasetH, TSDataSampler
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset
from qlib.utils import get_or_create_path
from qlib.contrib.model.pytorch_gru_ts import GRU
from qlib.contrib.evaluate import risk_analysis

# --- Register date operators ---
from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear
_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear]

# --- Custom dataset classes (must match training) ---

class FixedNormalizedTSDataSampler(TSDataSampler):
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


# pickle compat
class GRUWithProgress(GRU):
    def fit(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
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
        save_path = get_or_create_path(save_path)
        stop_steps, best_score, best_epoch = 0, -np.inf, 0
        best_param = copy.deepcopy(self.GRU_model.state_dict())
        evals_result["train"], evals_result["valid"] = [], []
        self.fitted = True
        from tqdm import tqdm
        pbar = tqdm(range(self.n_epochs), desc="Training", unit="epoch")
        for step in pbar:
            self.train_epoch(train_loader)
            train_loss, train_score = self.test_epoch(train_loader)
            val_loss, val_score = self.test_epoch(valid_loader)
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            if val_score > best_score:
                best_score, stop_steps, best_epoch = val_score, 0, step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    break
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest Alpha158Date (170-feat) model")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--instruments", default="mid_cap")
    parser.add_argument("--step_len", type=int, default=60)
    parser.add_argument("--backtest_start", default="2025-07-01")
    parser.add_argument("--backtest_end", default="2026-05-27")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n_drop", type=int, default=5)
    parser.add_argument("--account", type=int, default=10_000_000)
    args = parser.parse_args()

    EXP_NAME = "GRU_mid_cap_60d"
    BACKTEST_EXP = "backtest_gru_date"

    qlib.init(
        provider_uri=r"C:\Users\pp\.qlib\qlib_data\cn_data_10y",
        region=REG_CN,
        custom_ops=_CUSTOM_OPS,
    )

    # --- Load model ---
    recs_dict = R.list_recorders(experiment_name=EXP_NAME)
    recs = [r for r in recs_dict.values() if "trained_model" in r.list_artifacts()]
    if not recs:
        raise ValueError(f"No model found in '{EXP_NAME}'. Run train_date.py first.")
    recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
    model_id = args.model_id or recs[0].id
    tags = recs_dict[model_id].info.get("tags", {})
    print(f"Model: {model_id}")
    print(f"Tags: {tags}")

    recorder = R.get_recorder(recorder_id=model_id, experiment_name=EXP_NAME)
    model = recorder.load_object("trained_model")

    # --- Build dataset ---
    data_handler_config = {
        "start_time": "2022-01-01",
        "end_time": args.backtest_end,
        "fit_start_time": "2022-01-01",
        "fit_end_time": "2024-12-31",
        "instruments": args.instruments,
    }

    dataset = FixedNormalizedTSDatasetH(
        handler={
            "class": "Alpha158Date",
            "module_path": "daily_quant.handler.alpha158_date",
            "kwargs": data_handler_config,
        },
        segments={
            "train": ("2022-01-01", "2024-12-31"),
            "valid": ("2025-01-01", "2025-06-30"),
            "test": (args.backtest_start, args.backtest_end),
        },
        step_len=args.step_len,
    )
    print("Dataset ready.")

    # --- Backtest ---
    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {"model": model, "dataset": dataset, "topk": args.topk, "n_drop": args.n_drop},
        },
        "backtest": {
            "start_time": args.backtest_start,
            "end_time": args.backtest_end,
            "account": args.account,
            "benchmark": "SH000300",
            "exchange_kwargs": {
                "freq": "day",
                "deal_price": "open", "open_cost": 0.0005,
                "close_cost": 0.0015, "min_cost": 5,
                "limit_threshold": (
                    "Greater($change, BoardLimit($close) * 0.98)",
                    "Less($change, BoardLimit($close) * -0.98)",
                ),
            },
        },
    }

    with R.start(experiment_name=BACKTEST_EXP):
        recorder_train = R.get_recorder(recorder_id=model_id, experiment_name=EXP_NAME)
        trained_model = recorder_train.load_object("trained_model")
        sr_recorder = R.get_recorder()
        ba_rid = sr_recorder.id
        sr = SignalRecord(trained_model, dataset, sr_recorder)
        sr.generate()
        par = PortAnaRecord(sr_recorder, port_analysis_config, "day")
        par.generate()
        print(f"Backtest done, recorder_id: {ba_rid}")

    # --- Results ---
    sr_recorder = R.get_recorder(recorder_id=ba_rid, experiment_name=BACKTEST_EXP)
    report = sr_recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis = sr_recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    print("\n" + "=" * 60)
    print("=== 基准 (沪深300) ===")
    print(risk_analysis(report["bench"], freq="1day"))
    print("\n=== 超额收益（未扣费） ===")
    print(risk_analysis(report["return"] - report["bench"], freq="1day"))
    print("\n=== 超额收益（扣费） ===")
    excess = report["return"] - report["bench"]
    cost = report.get("cost", pd.Series(0, index=excess.index))
    excess_cost = excess - cost
    print(f"  年化超额: {excess_cost.mean()*238:.2%}")
    print(f"  跑赢胜率: {(excess_cost > 0).mean():.1%}")

    print("\n=== 详细分析 ===")
    print(analysis)

    print("\n=== 策略 vs HS300 ===")
    strat_ret = report["return"].mean() * 238
    bench_ret = report["bench"].mean() * 238
    strat_sharpe = report["return"].mean() / report["return"].std() * 238**0.5
    bench_sharpe = report["bench"].mean() / report["bench"].std() * 238**0.5
    cum_strat = (report["return"] + 1).cumprod().iloc[-1] - 1
    cum_bench = (report["bench"] + 1).cumprod().iloc[-1] - 1
    print(f"  策略年化: {strat_ret:.2%}  夏普: {strat_sharpe:.2f}  累计: {cum_strat:.2%}")
    print(f"  基准年化: {bench_ret:.2%}  夏普: {bench_sharpe:.2f}  累计: {cum_bench:.2%}")


if __name__ == "__main__":
    main()

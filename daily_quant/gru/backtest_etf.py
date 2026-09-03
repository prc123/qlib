"""
Backtest GRU model trained on ETF sharpe label (TSDatasetH + Alpha158ETF).

Usage
-----
    python daily_quant/gru/backtest_etf.py --exp_name GRU_ETF_sharpe_40d --step_len 40
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

import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.utils import ConcatDataset
from qlib.contrib.model.pytorch_gru_ts import GRU
from qlib.contrib.evaluate import risk_analysis

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


class GRUWithProgress(GRU):
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
    parser = argparse.ArgumentParser(description="Backtest GRU ETF model")
    parser.add_argument("--instruments", default="stock")
    parser.add_argument("--step_len", type=int, default=40)
    parser.add_argument("--backtest_start", default="2025-07-01")
    parser.add_argument("--backtest_end", default="2026-06-29")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n_drop", type=int, default=3)
    parser.add_argument("--benchmark", default="SH510050")
    parser.add_argument("--freq", default="day", choices=["day", "week"])
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\etf_data")
    parser.add_argument("--exp_name", default="GRU_ETF_sharpe_40d")
    args = parser.parse_args()

    qlib.init(provider_uri=args.qlib_data_dir, region=REG_CN, custom_ops=_CUSTOM_OPS, kernels=1)

    recs = R.list_recorders(experiment_name=args.exp_name)
    recs = [r for r in recs.values() if "trained_model" in r.list_artifacts()]
    if not recs:
        raise ValueError(f"No model in '{args.exp_name}'")
    recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
    recorder = R.get_recorder(recorder_id=recs[0].id, experiment_name=args.exp_name)
    model = recorder.load_object("trained_model")
    print(f"Model: {recs[0].id[:16]}, d_feat={model.d_feat}")

    data_handler_config = {
        "start_time": "2022-01-01",
        "end_time": args.backtest_end,
        "fit_start_time": "2022-01-01",
        "fit_end_time": "2024-12-31",
        "instruments": args.instruments,
        "use_alpha_factors": False,
        "label_type": "sharpe",
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
            "test": (args.backtest_start, args.backtest_end),
        },
        step_len=args.step_len,
    )

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": args.freq, "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "BoardLimitTopkDropoutStrategy",
            "module_path": "daily_quant.strategies",
            "kwargs": {
                "model": model, "dataset": dataset,
                "topk": args.topk, "n_drop": args.n_drop,
                "check_limit": True,
            },
        },
        "backtest": {
            "start_time": args.backtest_start,
            "end_time": args.backtest_end,
            "account": 100_000_000,
            "benchmark": args.benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "deal_price": "open",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
                "limit_threshold": None,
            },
        },
    }

    with R.start(experiment_name="backtest_gru_etf"):
        sr_recorder = R.get_recorder()
        ba_rid = sr_recorder.id
        sr = SignalRecord(model, dataset, sr_recorder)
        sr.generate()
        par = PortAnaRecord(sr_recorder, port_analysis_config, args.freq)
        par.generate()

    sr_recorder = R.get_recorder(recorder_id=ba_rid, experiment_name="backtest_gru_etf")
    report = sr_recorder.load_object(f"portfolio_analysis/report_normal_1{args.freq}.pkl")

    print("\n=== 基准 ===")
    print(risk_analysis(report["bench"], freq=args.freq))
    print("\n=== 策略收益 ===")
    print(risk_analysis(report["return"], freq=args.freq))
    print("\n=== 超额（未扣费） ===")
    excess = report["return"] - report["bench"]
    print(risk_analysis(excess, freq=args.freq))
    print("\n=== 超额（扣费） ===")
    net_excess = report["return"] - report["cost"] - report["bench"]
    print(risk_analysis(net_excess, freq=args.freq))

    ann = 238 if args.freq == "day" else 50
    print(f"\n=== 汇总 ===")
    print(f"  超额扣费年化: {net_excess.mean()*ann:.2%}  IR={net_excess.mean()/net_excess.std()*ann**0.5:.2f}")


if __name__ == "__main__":
    main()

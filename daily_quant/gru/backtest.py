"""
Backtest evaluation for Alpha158Date model (171 features with BoardLimit).

Usage
-----
    python backtest_eval_date.py
    python backtest_eval_date.py --model_id <id> --topk 30 --no_limit
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
from qlib.data import D

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


class FixedNormalizedTSDataSampler(TSDataSampler):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        process_data = data[:, 0:-1]
        if process_data.shape[0] == 0:
            return data
        with np.errstate(all="ignore"):
            data_mean = np.nanmean(process_data, axis=0)
            data_std = np.nanstd(process_data, axis=0)
        data_mean = np.where(np.isnan(data_mean), 0, data_mean)
        data_std = np.where(np.isnan(data_std) | (data_std < 1e-5), 1.0, data_std)
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
    parser = argparse.ArgumentParser(description="Backtest Alpha158Date (171-feat) model")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--instruments", default="mid_cap")
    parser.add_argument("--step_len", type=int, default=60)
    parser.add_argument("--backtest_start", default="2025-07-01")
    parser.add_argument("--backtest_end", default="2026-05-27")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n_drop", type=int, default=5)
    parser.add_argument("--account", type=int, default=10_000_000)
    parser.add_argument("--no_limit", action="store_true", help="Skip limit_threshold for diagnosis")
    parser.add_argument("--board_limit", action="store_true", help="Use BoardLimit expression instead of float threshold")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_fwd")
    parser.add_argument("--exp_name", default="GRU_mid_cap_60d")
    args = parser.parse_args()

    PROVIDER_URI = args.qlib_data_dir
    EXP_NAME = args.exp_name
    BACKTEST_EXP = "backtest_gru_date_v3"

    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN, custom_ops=_CUSTOM_OPS)
    print(f"Data: {PROVIDER_URI}")

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
    print(f"Model d_feat: {model.d_feat}")

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

    # ========================================================================
    # Step 1: Quick diagnosis — prediction distribution on first day
    # ========================================================================
    print("\n" + "=" * 60)
    print("=== 诊断：第一天预测分数 ===")
    pred_all = model.predict(dataset)
    first_day = pred_all.index.get_level_values("datetime").min()
    pred_day = pred_all.loc[pred_all.index.get_level_values("datetime") == first_day]
    print(f"日期: {first_day.strftime('%Y-%m-%d')}, 预测数: {len(pred_day)}")
    print(f"Score: min={pred_day.min():.4f}, max={pred_day.max():.4f}, mean={pred_day.mean():.4f}, std={pred_day.std():.4f}")
    print(f"Score > 0: {(pred_day > 0).sum()}, NaN: {pred_day.isna().sum()}")
    print(f"\nTop 10:")
    for (dt, inst), val in pred_day.nlargest(10).items():
        print(f"  {inst}  {val:.6f}")
    print(f"\nBottom 5:")
    for (dt, inst), val in pred_day.nsmallest(5).items():
        print(f"  {inst}  {val:.6f}")

    # Check $open for top stocks
    top5_instruments = [inst for (dt, inst) in pred_day.nlargest(5).index]
    opens = D.features(top5_instruments, ["$open", "$close"], start_time=first_day.strftime("%Y-%m-%d"), end_time=first_day.strftime("%Y-%m-%d"))
    print(f"\nTop5 $open/$close on {first_day.strftime('%Y-%m-%d')}:")

    for inst in top5_instruments:
        try:
            row = opens.loc[(inst, slice(None)), :]
            if len(row) > 0:
                r = row.iloc[0]
                print(f"  {inst}  open={float(r['$open']):.3f}  close={float(r['$close']):.3f}")
            else:
                print(f"  {inst}  NO DATA")
        except Exception:
            print(f"  {inst}  ERROR")

    # ========================================================================
    # Step 2: Backtest WITHOUT limit_threshold
    # ========================================================================
    print("\n" + "=" * 60)
    print("=== 回测（无涨跌停限制） ===")

    exchange_kwargs = {
        "freq": "day",
        "deal_price": "open", "open_cost": 0.0005,
        "close_cost": 0.0015, "min_cost": 5,
        "limit_threshold": None,  # strategy handles limit checks, not exchange
    }
    print("limit checks: Strategy-level BoardLimit (10/20/30%)" if not args.no_limit else "limit checks: DISABLED")

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "strategy": {
            "class": "BoardLimitTopkDropoutStrategy",
            "module_path": "daily_quant.strategies",
            "kwargs": {
                "model": model, "dataset": dataset,
                "topk": args.topk, "n_drop": args.n_drop,
                "check_limit": not args.no_limit,
            },
        },
        "backtest": {
            "start_time": args.backtest_start,
            "end_time": args.backtest_end,
            "account": args.account,
            "benchmark": "SH000300",
            "exchange_kwargs": exchange_kwargs,
        },
    }

    with R.start(experiment_name=BACKTEST_EXP):
        sr_recorder = R.get_recorder()
        ba_rid = sr_recorder.id
        sr = SignalRecord(model, dataset, sr_recorder)
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

    print("\n=== 策略收益 ===")
    print(risk_analysis(report["return"], freq="1day"))

    print("\n=== 超额收益（未扣费） ===")
    excess = report["return"] - report["bench"]
    print(risk_analysis(excess, freq="1day"))

    print("\n=== 汇总 ===")
    ann_factor = 238
    strat_ret = report["return"].mean() * ann_factor
    bench_ret = report["bench"].mean() * ann_factor
    cum_strat = (report["return"] + 1).cumprod().iloc[-1] - 1
    cum_bench = (report["bench"] + 1).cumprod().iloc[-1] - 1
    dd = (report["return"] + 1).cumprod() / (report["return"] + 1).cumprod().cummax() - 1
    print(f"  策略: 年化={strat_ret:.2%}  累计={cum_strat:.2%}  夏普={report['return'].mean()/report['return'].std()*ann_factor**0.5:.2f}  最大回撤={dd.min():.2%}")
    print(f"  基准: 年化={bench_ret:.2%}  累计={cum_bench:.2%}")
    print(f"  跑赢天数: {(excess > 0).mean():.1%}")
    print(f"  策略胜率: {(report['return'] > 0).mean():.1%}")


if __name__ == "__main__":
    main()

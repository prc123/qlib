"""Check NaN ratio in features after ZScoreNorm — same pipeline as training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.config import C
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import TSDatasetH

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]


def main():
    C["joblib_backend"] = "sequential"

    qlib.init(custom_ops=_CUSTOM_OPS,
              provider_uri=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd", region=REG_CN)

    # Same config as train_date.py
    cfg = {
        "start_time": "2022-01-01", "end_time": "2026-05-28",
        "fit_start_time": "2022-01-01", "fit_end_time": "2024-12-31",
        "instruments": "mid_cap",
        "infer_processors": [
            {"class": "ZScoreNorm"},
            {"class": "FillnaFeature", "module_path": "daily_quant.ops.fillna_processor",
             "kwargs": {"fields_group": "feature"}},
        ],
    }

    dataset = TSDatasetH(
        handler={"class": "Alpha158Date", "module_path": "daily_quant.handler.alpha158_date", "kwargs": cfg},
        segments={"train": ("2022-01-01", "2024-12-31"), "valid": ("2025-01-01", "2025-06-30")},
        step_len=60,
    )

    # Check raw handler data (_infer = after ZScoreNorm)
    dh = dataset.handler
    infer_data = dh._infer
    print(f"_infer (after ZScoreNorm) shape: {infer_data.shape}")

    # NaN ratio per feature
    nan_ratio = infer_data.isna().mean()
    nan_features = nan_ratio[nan_ratio > 0].sort_values(ascending=False)
    print(f"Features with NaN: {len(nan_features)} / {len(nan_ratio)}")
    if len(nan_features) > 0:
        print(f"\nTop 15 features by NaN ratio:")
        for col, ratio in nan_features.head(15).items():
            print(f"  {col}: {ratio:.2%}")

    # Check a sample through the sampler
    train_sampler = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    sample = train_sampler[0]
    nan_in_sample = np.isnan(sample[:, :-1]).sum()
    total_values = sample[:, :-1].size
    print(f"\nSample 0 (train): shape={sample.shape}, NaN features={nan_in_sample}/{total_values} ({nan_in_sample/total_values:.2%})")

    # After fillna (as model does)
    train_sampler.config(fillna_type="ffill+bfill")
    sample2 = train_sampler[0]
    nan_after = np.isnan(sample2[:, :-1]).sum()
    print(f"After ffill+bfill: NaN features={nan_after}/{total_values} ({nan_after/total_values:.2%})")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()

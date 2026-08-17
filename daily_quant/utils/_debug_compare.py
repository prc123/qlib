"""
Debug v8: test cuDNN non-determinism — run with deterministic mode + compare.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import get_or_create_path
from qlib.model.utils import ConcatDataset
from qlib.contrib.model.pytorch_gru_ts import GRU

from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]

class GRUWithProgress(GRU):
    def fit(self, dataset, evals_result=dict(), save_path=None, reweighter=None):
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset")
        dl_train.config(fillna_type="ffill+bfill"); dl_valid.config(fillna_type="ffill+bfill")
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
            evals_result["train"].append(train_score); evals_result["valid"].append(val_score)
            if val_score > best_score:
                best_score, stop_steps, best_epoch = val_score, 0, step
                best_param = copy.deepcopy(self.GRU_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    break
        self.GRU_model.load_state_dict(best_param)
        torch.save(best_param, save_path)


PROVIDER_URI = "C:/Users/pp/.qlib/qlib_data/cn_data_bwd"
TEST_DATE = "2026-06-29"

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # === Force deterministic cuDNN ===
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("cuDNN deterministic:", torch.backends.cudnn.deterministic)
    print("cuDNN benchmark:", torch.backends.cudnn.benchmark)

    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN, custom_ops=_CUSTOM_OPS)

    EXP_NAME = "GRU_mid_cap_60d"
    recs_dict = R.list_recorders(experiment_name=EXP_NAME)
    recs = [r for r in recs_dict.values() if "trained_model" in r.list_artifacts()]
    recs.sort(key=lambda r: r.info.get("end_time") or "", reverse=True)
    recorder = R.get_recorder(recorder_id=recs[0].id, experiment_name=EXP_NAME)
    model = recorder.load_object("trained_model")
    print(f"Model batch_size: {model.batch_size}, device: {model.device}")

    base_hcfg = {"start_time": "2016-01-01", "fit_start_time": "2016-01-01",
                 "fit_end_time": "2024-12-31", "instruments": "mid_cap_filtered",
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
                 ]}

    scores = {}
    for label, seg, hcfg_end in [
        ("short (06-29~06-30)", ("2026-06-29", "2026-06-30"), "2026-06-29"),
        ("long  (05-01~07-01)", ("2026-05-01", "2026-07-01"), "2026-07-01"),
    ]:
        cfg = {**base_hcfg, "end_time": hcfg_end}
        ds = TSDatasetH(
            handler={"class": "Alpha158Date", "module_path": "daily_quant.handler.alpha158_date", "kwargs": cfg},
            segments={"test": seg}, step_len=60,
        )
        pred = model.predict(ds)
        day = pred.loc[pred.index.get_level_values("datetime") == pd.Timestamp(TEST_DATE)]
        scores[label] = day
        print(f"\n{label}: {len(day)} stocks, mean={day.mean():.4f}")
        del ds; import gc; gc.collect()

    # Compare
    s_short = scores["short (06-29~06-30)"]
    s_long = scores["long  (05-01~07-01)"]
    common = s_short.index.intersection(s_long.index)
    diff = (s_short.loc[common] - s_long.loc[common]).abs()
    print(f"\n=== With cuDNN deterministic=True ===")
    print(f"Common stocks: {len(common)}")
    print(f"Max abs diff: {diff.max():.10f}")
    print(f"Corr: {s_short.loc[common].corr(s_long.loc[common]):.6f}")
    for sym in ["SZ000753", "SZ002263"]:
        if sym in common:
            print(f"  {sym}: short={float(s_short.loc[(pd.Timestamp(TEST_DATE), sym)]):.6f}  long={float(s_long.loc[(pd.Timestamp(TEST_DATE), sym)]):.6f}")

"""
ETF daily pipeline orchestrator: update -> retrain -> predict.

Runs the three steps in sequence and exits non-zero on any failure so the
Windows Task Scheduler can report a clean success/failure code.

Usage (scheduled task points directly at this file):
    C:\\Users\\pp\\.conda\\envs\\qlib\\python.exe E:\\kaggle_code\\qlib\\scripts\\data_collector\\tushare\\etf_daily_run.py
"""
import os
import subprocess
import sys
from datetime import datetime

# --- proxy: Tushare requires NO_PROXY=*, otherwise ProxyError ---
for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(key, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

PY = r"C:\Users\pp\.conda\envs\qlib\python.exe"
ROOT = r"E:\kaggle_code\qlib"
TUSHARE_DIR = os.path.join(ROOT, "scripts", "data_collector", "tushare")
QLIB_DATA = r"C:\Users\pp\.qlib\qlib_data\etf_data"

LOG_PATH = os.path.join(TUSHARE_DIR, "etf_daily_run.log")


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(cmd: str, cwd: str) -> int:
    log(f"$ {cmd}")
    result = subprocess.run(cmd, cwd=cwd, shell=True)
    return result.returncode


def main() -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    log("=" * 60)
    log(f"ETF daily run start, today={today}")

    steps = [
        (
            f'"{PY}" etf_daily_update.py --qlib_data_1d_dir "{QLIB_DATA}"',
            TUSHARE_DIR,
            "Step 1/3: data update",
        ),
        (
            f'"{PY}" daily_quant\\xgboost\\train_kfold.py --instruments stock '
            f"--n_folds 3 --label_type sharpe --share_only "
            f"--valid_end {today} --exp_prefix XGB_Current",
            ROOT,
            "Step 2/3: retrain model",
        ),
        (
            f'"{PY}" daily_quant\\xgboost\\predict_kfold.py '
            f"--exp_prefix XGB_Current --n_folds 3 --topk 30",
            ROOT,
            "Step 3/3: predict",
        ),
    ]

    for cmd, cwd, label in steps:
        log(label)
        code = run_step(cmd, cwd)
        if code != 0:
            log(f"[ERROR] {label} failed with exit code {code}")
            return 1

    log("ETF daily run completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())

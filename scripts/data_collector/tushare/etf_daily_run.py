"""
ETF daily pipeline orchestrator: update -> retrain -> predict.

Runs the steps in sequence and exits non-zero on any failure so the
Windows Task Scheduler can report a clean success/failure code.

Usage
-----
    python etf_daily_run.py                 # update data, retrain, then predict
    python etf_daily_run.py --no-train      # predict with the current models
    python etf_daily_run.py --no-update     # predict with current data only
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# --- proxy: Tushare requires NO_PROXY=*, otherwise ProxyError ---
for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(key, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"
# Newer MLflow refuses the file-based `mlruns` backend unless opted in.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
# Keep daily runs away from metadata copied from older workspace clones.
os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    "file:" + str(Path(__file__).resolve().parents[3] / "mlruns_daily"),
)

ROOT = Path(__file__).resolve().parents[3]
TUSHARE_DIR = Path(__file__).resolve().parent
PY = sys.executable
QLIB_DATA = str(Path.home() / ".qlib" / "qlib_data" / "etf_data")

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
    parser = argparse.ArgumentParser(description="ETF daily update + retrain + predict")
    parser.add_argument("--no-train", action="store_true",
                        help="Skip retraining and use the current k-fold models")
    parser.add_argument("--no-update", action="store_true",
                        help="Skip the data update step")
    parser.add_argument("--instruments", default="all")
    parser.add_argument("--exp_prefix", default="XGB_KFold")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--topk", type=int, default=30)
    args = parser.parse_args()

    today = datetime.now().strftime("%Y-%m-%d")
    log("=" * 60)
    log(f"ETF daily run start, today={today}")

    steps = []
    if not args.no_update:
        steps.append((
            f'"{PY}" "{TUSHARE_DIR / "etf_daily_update.py"}" '
            f'--qlib_data_1d_dir "{QLIB_DATA}"',
            str(TUSHARE_DIR),
            "Step: data update",
        ))
    if not args.no_train:
        steps.append((
            f'"{PY}" "{ROOT / "daily_quant" / "xgboost" / "train_kfold.py"}" '
            f"--instruments {args.instruments} "
            f"--qlib_data_dir \"{QLIB_DATA}\" "
            f"--n_folds {args.n_folds} --label_type sharpe --share_only "
            f"--valid_end {today} --exp_prefix {args.exp_prefix}",
            str(ROOT),
            "Step: retrain model",
        ))
    steps.append((
        f'"{PY}" "{ROOT / "daily_quant" / "xgboost" / "predict_kfold.py"}" '
        f"--instruments {args.instruments} --qlib_data_dir \"{QLIB_DATA}\" "
        f"--exp_prefix {args.exp_prefix} --n_folds {args.n_folds} --topk {args.topk}",
        str(ROOT),
        "Step: predict",
    ))

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

"""
每日收盘后一键执行：更新数据 → 预测 → 生成交易信号。

Usage
-----
    $ python daily_quant/daily_run.py

Windows Task Scheduler
-----------------------
    Program: C:\\Users\\pp\\.conda\\envs\\qlib\\python.exe
    Arguments: daily_run.py
    Start in: E:\\kaggle_code\\qlib\\daily_quant
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
CUR_DIR = Path(__file__).resolve().parent


def _clean_env():
    """Copy environ without proxy vars (avoid dead 127.0.0.1:7897 timeout)."""
    env = os.environ.copy()
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
        env.pop(k, None)
    return env


def _run(python: str, script: Path, cwd: Path, args: list[str] = None):
    cmd = [python, str(script)] + (args or [])
    print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), env=_clean_env())
    if result.returncode != 0:
        print(f"[FAIL] {script.name} returned {result.returncode}")
        sys.exit(result.returncode)
    print()


def main():
    python = sys.executable
    qlib_dir = r"C:\Users\pp\.qlib\qlib_data\cn_data_fwd"
    tushare_dir = ROOT / "scripts" / "data_collector" / "tushare"

    print(f"{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 每日任务开始")

    # Step 1: 更新数据 (前复权版)
    print(f"\n{'='*60}")
    print("Step 1/3: 更新 qlib 数据 (Tushare → qlib binary)")
    _run(python,
         tushare_dir / "daily_update_fwd.py",
         cwd=tushare_dir,
         args=["--qlib_data_1d_dir", qlib_dir, "--delay", "0.3"])

    # Step 2: 预测
    print(f"{'='*60}")
    print("Step 2/3: 生成预测分数")
    _run(python,
         CUR_DIR / "predict.py",
         cwd=ROOT,
         args=["--qlib_dir", qlib_dir, "--no-update"])

    # Step 3: 交易信号
    print(f"{'='*60}")
    print("Step 3/3: 生成交易信号")
    _run(python,
         CUR_DIR / "trade_signals.py",
         cwd=CUR_DIR)

    print(f"{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 每日任务完成")


if __name__ == "__main__":
    main()

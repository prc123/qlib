"""
每日收盘后一键执行：更新数据 → 预测 → 生成交易信号。

输出同时显示在终端和写入日志文件 ``daily_run.log``。

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
import io
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
CUR_DIR = Path(__file__).resolve().parent

LOG_FILE = CUR_DIR / "daily_run.log"
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy")


def _clean_env():
    """Copy environ without proxy vars (avoid dead 127.0.0.1:7897 timeout)."""
    env = os.environ.copy()
    for k in PROXY_KEYS:
        env.pop(k, None)
    return env


class Tee:
    """Write to both console and log file."""
    def __init__(self, log_path: Path):
        self.file = open(str(log_path), "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        try:
            self.stdout.write(s)
        except UnicodeEncodeError:
            # Fallback: encode to GBK (console encoding), replace unencodable chars
            self.stdout.write(s.encode("gbk", errors="replace").decode("gbk"))
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def log(msg: str = ""):
    print(msg)


def _run(python: str, script: Path, cwd: Path, args: list[str] = None):
    cmd = [python, str(script)] + (args or [])
    log(f"[run] {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=str(cwd), env=_clean_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    log(result.stdout)
    if result.returncode != 0:
        log(f"[FAIL] {script.name} returned {result.returncode}")
        sys.exit(result.returncode)


def main():
    python = sys.executable
    qlib_dir = r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd"
    tushare_dir = ROOT / "scripts" / "data_collector" / "tushare"

    tee = Tee(LOG_FILE)
    sys.stdout = tee

    try:
        log(f"{'='*60}")
        log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 每日任务开始")
        log()

        # Step 1: 更新数据 (后复权版)
        log(f"{'='*60}")
        log("Step 1/3: 更新 qlib 数据 (Tushare → qlib binary)")
        _run(python,
             tushare_dir / "daily_update_bwd.py",
             cwd=tushare_dir,
             args=["--qlib_data_1d_dir", qlib_dir, "--delay", "0.3"])

        # Step 2: 预测
        log(f"{'='*60}")
        log("Step 2/3: 生成预测分数")
        _run(python,
             CUR_DIR / "gru" / "predict.py",
             cwd=CUR_DIR,
             args=["--qlib_dir", qlib_dir, "--instruments", "mid_cap_filtered"])

        # Step 3: 交易信号
        log(f"{'='*60}")
        log("Step 3/3: 生成交易信号")
        _run(python,
             CUR_DIR / "trade_signals.py",
             cwd=CUR_DIR)

        log(f"{'='*60}")
        log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 每日任务完成")
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()

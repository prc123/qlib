# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Microsoft's Qlib (AI-oriented quant platform) with a custom daily trading pipeline for Chinese A-shares. The hybrid structure: official `qlib/` library for infrastructure/backtesting, plus `daily_quant/` for daily GRU/CatBoost prediction and signal generation, and `scripts/data_collector/tushare/` for Tushare-based data ingestion.

## Environment

- Python: `C:\Users\pp\.conda\envs\qlib\python.exe` (conda env `qlib`)
- Qlib data dir: `C:\Users\pp\.qlib\qlib_data\cn_data_fwd` (forward-adjusted)
- Tushare token hardcoded in collector and several daily_quant scripts

## Common Commands

```bash
# Install in dev mode (includes Cython compilation)
make prerequisite && pip install -e .

# Run tests
pytest qlib/tests/test_all_pipeline.py

# Lint
make lint  # black + pylint + flake8 + mypy

# Daily quant pipeline (orchestrates update → predict → signals)
python daily_quant/daily_run.py

# Individual steps of the pipeline
python scripts/data_collector/tushare/daily_update_fwd.py --qlib_data_1d_dir C:\Users\pp\.qlib\qlib_data\cn_data_fwd
python daily_quant/predict.py
python daily_quant/trade_signals.py

# Training
python daily_quant/train_date.py --instruments CSI300 --n_epochs 50   # GRU
python daily_quant/train_catboost.py --step_len 60                    # CatBoost

# Quick validation (~5 min, uses SimpleRNN)
python daily_quant/quick_test.py

# Backtest evaluation
python daily_quant/backtest_eval_date.py --topk 30

# CLI workflow runner (official qrun)
python -m qlib.cli.run <config.yaml>
```

## Architecture

**Daily pipeline flow** (orchestrated by `daily_quant/daily_run.py`):
```
Step 1: Tushare API → daily_update_fwd.py → normalize → dump_bin → qlib binary data
Step 2: Load GRU model from MLflow → predict.py → pred_score_*.csv
Step 3: RiskMonitor check → TopkDropout logic → trade_signals.csv
```

**Key relationships:**
- All `daily_quant/` scripts use `sys.path.insert(0, root)` so `import qlib` works from any directory.
- `daily_quant/ops/date_ops.py` registers custom operators (DayOfWeek, Month, BoardLimit, etc.) via `C.custom_ops` — must be imported before any feature expression uses them.
- `daily_quant/handler/alpha158_date.py` extends Alpha158 with 6 fundamental + 7 date/board fields → 171 features total.
- `daily_quant/strategies.py` extends Qlib's TopkDropoutStrategy with board-specific price-limit filtering (5% ST, 10% main, 20% ChiNext/STAR, 30% BJ).
- `daily_quant/risk_monitor.py` computes 9 risk signals (rolling win rate, Sharpe, drawdown, etc.) with GREEN/YELLOW/RED thresholds — called before trade signal generation.
- `daily_quant/flatten_ts_dataset.py` provides `FlattenedTSDatasetH` for tree models (CatBoost) that need tabular data instead of time-series tensors.

**Data pipeline:**
```
Tushare API → raw CSV (source/) → normalize (normalize/) → qlib binary (.bin) → D.features() API
```

**Qlib core:**
- `qlib/data/data.py` — `D.features()`, `D.instruments()`, `D.calendar()` are the main data API surface.
- `qlib/backtest/` — backtesting engine; strategies consume signals, simulate execution, produce reports.
- `qlib/workflow/` — MLflow-based experiment management; `R` global recorder tracks params/metrics/artifacts.
- `qlib/data/_libs/` — Cython-accelerated rolling/expanding window ops (compiled via `setup.py`).

## Windows-Specific

- Paths use backslashes in .bat files, forward slashes in Python.
- Windows Task Scheduler runs Mon-Fri at 18:00 via `daily_quant/setup_task.bat`.
- Proxy env vars (`HTTP_PROXY`, `HTTPS_PROXY`) are cleared before Tushare API calls.

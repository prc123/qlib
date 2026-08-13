# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Microsoft's Qlib (AI-oriented quant platform) with a custom daily trading pipeline for Chinese A-shares and ETFs. The hybrid structure: official `qlib/` library for infrastructure/backtesting, plus `daily_quant/` for daily GRU/CatBoost/XGBoost prediction and signal generation, and `scripts/data_collector/tushare/` for Tushare-based data ingestion.

## Environment

- Python: `C:\Users\pp\.conda\envs\qlib\python.exe` (conda env `qlib`)
- Qlib data dirs:
  - Stock: `C:\Users\pp\.qlib\qlib_data\cn_data_bwd` (backward-adjusted)
  - ETF: `C:\Users\pp\.qlib\qlib_data\etf_data` (fund_daily + fund_adj + share + nav)
- Tushare token hardcoded in collector and several daily_quant scripts
- Proxy env vars must be cleared (`NO_PROXY=*`) before Tushare API calls, otherwise calls fail with `ProxyError`

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
python scripts/data_collector/tushare/daily_update_bwd.py --qlib_data_1d_dir C:\Users\pp\.qlib\qlib_data\cn_data_bwd
python daily_quant/gru/predict.py
python daily_quant/trade_signals.py

# Training
python daily_quant/gru/train.py --instruments CSI300 --n_epochs 50   # GRU
python daily_quant/catboost/train.py --step_len 60                   # CatBoost

# Quick validation (~5 min, uses SimpleRNN)
python daily_quant/gru/quick_test.py

# Backtest evaluation
python daily_quant/gru/backtest.py --topk 30

# --- ETF pipeline (XGBoost kfold) ---
# Full download
python scripts/data_collector/tushare/etf_full_download.py
# Daily update (Windows scheduled task: Mon-Fri 18:07)
python scripts/data_collector/tushare/etf_daily_update.py --qlib_data_1d_dir C:\Users\pp\.qlib\qlib_data\etf_data
# K-fold training + ensemble backtest + prediction
python daily_quant/xgboost/train_kfold.py --instruments stock --n_folds 3 --label_type sharpe --share_only
python daily_quant/xgboost/backtest_kfold.py --instruments stock --n_folds 3
python daily_quant/xgboost/predict_kfold.py --exp_prefix XGB_Current --n_folds 3

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
- `daily_quant/handler/alpha158_date.py` extends Alpha158 with 6 fundamental + 7 date/board fields → 170 features total (VWAP removed, data source lacks `$vwap`).
- `daily_quant/handler/alpha158_etf.py` adds ETF-specific factors on top of Alpha158Date: `SHARE_CHG` ($share change rate, requires fund_share data) + `PREM_DISC` ($close/$factor/$nav premium, requires fund_nav). Supports `label_type`: return/sharpe/ret_vol/score.
- `daily_quant/strategies.py` extends Qlib's TopkDropoutStrategy with board-specific price-limit filtering (5% ST, 10% main, 20% ChiNext/STAR, 30% BJ).
- `daily_quant/risk_monitor.py` computes 9 risk signals (rolling win rate, Sharpe, drawdown, etc.) with GREEN/YELLOW/RED thresholds — called before trade signal generation.
- `daily_quant/flatten_ts_dataset.py` provides `FlattenedTSDatasetH` for tree models (CatBoost) that need tabular data instead of time-series tensors.

**ETF pipeline** (`daily_quant/xgboost/` + `scripts/data_collector/tushare/etf_*.py`):
```
Tushare fund_daily/fund_adj/fund_share/fund_nav → etf_source/*.csv → normalize → qlib binary → Alpha158ETF handler → XGBoost kfold → ensemble predict
```
- `etf_collector.py` — ETF download (fund_daily + fund_adj), bulk-by-date.
- `etf_extra_fields.py` — downloads share (fund_share) + nav (fund_nav), merges into source CSVs.
- `etf_daily_update.py` — incremental daily update (OHLCV + share + nav), full re-normalize + re-dump.
- `etf_daily_run.bat` — Windows scheduled task: update → retrain → predict.
- `train_kfold.py` — walk-forward k-fold training (expanding window, no look-ahead).
- `backtest_kfold.py` / `predict_kfold.py` — ensemble K models (average predictions), backtest/predict.
- Best config: Alpha158ETF (share-only) + sharpe label, XGBoost depth=8, 3-fold ensemble, daily freq, n_drop=3, topk=30 → net excess ~32% (IR 2.1).

**Data pipeline:**
```
Tushare API → raw CSV (source/) → normalize (normalize/) → qlib binary (.bin) → D.features() API
```

**Qlib core:**
- `qlib/data/data.py` — `D.features()`, `D.instruments()`, `D.calendar()` are the main data API surface.
- `qlib/backtest/` — backtesting engine; strategies consume signals, simulate execution, produce reports.
- `qlib/workflow/` — MLflow-based experiment management; `R` global recorder tracks params/metrics/artifacts.
- `qlib/data/_libs/` — Cython-accelerated rolling/expanding window ops (compiled via `setup.py`).

## Constraints

- **Do NOT modify files under `qlib/`.** The `qlib/` directory is the upstream library source and must remain untouched. All customizations go in `daily_quant/` or `scripts/`. Use string-based config (e.g., `{"class": "ZScoreNorm"}`) or subclassing to extend qlib behavior without editing its source.
- Feature count was changed from 171 to 170 (removed VWAP — data source doesn't provide `$vwap`).

## Windows-Specific

- Paths use backslashes in .bat files, forward slashes in Python.
- Windows Task Scheduler:
  - Stock pipeline: Mon-Fri at 18:00 via `daily_quant/setup_task.bat`.
  - ETF pipeline: Mon-Fri at 18:07 via task `ETF_Daily_Update` (runs `scripts/data_collector/tushare/etf_daily_run.bat`).
- Proxy env vars (`HTTP_PROXY`, `HTTPS_PROXY`) must be cleared (`NO_PROXY=*`) before Tushare API calls, otherwise `ProxyError` is raised.
- Chinese Windows `%date%` format is `YYYY/MM/DD`; .bat scripts convert `/` → `-` for date-stamped filenames.

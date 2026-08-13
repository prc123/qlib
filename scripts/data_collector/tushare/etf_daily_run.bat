@echo off
REM ============================================================
REM  ETF daily incremental update + retrain + predict
REM  Schedule: Windows Task Scheduler, Mon-Fri 18:00
REM ============================================================

chcp 65001 >nul
cd /d E:\kaggle_code\qlib\scripts\data_collector\tushare

set NO_PROXY=*
set no_proxy=*
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=

set PY=C:\Users\pp\.conda\envs\qlib\python.exe
set QLIB_DATA=C:\Users\pp\.qlib\qlib_data\etf_data

REM 转换日期格式 2026/08/13 -> 2026-08-13
set TODAY=%date:~0,10%
set TODAY=%TODAY:/=-%

echo [%date% %time%] === ETF daily update start ===

REM Step 1: Incremental data update (OHLCV + share + nav)
echo [%date% %time%] Step 1: data update
%PY% etf_daily_update.py --qlib_data_1d_dir %QLIB_DATA%
if errorlevel 1 (
    echo [ERROR] ETF data update failed
    exit /b 1
)

REM Step 2: Retrain k-fold model (current)
echo [%date% %time%] Step 2: retrain model
cd /d E:\kaggle_code\qlib
%PY% daily_quant\xgboost\train_kfold.py --instruments stock --n_folds 3 --label_type sharpe --share_only --valid_end %TODAY% --exp_prefix XGB_Current
if errorlevel 1 (
    echo [ERROR] Model training failed
    exit /b 1
)

REM Step 3: Predict all ETF scores (top30 recommended)
echo [%date% %time%] Step 3: predict
%PY% daily_quant\xgboost\predict_kfold.py --exp_prefix XGB_Current --n_folds 3 --topk 30
if errorlevel 1 (
    echo [ERROR] Prediction failed
    exit /b 1
)

echo [%date% %time%] === ETF daily update done ===
exit /b 0

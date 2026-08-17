@echo off
cd /d E:\kaggle_code\qlib\daily_quant
call C:\Users\pp\.conda\envs\qlib\python.exe daily_run.py
exit /b %errorlevel%

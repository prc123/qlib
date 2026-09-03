@echo off
cd /d "%~dp0..\.."
call "C:\ProgramData\anaconda3\envs\qlib\python.exe" "%~dp0etf_daily_run.py" %*
exit /b %errorlevel%

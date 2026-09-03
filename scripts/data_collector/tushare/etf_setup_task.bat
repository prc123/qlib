@echo off
:: Register the daily ETF update + predict task (weekdays 18:00).
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

set TASK_BAT=%~dp0etf_daily_run.bat
schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /TN "QlibEtfDaily" /TR "\"%TASK_BAT%\"" /ST 18:00 /F
if %errorlevel% equ 0 (
    echo Task QlibEtfDaily created: weekdays 18:00, command: %TASK_BAT%
) else (
    echo Failed to create task QlibEtfDaily.
)
pause

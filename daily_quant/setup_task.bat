@echo off
:: Check admin privileges
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /TN "QlibDailyUpdate" /TR "E:\kaggle_code\qlib\daily_quant\daily_run.bat" /ST 18:00 /F
echo Task created successfully.
pause
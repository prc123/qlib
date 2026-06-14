schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /TN "QlibDailyUpdate" /TR "E:\kaggle_code\qlib\daily_quant\daily_run.bat" /ST 18:00 /F

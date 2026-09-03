import sys
from pathlib import Path
sys.path.insert(0, str(Path("C:/Users/Administrator/Documents/qlib")))
from daily_quant.ops.date_ops import DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit
import qlib
from qlib.data import D
from qlib.constant import REG_CN

_CUSTOM_OPS = [DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit]

qlib.init(
    provider_uri=r"C:\Users\Administrator\.qlib\qlib_data\etf_data",
    region=REG_CN,
    custom_ops=_CUSTOM_OPS,
    kernels=4,
)

inst = D.instruments()
fields = ["$close", "$volume"]
print("fields:", fields)
try:
    df = D.features(inst, fields, start_time="2024-01-01", end_time="2024-01-05", freq="day")
    print("OK, shape:", df.shape)
except Exception as e:
    print("FAILED:", e)
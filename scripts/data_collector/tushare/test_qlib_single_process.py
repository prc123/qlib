import qlib
from qlib.data import D
from qlib.constant import REG_CN

qlib.init(
    provider_uri=r"C:\Users\Administrator\.qlib\qlib_data\etf_data",
    region=REG_CN,
    kernels=1,  # 单进程模式
)

inst = D.instruments()
cal = D.calendar()
fields = ["$close", "$volume"]

print(f"instruments count: {len(list(D.list_instruments(inst)))}")
print(f"calendar range: {cal[0].date()} -> {cal[-1].date()}")
print(f"fields: {fields}")

df = D.features(inst, fields, start_time="2024-01-01", end_time="2024-01-05", freq="day")
print("D.features succeeded!")
print(f"shape: {df.shape}")
print(df.head())
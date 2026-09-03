import time
import qlib
from qlib.data import D
from qlib.constant import REG_CN

def test_kernels(n):
    print(f"\n=== Testing kernels={n} ===")
    qlib.init(
        provider_uri=r"C:\Users\Administrator\.qlib\qlib_data\etf_data",
        region=REG_CN,
        kernels=n,
    )
    inst = D.instruments()
    fields = ["$close", "$volume"]
    start = time.time()
    try:
        df = D.features(inst, fields, start_time="2024-01-01", end_time="2024-06-30", freq="day")
        print(f"OK, shape={df.shape}, elapsed={time.time()-start:.2f}s")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    test_kernels(1)
    test_kernels(2)
    test_kernels(4)
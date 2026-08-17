"""Quick check: are alpha factor fields available in qlib binary data?"""
import os
import sys
from pathlib import Path

# Limit parallelism to reduce memory pressure on Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from daily_quant.ops.date_ops import (
    DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit,
)


def main():
    qlib.init(
        provider_uri=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd",
        region=REG_CN,
        expression_cache=None,
        dataset_cache=None,
        custom_ops=[DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit],
    )
    print("qlib init done")

    last_date = str(D.calendar()[-1])
    print(f"Last trading date: {last_date}")
    print(f"Calendar days: {len(D.calendar())}")
    print(f"All instruments: {len(D.instruments('all'))}")

    fields = [
        "$net_mf_amount", "$buy_lg_amount", "$sell_lg_amount",
        "$buy_elg_amount", "$sell_elg_amount",
        "$rzye", "$rqye", "$rzmre", "$rzche", "$rqyl",
    ]
    print(f"\nChecking {len(fields)} alpha fields on CSI300 at {last_date}...")
    val = D.features(D.instruments("csi300"), fields, start_time=last_date, end_time=last_date)

    for f in fields:
        n = val[f].notna().sum() if f in val.columns else 0
        pct = n / len(val) * 100 if len(val) > 0 else 0
        print(f"  {f:25s}: {n:3d}/300 ({pct:.0f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()

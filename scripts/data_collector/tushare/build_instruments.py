"""
Build instrument files by industry x market cap cross-classification.

Usage
-----
    $ python build_instruments.py --qlib_dir C:/Users/pp/.qlib/qlib_data/cn_data_bwd
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import tushare as ts
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

TUSHARE_TOKEN = "a2d2e11f92720e2a69fb021f4fa098a8a8e56c4800d0040b72306202"

# Tier definitions: within each industry, split by market cap percentile
# large: top 15%, mid: 15-50%, small: bottom 50%
TIERS = {
    "large_cap":  (0.0, 0.15),
    "mid_cap":    (0.15, 0.50),
    "small_cap":  (0.50, 1.0),
}


def build_instruments(
    qlib_dir: str = r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd",
    date: str = None,
):
    """Build industry x market-cap instrument files.

    Parameters
    ----------
    qlib_dir : str
        Path to qlib 1d data (containing features/, instruments/, calendars/).
    date : str
        Reference date for market cap snapshot. Default: latest calendar date.
    """
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    qlib_dir = str(Path(qlib_dir).expanduser().resolve())
    instr_dir = Path(qlib_dir) / "instruments"

    # --- Step 1: Get all symbols from qlib ---
    all_df = pd.read_csv(
        instr_dir / "all.txt",
        sep="\t",
        header=None,
        names=["symbol", "start", "end"],
    )
    qlib_symbols = set(all_df["symbol"].str.upper().tolist())
    logger.info(f"Qlib instruments: {len(qlib_symbols)}")

    # --- Step 2: Fetch industry from Tushare ---
    pro = ts.pro_api(TUSHARE_TOKEN)
    df_tushare = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,name,industry",
    )
    df_tushare["qlib_sym"] = df_tushare["ts_code"].apply(
        lambda x: (x.split(".")[1] + x.split(".")[0]).upper()
    )
    df_tushare = df_tushare[df_tushare["qlib_sym"].isin(qlib_symbols)]
    logger.info(f"Industry data matched: {len(df_tushare)} stocks")

    # --- Step 3: Get market cap from qlib ---
    if date is None:
        cal = pd.read_csv(Path(qlib_dir) / "calendars" / "day.txt")
        date = str(cal.iloc[-1, 0])

    qlib.init(
        provider_uri=qlib_dir,
        region=REG_CN,
        expression_cache=None,
        dataset_cache=None,
    )
    mv_df = D.features(
        D.instruments("all"),
        ["$total_mv"],
        start_time=date,
        end_time=date,
    )
    mv_df.columns = ["total_mv"]
    mv_df["qlib_sym"] = [str(s).upper() for (s, _) in mv_df.index]
    logger.info(f"Market cap data: {len(mv_df)} stocks on {date}")

    # --- Step 4: Merge industry + market cap ---
    merged = df_tushare[["qlib_sym", "name", "industry"]].merge(
        mv_df[["qlib_sym", "total_mv"]], on="qlib_sym", how="inner"
    )
    merged = merged.dropna(subset=["total_mv", "industry"])
    merged = merged[merged["industry"] != ""]
    merged["mv_yuan"] = merged["total_mv"] / 1e4  # 万 -> 亿

    n_industries = merged["industry"].nunique()
    n_valid = len(merged)
    logger.info(
        f"Merged: {n_valid} stocks, {n_industries} industries "
        f"(excluded: {len(qlib_symbols) - n_valid} without industry/mv data)"
    )

    # --- Step 5: Cross-classify ---
    results = {}
    for industry, group in merged.groupby("industry"):
        if len(group) < 5:
            continue
        group_sorted = group.sort_values("total_mv", ascending=False)
        for tier_name, (lo, hi) in TIERS.items():
            n_group = len(group_sorted)
            lo_idx = int(n_group * lo)
            hi_idx = max(int(n_group * hi) - 1, 0)
            tier_syms = group_sorted.iloc[lo_idx : hi_idx + 1]["qlib_sym"].tolist()
            results.setdefault(tier_name, []).extend(tier_syms)

    # --- Step 6: Write instrument files ---
    sym_info = dict(zip(all_df["symbol"].str.upper(), zip(all_df["start"], all_df["end"])))

    for tier_name, syms in results.items():
        syms = sorted(set(syms))
        rows = []
        for s in syms:
            start, end = sym_info.get(s, (date, date))
            rows.append((s, str(start), str(end)))
        df_out = pd.DataFrame(rows, columns=["symbol", "start", "end"])
        out_path = instr_dir / f"{tier_name}.txt"
        df_out.to_csv(out_path, sep="\t", header=False, index=False)
        logger.info(f"  {tier_name}: {len(df_out)} stocks -> {out_path}")

    # --- Step 7: Industry report ---
    print()
    print(f"{'Industry':<16} {'Total':>6} {'Large':>6} {'Mid':>6} {'Small':>6}")
    print("-" * 46)
    for industry, group in merged.groupby("industry"):
        group_sorted = group.sort_values("total_mv", ascending=False)
        n = len(group_sorted)
        l = int(n * 0.15)
        m = int(n * 0.50) - l
        s = n - l - m
        print(f"{industry:<16} {n:>6} {l:>6} {m:>6} {s:>6}")


if __name__ == "__main__":
    import fire

    fire.Fire(build_instruments)

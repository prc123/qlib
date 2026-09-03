"""Filter ETF qlib data by fund size (规模 = share × nav).

Excludes ETFs whose 12-month mean size < threshold, plus those without
any size snapshot data (cannot be traded / sized). Appends the removals
to the existing exclude list for provenance.

Usage
-----
    python filter_etf_by_size.py --min_size_yi 2.0
"""
import shutil
import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


def filter_etf_by_size(
    qlib_data_dir: str = r"C:\Users\Administrator\.qlib\qlib_data\etf_data",
    min_size_yi: float = 2.0,
):
    qlib_dir = Path(qlib_data_dir).expanduser().resolve()
    size_stats = pd.read_csv(qlib_dir / "etf_size_stats.csv")
    size_map = size_stats.set_index("symbol")["size_mean"]

    instruments_file = qlib_dir / "instruments" / "all.txt"
    with open(instruments_file) as f:
        lines = [line.strip() for line in f if line.strip()]
    symbols = [line.split("\t")[0].lower() for line in lines]
    logger.info(f"Current instruments: {len(symbols)}")

    exclude = []
    for sym in symbols:
        if sym not in size_map.index:
            exclude.append((sym, "no_size_data"))
        elif size_map[sym] < min_size_yi:
            exclude.append((sym, f"size<{min_size_yi}yi"))

    keep_syms = set(symbols) - {s for s, _ in exclude}
    logger.info(f"Excluding {len(exclude)} (size<{min_size_yi}yi: "
                f"{sum(1 for _, r in exclude if r.startswith('size'))}, "
                f"no_data: {sum(1 for _, r in exclude if r == 'no_size_data')}), "
                f"keeping {len(keep_syms)}")

    backup = instruments_file.with_suffix(".txt.pre_size")
    shutil.copy2(instruments_file, backup)
    logger.info(f"Backed up instruments to {backup}")

    features_dir = qlib_dir / "features"
    removed = 0
    for sym, _ in exclude:
        sym_dir = features_dir / sym
        if sym_dir.is_dir():
            shutil.rmtree(sym_dir)
            removed += 1
    logger.info(f"Removed {removed} feature dirs")

    kept_lines = [line for line in lines if line.split("\t")[0].lower() in keep_syms]
    with open(instruments_file, "w") as f:
        f.write("\n".join(kept_lines) + "\n")
    logger.info(f"Updated instruments: {len(lines)} -> {len(kept_lines)}")

    exclude_csv = qlib_dir / "etf_exclude_list.csv"
    old = pd.read_csv(exclude_csv)
    new_rows = pd.DataFrame(
        [{"symbol": s.upper(), "exclude_reason": r} for s, r in exclude]
    )
    old_syms = set(old["symbol"].str.lower())
    new_rows = new_rows[~new_rows["symbol"].str.lower().isin(old_syms)]
    merged = pd.concat([old, new_rows], ignore_index=True)
    merged.to_csv(exclude_csv, index=False)
    logger.info(f"Appended {len(new_rows)} rows to {exclude_csv.name} "
                f"(total {len(merged)})")

    if "SH510050" not in {s.upper() for s in keep_syms}:
        logger.error("Benchmark SH510050 was excluded!")
    logger.info("Size filtering completed!")


if __name__ == "__main__":
    import fire
    fire.Fire(filter_etf_by_size)

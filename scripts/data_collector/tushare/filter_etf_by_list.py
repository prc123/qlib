import shutil
import pandas as pd
from pathlib import Path
from loguru import logger

def filter_etf_qlib_data(
    qlib_data_dir: str = r"C:\Users\Administrator\.qlib\qlib_data\etf_data",
    exclude_list: str = r"C:\Users\Administrator\.qlib\qlib_data\etf_data\etf_exclude_list.csv",
    backup: bool = True,
):
    """Filter ETF qlib data by exclude list.

    Removes excluded ETF feature directories and updates instruments file.
    """
    qlib_dir = Path(qlib_data_dir).expanduser().resolve()
    exclude_path = Path(exclude_list).expanduser().resolve()

    if not exclude_path.exists():
        logger.warning(f"Exclude list not found: {exclude_path}")
        return

    exclude_df = pd.read_csv(exclude_path)
    exclude_symbols = set(exclude_df["symbol"].str.lower().tolist())
    logger.info(f"Loaded {len(exclude_symbols)} symbols to exclude")

    features_dir = qlib_dir / "features"
    if not features_dir.exists():
        logger.warning(f"Features dir not found: {features_dir}")
        return

    # Backup original instruments
    instruments_file = qlib_dir / "instruments" / "all.txt"
    if backup and instruments_file.exists():
        backup_file = instruments_file.with_suffix(".txt.bak")
        shutil.copy2(instruments_file, backup_file)
        logger.info(f"Backed up instruments to {backup_file}")

    removed = 0
    for sym_dir in features_dir.iterdir():
        if not sym_dir.is_dir():
            continue
        if sym_dir.name in exclude_symbols:
            shutil.rmtree(sym_dir)
            removed += 1
            logger.info(f"Removed {sym_dir.name}")

    logger.info(f"Removed {removed} ETF feature directories")

    # Update instruments file
    if instruments_file.exists():
        with open(instruments_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        filtered_lines = [
            line for line in lines
            if line.split("\t")[0].lower() not in exclude_symbols
        ]
        with open(instruments_file, "w") as f:
            f.write("\n".join(filtered_lines) + "\n")
        logger.info(f"Updated instruments: {len(lines)} -> {len(filtered_lines)}")

    logger.info("ETF data filtering completed!")

if __name__ == "__main__":
    import fire
    fire.Fire(filter_etf_qlib_data)
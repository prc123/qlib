"""Feature screening by Rank IC. Outputs top-K features for FilterCol."""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from scipy import stats as st

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from daily_quant.ops.date_ops import (
    DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit,
)
from daily_quant.handler.alpha158_date import Alpha158DateV2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", default="CSI300")
    parser.add_argument("--topk", type=int, default=50, help="Number of features to keep")
    parser.add_argument("--qlib_data_dir", default=r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd")
    args = parser.parse_args()

    qlib.init(
        provider_uri=args.qlib_data_dir, region=REG_CN,
        expression_cache=None, dataset_cache=None,
        custom_ops=[DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit],
    )
    print(f"qlib init done")
    print(f"Instruments: {args.instruments}")

    # Load all feature names
    h = Alpha158DateV2(
        instruments=args.instruments,
        start_time="2022-01-01", end_time="2026-05-27",
        fit_start_time="2022-01-01", fit_end_time="2024-12-31",
    )
    feat_config = h._build_group_config()
    # Build (field, name, group) triples
    triples = []
    for group, (fields, names) in feat_config.items():
        for f, n in zip(fields, names):
            triples.append((f, n, group))
    print(f"Total features: {len(triples)}")

    # Compute per-feature IC on test period (2025-07-01 ~ 2026-05-27)
    test_start, test_end = "2025-07-01", "2026-05-27"
    print(f"\nComputing per-feature IC on {test_start} ~ {test_end}...")

    # Load label
    label_h = Alpha158DateV2(
        instruments=args.instruments,
        start_time=test_start, end_time=test_end,
        fit_start_time=test_start, fit_end_time=test_end,
        infer_processors=[
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ],
    )
    label_df = label_h.fetch(data_key=DataHandlerLP.DK_I).swaplevel().sort_index()

    # Compute IC per feature
    ic_results = []
    total = len(triples)
    for i, (field, name, group) in enumerate(triples):
        try:
            feat_df = D.features(
                D.instruments(args.instruments),
                [field], start_time=test_start, end_time=test_end
            )
            if feat_df.empty:
                continue

            # Align and compute daily Rank IC
            feat_s = feat_df.swaplevel().sort_index().iloc[:, 0]
            label_s = label_df.iloc[:, 0]
            # Normalize instrument names to uppercase (level 1 = instrument)
            def upper_level(s, pos):
                levels = [s.index.get_level_values(j) for j in range(s.index.nlevels)]
                levels[pos] = levels[pos].astype(str).str.upper()
                return s.set_axis(pd.MultiIndex.from_arrays(levels, names=s.index.names))
            feat_s = upper_level(feat_s, 1)
            label_s = upper_level(label_s, 1)

            # Merge to get aligned data
            merged = pd.DataFrame(
                {'feat': feat_s.values, 'label': label_s.values},
                index=feat_s.index
            ).dropna()
            if len(merged) < 100:
                continue

            ic_daily = merged.groupby("datetime").apply(
                lambda g: st.spearmanr(g['feat'], g['label'])[0]
                if len(g) > 5 else np.nan
            ).dropna()

            if len(ic_daily) > 10:
                ic_results.append({
                    "feature": name, "field": field,
                    "group": group,
                    "mean_ic": ic_daily.mean(),
                    "abs_ic": abs(ic_daily.mean()),
                    "ic_std": ic_daily.std(),
                    "icir": ic_daily.mean() / ic_daily.std() if ic_daily.std() > 0 else 0,
                    "ic_pos": (ic_daily > 0).mean(),
                })
        except Exception as e:
            if i < 5:
                print(f"  [{i}] {name}: ERROR - {e}")

        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{total}... (found {len(ic_results)} valid so far)")

    if not ic_results:
        print("No features passed screening — all IC values below threshold or errors.")
        return

    # Rank by abs(IC)
    results = pd.DataFrame(ic_results).sort_values("abs_ic", ascending=False)
    print(f"\n{'='*80}")
    print(f"Top {min(args.topk, len(results))} features by |IC|:\n")

    top_k = results.head(args.topk)
    for rank, (_, row) in enumerate(top_k.iterrows(), 1):
        print(f"  {rank:3d}. {row['feature']:20s}  |IC|={row['abs_ic']:.4f}  "
              f"mean={row['mean_ic']:+.4f}  std={row['ic_std']:.4f}  [{row['group']}]")

    # Print FilterCol config
    print(f"\n{'='*80}")
    print(f"FilterCol config (copy to train.py):\n")
    print(f'{{"class": "FilterCol", "kwargs": {{')
    print(f'    "fields_group": "feature",')
    print(f'    "col_list": [')
    top_names = list(top_k["feature"])
    for j, n in enumerate(top_names):
        comma = "," if j < len(top_names) - 1 else ""
        print(f'        "{n}"{comma}')
    print(f'    ],')
    print(f'    }},')
    print(f'}}')

    # Group summary
    print(f"\n{'='*80}")
    print(f"Features by group in top {args.topk}:")
    for g in sorted(top_k["group"].unique()):
        n = (top_k["group"] == g).sum()
        print(f"  {g}: {n}/{top_k.head(args.topk).shape[0]}")

    # Full results to CSV
    csv_path = Path(__file__).resolve().parent / "feature_ic_ranking.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nFull ranking saved to: {csv_path}")


if __name__ == "__main__":
    main()

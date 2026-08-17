"""Feature analysis: compare 170 vs 180 features with alpha factors."""
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset import TSDatasetH
from daily_quant.ops.date_ops import (
    DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit,
)
from daily_quant.handler.alpha158_date import (
    Alpha158Date,
    _ALPHA_MONEYFLOW_FIELDS, _ALPHA_MONEYFLOW_NAMES,
    _ALPHA_MARGIN_FIELDS, _ALPHA_MARGIN_NAMES,
)

QLIB_DIR = r"C:\Users\pp\.qlib\qlib_data\cn_data_bwd"


def main():
    qlib.init(
        provider_uri=QLIB_DIR, region=REG_CN,
        expression_cache=None, dataset_cache=None,
        custom_ops=[DayOfWeek, Month, Quarter, DayOfMonth, WeekOfYear, DayOfYear, BoardLimit],
    )
    print(f"qlib init done, calendar: {len(D.calendar())} days\n")

    # ── 1. Load data with 180 features ──
    print("=" * 60)
    print("Loading 180-feature dataset (CSI300, 2022-2026)...")
    handler_config = {
        "start_time": "2022-01-01", "end_time": "2026-07-06",
        "fit_start_time": "2022-01-01", "fit_end_time": "2024-12-31",
        "instruments": "csi300",
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
    }
    h = Alpha158Date(use_alpha_factors=True, **handler_config)
    dataset = TSDatasetH(handler=h,
                         segments={"train": ("2022-01-01", "2024-12-31"),
                                   "valid": ("2025-01-01", "2025-12-31"),
                                   "test": ("2026-01-01", "2026-07-01")},
                         step_len=60)

    # ── 2. Get feature names ──
    fields, names = h.get_feature_config()
    print(f"Total features: {len(fields)}")

    # Identify alpha factor positions
    alpha_names = _ALPHA_MONEYFLOW_NAMES + _ALPHA_MARGIN_NAMES
    alpha_indices = [i for i, n in enumerate(names) if n in alpha_names]
    non_alpha = [i for i in range(len(names)) if i not in alpha_indices]
    print(f"Alpha features: {len(alpha_indices)}")
    for i in alpha_indices:
        print(f"  [{i:3d}] {names[i]:20s} <- {fields[i]}")

    # ── 3. Extract features & labels via prepare() ──
    print("\nLoading feature matrix (subset for analysis)...")
    train_feat = dataset.prepare("train", col_set="feature")
    train_label = dataset.prepare("train", col_set="label")

    # Use subset to avoid qlib index out-of-bounds issue and save memory
    N_MAX = 20000
    total = min(len(train_feat) - 1, N_MAX)  # -1 to avoid off-by-one
    print(f"Using {total} / {len(train_feat)} samples")

    X_batch, y_batch = [], []
    first_feat = train_feat[0]
    first_label = train_label[0]
    print(f"Feature shape: {np.array(first_feat).shape}, Label shape: {np.array(first_label).shape}")
    for i in range(total):
        f = train_feat[i]
        l = train_label[i]
        f_arr = np.array(f)  # (60, d_feat)
        l_arr = np.array(l).ravel()
        # Take last-timestep features + corresponding label
        X_batch.append(f_arr[-1, :])  # last day features
        y_batch.append(float(l_arr[-1]))  # last day label
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{total}...")

    X = np.stack(X_batch).astype(np.float32)  # (N, d_feat)
    y = np.array(y_batch)  # (N,)
    n_feat = X.shape[1]
    print(f"X={X.shape}, y={y.shape}")
    print(f"X NaN count: {np.isnan(X).sum()} / {X.size}")
    print(f"y NaN count: {np.isnan(y).sum()} / {len(y)}")
    print(f"y range: [{np.nanmin(y):.6f}, {np.nanmax(y):.6f}]")

    # Quick check alpha feature values
    for i in alpha_indices:
        if i < n_feat:
            col = X[:, i]
            n_nan = np.isnan(col).sum()
            print(f"  {names[i]:20s}: nan={n_nan}/{len(col)}, mean={np.nanmean(col):.4f}, std={np.nanstd(col):.4f}")

    # ── 4. Feature-target correlation ──
    print("\n" + "=" * 60)
    print("Feature correlation with label (return):")
    correlations = []
    for i in range(X.shape[1]):
        mask = ~np.isnan(X[:, i]) & ~np.isnan(y)
        if mask.sum() > 100:
            corr = np.corrcoef(X[mask, i], y[mask])[0, 1]
            correlations.append((i, names[i], abs(corr), corr))

    correlations.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Rank':<5} {'Feature':<20} {'|Corr|':<8} {'Corr':<8} {'Type'}")
    print("-" * 55)
    for rank, (idx, name, abs_corr, corr) in enumerate(correlations[:20]):
        ftype = "ALPHA" if idx in alpha_indices else "base"
        marker = " ***" if ftype == "ALPHA" else ""
        print(f"{rank+1:<5} {name:<20} {abs_corr:<8.4f} {corr:<+8.4f} {ftype}{marker}")

    # ── 5. Alpha factor summary ──
    print("\n" + "=" * 60)
    print("Alpha factor correlations:")
    alpha_corrs = [(idx, names[idx], abs_corr, corr)
                   for idx, name, abs_corr, corr in correlations
                   if idx in alpha_indices]
    if alpha_corrs:
        for idx, name, abs_corr, corr in alpha_corrs:
            print(f"  {name:20s}: |r|={abs_corr:.4f}, r={corr:+.4f}")
        avg_abs = np.mean([c[2] for c in alpha_corrs])
        print(f"\n  Average |corr| of alpha factors: {avg_abs:.4f}")
        top20_alpha = sum(1 for i, _, _, _ in alpha_corrs if i in [c[0] for c in correlations[:20]])
        top50_alpha = sum(1 for i, _, _, _ in alpha_corrs if i in [c[0] for c in correlations[:50]])
        print(f"  Alpha factors in top-20:  {top20_alpha}/10")
        print(f"  Alpha factors in top-50:  {top50_alpha}/10")
    else:
        print("  No alpha features found in correlation list!")

    # ── 6. Feature correlation heatmap (alpha vs alpha) ──
    print("\n" + "=" * 60)
    print("Alpha feature inter-correlation matrix:")
    if len(alpha_indices) >= 2 and max(alpha_indices) < X.shape[1]:
        X_alpha = X[:, [i for i in alpha_indices if i < X.shape[1]]]
        # Sample to avoid memory issues
        sample_n = min(50000, X_alpha.shape[0])
        idx_sample = np.random.choice(X_alpha.shape[0], sample_n, replace=False)
        corr_matrix = np.corrcoef(X_alpha[idx_sample].T)
        print(f"  {'':>20}", end="")
        for name in alpha_names[:5]:
            print(f"{name:>10}", end="")
        print()
        for i, name_i in enumerate(alpha_names):
            if i >= 5:
                break
            print(f"  {name_i:>20}", end="")
            for j in range(min(5, len(alpha_names))):
                print(f"{corr_matrix[i, j]:>10.3f}", end="")
            print()
        # Check redundancy
        high_corr = np.sum(np.abs(np.triu(corr_matrix, 1)) > 0.8)
        print(f"\n  Pairs with |r| > 0.8: {high_corr}")

    print("\nDone.")


if __name__ == "__main__":
    main()

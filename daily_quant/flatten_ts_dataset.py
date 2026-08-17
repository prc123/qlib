"""
Flattened time-series dataset for tree models (CatBoost, XGBoost, LightGBM).

Converts TSDatasetH time windows into tabular features by computing
statistical summaries over the time dimension.

Usage
-----
    from daily_quant.flatten_ts_dataset import FlattenedTSDatasetH

    dataset = FlattenedTSDatasetH(
        handler={"class": "Alpha158Date", "module_path": "daily_quant.handler.alpha158_date",
                 "kwargs": {...}},
        segments={"train": ("2022-01-01", "2024-12-31"),
                  "valid": ("2025-01-01", "2025-06-30"),
                  "test":  ("2025-07-01", "2026-05-27")},
        step_len=20,
        flatten_mode="stats",  # "stats" or "full"
    )
"""

import numpy as np
import pandas as pd
from qlib.data.dataset import TSDatasetH
from qlib.data.dataset.handler import DataHandlerLP


class FlattenedTSDatasetH:
    """Flatten time-series windows into tabular features for tree models.

    For each (stock, date) sample, extracts a time window of ``step_len``
    past trading days and computes statistical summaries over the window.

    Parameters
    ----------
    handler : dict
        Handler config (same as TSDatasetH).
    segments : dict
        Segment definitions: {"train": (start, end), ...}.
    step_len : int
        Number of past trading days in each time window.
    flatten_mode : str
        - "stats": 6 stats per feature (mean, std, last, trend, min, max).
          Total features = n_raw_features * 6.
        - "full": All time steps flattened. Total = step_len * n_raw_features.
    """

    def __init__(self, handler, segments, step_len=20, flatten_mode="stats"):
        self._ts_dataset = TSDatasetH(
            handler=handler, segments=segments, step_len=step_len,
        )
        self.step_len = step_len
        self.flatten_mode = flatten_mode
        self.segments = segments  # needed by SignalRecord
        self.handler = self._ts_dataset.handler
        self._cache = {}

    def _flatten_batch(self, feat_3d):
        """Vectorized flatten: (batch, step_len, n_feat) -> (batch, n_flat).

        Parameters
        ----------
        feat_3d : np.ndarray, shape (batch, step_len, n_feat)

        Returns
        -------
        np.ndarray, shape (batch, n_flat)
        """
        if self.flatten_mode == "stats":
            with np.errstate(all="ignore"):
                f_mean = np.nanmean(feat_3d, axis=1)
                f_std = np.nanstd(feat_3d, axis=1)
                f_last = feat_3d[:, -1, :]
                f_trend = (feat_3d[:, -1, :] - feat_3d[:, 0, :]) / max(self.step_len, 1)
                f_min = np.nanmin(feat_3d, axis=1)
                f_max = np.nanmax(feat_3d, axis=1)
            flat = np.concatenate([f_mean, f_std, f_last, f_trend, f_min, f_max], axis=1)
            return np.nan_to_num(flat, nan=0.0).astype(np.float32)
        else:
            return feat_3d.reshape(feat_3d.shape[0], -1).astype(np.float32)

    def _build_flat_dataframe(self, segment):
        """Convert a TS segment to a flat DataFrame with MultiIndex columns.

        Uses batched access to the TSDataSampler for efficiency:
        ``sampler[list_of_indices]`` returns a stacked (N, step_len, F+1) array,
        which we process with vectorized numpy operations.

        Returns
        -------
        df_features : pd.DataFrame with columns like ("feature", "f0"), ...
        df_labels : pd.Series
        """
        sampler = self._ts_dataset.prepare(
            segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,
        )
        n_samples = len(sampler)
        if n_samples == 0:
            raise ValueError(f"Empty segment: {segment}")

        sample_0 = sampler[0]
        n_feat = sample_0.shape[1] - 1

        if self.flatten_mode == "stats":
            n_flat = n_feat * 6
        else:
            n_flat = self.step_len * n_feat

        X = np.zeros((n_samples, n_flat), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.float32)

        # Process in batches — sampler[list] returns a stacked (B, step_len, F+1) array
        batch_size = 4096
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_idx = list(range(batch_start, batch_end))
            batch_data = sampler[batch_idx]  # (B, step_len, n_feat+1)

            feat_3d = batch_data[:, :, :-1]          # (B, step_len, n_feat)
            labels_1d = batch_data[:, -1, -1]         # (B,)

            X[batch_start:batch_end] = self._flatten_batch(feat_3d)
            y[batch_start:batch_end] = np.nan_to_num(labels_1d, nan=0.0)

        idx = sampler.get_index()
        col_tuples = [("feature", f"f{j}") for j in range(n_flat)]
        df_X = pd.DataFrame(X, index=idx, columns=pd.MultiIndex.from_tuples(col_tuples))
        df_y = pd.Series(y, index=idx, name=("label", "LABEL0"))
        return df_X, df_y

    def _get_flat(self, segment):
        if segment not in self._cache:
            X, y = self._build_flat_dataframe(segment)
            self._cache[segment] = (X, y)
        return self._cache[segment]

    def prepare(self, segments, col_set="feature", data_key=DataHandlerLP.DK_I, **kwargs):
        """Prepare data in CatBoostModel-compatible format.

        Returns
        -------
        For DK_L + ["feature", "label"]: pd.DataFrame with MultiIndex columns
            ("feature", ...) and a "label" column.
        For DK_I: pd.DataFrame with MultiIndex columns ("feature", ...).
        """
        if isinstance(segments, str):
            segments = [segments]
            single = True
        else:
            single = False

        results = []
        for seg in segments:
            X, y = self._get_flat(seg)
            if data_key == DataHandlerLP.DK_L:
                df = X.copy()
                df[("label", "LABEL0")] = y.values
                results.append(df)
            else:
                results.append(X.copy())

        return results[0] if single else results

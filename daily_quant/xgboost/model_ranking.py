"""Ranking-aware XGBoost: uses pairwise/NDCG loss for TopK selection.

Standard XGBoost with MSE minimizes (pred - actual)^2.  This optimises
absolute prediction accuracy.  For a TopK stock selection strategy, we
only care about the relative ORDER of predictions within each trading day.

XGBoost has built-in ranking objectives that do exactly this:

    rank:pairwise  — minimises pairwise swap errors (faster)
    rank:ndcg      — maximises NDCG@k (slower, more precise)

Both require the samples to be grouped by query (trading day), so that
the loss is computed within each day's cross-section independently.

Usage
-----
    model = RankingXGBModel(objective="rank:pairwise", ...)
    model.fit(dataset)   # auto-computes group sizes by date
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.xgboost import XGBModel


class RankingXGBModel(XGBModel):
    """XGBoost with pairwise ranking loss.

    Each trading day is treated as a query group — the model learns to
    rank stocks correctly within each day rather than predicting exact
    returns.

    Parameters
    ----------
    objective : str, default "rank:pairwise"
        XGBoost ranking objective: "rank:pairwise" or "rank:ndcg".
    lambdarank_num_pair_per_sample : int, default None
        For rank:ndcg, number of pairs per sample (higher = more accurate,
        slower).
    **kwargs : passed to XGBModel.
    """

    def __init__(self, objective="rank:pairwise", **kwargs):
        super().__init__(**kwargs)
        self._ranking_objective = objective
        self._params["objective"] = objective
        # Remove eval_metric — ranking objectives use NDCG/MAP internally
        self._params.pop("eval_metric", None)

    def fit(self, dataset, num_boost_round=1000, early_stopping_rounds=50,
            verbose_eval=50, evals_result=dict(), **kwargs):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train = df_train["feature"].values
        y_train = df_train["label"].values
        x_valid = df_valid["feature"].values
        y_valid = df_valid["label"].values

        if y_train.ndim == 2:
            y_train = y_train.ravel()
            y_valid = y_valid.ravel()

        # Compute group sizes (samples per trading day)
        train_groups = _count_per_day(df_train)
        valid_groups = _count_per_day(df_valid)

        # Convert raw returns to within-day ranks (0..N-1).
        # This is what rank:pairwise expects — integer relevance scores.
        y_train = _rank_by_group(y_train, train_groups)
        y_valid = _rank_by_group(y_valid, valid_groups)

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtrain.set_group(train_groups)
        dvalid = xgb.DMatrix(x_valid, label=y_valid)
        dvalid.set_group(valid_groups)

        params = dict(self._params)
        params.setdefault("verbosity", 0)
        params.setdefault("ndcg_exp_gain", False)  # allow labels > 31

        print(f"Ranking loss: {self._ranking_objective}, "
              f"train days={len(train_groups)}, valid days={len(valid_groups)}")

        evals = [(dtrain, "train"), (dvalid, "valid")]
        self.model = xgb.train(
            params, dtrain, num_boost_round=num_boost_round,
            evals=evals, early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval, evals_result=evals_result,
        )
        self.fitted = True

    def predict(self, dataset: DatasetH, segment="test"):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        raw = self.model.predict(xgb.DMatrix(x_test))
        return pd.Series(raw, index=x_test.index)


def _count_per_day(df):
    """Count samples per trading day for xgboost group."""
    if df.index.nlevels == 2:
        dates = df.index.get_level_values("datetime")
    else:
        dates = df.index
    _, counts = np.unique(dates, return_counts=True)
    return counts.tolist()


def _rank_by_group(y, groups):
    """Convert values to within-group integer ranks (0 to N-1).

    For rank:pairwise, labels must be non-negative integers.  Pure daily
    ranking captures exactly what we care about — the relative order.
    """
    result = np.empty_like(y, dtype=int)
    offset = 0
    for g_size in groups:
        vals = y[offset:offset + g_size]
        result[offset:offset + g_size] = np.argsort(np.argsort(vals))
        offset += g_size
    return result

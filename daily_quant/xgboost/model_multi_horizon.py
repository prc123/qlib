"""Multi-horizon XGBoost model: predicts T+1, T+3, T+5 returns simultaneously.

XGBoost natively supports multi-output regression. This wrapper trains on
3 labels at once and combines the 3 predictions into a single score for
the backtest strategy.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.model.xgboost import XGBModel


class MultiHorizonXGBModel(XGBModel):
    """XGBoost model that predicts T+1, T+3, T+5 returns simultaneously.

    Training: uses xgboost's multi-output regression (3 targets).
    Prediction: combines 3 predictions via weighted average into a single score.

    Parameters
    ----------
    horizon_weights : tuple of float, default (0.2, 0.3, 0.5)
        Weights for combining T+1, T+3, T+5 predictions.
    **kwargs : passed to XGBModel.
    """

    def __init__(self, horizon_weights=(0.2, 0.3, 0.5), **kwargs):
        super().__init__(**kwargs)
        self.horizon_weights = np.array(horizon_weights, dtype=np.float64)
        self._multi_label = True

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

        # Convert DataFrame columns to numpy (handle multi-label)
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values
            y_valid = y_valid.values
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
            y_valid = y_valid.reshape(-1, 1)

        n_labels = y_train.shape[1]
        self._n_labels = n_labels
        print(f"Multi-horizon: {n_labels} labels, weights={tuple(self.horizon_weights[:n_labels])}")

        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_valid, label=y_valid)

        params = dict(self._params)
        params.update({
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "verbosity": 0,
            "num_target": n_labels,
        })

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
        # raw shape: (n_samples, n_labels)
        weights = self.horizon_weights[:raw.shape[1]] if raw.ndim == 2 else np.array([1.0])
        combined = np.average(raw, axis=1, weights=weights)
        return pd.Series(combined, index=x_test.index)

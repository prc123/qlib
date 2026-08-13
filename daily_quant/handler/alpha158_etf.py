"""ETF-specific Alpha158 handler: Alpha158Date + ETF factors.

Adds 2 ETF-specific factors that Alpha158 (stock factors) lacks:

    SHARE_CHG    = $share / Ref($share, 1) - 1    (份额变化率)
    PREM_DISC    = $close / $nav - 1              (折溢价率)

These require the ETF data to have been downloaded with fund_share
(``$share``) and fund_nav (``$nav``) fields via ``etf_extra_fields.py``.

Usage::

    handler={
        "class": "Alpha158ETF",
        "module_path": "daily_quant.handler.alpha158_etf",
        "kwargs": {...},
    }

Feature count: 170 + 2 = 172 (use_alpha_factors=False) or 182 (True).
"""

from daily_quant.handler.alpha158_date import Alpha158Date

_ETF_FIELDS = [
    "$share / Ref($share, 1) - 1",
    "$close / $factor / $nav - 1",
]

_ETF_NAMES = [
    "SHARE_CHG",
    "PREM_DISC",
]


class Alpha158ETF(Alpha158Date):
    """Alpha158Date + share-change + premium/discount ETF factors.

    Parameters
    ----------
    use_alpha_factors : bool, default False
        Include moneyflow/margin alpha factor fields.
    label_type : str, default "return"
        Label formulation:
          - "return"   : 5-day forward return (default)
          - "sharpe"   : forward return / historical vol (risk-adjusted)
          - "ret_vol"  : forward return - 0.5 * historical vol
    """

    def __init__(self, use_alpha_factors=False, label_type="return", include_prem_disc=True,
                 score_weights=(1.0, 0.5, 0.5), **kwargs):
        self._label_type = label_type
        self._include_prem_disc = include_prem_disc
        self._score_weights = score_weights
        super().__init__(use_alpha_factors=use_alpha_factors, **kwargs)

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        if self._include_prem_disc:
            fields += _ETF_FIELDS
            names += _ETF_NAMES
        else:
            fields += _ETF_FIELDS[:1]  # only SHARE_CHG
            names += _ETF_NAMES[:1]
        return fields, names

    def get_label_config(self):
        if self._label_type == "sharpe":
            return (
                ["(Ref($close, -5) / $close - 1) / Std($close / Ref($close, 1) - 1, 20)"],
                ["LABEL0"],
            )
        elif self._label_type == "ret_vol":
            return (
                ["Ref($close, -5) / $close - 1 - 0.5 * Std($close / Ref($close, 1) - 1, 20)"],
                ["LABEL0"],
            )
        elif self._label_type == "score":
            w_ret, w_dd, w_vol = self._score_weights
            ret = "Ref($close, -5) / $close - 1"
            dd = "Ref(Min($close, 5), -5) / $close - 1"
            vol = "Std($close / Ref($close, 1) - 1, 20)"
            expr = f"{w_ret} * ({ret}) + {w_dd} * ({dd}) - {w_vol} * ({vol})"
            return ([expr], ["LABEL0"])
        else:
            return super().get_label_config()

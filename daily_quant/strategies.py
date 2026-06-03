"""Board-limit-aware strategies for A-share market.

Board-specific limits:
- ST / *ST stocks: 5%
- Main board (SH600/SZ000 etc.): 10%
- ChiNext / 创业板 (SZ300): 20%
- STAR Market / 科创板 (SH688): 20%
- Beijing Stock Exchange (BJ): 30%

Usage in qlib config::

    "strategy": {
        "class": "BoardLimitTopkDropoutStrategy",
        "module_path": "daily_quant.strategies",
        "kwargs": {"model": ..., "dataset": ..., "topk": 30, "n_drop": 5},
    }
"""

import copy
import numpy as np
import pandas as pd

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.position import Position
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class BoardLimitTopkDropoutStrategy(TopkDropoutStrategy):
    """TopkDropoutStrategy with per-stock board-level price-limit filtering.

    Checks ``$change`` against the stock's board limit (5/10/20/30%) to
    skip buy candidates at limit-up and sell candidates at limit-down.
    Does NOT rely on the exchange's ``limit_threshold`` expression.

    Parameters
    ----------
    check_limit : bool
        Enable board-specific limit-up/down checks (default True).
    Other params are forwarded to :class:`TopkDropoutStrategy`.
    """

    LIMIT_MARGIN = 0.98  # 2% margin below actual limit

    # ---- Board limit logic ----

    @staticmethod
    def get_board_limit(instrument: str, is_st: bool = False) -> float:
        """Daily price-change limit for *instrument*.

        ST detection is best-effort; if *is_st* is not provided externally
        the strategy will try to infer it from the exchange's ``$name`` field
        during trading.
        """
        if is_st:
            return 0.05
        code = str(instrument).upper()
        if code.startswith("BJ"):
            return 0.30
        if code.startswith("SH688") or code.startswith("SZ30"):
            return 0.20
        return 0.10

    def _infer_st(self, stock_id, start_time, end_time) -> bool:
        """Best-effort ST detection via stock name in quote data."""
        try:
            name = self.trade_exchange.get_quote_info(
                stock_id, start_time, end_time, field="$name"
            )
            if name is not None:
                name_str = str(name).upper().strip()
                return name_str.startswith("ST") or name_str.startswith("*ST")
        except Exception:
            pass
        return False

    def _is_at_limit_up(self, stock_id, start_time, end_time) -> bool:
        """True if stock is at limit-up → cannot buy."""
        change = self.trade_exchange.get_quote_info(
            stock_id, start_time, end_time, field="$change"
        )
        if change is None or (isinstance(change, float) and np.isnan(change)):
            return True
        is_st = self._infer_st(stock_id, start_time, end_time)
        limit = self.get_board_limit(stock_id, is_st)
        return float(change) >= limit * self.LIMIT_MARGIN

    def _is_at_limit_down(self, stock_id, start_time, end_time) -> bool:
        """True if stock is at limit-down → cannot sell."""
        change = self.trade_exchange.get_quote_info(
            stock_id, start_time, end_time, field="$change"
        )
        if change is None or (isinstance(change, float) and np.isnan(change)):
            return True
        is_st = self._infer_st(stock_id, start_time, end_time)
        limit = self.get_board_limit(stock_id, is_st)
        return float(change) <= -limit * self.LIMIT_MARGIN

    # ---- Construction ----

    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        check_limit=True,
        **kwargs,
    ):
        super().__init__(
            topk=topk,
            n_drop=n_drop,
            method_sell=method_sell,
            method_buy=method_buy,
            hold_thresh=hold_thresh,
            only_tradable=only_tradable,
            forbid_all_trade_at_limit=forbid_all_trade_at_limit,
            **kwargs,
        )
        self.check_limit = check_limit

    # ---- Override generate_trade_decision ----

    def generate_trade_decision(self, execute_result=None):
        """Same logic as TopkDropoutStrategy, with board-specific limit checks
        added on top of the existing tradability filters."""
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        # ---- Stock filtering helpers (mirrors TopkDropoutStrategy) ----
        def get_first_n(li, n, reverse=False):
            cur_n = 0
            res = []
            for si in reversed(li) if reverse else li:
                if self.trade_exchange.is_stock_tradable(
                    stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                ):
                    res.append(si)
                    cur_n += 1
                    if cur_n >= n:
                        break
            return res[::-1] if reverse else res

        def get_last_n(li, n):
            return get_first_n(li, n, reverse=True)

        def filter_stock(li):
            return [
                si
                for si in li
                if self.trade_exchange.is_stock_tradable(
                    stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                )
            ]

        # ---- Position & candidate selection ----
        current_temp = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index

        if self.method_buy == "top":
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"method_buy={self.method_buy}")

        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:
                sell = candi
        else:
            raise NotImplementedError(f"method_sell={self.method_sell}")

        buy = today[: len(sell) + self.topk - len(last)]

        # ---- Sell loop ----
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code in sell:
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                if self.check_limit and self._is_at_limit_down(code, trade_start_time, trade_end_time):
                    continue
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code, amount=sell_amount,
                    start_time=trade_start_time, end_time=trade_end_time,
                    direction=Order.SELL,
                )
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    cash += trade_val - trade_cost

        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0

        # ---- Buy loop ----
        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            if self.check_limit and self._is_at_limit_up(code, trade_start_time, trade_end_time):
                continue
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time
            )
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code, amount=buy_amount,
                start_time=trade_start_time, end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)

        return TradeDecisionWO(sell_order_list + buy_order_list, self)

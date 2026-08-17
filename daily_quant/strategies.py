"""Board-limit-aware strategies for A-share market.

Board-specific limits:
- ST / *ST stocks: 5%
- Main board (SH600/SZ000 etc.): 10%
- ChiNext / 创业板 (SZ300): 20%
- STAR Market / 科创板 (SH688): 20%
- Beijing Stock Exchange (BJ): 30%
"""

import copy
import numpy as np
import pandas as pd

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class BoardLimitTopkDropoutStrategy(TopkDropoutStrategy):

    LIMIT_MARGIN = 0.98

    @staticmethod
    def get_board_limit(instrument: str, is_st: bool = False) -> float:
        if is_st:
            return 0.05
        code = str(instrument).upper()
        if code.startswith("BJ"):
            return 0.30
        if code.startswith("SH688") or code.startswith("SZ30"):
            return 0.20
        return 0.10

    def __init__(self, *, topk, n_drop, method_sell="bottom", method_buy="top",
                 hold_thresh=1, only_tradable=False, forbid_all_trade_at_limit=True,
                 check_limit=True, score_thresh=None, **kwargs):
        super().__init__(topk=topk, n_drop=n_drop, method_sell=method_sell,
                         method_buy=method_buy, hold_thresh=hold_thresh,
                         only_tradable=only_tradable,
                         forbid_all_trade_at_limit=forbid_all_trade_at_limit,
                         **kwargs)
        self.check_limit = check_limit
        self.score_thresh = score_thresh
        self._st_cache = {}  # instrument_upper → bool

    def _infer_st(self, stock_id, start_time, end_time):
        key = str(stock_id).upper()
        if key in self._st_cache:
            return self._st_cache[key]
        try:
            name = self.trade_exchange.get_quote_info(
                stock_id, start_time, end_time, field="$name"
            )
            if name is not None:
                is_st = str(name).upper().strip().startswith(("ST", "*ST"))
                self._st_cache[key] = is_st
                return is_st
        except Exception:
            pass
        self._st_cache[key] = False
        return False

    def _is_at_limit_up(self, stock_id, start_time, end_time):
        change = self.trade_exchange.get_quote_info(
            stock_id, start_time, end_time, field="$change"
        )
        if change is None or (isinstance(change, float) and np.isnan(change)):
            return True
        is_st = self._infer_st(stock_id, start_time, end_time)
        limit = self.get_board_limit(stock_id, is_st)
        return float(change) >= limit * self.LIMIT_MARGIN

    def _is_at_limit_down(self, stock_id, start_time, end_time):
        change = self.trade_exchange.get_quote_info(
            stock_id, start_time, end_time, field="$change"
        )
        if change is None or (isinstance(change, float) and np.isnan(change)):
            return True
        is_st = self._infer_st(stock_id, start_time, end_time)
        limit = self.get_board_limit(stock_id, is_st)
        return float(change) <= -limit * self.LIMIT_MARGIN

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        def get_first_n(li, n, reverse=False):
            cur_n, res = 0, []
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
            return [si for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time)]

        current_temp = copy.deepcopy(self.trade_position)
        sell_order_list, buy_order_list = [], []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index

        # Filter by score threshold before selecting top-k
        candidates = pred_score[~pred_score.index.isin(last)]
        if self.score_thresh is not None:
            candidates = candidates[candidates >= self.score_thresh]

        if self.method_buy == "top":
            today = get_first_n(
                candidates.sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last))
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

        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL):
                continue
            if code in sell:
                if current_temp.get_stock_count(code, bar=self.trade_calendar.get_freq()) < self.hold_thresh:
                    continue
                if self.check_limit and self._is_at_limit_down(code, trade_start_time, trade_end_time):
                    continue
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(stock_id=code, amount=sell_amount,
                                   start_time=trade_start_time, end_time=trade_end_time,
                                   direction=Order.SELL)
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp)
                    cash += trade_val - trade_cost

        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0

        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY):
                continue
            if self.check_limit and self._is_at_limit_up(code, trade_start_time, trade_end_time):
                continue
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY)
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(stock_id=code, amount=buy_amount,
                              start_time=trade_start_time, end_time=trade_end_time,
                              direction=Order.BUY)
            buy_order_list.append(buy_order)

        return TradeDecisionWO(sell_order_list + buy_order_list, self)

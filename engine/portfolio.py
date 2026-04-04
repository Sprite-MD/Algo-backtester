from datetime import datetime

import pandas as pd

from config import CONFIG, BacktestConfig
from engine.order import Order


class Portfolio:

    def __init__(self, config: BacktestConfig = CONFIG):
        self.commission = config.commission
        self.slippage   = config.slippage

        self.cash       = config.initial_capital
        self.positions  = {}   # ticker -> quantity
        self.trade_log  = []   # list of dicts, one per executed trade
        self.equity_curve = [] # list of (timestamp, total_equity)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def execute_order(self, order: Order, current_price: float) -> None:
        if order.direction == "buy":
            fill_price = current_price * (1 + self.slippage)
            cost = order.quantity * fill_price * (1 + self.commission)
            self.cash -= cost
            self.positions[order.ticker] = (
                self.positions.get(order.ticker, 0) + order.quantity
            )
        else:
            fill_price = current_price * (1 - self.slippage)
            proceeds = order.quantity * fill_price * (1 - self.commission)
            self.cash += proceeds
            self.positions[order.ticker] = (
                self.positions.get(order.ticker, 0) - order.quantity
            )
            if self.positions[order.ticker] == 0:
                del self.positions[order.ticker]

        self.trade_log.append({
            "timestamp":  order.timestamp,
            "ticker":     order.ticker,
            "direction":  order.direction,
            "quantity":   order.quantity,
            "fill_price": fill_price,
            "cost":       cost if order.direction == "buy" else None,
            "proceeds":   proceeds if order.direction == "sell" else None,
        })

    # ------------------------------------------------------------------
    # Equity tracking
    # ------------------------------------------------------------------

    def update_equity(self, timestamp: datetime, current_prices: dict) -> None:
        position_value = sum(
            qty * current_prices.get(ticker, 0)
            for ticker, qty in self.positions.items()
        )
        self.equity_curve.append((timestamp, self.cash + position_value))

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def get_equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        timestamps, values = zip(*self.equity_curve)
        return pd.Series(values, index=pd.DatetimeIndex(timestamps), name="equity")

    def get_trade_df(self) -> pd.DataFrame:
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)

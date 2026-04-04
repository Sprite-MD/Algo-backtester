import pandas as pd

from config import CONFIG, BacktestConfig
from engine.order import Order
from engine.portfolio import Portfolio
from strategy.base import BaseStrategy


class Backtester:

    def __init__(
        self,
        config:    BacktestConfig,
        data:      pd.DataFrame,
        strategy:  BaseStrategy,
        portfolio: Portfolio,
    ):
        self.config    = config
        self.data      = data
        self.strategy  = strategy
        self.portfolio = portfolio

    def run(self) -> tuple[pd.Series, pd.DataFrame]:
        for i in range(len(self.data)):
            bar       = self.data.index[i]
            close     = float(self.data["close"].iloc[i])
            slice_    = self.data.iloc[: i + 1]

            signals   = self.strategy.generate_signals(slice_)
            signal    = int(signals.iloc[-1])

            held      = self.portfolio.positions.get(self.config.ticker, 0)

            if signal == 1 and held == 0:
                quantity = (self.portfolio.cash * self.config.position_size) / close
                order = Order(
                    ticker=self.config.ticker,
                    direction="buy",
                    order_type="market",
                    quantity=quantity,
                    price=close,
                    timestamp=bar,
                )
                self.portfolio.execute_order(order, close)

            elif signal == -1 and held > 0:
                order = Order(
                    ticker=self.config.ticker,
                    direction="sell",
                    order_type="market",
                    quantity=held,
                    price=close,
                    timestamp=bar,
                )
                self.portfolio.execute_order(order, close)

            self.portfolio.update_equity(bar, {self.config.ticker: close})

        return self.portfolio.get_equity_series(), self.portfolio.get_trade_df()

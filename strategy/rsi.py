import pandas as pd

from config import CONFIG, BacktestConfig
from strategy.base import BaseStrategy


class RSIStrategy(BaseStrategy):

    def __init__(self, config: BacktestConfig = CONFIG):
        self.period     = config.rsi.period
        self.oversold   = config.rsi.oversold
        self.overbought = config.rsi.overbought

    def _compute_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1 / self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, adjust=False).mean()

        # When avg_loss == 0: RS = inf → RSI = 100 (pure gain, no losses)
        # When avg_gain == avg_loss == 0: RS = NaN → RSI = NaN (no data yet)
        rs  = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = self._compute_rsi(df["close"])

        was_oversold   = rsi.shift(1, fill_value=float("nan")) >= self.oversold
        was_overbought = rsi.shift(1, fill_value=float("nan")) <= self.overbought

        buy_signal  = (rsi < self.oversold)  & was_oversold
        sell_signal = (rsi > self.overbought) & was_overbought

        signals = pd.Series(0, index=df.index)
        signals[buy_signal]  = 1
        signals[sell_signal] = -1

        return signals

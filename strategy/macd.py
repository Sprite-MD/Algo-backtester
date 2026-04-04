import pandas as pd

from config import CONFIG, BacktestConfig
from strategy.base import BaseStrategy


class MACDStrategy(BaseStrategy):

    def __init__(self, config: BacktestConfig = CONFIG):
        self.fast_period   = config.macd.fast_period
        self.slow_period   = config.macd.slow_period
        self.signal_period = config.macd.signal_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        fast_ema    = close.ewm(span=self.fast_period,   adjust=False).mean()
        slow_ema    = close.ewm(span=self.slow_period,   adjust=False).mean()
        macd_line   = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        macd_above      = macd_line > signal_line
        prev_macd_above = macd_above.shift(1, fill_value=False)

        bullish_cross = macd_above  & ~prev_macd_above
        bearish_cross = ~macd_above &  prev_macd_above

        signals = pd.Series(0, index=df.index)
        signals[bullish_cross] = 1
        signals[bearish_cross] = -1

        return signals

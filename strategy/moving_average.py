import pandas as pd

from config import CONFIG, BacktestConfig
from strategy.base import BaseStrategy


class MovingAverageStrategy(BaseStrategy):

    def __init__(self, config: BacktestConfig = CONFIG):
        self.fast_window = config.moving_average.fast_window
        self.slow_window = config.moving_average.slow_window
        self.ma_type = config.moving_average.ma_type

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]

        if self.ma_type == "ema":
            fast_ma = close.ewm(span=self.fast_window, adjust=False).mean()
            slow_ma = close.ewm(span=self.slow_window, adjust=False).mean()
        else:
            fast_ma = close.rolling(window=self.fast_window).mean()
            slow_ma = close.rolling(window=self.slow_window).mean()

        fast_above = fast_ma > slow_ma
        prev_fast_above = fast_above.shift(1, fill_value=False)
        golden_cross = fast_above & ~prev_fast_above
        death_cross = ~fast_above & prev_fast_above

        signals = pd.Series(0, index=df.index)
        signals[golden_cross] = 1
        signals[death_cross] = -1

        return signals

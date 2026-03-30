from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV price data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with a DatetimeIndex and lowercase columns:
            open, high, low, close, volume. Must contain only the bars
            available up to the current point in time — no future data.

        Returns
        -------
        pd.Series
            A Series of integer signals aligned to df.index:
              1  = buy
             -1  = sell
              0  = hold / no action
        """

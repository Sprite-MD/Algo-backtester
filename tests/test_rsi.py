import pytest
import pandas as pd
import numpy as np

from config import BacktestConfig, RSIParams
from strategy.rsi import RSIStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(period=14, oversold=30.0, overbought=70.0):
    return BacktestConfig(
        rsi=RSIParams(period=period, oversold=oversold, overbought=overbought)
    )


def make_df(prices: list) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "open":   prices,
            "high":   prices,
            "low":    prices,
            "close":  prices,
            "volume": [1_000_000] * len(prices),
        },
        index=idx,
    )


def sharp_drop_then_recover():
    """
    Oscillating warmup so RSI is established near 50, then a sharp drop
    that pushes RSI below oversold, then a recovery.
    """
    warmup  = [100.0 + (i % 2) * 1.0 for i in range(30)]  # RSI ~50
    drop    = [100.0 - i * 4.0 for i in range(20)]         # rapid fall → RSI < 30
    recover = [100.0 - 76.0 + i * 4.0 for i in range(20)] # recovery
    return warmup + drop + recover


def sharp_rise_then_fall():
    """
    Oscillating warmup so RSI is established near 50, then a sharp rise
    that pushes RSI above overbought, then a fall.
    """
    warmup = [100.0 + (i % 2) * 1.0 for i in range(30)]   # RSI ~50
    rise   = [100.0 + i * 4.0 for i in range(20)]          # rapid rise → RSI > 70
    fall   = [100.0 + 76.0 - i * 4.0 for i in range(20)]  # fall
    return warmup + rise + fall


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReturnShape:

    def test_returns_series(self):
        df = make_df([100.0] * 30)
        s = RSIStrategy(make_config())
        assert isinstance(s.generate_signals(df), pd.Series)

    def test_index_matches_input(self):
        df = make_df([100.0] * 30)
        s = RSIStrategy(make_config())
        signals = s.generate_signals(df)
        pd.testing.assert_index_equal(signals.index, df.index)

    def test_only_valid_signal_values(self):
        df = make_df(sharp_drop_then_recover())
        s = RSIStrategy(make_config())
        signals = s.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestRSIComputation:

    def test_rsi_bounded_between_0_and_100(self):
        df = make_df(sharp_drop_then_recover())
        s = RSIStrategy(make_config())
        rsi = s._compute_rsi(df["close"])
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_low_after_sustained_drop(self):
        """After a long series of falling prices RSI should be well below 50."""
        prices = [100.0 - i for i in range(40)]
        df = make_df(prices)
        s = RSIStrategy(make_config())
        rsi = s._compute_rsi(df["close"])
        assert rsi.dropna().iloc[-1] < 50

    def test_rsi_high_after_sustained_rise(self):
        """After a long series of rising prices RSI should be well above 50."""
        prices = [50.0 + i for i in range(40)]
        df = make_df(prices)
        s = RSIStrategy(make_config())
        rsi = s._compute_rsi(df["close"])
        assert rsi.dropna().iloc[-1] > 50

    def test_flat_prices_produce_nan_rsi(self):
        """Flat prices → zero gain and zero loss → RSI is undefined (NaN)."""
        df = make_df([100.0] * 30)
        s = RSIStrategy(make_config())
        rsi = s._compute_rsi(df["close"])
        # After the first diff, all changes are 0 — avg_loss is 0 → NaN
        assert rsi.dropna().empty or (rsi.dropna() == 100.0).all()


class TestBuySignals:

    def test_buy_signal_fires_on_oversold_cross(self):
        df = make_df(sharp_drop_then_recover())
        s = RSIStrategy(make_config(period=5, oversold=40.0, overbought=60.0))
        signals = s.generate_signals(df)
        assert (signals == 1).any()

    def test_no_buy_on_flat_prices(self):
        df = make_df([100.0] * 60)
        s = RSIStrategy(make_config())
        signals = s.generate_signals(df)
        assert not (signals == 1).any()


class TestSellSignals:

    def test_sell_signal_fires_on_overbought_cross(self):
        df = make_df(sharp_rise_then_fall())
        s = RSIStrategy(make_config(period=5, oversold=40.0, overbought=60.0))
        signals = s.generate_signals(df)
        assert (signals == -1).any()

    def test_no_sell_on_flat_prices(self):
        df = make_df([100.0] * 60)
        s = RSIStrategy(make_config())
        signals = s.generate_signals(df)
        assert not (signals == -1).any()


class TestNoLookahead:

    def test_single_bar_returns_zero(self):
        df = make_df([100.0])
        s = RSIStrategy(make_config())
        signals = s.generate_signals(df)
        assert signals.iloc[0] == 0

    def test_signals_stable_when_future_bars_added(self):
        prices = sharp_drop_then_recover()
        df_partial = make_df(prices[:35])
        df_full    = make_df(prices)
        s = RSIStrategy(make_config(period=5, oversold=40.0, overbought=60.0))

        sig_partial = s.generate_signals(df_partial)
        sig_full    = s.generate_signals(df_full)

        pd.testing.assert_series_equal(
            sig_partial,
            sig_full.iloc[:35],
            check_names=False,
        )


class TestConfigPassthrough:

    def test_reads_period_from_config(self):
        s = RSIStrategy(make_config(period=7))
        assert s.period == 7

    def test_reads_oversold_from_config(self):
        s = RSIStrategy(make_config(oversold=25.0))
        assert s.oversold == 25.0

    def test_reads_overbought_from_config(self):
        s = RSIStrategy(make_config(overbought=75.0))
        assert s.overbought == 75.0

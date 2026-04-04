import pytest
import pandas as pd

from config import BacktestConfig, MACDParams
from strategy.macd import MACDStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(fast=5, slow=12, signal=4):
    return BacktestConfig(
        macd=MACDParams(fast_period=fast, slow_period=slow, signal_period=signal)
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


def rising_then_falling(n=80):
    rise = [100.0 + i * 1.5 for i in range(n // 2)]
    fall = [100.0 + (n // 2) * 1.5 - i * 1.5 for i in range(n // 2)]
    return rise + fall


def falling_then_rising(n=80):
    fall = [200.0 - i * 1.5 for i in range(n // 2)]
    rise = [200.0 - (n // 2) * 1.5 + i * 1.5 for i in range(n // 2)]
    return fall + rise


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReturnShape:

    def test_returns_series(self):
        df = make_df(list(range(1, 61)))
        s = MACDStrategy(make_config())
        assert isinstance(s.generate_signals(df), pd.Series)

    def test_index_matches_input(self):
        df = make_df(list(range(1, 61)))
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        pd.testing.assert_index_equal(signals.index, df.index)

    def test_only_valid_signal_values(self):
        df = make_df(rising_then_falling())
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        assert set(signals.unique()).issubset({-1, 0, 1})


class TestBullishCross:

    def test_buy_signal_fires_on_bullish_cross(self):
        """Falling then rising prices should cause MACD to cross above signal line."""
        df = make_df(falling_then_rising())
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        assert (signals == 1).any()

    def test_rising_prices_produce_buy_signal(self):
        """Steadily rising prices should produce at least one bullish cross."""
        df = make_df(list(range(1, 81)))
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        assert (signals == 1).any()

    def test_flat_prices_produce_no_signals(self):
        """Identical prices → fast EMA == slow EMA always → no crossover."""
        df = make_df([100.0] * 60)
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        assert (signals == 0).all()


class TestBearishCross:

    def test_sell_signal_fires_on_bearish_cross(self):
        """Rising then falling prices should cause MACD to cross below signal line."""
        df = make_df(rising_then_falling())
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        assert (signals == -1).any()

    def test_falling_prices_produce_sell_signal(self):
        """Rising then falling prices produce a bearish cross."""
        df = make_df(rising_then_falling())
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        assert (signals == -1).any()


class TestCrossoverFires_Once:

    def test_single_bullish_cross_fires_exactly_once(self):
        """One smooth upswing should produce exactly one buy crossover."""
        df = make_df(list(range(1, 81)))
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        assert (signals == 1).sum() == 1

    def test_single_bearish_cross_fires_exactly_once(self):
        """One smooth rise then fall should produce exactly one bearish crossover."""
        df = make_df(rising_then_falling())
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        assert (signals == -1).sum() == 1


class TestNoLookahead:

    def test_single_bar_returns_zero(self):
        df = make_df([100.0])
        s = MACDStrategy(make_config())
        signals = s.generate_signals(df)
        assert signals.iloc[0] == 0

    def test_signals_stable_when_future_bars_added(self):
        prices = rising_then_falling()
        df_partial = make_df(prices[:50])
        df_full    = make_df(prices)
        s = MACDStrategy(make_config())

        sig_partial = s.generate_signals(df_partial)
        sig_full    = s.generate_signals(df_full)

        pd.testing.assert_series_equal(
            sig_partial,
            sig_full.iloc[:50],
            check_names=False,
        )


class TestConfigPassthrough:

    def test_reads_fast_period_from_config(self):
        s = MACDStrategy(make_config(fast=3))
        assert s.fast_period == 3

    def test_reads_slow_period_from_config(self):
        s = MACDStrategy(make_config(slow=10))
        assert s.slow_period == 10

    def test_reads_signal_period_from_config(self):
        s = MACDStrategy(make_config(signal=3))
        assert s.signal_period == 3

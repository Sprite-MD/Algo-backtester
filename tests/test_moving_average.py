import pandas as pd
import numpy as np
import pytest

from config import BacktestConfig, MovingAverageParams
from strategy.moving_average import MovingAverageStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(fast=5, slow=20, ma_type="sma"):
    return BacktestConfig(
        moving_average=MovingAverageParams(
            fast_window=fast,
            slow_window=slow,
            ma_type=ma_type,
        )
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


def rising_then_falling(n=60):
    """Prices rise for the first half then fall for the second half."""
    rise = list(range(100, 100 + n // 2))
    fall = list(range(100 + n // 2, 100, -1))
    return rise + fall


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReturnShape:

    def test_returns_series(self):
        df = make_df(list(range(1, 51)))
        strategy = MovingAverageStrategy(make_config())
        result = strategy.generate_signals(df)
        assert isinstance(result, pd.Series)

    def test_index_matches_input(self):
        df = make_df(list(range(1, 51)))
        strategy = MovingAverageStrategy(make_config())
        result = strategy.generate_signals(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_only_valid_signal_values(self):
        df = make_df(rising_then_falling())
        strategy = MovingAverageStrategy(make_config())
        result = strategy.generate_signals(df)
        assert set(result.unique()).issubset({-1, 0, 1})


class TestSMASignals:

    def test_golden_cross_produces_buy_signal(self):
        """Steadily rising prices should produce at least one buy signal."""
        prices = list(range(1, 101))
        df = make_df(prices)
        strategy = MovingAverageStrategy(make_config(fast=5, slow=20, ma_type="sma"))
        signals = strategy.generate_signals(df)
        assert (signals == 1).any()

    def test_death_cross_produces_sell_signal(self):
        """Rising then falling prices should produce at least one sell signal."""
        df = make_df(rising_then_falling(60))
        strategy = MovingAverageStrategy(make_config(fast=5, slow=20, ma_type="sma"))
        signals = strategy.generate_signals(df)
        assert (signals == -1).any()

    def test_flat_prices_produce_no_signals(self):
        """Flat prices produce identical MAs — no crossover ever fires."""
        df = make_df([100.0] * 60)
        strategy = MovingAverageStrategy(make_config(fast=5, slow=20, ma_type="sma"))
        signals = strategy.generate_signals(df)
        assert (signals == 0).all()

    def test_no_signals_before_slow_window_fills(self):
        """No crossover can occur until the slow MA has enough data."""
        prices = list(range(1, 51))
        df = make_df(prices)
        strategy = MovingAverageStrategy(make_config(fast=5, slow=20, ma_type="sma"))
        signals = strategy.generate_signals(df)
        assert (signals.iloc[:19] == 0).all()

    def test_crossover_fires_exactly_once_per_cross(self):
        """A single golden cross should produce exactly one buy signal."""
        prices = list(range(1, 101))
        df = make_df(prices)
        strategy = MovingAverageStrategy(make_config(fast=5, slow=20, ma_type="sma"))
        signals = strategy.generate_signals(df)
        assert (signals == 1).sum() == 1


class TestEMASignals:

    def test_ema_golden_cross_produces_buy_signal(self):
        prices = list(range(1, 101))
        df = make_df(prices)
        strategy = MovingAverageStrategy(make_config(fast=5, slow=20, ma_type="ema"))
        signals = strategy.generate_signals(df)
        assert (signals == 1).any()

    def test_ema_death_cross_produces_sell_signal(self):
        df = make_df(rising_then_falling(60))
        strategy = MovingAverageStrategy(make_config(fast=5, slow=20, ma_type="ema"))
        signals = strategy.generate_signals(df)
        assert (signals == -1).any()

    def test_ema_flat_prices_produce_no_signals(self):
        df = make_df([100.0] * 60)
        strategy = MovingAverageStrategy(make_config(fast=5, slow=20, ma_type="ema"))
        signals = strategy.generate_signals(df)
        assert (signals == 0).all()


class TestNoLookahead:

    def test_single_bar_returns_zero(self):
        """With only one bar there is no previous bar to compare — must be hold."""
        df = make_df([100.0])
        strategy = MovingAverageStrategy(make_config(fast=2, slow=3, ma_type="sma"))
        signals = strategy.generate_signals(df)
        assert signals.iloc[0] == 0

    def test_signals_do_not_change_when_future_bars_appended(self):
        """
        Signals on bars 0..N must be identical whether or not bars N+1..M
        are present. If they change, the strategy is using future data.
        """
        prices = rising_then_falling(60)
        df_partial = make_df(prices[:40])
        df_full = make_df(prices)

        strategy = MovingAverageStrategy(make_config(fast=5, slow=20, ma_type="sma"))
        signals_partial = strategy.generate_signals(df_partial)
        signals_full = strategy.generate_signals(df_full)

        pd.testing.assert_series_equal(
            signals_partial,
            signals_full.iloc[:40],
            check_names=False,
        )


class TestConfigPassthrough:

    def test_reads_fast_window_from_config(self):
        config = make_config(fast=10, slow=30)
        strategy = MovingAverageStrategy(config)
        assert strategy.fast_window == 10

    def test_reads_slow_window_from_config(self):
        config = make_config(fast=10, slow=30)
        strategy = MovingAverageStrategy(config)
        assert strategy.slow_window == 30

    def test_reads_ma_type_from_config(self):
        config = make_config(ma_type="ema")
        strategy = MovingAverageStrategy(config)
        assert strategy.ma_type == "ema"

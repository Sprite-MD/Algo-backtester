import pytest
import pandas as pd
import numpy as np
from datetime import date

from config import BacktestConfig, MovingAverageParams
from engine.backtester import Backtester
from engine.portfolio import Portfolio
from strategy.moving_average import MovingAverageStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(capital=100_000.0, commission=0.0, slippage=0.0,
                fast=5, slow=20, position_size=1.0):
    return BacktestConfig(
        ticker="AAPL",
        initial_capital=capital,
        commission=commission,
        slippage=slippage,
        position_size=position_size,
        moving_average=MovingAverageParams(fast_window=fast, slow_window=slow),
    )


def make_df(prices: list) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=len(prices), freq="B")
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
    rise = list(range(100, 100 + n // 2))
    fall = list(range(100 + n // 2, 100, -1))
    return rise + fall


def run(prices, config=None):
    if config is None:
        config = make_config()
    df = make_df(prices)
    portfolio = Portfolio(config)
    strategy = MovingAverageStrategy(config)
    bt = Backtester(config, df, strategy, portfolio)
    return bt.run()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_returns_tuple_of_two(self):
        result = run(list(range(1, 61)))
        assert isinstance(result, tuple) and len(result) == 2

    def test_first_element_is_series(self):
        equity, _ = run(list(range(1, 61)))
        assert isinstance(equity, pd.Series)

    def test_second_element_is_dataframe(self):
        _, trades = run(list(range(1, 61)))
        assert isinstance(trades, pd.DataFrame)

    def test_equity_length_matches_data_length(self):
        prices = list(range(1, 61))
        equity, _ = run(prices)
        assert len(equity) == len(prices)

    def test_equity_index_is_datetimeindex(self):
        equity, _ = run(list(range(1, 61)))
        assert isinstance(equity.index, pd.DatetimeIndex)


class TestSimulationLogic:

    def test_buy_executed_on_golden_cross(self):
        _, trades = run(list(range(1, 61)))
        buys = trades[trades["direction"] == "buy"] if not trades.empty else pd.DataFrame()
        assert len(buys) >= 1

    def test_sell_executed_after_buy(self):
        _, trades = run(rising_then_falling(60))
        if not trades.empty:
            directions = trades["direction"].tolist()
            # First trade must be a buy before any sell
            assert directions[0] == "buy"

    def test_no_double_buy_without_sell(self):
        _, trades = run(list(range(1, 61)))
        if not trades.empty:
            directions = trades["direction"].tolist()
            for i in range(len(directions) - 1):
                assert not (directions[i] == "buy" and directions[i + 1] == "buy")

    def test_no_sell_without_prior_buy(self):
        _, trades = run(rising_then_falling(60))
        if not trades.empty:
            directions = trades["direction"].tolist()
            assert directions[0] != "sell"

    def test_equity_never_negative(self):
        equity, _ = run(rising_then_falling(60))
        assert (equity >= 0).all()

    def test_flat_prices_produce_no_trades(self):
        _, trades = run([100.0] * 60)
        assert trades.empty

    def test_equity_starts_at_initial_capital_on_flat_prices(self):
        equity, _ = run([100.0] * 60)
        assert equity.iloc[0] == pytest.approx(100_000.0)


class TestPositionSizing:

    def test_full_position_size_depletes_cash(self):
        config = make_config(commission=0.0, slippage=0.0, position_size=1.0)
        df = make_df(list(range(1, 61)))
        portfolio = Portfolio(config)
        strategy = MovingAverageStrategy(config)
        bt = Backtester(config, df, strategy, portfolio)
        bt.run()
        # After a buy with position_size=1.0, cash should be near zero
        if portfolio.positions:
            assert portfolio.cash == pytest.approx(0.0, abs=1e-6)

    def test_half_position_size_leaves_cash_remaining(self):
        config = make_config(commission=0.0, slippage=0.0, position_size=0.5)
        df = make_df(list(range(1, 61)))
        portfolio = Portfolio(config)
        strategy = MovingAverageStrategy(config)
        bt = Backtester(config, df, strategy, portfolio)
        bt.run()
        if portfolio.positions:
            assert portfolio.cash > 0


class TestNoLookahead:

    def test_equity_length_equals_bar_count(self):
        """One equity update per bar — confirms the loop runs bar by bar."""
        prices = list(range(1, 61))
        df = make_df(prices)
        config = make_config()
        portfolio = Portfolio(config)
        strategy = MovingAverageStrategy(config)
        bt = Backtester(config, df, strategy, portfolio)
        bt.run()
        assert len(portfolio.equity_curve) == len(prices)

    def test_results_consistent_with_extended_data(self):
        """
        Running on bars 0..N must produce the same equity curve as running
        on 0..M (M > N) and slicing to N. If not, future data is leaking in.
        """
        prices = rising_then_falling(60)
        equity_full, _ = run(prices)
        equity_partial, _ = run(prices[:40])

        pd.testing.assert_series_equal(
            equity_partial,
            equity_full.iloc[:40],
            check_names=False,
        )

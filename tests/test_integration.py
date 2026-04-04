"""
Integration tests — test the full pipeline from config through to metrics.
These tests exercise all modules working together, using synthetic price data
so no network calls are made.
"""
import argparse
import pytest
import pandas as pd
import numpy as np
from datetime import date
from unittest.mock import patch

from config import BacktestConfig, MovingAverageParams, RSIParams, MACDParams
from main import build_strategy, run_backtest, build_config, parse_args


# ---------------------------------------------------------------------------
# Shared synthetic data fixture
# ---------------------------------------------------------------------------

def make_ohlcv(prices: list) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "open":   prices,
            "high":   [p * 1.01 for p in prices],
            "low":    [p * 0.99 for p in prices],
            "close":  prices,
            "volume": [1_000_000] * len(prices),
        },
        index=idx,
    )


def rising_then_falling(n=100):
    rise = [100.0 + i * 1.0 for i in range(n // 2)]
    fall = [100.0 + (n // 2) * 1.0 - i * 1.0 for i in range(n // 2)]
    return rise + fall


PRICES = rising_then_falling()
DATA   = make_ohlcv(PRICES)


# ---------------------------------------------------------------------------
# build_strategy
# ---------------------------------------------------------------------------

class TestBuildStrategy:

    def test_returns_moving_average_strategy(self):
        from strategy.moving_average import MovingAverageStrategy
        config = BacktestConfig(strategy="moving_average")
        assert isinstance(build_strategy(config), MovingAverageStrategy)

    def test_returns_rsi_strategy(self):
        from strategy.rsi import RSIStrategy
        config = BacktestConfig(strategy="rsi")
        assert isinstance(build_strategy(config), RSIStrategy)

    def test_returns_macd_strategy(self):
        from strategy.macd import MACDStrategy
        config = BacktestConfig(strategy="macd")
        assert isinstance(build_strategy(config), MACDStrategy)


# ---------------------------------------------------------------------------
# Full pipeline — moving average
# ---------------------------------------------------------------------------

class TestPipelineMovingAverage:

    @pytest.fixture
    def result(self):
        config = BacktestConfig(
            strategy="moving_average",
            moving_average=MovingAverageParams(fast_window=5, slow_window=20),
            commission=0.0,
            slippage=0.0,
        )
        with patch("data.fetcher.yf.download", return_value=pd.DataFrame()):
            with patch("data.fetcher.os.path.exists", return_value=False):
                pass
        # Run directly with synthetic data
        from engine.backtester import Backtester
        from engine.portfolio import Portfolio
        from strategy.moving_average import MovingAverageStrategy
        from risk.metrics import compute_all

        portfolio  = Portfolio(config)
        strategy   = MovingAverageStrategy(config)
        backtester = Backtester(config, DATA, strategy, portfolio)
        equity, trade_df = backtester.run()
        metrics = compute_all(equity, trade_df)
        return equity, trade_df, metrics

    def test_equity_is_series(self, result):
        equity, _, _ = result
        assert isinstance(equity, pd.Series)

    def test_equity_length_matches_data(self, result):
        equity, _, _ = result
        assert len(equity) == len(DATA)

    def test_trade_df_is_dataframe(self, result):
        _, trade_df, _ = result
        assert isinstance(trade_df, pd.DataFrame)

    def test_metrics_is_dict_with_all_keys(self, result):
        _, _, metrics = result
        expected = {
            "total_return", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "win_rate",
            "profit_factor", "avg_trade_duration", "calmar_ratio",
        }
        assert expected == set(metrics.keys())

    def test_equity_never_negative(self, result):
        equity, _, _ = result
        assert (equity >= 0).all()

    def test_trades_alternating_buy_sell(self, result):
        _, trade_df, _ = result
        if not trade_df.empty:
            directions = trade_df["direction"].tolist()
            for i in range(len(directions) - 1):
                assert not (directions[i] == "buy" and directions[i + 1] == "buy")
                assert not (directions[i] == "sell" and directions[i + 1] == "sell")


# ---------------------------------------------------------------------------
# Full pipeline — RSI
# ---------------------------------------------------------------------------

class TestPipelineRSI:

    @pytest.fixture
    def result(self):
        config = BacktestConfig(
            strategy="rsi",
            rsi=RSIParams(period=5, oversold=40.0, overbought=60.0),
            commission=0.0,
            slippage=0.0,
        )
        from engine.backtester import Backtester
        from engine.portfolio import Portfolio
        from strategy.rsi import RSIStrategy
        from risk.metrics import compute_all

        # Warmup + drop + recover pattern that triggers RSI signals
        warmup  = [100.0 + (i % 2) for i in range(30)]
        drop    = [100.0 - i * 4.0 for i in range(20)]
        recover = [100.0 - 76.0 + i * 4.0 for i in range(20)]
        data = make_ohlcv(warmup + drop + recover)

        portfolio  = Portfolio(config)
        strategy   = RSIStrategy(config)
        backtester = Backtester(config, data, strategy, portfolio)
        equity, trade_df = backtester.run()
        metrics = compute_all(equity, trade_df)
        return equity, trade_df, metrics

    def test_equity_is_series(self, result):
        equity, _, _ = result
        assert isinstance(equity, pd.Series)

    def test_metrics_has_all_keys(self, result):
        _, _, metrics = result
        assert len(metrics) == 9

    def test_equity_never_negative(self, result):
        equity, _, _ = result
        assert (equity >= 0).all()


# ---------------------------------------------------------------------------
# Full pipeline — MACD
# ---------------------------------------------------------------------------

class TestPipelineMACD:

    @pytest.fixture
    def result(self):
        config = BacktestConfig(
            strategy="macd",
            macd=MACDParams(fast_period=5, slow_period=12, signal_period=4),
            commission=0.0,
            slippage=0.0,
        )
        from engine.backtester import Backtester
        from engine.portfolio import Portfolio
        from strategy.macd import MACDStrategy
        from risk.metrics import compute_all

        portfolio  = Portfolio(config)
        strategy   = MACDStrategy(config)
        backtester = Backtester(config, DATA, strategy, portfolio)
        equity, trade_df = backtester.run()
        metrics = compute_all(equity, trade_df)
        return equity, trade_df, metrics

    def test_equity_is_series(self, result):
        equity, _, _ = result
        assert isinstance(equity, pd.Series)

    def test_metrics_has_all_keys(self, result):
        _, _, metrics = result
        assert len(metrics) == 9

    def test_equity_never_negative(self, result):
        equity, _, _ = result
        assert (equity >= 0).all()


# ---------------------------------------------------------------------------
# build_config — CLI override tests
# ---------------------------------------------------------------------------

class TestBuildConfig:

    def _args(self, **kwargs):
        defaults = dict(ticker=None, strategy=None, start=None, end=None,
                        capital=None, no_dashboard=False)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_defaults_produce_valid_config(self):
        config = build_config(self._args())
        assert isinstance(config, BacktestConfig)

    def test_ticker_override(self):
        config = build_config(self._args(ticker="TSLA"))
        assert config.ticker == "TSLA"

    def test_strategy_override(self):
        config = build_config(self._args(strategy="rsi"))
        assert config.strategy == "rsi"

    def test_start_date_override(self):
        config = build_config(self._args(start="2021-01-01"))
        assert config.start_date == date(2021, 1, 1)

    def test_end_date_override(self):
        config = build_config(self._args(end="2023-06-01"))
        assert config.end_date == date(2023, 6, 1)

    def test_capital_override(self):
        config = build_config(self._args(capital=50_000.0))
        assert config.initial_capital == 50_000.0

    def test_multiple_overrides(self):
        config = build_config(self._args(ticker="MSFT", strategy="macd", capital=200_000.0))
        assert config.ticker == "MSFT"
        assert config.strategy == "macd"
        assert config.initial_capital == 200_000.0


# ---------------------------------------------------------------------------
# run_backtest — pipeline integration with mocked fetcher
# ---------------------------------------------------------------------------

class TestRunBacktest:

    def test_returns_four_values(self):
        config = BacktestConfig(
            strategy="moving_average",
            moving_average=MovingAverageParams(fast_window=5, slow_window=20),
        )
        with patch("main.fetch_data", return_value=DATA):
            result = run_backtest(config)
        assert len(result) == 4

    def test_data_passthrough(self):
        config = BacktestConfig(
            strategy="moving_average",
            moving_average=MovingAverageParams(fast_window=5, slow_window=20),
        )
        with patch("main.fetch_data", return_value=DATA) as mock_fetch:
            run_backtest(config)
        mock_fetch.assert_called_once_with(config)

    def test_equity_shape(self):
        config = BacktestConfig(
            strategy="moving_average",
            moving_average=MovingAverageParams(fast_window=5, slow_window=20),
        )
        with patch("main.fetch_data", return_value=DATA):
            data, equity, trade_df, metrics = run_backtest(config)
        assert len(equity) == len(DATA)

    def test_metrics_all_floats(self):
        config = BacktestConfig(
            strategy="moving_average",
            moving_average=MovingAverageParams(fast_window=5, slow_window=20),
        )
        with patch("main.fetch_data", return_value=DATA):
            _, _, _, metrics = run_backtest(config)
        for k, v in metrics.items():
            assert isinstance(v, float), f"{k} is not float"

    def test_all_three_strategies_complete(self):
        for strategy_name in ["moving_average", "rsi", "macd"]:
            config = BacktestConfig(strategy=strategy_name)
            with patch("main.fetch_data", return_value=DATA):
                data, equity, trade_df, metrics = run_backtest(config)
            assert isinstance(equity, pd.Series)
            assert isinstance(metrics, dict)

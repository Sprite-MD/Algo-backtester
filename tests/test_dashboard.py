import pytest
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from config import BacktestConfig
from visualization.dashboard import (
    _equity_chart,
    _drawdown_chart,
    _candlestick_chart,
    _metrics_table,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_equity(n=50) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    values = [100_000 + i * 200 for i in range(n)]
    return pd.Series(values, index=idx, name="equity", dtype=float)


def make_data(n=50) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    price = [100.0 + i * 0.5 for i in range(n)]
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": [1_000_000] * n},
        index=idx,
    )


def make_trade_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"direction": "buy",  "quantity": 10, "fill_price": 100.0, "timestamp": datetime(2020, 1, 10)},
        {"direction": "sell", "quantity": 10, "fill_price": 120.0, "timestamp": datetime(2020, 2, 10)},
    ])


def make_metrics() -> dict:
    return {
        "total_return":       0.15,
        "annualized_return":  0.12,
        "sharpe_ratio":       1.4,
        "sortino_ratio":      1.8,
        "max_drawdown":      -0.08,
        "win_rate":           0.65,
        "profit_factor":      2.1,
        "avg_trade_duration": 22.5,
    }


config = BacktestConfig()


# ---------------------------------------------------------------------------
# Equity chart
# ---------------------------------------------------------------------------

class TestEquityChart:

    def test_returns_figure(self):
        fig = _equity_chart(make_equity(), make_data(), config)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self):
        fig = _equity_chart(make_equity(), make_data(), config)
        assert len(fig.data) == 2

    def test_first_trace_is_strategy(self):
        fig = _equity_chart(make_equity(), make_data(), config)
        assert fig.data[0].name == "Strategy"

    def test_second_trace_is_buy_and_hold(self):
        fig = _equity_chart(make_equity(), make_data(), config)
        assert fig.data[1].name == "Buy & Hold"

    def test_benchmark_starts_at_initial_equity(self):
        equity = make_equity()
        fig = _equity_chart(equity, make_data(), config)
        bnh_first = fig.data[1].y[0]
        assert bnh_first == pytest.approx(equity.iloc[0])


# ---------------------------------------------------------------------------
# Drawdown chart
# ---------------------------------------------------------------------------

class TestDrawdownChart:

    def test_returns_figure(self):
        fig = _drawdown_chart(make_equity())
        assert isinstance(fig, go.Figure)

    def test_has_one_trace(self):
        fig = _drawdown_chart(make_equity())
        assert len(fig.data) == 1

    def test_drawdown_non_positive_for_rising_equity(self):
        fig = _drawdown_chart(make_equity())
        assert all(v <= 0 for v in fig.data[0].y)

    def test_drawdown_negative_after_peak(self):
        values = [100_000, 120_000, 90_000, 95_000]
        idx = pd.date_range("2020-01-01", periods=4, freq="B")
        equity = pd.Series(values, index=idx)
        fig = _drawdown_chart(equity)
        assert min(fig.data[0].y) < 0


# ---------------------------------------------------------------------------
# Candlestick chart
# ---------------------------------------------------------------------------

class TestCandlestickChart:

    def test_returns_figure(self):
        fig = _candlestick_chart(make_data(), make_trade_df())
        assert isinstance(fig, go.Figure)

    def test_has_candlestick_trace(self):
        fig = _candlestick_chart(make_data(), make_trade_df())
        types = [type(t).__name__ for t in fig.data]
        assert "Candlestick" in types

    def test_buy_marker_trace_present(self):
        fig = _candlestick_chart(make_data(), make_trade_df())
        names = [t.name for t in fig.data]
        assert "Buy" in names

    def test_sell_marker_trace_present(self):
        fig = _candlestick_chart(make_data(), make_trade_df())
        names = [t.name for t in fig.data]
        assert "Sell" in names

    def test_no_signal_traces_for_empty_trade_df(self):
        fig = _candlestick_chart(make_data(), pd.DataFrame())
        names = [t.name for t in fig.data]
        assert "Buy" not in names
        assert "Sell" not in names

    def test_only_candlestick_when_no_trades(self):
        fig = _candlestick_chart(make_data(), pd.DataFrame())
        assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

class TestMetricsTable:

    def test_returns_dataframe(self):
        df = _metrics_table(make_metrics())
        assert isinstance(df, pd.DataFrame)

    def test_has_metric_and_value_columns(self):
        df = _metrics_table(make_metrics())
        assert "Metric" in df.columns
        assert "Value" in df.columns

    def test_has_eight_rows(self):
        df = _metrics_table(make_metrics())
        assert len(df) == 8

    def test_total_return_formatted_as_percent(self):
        df = _metrics_table(make_metrics())
        row = df[df["Metric"] == "Total Return"]["Value"].iloc[0]
        assert "%" in row

    def test_max_drawdown_formatted_as_percent(self):
        df = _metrics_table(make_metrics())
        row = df[df["Metric"] == "Max Drawdown"]["Value"].iloc[0]
        assert "%" in row

    def test_sharpe_ratio_formatted_as_decimal(self):
        df = _metrics_table(make_metrics())
        row = df[df["Metric"] == "Sharpe Ratio"]["Value"].iloc[0]
        assert "%" not in row

    def test_all_metrics_keys_represented(self):
        df = _metrics_table(make_metrics())
        assert len(df) == len(make_metrics())

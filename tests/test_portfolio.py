import pytest
from datetime import datetime

import pandas as pd

from config import BacktestConfig
from engine.order import Order
from engine.portfolio import Portfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(capital=100_000.0, commission=0.001, slippage=0.0005):
    return BacktestConfig(
        initial_capital=capital,
        commission=commission,
        slippage=slippage,
    )


def make_order(direction="buy", quantity=10.0, price=100.0, ticker="AAPL"):
    return Order(
        ticker=ticker,
        direction=direction,
        order_type="market",
        quantity=quantity,
        price=price,
        timestamp=datetime(2023, 1, 3),
    )


TS = datetime(2023, 1, 3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInitialState:

    def test_cash_equals_initial_capital(self):
        p = Portfolio(make_config(capital=50_000.0))
        assert p.cash == 50_000.0

    def test_positions_start_empty(self):
        p = Portfolio(make_config())
        assert p.positions == {}

    def test_trade_log_starts_empty(self):
        p = Portfolio(make_config())
        assert p.trade_log == []

    def test_equity_curve_starts_empty(self):
        p = Portfolio(make_config())
        assert p.equity_curve == []


class TestBuyOrder:

    def test_cash_decreases_after_buy(self):
        p = Portfolio(make_config(commission=0.0, slippage=0.0))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        assert p.cash == 100_000.0 - 1_000.0

    def test_position_created_after_buy(self):
        p = Portfolio(make_config(commission=0.0, slippage=0.0))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        assert p.positions["AAPL"] == 10.0

    def test_commission_applied_on_buy(self):
        p = Portfolio(make_config(commission=0.01, slippage=0.0))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        expected_cash = 100_000.0 - (10.0 * 100.0 * 1.01)
        assert p.cash == pytest.approx(expected_cash)

    def test_slippage_applied_on_buy(self):
        p = Portfolio(make_config(commission=0.0, slippage=0.01))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        fill_price = 100.0 * 1.01
        expected_cash = 100_000.0 - (10.0 * fill_price)
        assert p.cash == pytest.approx(expected_cash)

    def test_trade_logged_after_buy(self):
        p = Portfolio(make_config())
        p.execute_order(make_order(direction="buy"), 100.0)
        assert len(p.trade_log) == 1
        assert p.trade_log[0]["direction"] == "buy"

    def test_multiple_buys_accumulate_position(self):
        p = Portfolio(make_config(commission=0.0, slippage=0.0))
        p.execute_order(make_order(direction="buy", quantity=5.0, price=100.0), 100.0)
        p.execute_order(make_order(direction="buy", quantity=5.0, price=100.0), 100.0)
        assert p.positions["AAPL"] == 10.0


class TestSellOrder:

    def test_cash_increases_after_sell(self):
        p = Portfolio(make_config(commission=0.0, slippage=0.0))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        cash_after_buy = p.cash
        p.execute_order(make_order(direction="sell", quantity=10.0, price=100.0), 100.0)
        assert p.cash > cash_after_buy

    def test_position_removed_after_full_sell(self):
        p = Portfolio(make_config(commission=0.0, slippage=0.0))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        p.execute_order(make_order(direction="sell", quantity=10.0, price=100.0), 100.0)
        assert "AAPL" not in p.positions

    def test_commission_applied_on_sell(self):
        p = Portfolio(make_config(commission=0.01, slippage=0.0))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        cash_before_sell = p.cash
        p.execute_order(make_order(direction="sell", quantity=10.0, price=100.0), 100.0)
        proceeds = 10.0 * 100.0 * (1 - 0.01)
        assert p.cash == pytest.approx(cash_before_sell + proceeds)

    def test_slippage_applied_on_sell(self):
        p = Portfolio(make_config(commission=0.0, slippage=0.01))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        cash_before_sell = p.cash
        p.execute_order(make_order(direction="sell", quantity=10.0, price=100.0), 100.0)
        fill_price = 100.0 * (1 - 0.01)
        proceeds = 10.0 * fill_price
        assert p.cash == pytest.approx(cash_before_sell + proceeds)

    def test_trade_logged_after_sell(self):
        p = Portfolio(make_config())
        p.execute_order(make_order(direction="buy"), 100.0)
        p.execute_order(make_order(direction="sell", quantity=10.0, price=100.0), 100.0)
        assert p.trade_log[1]["direction"] == "sell"


class TestEquityTracking:

    def test_equity_equals_cash_when_no_positions(self):
        p = Portfolio(make_config(capital=100_000.0, commission=0.0, slippage=0.0))
        p.update_equity(TS, {})
        assert p.equity_curve[0][1] == pytest.approx(100_000.0)

    def test_equity_includes_position_value(self):
        p = Portfolio(make_config(capital=100_000.0, commission=0.0, slippage=0.0))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        p.update_equity(TS, {"AAPL": 150.0})
        expected = p.cash + 10.0 * 150.0
        assert p.equity_curve[0][1] == pytest.approx(expected)

    def test_equity_curve_grows_with_each_update(self):
        p = Portfolio(make_config())
        p.update_equity(datetime(2023, 1, 3), {})
        p.update_equity(datetime(2023, 1, 4), {})
        p.update_equity(datetime(2023, 1, 5), {})
        assert len(p.equity_curve) == 3

    def test_get_equity_series_returns_series(self):
        p = Portfolio(make_config())
        p.update_equity(TS, {})
        result = p.get_equity_series()
        assert isinstance(result, pd.Series)

    def test_get_equity_series_has_datetimeindex(self):
        p = Portfolio(make_config())
        p.update_equity(TS, {})
        result = p.get_equity_series()
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_get_equity_series_empty_when_no_updates(self):
        p = Portfolio(make_config())
        result = p.get_equity_series()
        assert result.empty


class TestTradeLog:

    def test_get_trade_df_returns_dataframe(self):
        p = Portfolio(make_config())
        p.execute_order(make_order(direction="buy"), 100.0)
        result = p.get_trade_df()
        assert isinstance(result, pd.DataFrame)

    def test_get_trade_df_empty_when_no_trades(self):
        p = Portfolio(make_config())
        result = p.get_trade_df()
        assert result.empty

    def test_trade_df_has_expected_columns(self):
        p = Portfolio(make_config())
        p.execute_order(make_order(direction="buy"), 100.0)
        df = p.get_trade_df()
        for col in ["timestamp", "ticker", "direction", "quantity", "fill_price"]:
            assert col in df.columns

    def test_trade_count_matches_orders_executed(self):
        p = Portfolio(make_config(commission=0.0, slippage=0.0))
        p.execute_order(make_order(direction="buy", quantity=10.0, price=100.0), 100.0)
        p.execute_order(make_order(direction="sell", quantity=10.0, price=100.0), 100.0)
        assert len(p.get_trade_df()) == 2

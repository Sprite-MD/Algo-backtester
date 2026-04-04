import pytest
from datetime import datetime
from dataclasses import fields

from engine.order import Order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_order(**kwargs):
    defaults = dict(
        ticker="AAPL",
        direction="buy",
        order_type="market",
        quantity=10.0,
        price=150.0,
        timestamp=datetime(2023, 1, 3, 9, 30),
    )
    defaults.update(kwargs)
    return Order(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInstantiation:

    def test_creates_with_valid_fields(self):
        order = make_order()
        assert order.ticker == "AAPL"
        assert order.direction == "buy"
        assert order.order_type == "market"
        assert order.quantity == 10.0
        assert order.price == 150.0
        assert order.timestamp == datetime(2023, 1, 3, 9, 30)

    def test_buy_direction(self):
        order = make_order(direction="buy")
        assert order.direction == "buy"

    def test_sell_direction(self):
        order = make_order(direction="sell")
        assert order.direction == "sell"

    def test_market_order_type(self):
        order = make_order(order_type="market")
        assert order.order_type == "market"

    def test_limit_order_type(self):
        order = make_order(order_type="limit")
        assert order.order_type == "limit"

    def test_stop_order_type(self):
        order = make_order(order_type="stop")
        assert order.order_type == "stop"


class TestFields:

    def test_has_all_required_fields(self):
        field_names = {f.name for f in fields(Order)}
        assert field_names == {"ticker", "direction", "order_type", "quantity", "price", "timestamp"}

    def test_ticker_is_str(self):
        order = make_order(ticker="TSLA")
        assert isinstance(order.ticker, str)

    def test_quantity_is_float(self):
        order = make_order(quantity=25.0)
        assert isinstance(order.quantity, float)

    def test_price_is_float(self):
        order = make_order(price=200.0)
        assert isinstance(order.price, float)

    def test_timestamp_is_datetime(self):
        order = make_order()
        assert isinstance(order.timestamp, datetime)


class TestMutability:

    def test_fields_are_mutable(self):
        order = make_order(quantity=10.0)
        order.quantity = 20.0
        assert order.quantity == 20.0

    def test_different_tickers_are_independent(self):
        order_a = make_order(ticker="AAPL")
        order_b = make_order(ticker="TSLA")
        assert order_a.ticker != order_b.ticker


class TestEquality:

    def test_identical_orders_are_equal(self):
        order_a = make_order()
        order_b = make_order()
        assert order_a == order_b

    def test_orders_with_different_prices_are_not_equal(self):
        order_a = make_order(price=100.0)
        order_b = make_order(price=200.0)
        assert order_a != order_b

    def test_orders_with_different_directions_are_not_equal(self):
        order_a = make_order(direction="buy")
        order_b = make_order(direction="sell")
        assert order_a != order_b

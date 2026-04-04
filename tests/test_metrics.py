import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from risk.metrics import (
    total_return,
    annualized_return,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    win_rate,
    profit_factor,
    avg_trade_duration,
    compute_all,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_equity(values: list, start="2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="equity", dtype=float)


def make_trade_df(trades: list) -> pd.DataFrame:
    """
    trades: list of dicts with keys:
        direction, quantity, fill_price, timestamp
    """
    return pd.DataFrame(trades)


def make_round_trip(buy_price: float, sell_price: float,
                    qty: float = 10.0,
                    buy_date=datetime(2020, 1, 2),
                    sell_date=datetime(2020, 1, 12)) -> pd.DataFrame:
    return make_trade_df([
        {"direction": "buy",  "quantity": qty, "fill_price": buy_price,  "timestamp": buy_date},
        {"direction": "sell", "quantity": qty, "fill_price": sell_price, "timestamp": sell_date},
    ])


# ---------------------------------------------------------------------------
# total_return
# ---------------------------------------------------------------------------

class TestTotalReturn:

    def test_positive_return(self):
        equity = make_equity([100_000, 110_000])
        assert total_return(equity) == pytest.approx(0.10)

    def test_negative_return(self):
        equity = make_equity([100_000, 90_000])
        assert total_return(equity) == pytest.approx(-0.10)

    def test_zero_return(self):
        equity = make_equity([100_000, 100_000])
        assert total_return(equity) == pytest.approx(0.0)

    def test_multi_bar(self):
        equity = make_equity([100_000, 105_000, 110_000, 120_000])
        assert total_return(equity) == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------

class TestAnnualizedReturn:

    def test_positive_over_252_days(self):
        # 10% over exactly 252 trading days → annualized = 10%
        equity = make_equity([100_000] + [110_000] * 251)
        result = annualized_return(equity)
        assert result == pytest.approx(0.10, abs=0.005)

    def test_single_bar_returns_zero(self):
        equity = make_equity([100_000])
        assert annualized_return(equity) == 0.0

    def test_flat_equity_returns_zero(self):
        equity = make_equity([100_000] * 252)
        assert annualized_return(equity) == pytest.approx(0.0)

    def test_negative_return_is_negative(self):
        equity = make_equity([100_000, 50_000])
        assert annualized_return(equity) < 0


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:

    def test_positive_sharpe_for_consistent_gains(self):
        # Steady 0.1% daily gain
        values = [100_000 * (1.001 ** i) for i in range(252)]
        equity = make_equity(values)
        assert sharpe_ratio(equity) > 0

    def test_negative_sharpe_for_consistent_losses(self):
        values = [100_000 * (0.999 ** i) for i in range(252)]
        equity = make_equity(values)
        assert sharpe_ratio(equity) < 0

    def test_zero_sharpe_for_flat_equity(self):
        equity = make_equity([100_000] * 100)
        assert sharpe_ratio(equity) == pytest.approx(0.0)

    def test_higher_rf_lowers_sharpe(self):
        values = [100_000 * (1.001 ** i) for i in range(252)]
        equity = make_equity(values)
        assert sharpe_ratio(equity, risk_free_rate=0.0) > sharpe_ratio(equity, risk_free_rate=0.05)


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:

    def test_positive_sortino_for_net_positive_returns(self):
        # Mostly gains with occasional small losses → positive Sortino
        np.random.seed(0)
        returns = np.random.choice([0.01, -0.002], size=252, p=[0.7, 0.3])
        values  = [100_000]
        for r in returns:
            values.append(values[-1] * (1 + r))
        equity = make_equity(values)
        assert sortino_ratio(equity) > 0

    def test_zero_sortino_for_flat_equity(self):
        equity = make_equity([100_000] * 100)
        assert sortino_ratio(equity) == pytest.approx(0.0)

    def test_sortino_returns_zero_when_no_downside(self):
        # Monotonically rising equity — no negative daily returns
        values = [100_000 + i * 100 for i in range(50)]
        equity = make_equity(values)
        assert sortino_ratio(equity) == pytest.approx(0.0)

    def test_sortino_higher_than_sharpe_for_right_skew(self):
        # Strategy with large gains and small losses → Sortino > Sharpe
        np.random.seed(42)
        returns = np.random.choice([0.02, -0.005], size=252, p=[0.6, 0.4])
        values  = [100_000]
        for r in returns:
            values.append(values[-1] * (1 + r))
        equity = make_equity(values)
        assert sortino_ratio(equity) >= sharpe_ratio(equity)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:

    def test_known_drawdown(self):
        # Peak at 120k, then drops to 90k → drawdown = (90-120)/120 = -0.25
        equity = make_equity([100_000, 120_000, 90_000, 95_000])
        assert max_drawdown(equity) == pytest.approx(-0.25)

    def test_no_drawdown_for_monotonic_rise(self):
        equity = make_equity([100_000, 110_000, 120_000, 130_000])
        assert max_drawdown(equity) == pytest.approx(0.0)

    def test_drawdown_is_negative_or_zero(self):
        equity = make_equity([100_000, 80_000, 90_000, 70_000])
        assert max_drawdown(equity) <= 0

    def test_full_loss_drawdown(self):
        equity = make_equity([100_000, 50_000])
        assert max_drawdown(equity) == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------

class TestWinRate:

    def test_all_winners(self):
        df = make_round_trip(buy_price=100.0, sell_price=120.0)
        assert win_rate(df) == pytest.approx(1.0)

    def test_all_losers(self):
        df = make_round_trip(buy_price=100.0, sell_price=80.0)
        assert win_rate(df) == pytest.approx(0.0)

    def test_half_winners(self):
        df = make_trade_df([
            {"direction": "buy",  "quantity": 10, "fill_price": 100.0, "timestamp": datetime(2020, 1, 2)},
            {"direction": "sell", "quantity": 10, "fill_price": 120.0, "timestamp": datetime(2020, 1, 12)},
            {"direction": "buy",  "quantity": 10, "fill_price": 100.0, "timestamp": datetime(2020, 2, 2)},
            {"direction": "sell", "quantity": 10, "fill_price": 80.0,  "timestamp": datetime(2020, 2, 12)},
        ])
        assert win_rate(df) == pytest.approx(0.5)

    def test_empty_trade_df_returns_zero(self):
        assert win_rate(pd.DataFrame()) == pytest.approx(0.0)

    def test_only_buys_returns_zero(self):
        df = make_trade_df([
            {"direction": "buy", "quantity": 10, "fill_price": 100.0, "timestamp": datetime(2020, 1, 2)},
        ])
        assert win_rate(df) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------

class TestProfitFactor:

    def test_profitable_strategy(self):
        # Gross profit = 200, gross loss = 100 → PF = 2.0
        df = make_trade_df([
            {"direction": "buy",  "quantity": 10, "fill_price": 100.0, "timestamp": datetime(2020, 1, 2)},
            {"direction": "sell", "quantity": 10, "fill_price": 120.0, "timestamp": datetime(2020, 1, 12)},
            {"direction": "buy",  "quantity": 10, "fill_price": 100.0, "timestamp": datetime(2020, 2, 2)},
            {"direction": "sell", "quantity": 10, "fill_price": 90.0,  "timestamp": datetime(2020, 2, 12)},
        ])
        assert profit_factor(df) == pytest.approx(2.0)

    def test_empty_trade_df_returns_zero(self):
        assert profit_factor(pd.DataFrame()) == pytest.approx(0.0)

    def test_no_losses_returns_zero(self):
        df = make_round_trip(buy_price=100.0, sell_price=120.0)
        assert profit_factor(df) == pytest.approx(0.0)

    def test_only_losses_returns_zero(self):
        df = make_round_trip(buy_price=100.0, sell_price=80.0)
        assert profit_factor(df) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# avg_trade_duration
# ---------------------------------------------------------------------------

class TestAvgTradeDuration:

    def test_known_duration(self):
        df = make_round_trip(
            buy_price=100.0, sell_price=110.0,
            buy_date=datetime(2020, 1, 2),
            sell_date=datetime(2020, 1, 12),
        )
        assert avg_trade_duration(df) == pytest.approx(10.0)

    def test_empty_trade_df_returns_zero(self):
        assert avg_trade_duration(pd.DataFrame()) == pytest.approx(0.0)

    def test_multiple_trades_averaged(self):
        df = make_trade_df([
            {"direction": "buy",  "quantity": 10, "fill_price": 100.0, "timestamp": datetime(2020, 1, 1)},
            {"direction": "sell", "quantity": 10, "fill_price": 110.0, "timestamp": datetime(2020, 1, 11)},
            {"direction": "buy",  "quantity": 10, "fill_price": 100.0, "timestamp": datetime(2020, 2, 1)},
            {"direction": "sell", "quantity": 10, "fill_price": 110.0, "timestamp": datetime(2020, 2, 21)},
        ])
        # durations: 10, 20 → mean = 15
        assert avg_trade_duration(df) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# compute_all
# ---------------------------------------------------------------------------

class TestComputeAll:

    def test_returns_dict(self):
        equity = make_equity([100_000, 110_000])
        df     = make_round_trip(100.0, 110.0)
        result = compute_all(equity, df)
        assert isinstance(result, dict)

    def test_all_keys_present(self):
        equity = make_equity([100_000, 110_000])
        df     = make_round_trip(100.0, 110.0)
        result = compute_all(equity, df)
        expected_keys = {
            "total_return", "annualized_return", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "win_rate",
            "profit_factor", "avg_trade_duration",
        }
        assert expected_keys == set(result.keys())

    def test_values_are_floats(self):
        equity = make_equity([100_000, 110_000, 105_000, 115_000])
        df     = make_round_trip(100.0, 115.0)
        result = compute_all(equity, df)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float"

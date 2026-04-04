import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    """(final - initial) / initial"""
    return (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]


def annualized_return(equity: pd.Series) -> float:
    """Compound annual growth rate over the number of trading days."""
    n_days = len(equity)
    if n_days < 2:
        return 0.0
    total = total_return(equity)
    return (1 + total) ** (252 / n_days) - 1


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sharpe ratio.
    risk_free_rate is the annual rate; converted to daily before subtraction.
    """
    daily_returns = equity.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    rf_daily = risk_free_rate / 252
    excess   = daily_returns - rf_daily
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualised Sortino ratio — like Sharpe but uses downside deviation only.
    """
    daily_returns = equity.pct_change().dropna()
    rf_daily      = risk_free_rate / 252
    excess        = daily_returns - rf_daily
    downside      = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(252))


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough decline as a fraction (negative value).
    e.g. -0.25 means the portfolio fell 25% from its peak at worst.
    """
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def win_rate(trade_df: pd.DataFrame) -> float:
    """
    Fraction of completed round-trip trades where the net PnL was positive.
    Pairs buys with the next sell chronologically.
    """
    if trade_df.empty:
        return 0.0

    buys  = trade_df[trade_df["direction"] == "buy"].reset_index(drop=True)
    sells = trade_df[trade_df["direction"] == "sell"].reset_index(drop=True)
    n     = min(len(buys), len(sells))
    if n == 0:
        return 0.0

    wins = 0
    for i in range(n):
        buy_cost     = buys.loc[i, "quantity"] * buys.loc[i, "fill_price"]
        sell_proceed = sells.loc[i, "quantity"] * sells.loc[i, "fill_price"]
        if sell_proceed > buy_cost:
            wins += 1
    return wins / n


def profit_factor(trade_df: pd.DataFrame) -> float:
    """
    Gross profit divided by gross loss across all completed round-trips.
    > 1.0 means the strategy made money overall.
    Returns 0.0 if there are no completed trades or no losses.
    """
    if trade_df.empty:
        return 0.0

    buys  = trade_df[trade_df["direction"] == "buy"].reset_index(drop=True)
    sells = trade_df[trade_df["direction"] == "sell"].reset_index(drop=True)
    n     = min(len(buys), len(sells))
    if n == 0:
        return 0.0

    gross_profit = 0.0
    gross_loss   = 0.0
    for i in range(n):
        pnl = (sells.loc[i, "quantity"] * sells.loc[i, "fill_price"]
               - buys.loc[i, "quantity"] * buys.loc[i, "fill_price"])
        if pnl > 0:
            gross_profit += pnl
        else:
            gross_loss += abs(pnl)

    if gross_loss == 0:
        return 0.0
    return gross_profit / gross_loss


def avg_trade_duration(trade_df: pd.DataFrame) -> float:
    """
    Mean number of calendar days between buy and sell for each round-trip.
    Returns 0.0 if there are no completed trades.
    """
    if trade_df.empty:
        return 0.0

    buys  = trade_df[trade_df["direction"] == "buy"].reset_index(drop=True)
    sells = trade_df[trade_df["direction"] == "sell"].reset_index(drop=True)
    n     = min(len(buys), len(sells))
    if n == 0:
        return 0.0

    durations = []
    for i in range(n):
        delta = pd.Timestamp(sells.loc[i, "timestamp"]) - pd.Timestamp(buys.loc[i, "timestamp"])
        durations.append(delta.days)

    return float(np.mean(durations))


def calmar_ratio(equity: pd.Series) -> float:
    """
    Annualized return divided by the absolute max drawdown.
    Higher is better. Returns 0.0 if max drawdown is zero.
    """
    ann_ret = annualized_return(equity)
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(ann_ret / abs(mdd))


def compute_all(equity: pd.Series, trade_df: pd.DataFrame) -> dict:
    """Run all metrics and return results in a single dictionary."""
    return {
        "total_return":       total_return(equity),
        "annualized_return":  annualized_return(equity),
        "sharpe_ratio":       sharpe_ratio(equity),
        "sortino_ratio":      sortino_ratio(equity),
        "max_drawdown":       max_drawdown(equity),
        "win_rate":           win_rate(trade_df),
        "profit_factor":      profit_factor(trade_df),
        "avg_trade_duration": avg_trade_duration(trade_df),
        "calmar_ratio":       calmar_ratio(equity),
    }

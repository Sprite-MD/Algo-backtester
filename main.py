import argparse
from datetime import date

from config import BacktestConfig, MovingAverageParams, RSIParams, MACDParams
from data.fetcher import fetch_data
from engine.backtester import Backtester
from engine.portfolio import Portfolio
from risk.metrics import compute_all
from strategy.moving_average import MovingAverageStrategy
from strategy.rsi import RSIStrategy
from strategy.macd import MACDStrategy
from visualization.dashboard import run_dashboard


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def build_strategy(config: BacktestConfig):
    """Return the correct strategy instance based on config.strategy."""
    if config.strategy == "moving_average":
        return MovingAverageStrategy(config)
    elif config.strategy == "rsi":
        return RSIStrategy(config)
    elif config.strategy == "macd":
        return MACDStrategy(config)
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_backtest(config: BacktestConfig) -> tuple:
    """
    Execute the full backtest pipeline.

    Returns
    -------
    tuple of (data, equity, trade_df, metrics)
    """
    data      = fetch_data(config)
    strategy  = build_strategy(config)
    portfolio = Portfolio(config)
    backtester = Backtester(config, data, strategy, portfolio)
    equity, trade_df = backtester.run()
    metrics = compute_all(equity, trade_df)
    return data, equity, trade_df, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Algorithmic Trading Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker",   type=str,  default=None, help="Stock ticker symbol")
    parser.add_argument("--strategy", type=str,  default=None, choices=["moving_average", "rsi", "macd"], help="Strategy to run")
    parser.add_argument("--start",    type=str,  default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",      type=str,  default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--capital",  type=float, default=None, help="Initial capital")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip launching the Streamlit dashboard")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BacktestConfig:
    """Build a BacktestConfig, applying any CLI overrides on top of defaults."""
    overrides = {}
    if args.ticker:
        overrides["ticker"] = args.ticker
    if args.strategy:
        overrides["strategy"] = args.strategy
    if args.start:
        overrides["start_date"] = date.fromisoformat(args.start)
    if args.end:
        overrides["end_date"] = date.fromisoformat(args.end)
    if args.capital:
        overrides["initial_capital"] = args.capital
    return BacktestConfig(**overrides)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = build_config(args)

    print(f"\nRunning backtest: {config.ticker} | {config.strategy} | {config.start_date} → {config.end_date}")
    print(f"Capital: ${config.initial_capital:,.0f} | Commission: {config.commission*100:.2f}% | Slippage: {config.slippage*100:.3f}%\n")

    data, equity, trade_df, metrics = run_backtest(config)

    # Print summary to terminal
    print("── Performance Summary ──────────────────────────")
    print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
    print(f"  Annualized Return:   {metrics['annualized_return']*100:>8.2f}%")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>8.2f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")
    print(f"  Win Rate:            {metrics['win_rate']*100:>8.2f}%")
    print(f"  Profit Factor:       {metrics['profit_factor']:>8.2f}")
    print(f"  Avg Trade Duration:  {metrics['avg_trade_duration']:>8.1f} days")
    print(f"  Total Trades:        {len(trade_df):>8}")
    print("─────────────────────────────────────────────────\n")

    if not args.no_dashboard:
        run_dashboard(config, data, equity, trade_df, metrics)


if __name__ == "__main__":
    main()

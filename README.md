# Algorithmic Trading Backtester

Test buy/sell strategies against historical price data. Simulates realistic trade execution with commission and slippage, then measures performance via risk-adjusted metrics and renders results in an interactive Streamlit dashboard.

## Strategies

| Strategy | Logic |
|---|---|
| Moving Average | Buy on golden cross (fast MA crosses above slow MA), sell on death cross |
| RSI | Buy when RSI drops below oversold threshold, sell when it rises above overbought |
| MACD | Buy when MACD line crosses above signal line, sell when it crosses below |

## Project Structure

```
Algo-backtester/
├── config.py                   # Single source of truth for all parameters
├── main.py                     # Pipeline functions and CLI
├── streamlit_app.py            # Streamlit entry point with interactive sidebar
├── data/
│   └── fetcher.py              # yfinance data download with local cache
├── strategy/
│   ├── base.py                 # Abstract base class all strategies implement
│   ├── moving_average.py       # SMA/EMA crossover strategy
│   ├── rsi.py                  # RSI mean-reversion strategy
│   └── macd.py                 # MACD crossover strategy
├── engine/
│   ├── order.py                # Order dataclass
│   ├── portfolio.py            # Cash, positions, equity curve
│   └── backtester.py           # Bar-by-bar simulation loop
├── risk/
│   └── metrics.py              # Sharpe, Sortino, drawdown, win rate, etc.
├── visualization/
│   └── dashboard.py            # Streamlit + Plotly results dashboard
└── tests/
    └── ...                     # Unit and integration tests
```

## How It Works

```
config.py
   ↓
data/fetcher.py         →  clean OHLCV DataFrame
   ↓
strategy/*.py           →  signal Series (1 = buy, -1 = sell, 0 = hold)
   ↓
engine/backtester.py    →  bar-by-bar simulation loop
   ├── engine/order.py        →  Order objects
   └── engine/portfolio.py    →  equity curve + trade log
   ↓
risk/metrics.py         →  performance statistics
   ↓
visualization/dashboard.py   →  interactive Streamlit app
```

## Setup

**Requirements:** Python 3.10+, WSL or Linux/macOS recommended.

```bash
# Clone the repo and navigate into it
git clone <repo-url>
cd Algo-backtester

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run with defaults

```bash
python main.py
```

Runs a Moving Average crossover backtest on AAPL from 2020-01-01 to 2024-01-01 with $100,000 starting capital.

### Override via CLI

```bash
python main.py --ticker TSLA --strategy rsi --start 2021-01-01 --end 2023-01-01
```

### Launch the interactive dashboard

```bash
streamlit run streamlit_app.py
```

Opens an interactive browser at `http://localhost:8501`. Use the sidebar to change the ticker, strategy, date range, capital, and all strategy parameters — the backtest reruns automatically on every change.

## Configuration

All parameters live in [config.py](config.py). They can be changed via the interactive sidebar in the dashboard, via CLI flags, or by editing the defaults directly.

| Parameter | Default | Description |
|---|---|---|
| `ticker` | `AAPL` | Stock symbol |
| `start_date` | `2020-01-01` | Start of simulation window |
| `end_date` | `2024-01-01` | End of simulation window |
| `initial_capital` | `100000` | Starting cash |
| `commission` | `0.001` | Fee per trade (0.1%) |
| `slippage` | `0.0005` | Fill price penalty (0.05%) |
| `position_size` | `1.0` | Fraction of capital deployed per signal |
| `strategy` | `moving_average` | Strategy to run |

### Strategy Parameters

**Moving Average**
| Parameter | Default | Description |
|---|---|---|
| `fast_window` | `20` | Short MA lookback period |
| `slow_window` | `50` | Long MA lookback period |
| `ma_type` | `sma` | `sma` or `ema` |

**RSI**
| Parameter | Default | Description |
|---|---|---|
| `period` | `14` | RSI lookback period |
| `oversold` | `30` | Buy threshold |
| `overbought` | `70` | Sell threshold |

**MACD**
| Parameter | Default | Description |
|---|---|---|
| `fast_period` | `12` | Fast EMA period |
| `slow_period` | `26` | Slow EMA period |
| `signal_period` | `9` | Signal line EMA period |

## Performance Metrics

The backtester computes the following after each run:

- **Total Return** — overall gain/loss as a percentage
- **Annualized Return** — return scaled to a yearly rate
- **Sharpe Ratio** — risk-adjusted return (penalises all volatility)
- **Sortino Ratio** — risk-adjusted return (penalises only downside volatility)
- **Max Drawdown** — largest peak-to-trough decline in equity
- **Win Rate** — percentage of trades that were profitable
- **Profit Factor** — gross profit divided by gross loss
- **Calmar Ratio** — annualized return divided by max drawdown
- **Avg Trade Duration** — mean number of days between entry and exit
- **Alpha vs Buy & Hold** — strategy return minus buy-and-hold return

## Running Tests

```bash
pytest tests/ -v
```

## Dependencies

```
yfinance
pandas
pydantic
plotly
streamlit
pyarrow
pytest
```

Install all at once:

```bash
pip install -r requirements.txt
```

## Disclaimer

This project is for educational purposes only. It is not financial advice and should not be used to make real investment decisions. Past performance of a backtested strategy does not guarantee future results.

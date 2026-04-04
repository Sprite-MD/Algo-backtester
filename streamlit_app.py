"""
Streamlit entry point — run from the project root:
    streamlit run streamlit_app.py

All sidebar widgets live here. Changing any control re-runs the script
automatically (standard Streamlit behaviour), which rebuilds the config,
re-runs the backtest, and refreshes every chart.
"""
import streamlit as st
from datetime import date

from config import BacktestConfig, MovingAverageParams, RSIParams, MACDParams
from main import run_backtest
from visualization.dashboard import run_dashboard

st.set_page_config(
    page_title="Algo Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — interactive controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    ticker = st.text_input("Ticker", value="AAPL").upper().strip()

    strategy = st.selectbox(
        "Strategy",
        options=["moving_average", "rsi", "macd"],
        format_func=lambda s: s.replace("_", " ").title(),
    )

    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start Date", value=date(2020, 1, 1), format="MM/DD/YYYY")
    end_date   = col2.date_input("End Date",   value=date(2024, 1, 1), format="MM/DD/YYYY")

    initial_capital = st.number_input(
        "Initial Capital ($)", min_value=1_000, max_value=10_000_000,
        value=100_000, step=1_000,
    )

    commission   = st.slider("Commission (%)",   0.0, 1.0,  0.10, step=0.01) / 100
    slippage     = st.slider("Slippage (%)",     0.0, 0.50, 0.05, step=0.005) / 100
    position_size = st.slider("Position Size (%)", 10,  100,  100,  step=10) / 100

    # Strategy-specific parameters
    st.markdown("---")
    st.subheader("Strategy Parameters")

    ma_params   = MovingAverageParams()
    rsi_params  = RSIParams()
    macd_params = MACDParams()

    if strategy == "moving_average":
        ma_type     = st.selectbox("MA Type", ["sma", "ema"], format_func=str.upper)
        fast_window = st.number_input("Fast Window", min_value=2,  max_value=200, value=20, step=1)
        slow_window = st.number_input("Slow Window", min_value=3,  max_value=500, value=50, step=1)
        try:
            ma_params = MovingAverageParams(
                ma_type=ma_type,
                fast_window=int(fast_window),
                slow_window=int(slow_window),
            )
        except Exception as e:
            st.error(f"MA config error: {e}")
            st.stop()

    elif strategy == "rsi":
        period    = st.number_input("Period",    min_value=2,  max_value=50, value=14, step=1)
        oversold  = st.slider("Oversold",        min_value=10, max_value=45, value=30)
        overbought = st.slider("Overbought",     min_value=55, max_value=90, value=70)
        try:
            rsi_params = RSIParams(
                period=int(period),
                oversold=float(oversold),
                overbought=float(overbought),
            )
        except Exception as e:
            st.error(f"RSI config error: {e}")
            st.stop()

    elif strategy == "macd":
        fast_period   = st.number_input("Fast Period",   min_value=2, max_value=50,  value=12, step=1)
        slow_period   = st.number_input("Slow Period",   min_value=3, max_value=100, value=26, step=1)
        signal_period = st.number_input("Signal Period", min_value=2, max_value=50,  value=9,  step=1)
        macd_params = MACDParams(
            fast_period=int(fast_period),
            slow_period=int(slow_period),
            signal_period=int(signal_period),
        )

# ---------------------------------------------------------------------------
# Build and validate config
# ---------------------------------------------------------------------------

try:
    config = BacktestConfig(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=float(initial_capital),
        commission=commission,
        slippage=slippage,
        position_size=position_size,
        strategy=strategy,
        moving_average=ma_params,
        rsi=rsi_params,
        macd=macd_params,
    )
except Exception as e:
    st.error(f"Invalid configuration: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Run backtest and render dashboard
# ---------------------------------------------------------------------------

with st.spinner("Running backtest…"):
    try:
        data, equity, trade_df, metrics = run_backtest(config)
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

run_dashboard(config, data, equity, trade_df, metrics)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from config import BacktestConfig  # kept for run_dashboard type hint


# ---------------------------------------------------------------------------
# Chart builders — each returns a go.Figure
# ---------------------------------------------------------------------------

def _equity_chart(equity: pd.Series, data: pd.DataFrame, config: BacktestConfig) -> go.Figure:
    """Portfolio equity vs buy-and-hold benchmark."""
    initial = equity.iloc[0]
    bnh = data["close"] / data["close"].iloc[0] * initial

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Strategy", line=dict(color="#2196F3"),
    ))
    fig.add_trace(go.Scatter(
        x=bnh.index, y=bnh.values,
        name="Buy & Hold", line=dict(color="#9E9E9E", dash="dash"),
    ))
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    return fig


def _drawdown_chart(equity: pd.Series) -> go.Figure:
    """Underwater equity curve showing loss periods."""
    rolling_max = equity.cummax()
    drawdown    = (equity - rolling_max) / rolling_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        name="Drawdown",
        fill="tozeroy",
        line=dict(color="#F44336"),
        fillcolor="rgba(244,67,54,0.2)",
    ))
    fig.update_layout(
        title="Drawdown (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )
    return fig


def _candlestick_chart(data: pd.DataFrame, trade_df: pd.DataFrame) -> go.Figure:
    """OHLC candlestick with buy/sell signal markers overlaid."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"],
        name="Price",
        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350",
    ))

    if not trade_df.empty:
        buys  = trade_df[trade_df["direction"] == "buy"]
        sells = trade_df[trade_df["direction"] == "sell"]

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(buys["timestamp"]),
                y=buys["fill_price"],
                mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=12, color="#26A69A"),
            ))

        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sells["timestamp"]),
                y=sells["fill_price"],
                mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=12, color="#EF5350"),
            ))

    fig.update_layout(
        title="Price & Signals",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    return fig


def _metrics_table(metrics: dict) -> pd.DataFrame:
    """Format metrics dict into a two-column display DataFrame."""
    labels = {
        "total_return":       "Total Return",
        "annualized_return":  "Annualized Return",
        "sharpe_ratio":       "Sharpe Ratio",
        "sortino_ratio":      "Sortino Ratio",
        "max_drawdown":       "Max Drawdown",
        "win_rate":           "Win Rate",
        "profit_factor":      "Profit Factor",
        "avg_trade_duration": "Avg Trade Duration (days)",
    }
    pct_keys = {"total_return", "annualized_return", "max_drawdown", "win_rate"}

    rows = []
    for key, label in labels.items():
        val = metrics.get(key, 0.0)
        if key in pct_keys:
            formatted = f"{val * 100:.2f}%"
        else:
            formatted = f"{val:.2f}"
        rows.append({"Metric": label, "Value": formatted})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_dashboard(
    config:   BacktestConfig,
    data:     pd.DataFrame,
    equity:   pd.Series,
    trade_df: pd.DataFrame,
    metrics:  dict,
) -> None:
    st.title("Algorithmic Trading Backtester")
    st.caption(f"{config.ticker} · {config.start_date.strftime('%m/%d/%Y')} → {config.end_date.strftime('%m/%d/%Y')} · {config.strategy.replace('_', ' ').title()}")

    # Row 1 — metrics table
    st.subheader("Performance Summary")
    st.dataframe(
        _metrics_table(metrics),
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")

    # Row 2 — equity curve and drawdown side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_equity_chart(equity, data, config), width="stretch")
    with col2:
        st.plotly_chart(_drawdown_chart(equity), width="stretch")

    st.markdown("---")

    # Row 3 — candlestick with signals, full width
    st.plotly_chart(_candlestick_chart(data, trade_df), width="stretch")

    # Row 4 — trade log
    if not trade_df.empty:
        st.subheader("Trade Log")
        st.dataframe(trade_df, width="stretch", hide_index=True)

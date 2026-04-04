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


def _candlestick_chart(
    data: pd.DataFrame,
    trade_df: pd.DataFrame,
    config: BacktestConfig,
) -> go.Figure:
    """OHLC candlestick with buy/sell signal markers and MA overlay."""
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

    # MA overlay — only for moving_average strategy since MAs are the signal
    if config.strategy == "moving_average":
        p = config.moving_average
        if p.ma_type == "sma":
            fast_ma = data["close"].rolling(p.fast_window).mean()
            slow_ma = data["close"].rolling(p.slow_window).mean()
        else:
            fast_ma = data["close"].ewm(span=p.fast_window, adjust=False).mean()
            slow_ma = data["close"].ewm(span=p.slow_window, adjust=False).mean()

        fig.add_trace(go.Scatter(
            x=data.index, y=fast_ma,
            name=f"Fast {p.ma_type.upper()}({p.fast_window})",
            line=dict(color="#FF9800", width=1),
        ))
        fig.add_trace(go.Scatter(
            x=data.index, y=slow_ma,
            name=f"Slow {p.ma_type.upper()}({p.slow_window})",
            line=dict(color="#9C27B0", width=1),
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
        xaxis_rangeslider_visible=True,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _render_metric_cards(metrics: dict, bnh_return: float) -> None:
    """Render performance metrics as st.metric() cards in a grid."""
    alpha = metrics["total_return"] - bnh_return

    row1 = st.columns(4)
    row1[0].metric("Total Return",      f"{metrics['total_return'] * 100:.2f}%")
    row1[1].metric("Annualized Return", f"{metrics['annualized_return'] * 100:.2f}%")
    row1[2].metric("Sharpe Ratio",      f"{metrics['sharpe_ratio']:.2f}")
    row1[3].metric("Sortino Ratio",     f"{metrics['sortino_ratio']:.2f}")

    row2 = st.columns(4)
    row2[0].metric("Max Drawdown",      f"{metrics['max_drawdown'] * 100:.2f}%")
    row2[1].metric("Win Rate",          f"{metrics['win_rate'] * 100:.2f}%")
    row2[2].metric("Profit Factor",     f"{metrics['profit_factor']:.2f}")
    row2[3].metric("Calmar Ratio",      f"{metrics['calmar_ratio']:.2f}")

    row3 = st.columns(4)
    row3[0].metric("Avg Trade Duration", f"{metrics['avg_trade_duration']:.1f} days")
    row3[1].metric("Alpha vs B&H",       f"{alpha * 100:.2f}%",
                   delta=f"{alpha * 100:.2f}%",
                   delta_color="normal")
    row3[2].metric("Buy & Hold Return",  f"{bnh_return * 100:.2f}%")


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

    # Buy-and-hold return for alpha calculation
    bnh_return = float((data["close"].iloc[-1] - data["close"].iloc[0]) / data["close"].iloc[0])

    # Row 1 — metric cards
    st.subheader("Performance Summary")
    _render_metric_cards(metrics, bnh_return)

    st.markdown("---")

    # Row 2 — equity curve and drawdown side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(_equity_chart(equity, data, config), width="stretch")
    with col2:
        st.plotly_chart(_drawdown_chart(equity), width="stretch")

    st.markdown("---")

    # Row 3 — candlestick with signals and MA overlay, full width
    st.plotly_chart(_candlestick_chart(data, trade_df, config), width="stretch")

    # Row 4 — trade log with download button
    if not trade_df.empty:
        st.subheader("Trade Log")
        st.dataframe(trade_df, width="stretch", hide_index=True)
        st.download_button(
            label="Download Trade Log as CSV",
            data=trade_df.to_csv(index=False),
            file_name=f"{config.ticker}_{config.strategy}_trades.csv",
            mime="text/csv",
        )

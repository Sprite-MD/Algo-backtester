"""
Streamlit entry point — run from the project root:
    streamlit run streamlit_app.py
"""
from main import run_backtest, build_config
from visualization.dashboard import run_dashboard
import argparse

config = build_config(argparse.Namespace(
    ticker=None, strategy=None, start=None, end=None,
    capital=None, no_dashboard=False,
))

data, equity, trade_df, metrics = run_backtest(config)
run_dashboard(config, data, equity, trade_df, metrics)

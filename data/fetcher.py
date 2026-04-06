import os
import pandas as pd
import yfinance as yf

from config import CONFIG, BacktestConfig


def fetch_data(config: BacktestConfig = CONFIG) -> pd.DataFrame:
    """
    Return a clean OHLCV DataFrame for the configured ticker and date range.
    Loads from local cache if available, otherwise downloads from yfinance.
    """
    os.makedirs(config.data_cache_dir, exist_ok=True)

    cache_path = os.path.join(
        config.data_cache_dir,
        f"{config.ticker}_{config.start_date}_{config.end_date}.parquet"
    )

    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    raw = yf.download(
        config.ticker,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=True,       # adjusts for splits and dividends
        progress=False,
        multi_level_index=False, # return flat columns instead of MultiIndex
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker '{config.ticker}' "
            f"between {config.start_date} and {config.end_date}. "
            "Check that the ticker is valid and the date range includes trading days."
        )

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]

    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    df.dropna(inplace=True)

    df.to_parquet(cache_path)

    return df

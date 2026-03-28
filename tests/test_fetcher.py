import os
import tempfile
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from config import BacktestConfig
from data.fetcher import fetch_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(tmpdir, ticker="AAPL", start=date(2023, 1, 1), end=date(2023, 3, 1)):
    """Return a BacktestConfig pointing its cache at a temporary directory."""
    return BacktestConfig(
        ticker=ticker,
        start_date=start,
        end_date=end,
        data_cache_dir=tmpdir,
    )


def make_raw_ohlcv(n_rows: int = 5) -> pd.DataFrame:
    """
    Return a minimal yfinance-style DataFrame with proper capitalised columns
    and a DatetimeIndex.
    """
    idx = pd.date_range("2023-01-03", periods=n_rows, freq="B")
    return pd.DataFrame(
        {
            "Open":   [150.0] * n_rows,
            "High":   [155.0] * n_rows,
            "Low":    [148.0] * n_rows,
            "Close":  [152.0] * n_rows,
            "Volume": [1_000_000] * n_rows,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReturnShape:
    """The DataFrame returned must have the right columns and index type."""

    def test_columns_are_lowercase_ohlcv(self, tmp_path):
        config = make_config(str(tmp_path))
        with patch("yfinance.download", return_value=make_raw_ohlcv()):
            df = fetch_data(config)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_index_is_datetimeindex(self, tmp_path):
        config = make_config(str(tmp_path))
        with patch("yfinance.download", return_value=make_raw_ohlcv()):
            df = fetch_data(config)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_index_name_is_date(self, tmp_path):
        config = make_config(str(tmp_path))
        with patch("yfinance.download", return_value=make_raw_ohlcv()):
            df = fetch_data(config)
        assert df.index.name == "date"

    def test_returns_dataframe(self, tmp_path):
        config = make_config(str(tmp_path))
        with patch("yfinance.download", return_value=make_raw_ohlcv()):
            df = fetch_data(config)
        assert isinstance(df, pd.DataFrame)

    def test_no_null_values(self, tmp_path):
        config = make_config(str(tmp_path))
        raw = make_raw_ohlcv()
        raw.iloc[2, 3] = float("nan")   # inject a NaN into the raw data
        with patch("yfinance.download", return_value=raw):
            df = fetch_data(config)
        assert not df.isnull().any().any()


class TestCaching:
    """Data should be saved on first fetch and loaded from disk on repeat calls."""

    def test_cache_file_is_created(self, tmp_path):
        config = make_config(str(tmp_path))
        with patch("yfinance.download", return_value=make_raw_ohlcv()):
            fetch_data(config)

        cache_files = list(tmp_path.glob("*.parquet"))
        assert len(cache_files) == 1

    def test_cache_filename_includes_ticker_and_dates(self, tmp_path):
        config = make_config(str(tmp_path))
        with patch("yfinance.download", return_value=make_raw_ohlcv()):
            fetch_data(config)

        cache_files = list(tmp_path.glob("*.parquet"))
        filename = cache_files[0].name
        assert "AAPL" in filename
        assert "2023-01-01" in filename
        assert "2023-03-01" in filename

    def test_yfinance_not_called_on_cache_hit(self, tmp_path):
        config = make_config(str(tmp_path))
        with patch("yfinance.download", return_value=make_raw_ohlcv()) as mock_dl:
            fetch_data(config)   # first call — populates cache
            fetch_data(config)   # second call — should hit cache

        assert mock_dl.call_count == 1

    def test_cached_data_matches_original(self, tmp_path):
        config = make_config(str(tmp_path))
        with patch("yfinance.download", return_value=make_raw_ohlcv()):
            df_first = fetch_data(config)
            df_second = fetch_data(config)

        pd.testing.assert_frame_equal(df_first, df_second, check_freq=False)

    def test_different_tickers_produce_separate_cache_files(self, tmp_path):
        config_aapl = make_config(str(tmp_path), ticker="AAPL")
        config_tsla = make_config(str(tmp_path), ticker="TSLA")

        with patch("yfinance.download", return_value=make_raw_ohlcv()):
            fetch_data(config_aapl)
            fetch_data(config_tsla)

        cache_files = list(tmp_path.glob("*.parquet"))
        assert len(cache_files) == 2


class TestErrorHandling:
    """Fetcher must raise clearly when yfinance returns no data."""

    def test_raises_on_empty_download(self, tmp_path):
        config = make_config(str(tmp_path))
        empty = pd.DataFrame()

        with patch("yfinance.download", return_value=empty):
            with pytest.raises(ValueError, match="No data returned"):
                fetch_data(config)

    def test_error_message_includes_ticker(self, tmp_path):
        config = make_config(str(tmp_path), ticker="FAKEXYZ")
        with patch("yfinance.download", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="FAKEXYZ"):
                fetch_data(config)


class TestMultiIndexHandling:
    """Fetcher must handle the MultiIndex columns yfinance sometimes returns."""

    def test_multiindex_columns_are_flattened(self, tmp_path):
        config = make_config(str(tmp_path))
        raw = make_raw_ohlcv()

        # Simulate yfinance MultiIndex columns like ("Close", "AAPL")
        raw.columns = pd.MultiIndex.from_tuples(
            [(col, "AAPL") for col in raw.columns]
        )

        with patch("yfinance.download", return_value=raw):
            df = fetch_data(config)

        assert list(df.columns) == ["open", "high", "low", "close", "volume"]


class TestCacheDirectory:
    """Cache directory should be created automatically if it does not exist."""

    def test_cache_dir_is_created_if_missing(self, tmp_path):
        nested = str(tmp_path / "new" / "nested" / "dir")
        config = make_config(nested)

        with patch("yfinance.download", return_value=make_raw_ohlcv()):
            fetch_data(config)

        assert os.path.isdir(nested)

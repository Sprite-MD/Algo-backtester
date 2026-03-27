from datetime import date
from typing import Literal
from pydantic import BaseModel, field_validator, model_validator


class MovingAverageParams(BaseModel):
    fast_window: int = 20
    slow_window: int = 50
    ma_type: Literal["sma", "ema"] = "sma"

    @model_validator(mode="after")
    def fast_must_be_less_than_slow(self):
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be less than slow_window")
        return self


class RSIParams(BaseModel):
    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0

    @model_validator(mode="after")
    def thresholds_must_not_overlap(self):
        if self.oversold >= self.overbought:
            raise ValueError("oversold must be less than overbought")
        return self


class MACDParams(BaseModel):
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9


class BacktestConfig(BaseModel):
    # Universe & date range
    ticker: str = "AAPL"
    start_date: date = date(2020, 1, 1)
    end_date: date = date(2024, 1, 1)

    # Capital & cost model
    initial_capital: float = 100_000.0
    commission: float = 0.001       # 0.1% per trade
    slippage: float = 0.0005        # 0.05% per fill
    position_size: float = 1.0      # fraction of capital to deploy per signal

    # Strategy selection
    strategy: Literal["moving_average", "rsi", "macd"] = "moving_average"

    # Strategy parameters
    moving_average: MovingAverageParams = MovingAverageParams()
    rsi: RSIParams = RSIParams()
    macd: MACDParams = MACDParams()

    # Infrastructure
    data_cache_dir: str = "data/cache"

    @field_validator("initial_capital")
    @classmethod
    def capital_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("initial_capital must be positive")
        return v

    @field_validator("commission", "slippage")
    @classmethod
    def rates_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("commission and slippage must be >= 0")
        return v

    @field_validator("position_size")
    @classmethod
    def position_size_valid(cls, v: float) -> float:
        if not (0 < v <= 1.0):
            raise ValueError("position_size must be between 0 (exclusive) and 1.0 (inclusive)")
        return v

    @model_validator(mode="after")
    def end_after_start(self):
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


CONFIG = BacktestConfig()

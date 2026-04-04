from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class Order:
    ticker:     str
    direction:  Literal["buy", "sell"]
    order_type: Literal["market", "limit", "stop"]
    quantity:   float
    price:      float
    timestamp:  datetime

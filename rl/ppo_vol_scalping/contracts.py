from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flax import struct

RAW_BAR_INPUT_COLUMNS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "average",
    "barCount",
)

RAW_BAR_INPUT_DTYPES = (
    "datetime64[us]",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "int64",
)

BAR_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "average_price",
    "trade_count",
    "day_id",
    "bar_in_day",
    "bars_in_day",
    "tau",
    "is_session_end",
)

STATIC_FEATURE_NAMES = (
    "atr_over_ema",
    "atr_imbalance",
)


@dataclass(frozen=True)
class PreprocessedArrays:
    ohlc: Any
    atr: Any
    atr_up: Any
    atr_down: Any
    atr_imbalance: Any
    atr_over_ema: Any
    day_ids: Any
    bar_in_day: Any
    session_end_mask: Any

    @property
    def num_bars(self) -> int:
        return int(self.ohlc.shape[0])

    def __getitem__(self, idx):
        return PreprocessedArrays(
            ohlc=self.ohlc[idx],
            atr=self.atr[idx],
            atr_up=self.atr_up[idx],
            atr_down=self.atr_down[idx],
            atr_imbalance=self.atr_imbalance[idx],
            atr_over_ema=self.atr_over_ema[idx],
            day_ids=self.day_ids[idx],
            bar_in_day=self.bar_in_day[idx],
            session_end_mask=self.session_end_mask[idx],
        )


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: int
    train_end: int
    inference_start: int
    inference_end: int

    @property
    def train_length(self) -> int:
        return self.train_end - self.train_start

    @property
    def inference_length(self) -> int:
        return self.inference_end - self.inference_start

    @property
    def validation_start(self) -> int:
        return self.inference_start

    @property
    def validation_end(self) -> int:
        return self.inference_end

    @property
    def validation_length(self) -> int:
        return self.inference_length


@struct.dataclass
class StrategyGrid:
    k_init: Any
    a_tp: Any
    k_act: Any
    entry_short: Any
    entry_long: Any

    @property
    def size(self) -> int:
        return int(self.k_init.shape[0])
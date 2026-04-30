from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple
import jax.numpy as jnp

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
    "tau",
    "log_return",
    "ema_variance",
    "srvi",
    "vslope",
    "vmacd",
    "vmacd_slope",
)

STATIC_FEATURE_DIM = len(STATIC_FEATURE_NAMES)
STATE_DIM = STATIC_FEATURE_DIM + 1


@dataclass(frozen=True)
class PreprocessedArrays:
    ohlc: Any
    static_features: Any
    atr: Any
    sigma_price: Any
    day_ids: Any
    bar_in_day: Any
    session_end_mask: Any

    @property
    def num_bars(self) -> int:
        return int(self.ohlc.shape[0])

    def __getitem__(self, idx):
        return PreprocessedArrays(
            ohlc=self.ohlc[idx],
            static_features=self.static_features[idx],
            atr=self.atr[idx],
            sigma_price=self.sigma_price[idx],
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
    episode_start_indices: Any

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

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
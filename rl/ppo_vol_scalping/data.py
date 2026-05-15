from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .contracts import BAR_COLUMNS, Fold, PreprocessedArrays, RAW_BAR_INPUT_COLUMNS
from data.database import Database

if TYPE_CHECKING:
    from .config import DataConfig, FeatureConfig

def load_raw_bars_dataframe(data_config: DataConfig) -> pd.DataFrame:
    database = Database()

    raw_bars_df = (
        database.load(data_config.root, data_config.start_date, data_config.end_date)
        .loc[:, RAW_BAR_INPUT_COLUMNS]
        .rename(
            columns={
                "date": "timestamp",
                "average": "average_price",
                "barCount": "trade_count",
            }
        )
        .copy()
    )

    raw_bars_df["timestamp"] = raw_bars_df["timestamp"].astype("datetime64[us]")
    raw_bars_df[["open", "high", "low", "close", "volume", "average_price"]] = raw_bars_df[
        ["open", "high", "low", "close", "volume", "average_price"]
    ].astype(np.float64)
    raw_bars_df["trade_count"] = raw_bars_df["trade_count"].astype(np.int64)
    assert not raw_bars_df.empty
    

    session_dates = raw_bars_df["timestamp"].dt.normalize()
    raw_bars_df["day_id"] = pd.factorize(session_dates, sort=False)[0].astype(np.int32)
    raw_bars_df["bar_in_day"] = raw_bars_df.groupby("day_id").cumcount().astype(np.int32)
    raw_bars_df["bars_in_day"] = raw_bars_df.groupby("day_id")["day_id"].transform("size").astype(np.int32)

    bar_in_day = raw_bars_df["bar_in_day"].to_numpy(dtype=np.float32)
    bars_in_day = raw_bars_df["bars_in_day"].to_numpy(dtype=np.float32)
    tau = np.zeros(len(raw_bars_df), dtype=np.float32)
    multi_bar_mask = bars_in_day > 1.0
    tau[multi_bar_mask] = bar_in_day[multi_bar_mask] / (bars_in_day[multi_bar_mask] - 1.0)

    raw_bars_df["tau"] = tau
    raw_bars_df["is_session_end"] = raw_bars_df["bar_in_day"] == (raw_bars_df["bars_in_day"] - 1)
    return raw_bars_df.loc[:, BAR_COLUMNS]


def build_preprocessed_arrays(raw_bars_df: pd.DataFrame, feature_config: FeatureConfig) -> PreprocessedArrays:
    close = raw_bars_df["close"].to_numpy(dtype=np.float64)
    open_ = raw_bars_df["open"].to_numpy(dtype=np.float64)
    high = raw_bars_df["high"].to_numpy(dtype=np.float64)
    low = raw_bars_df["low"].to_numpy(dtype=np.float64)
    assert close.size > 1
    prev_close = np.empty_like(close)
    prev_close[0] = close[0] if close.size else 0.0
    prev_close[1:] = close[:-1]

    true_range = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    atr = pd.Series(true_range, copy=False).ewm(
        alpha=1.0 / feature_config.atr_length,
        adjust=False,
    ).mean().to_numpy(dtype=np.float64)
    close_ema = pd.Series(close, copy=False).ewm(
        span=feature_config.atr_over_ema_length,
        adjust=False,
    ).mean().to_numpy(dtype=np.float64)

    close_delta = close - prev_close
    directional_true_range_up = np.where(close_delta > 0.0, true_range, 0.0)
    directional_true_range_down = np.where(close_delta < 0.0, true_range, 0.0)
    atr_up = pd.Series(directional_true_range_up, copy=False).ewm(
        span=feature_config.directional_atr_ema_length,
        adjust=False,
    ).mean().to_numpy(dtype=np.float64)
    atr_down = pd.Series(directional_true_range_down, copy=False).ewm(
        span=feature_config.directional_atr_ema_length,
        adjust=False,
    ).mean().to_numpy(dtype=np.float64)
    atr_imbalance = (atr_up - atr_down) / np.maximum(atr, feature_config.epsilon)
    atr_over_ema = atr / np.maximum(close_ema, feature_config.epsilon)

    ohlc = np.column_stack([open_, high, low, close])

    ohlc = np.ascontiguousarray(ohlc.astype(np.float32))
    atr = np.ascontiguousarray(atr.astype(np.float32))
    atr_up = np.ascontiguousarray(atr_up.astype(np.float32))
    atr_down = np.ascontiguousarray(atr_down.astype(np.float32))
    atr_imbalance = np.ascontiguousarray(atr_imbalance.astype(np.float32))
    atr_over_ema = np.ascontiguousarray(atr_over_ema.astype(np.float32))
    day_ids = np.ascontiguousarray(raw_bars_df["day_id"].to_numpy(dtype=np.int32))
    bar_in_day = np.ascontiguousarray(raw_bars_df["bar_in_day"].to_numpy(dtype=np.int32))
    session_end_mask = np.ascontiguousarray(raw_bars_df["is_session_end"].to_numpy(dtype=bool))

    if not np.isfinite(ohlc).all():
        raise ValueError("OHLC preprocessing produced non-finite values")
    if not np.isfinite(atr).all():
        raise ValueError("ATR preprocessing produced non-finite values")
    if not np.isfinite(atr_up).all():
        raise ValueError("ATR up preprocessing produced non-finite values")
    if not np.isfinite(atr_down).all():
        raise ValueError("ATR down preprocessing produced non-finite values")
    if not np.isfinite(atr_imbalance).all():
        raise ValueError("ATR imbalance preprocessing produced non-finite values")
    if not np.isfinite(atr_over_ema).all():
        raise ValueError("ATR over EMA preprocessing produced non-finite values")

    return PreprocessedArrays(
        ohlc=ohlc,
        atr=atr,
        atr_up=atr_up,
        atr_down=atr_down,
        atr_imbalance=atr_imbalance,
        atr_over_ema=atr_over_ema,
        day_ids=day_ids,
        bar_in_day=bar_in_day,
        session_end_mask=session_end_mask,
    )


def build_walk_forward_folds(
    preprocessed_arrays: PreprocessedArrays,
    train_window_bars: int,
    inference_window_bars: int,
    fold_stride_bars: int,
) -> list[Fold]:
    num_bars = preprocessed_arrays.num_bars
    full_window = train_window_bars + inference_window_bars
    assert num_bars >= full_window, "Not enough bars to create a single fold"

    folds: list[Fold] = []
    for fold_id, train_start in enumerate(range(0, num_bars - full_window + 1, fold_stride_bars)):
        train_end = train_start + train_window_bars
        inference_start = train_end
        inference_end = inference_start + inference_window_bars
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                inference_start=inference_start,
                inference_end=inference_end,
            )
        )

    return folds
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .contracts import BAR_COLUMNS, Fold, PreprocessedArrays, RAW_BAR_INPUT_COLUMNS
from data.database import Database

if TYPE_CHECKING:
    from .config import DataConfig, FeatureConfig


def _ema(values: np.ndarray, length: int) -> np.ndarray:
    return pd.Series(values, copy=False).ewm(span=length, adjust=False).mean().to_numpy(dtype=np.float64)


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values, copy=False).rolling(window=window, min_periods=1).sum().to_numpy(dtype=np.float64)


def _episode_start_indices(train_window_bars: int, episode_length: int, episode_stride: int) -> np.ndarray:
    train_window_bars = train_window_bars - 1 # The effective training window is one bar shorter than the nominal length. The last bar is used as the "next observation" for the final step of the final episode.
    max_start = train_window_bars - episode_length
    assert max_start >= 0, "Episode length is greater than training window."
    return np.arange(0, max_start + 1, episode_stride, dtype=np.int32)


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
    log_return = np.zeros_like(close)
    log_return[1:] = np.log(close[1:] / prev_close[1:])

    variance_proxy = np.square(log_return)
    ema_variance = _ema(variance_proxy, feature_config.variance_ema_length)
    ema_fast = _ema(close, feature_config.fast_ema_length)
    ema_slow = _ema(close, feature_config.slow_ema_length)

    true_range = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    atr = _ema(true_range, feature_config.atr_length)

    signed_variance = variance_proxy * np.sign(log_return)
    srvi_num = _rolling_sum(signed_variance, feature_config.srvi_length)
    srvi_den = _rolling_sum(variance_proxy, feature_config.srvi_length)
    srvi = srvi_num / np.maximum(srvi_den, feature_config.epsilon)
    sigma_price = close * (np.exp(np.sqrt(np.maximum(ema_variance, 0.0) + feature_config.epsilon)) - 1.0)

    vslope = np.zeros_like(close)
    vslope[1:] = (ema_slow[1:] - ema_slow[:-1]) / (atr[1:] + feature_config.epsilon)

    vmacd = (ema_fast - ema_slow) / (atr + feature_config.epsilon)
    vmacd_slope = np.zeros_like(close)
    vmacd_slope[1:] = vmacd[1:] - vmacd[:-1]

    static_features = np.column_stack(
        [
            raw_bars_df["tau"].to_numpy(dtype=np.float64),
            log_return,
            ema_variance,
            srvi,
            vslope,
            vmacd,
            vmacd_slope,
        ]
    )

    ohlc = np.column_stack([open_, high, low, close])

    static_features = np.ascontiguousarray(static_features.astype(np.float32))
    ohlc = np.ascontiguousarray(ohlc.astype(np.float32))
    atr = np.ascontiguousarray(atr.astype(np.float32))
    sigma_price = np.ascontiguousarray(sigma_price.astype(np.float32))
    day_ids = np.ascontiguousarray(raw_bars_df["day_id"].to_numpy(dtype=np.int32))
    bar_in_day = np.ascontiguousarray(raw_bars_df["bar_in_day"].to_numpy(dtype=np.int32))
    session_end_mask = np.ascontiguousarray(raw_bars_df["is_session_end"].to_numpy(dtype=bool))

    if not np.isfinite(static_features).all():
        raise ValueError("Feature preprocessing produced non-finite values")
    if not np.isfinite(ohlc).all():
        raise ValueError("OHLC preprocessing produced non-finite values")
    if not np.isfinite(atr).all():
        raise ValueError("ATR preprocessing produced non-finite values")
    if not np.isfinite(sigma_price).all():
        raise ValueError("Sigma-price preprocessing produced non-finite values")

    return PreprocessedArrays(
        ohlc=ohlc,
        static_features=static_features,
        atr=atr,
        sigma_price=sigma_price,
        day_ids=day_ids,
        bar_in_day=bar_in_day,
        session_end_mask=session_end_mask,
    )


def build_walk_forward_folds(
    preprocessed_arrays: PreprocessedArrays,
    train_window_bars: int,
    inference_window_bars: int,
    fold_stride_bars: int,
    episode_length: int,
    episode_stride: int,
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
                episode_start_indices=_episode_start_indices(
                    train_window_bars=train_window_bars,
                    episode_length=episode_length,
                    episode_stride=episode_stride,
                ),
            )
        )

    return folds
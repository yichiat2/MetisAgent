from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PACKAGE_NAME = "_test_ppo_vol_scalping"
PACKAGE_DIR = Path(__file__).resolve().parents[2] / "rl" / "ppo_vol_scalping"

package_module = ModuleType(PACKAGE_NAME)
package_module.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package_module)

contracts_module = _load_module(f"{PACKAGE_NAME}.contracts", PACKAGE_DIR / "contracts.py")
data_module = _load_module(f"{PACKAGE_NAME}.data", PACKAGE_DIR / "data.py")
BAR_COLUMNS = contracts_module.BAR_COLUMNS


def _make_database_bars() -> pd.DataFrame:
    day_one = pd.date_range("2024-01-02 09:30:00", periods=6, freq="5min")
    day_two = pd.date_range("2024-01-03 09:30:00", periods=6, freq="5min")
    timestamps = day_one.append(day_two)

    close = np.array(
        [100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 104.5, 105.0, 106.0, 105.5, 107.0, 108.0],
        dtype=np.float64,
    )
    open_ = close - 0.4
    high = close + 0.6
    low = close - 0.8
    volume = np.array([1000, 1025, 980, 1100, 1080, 1150, 1175, 1160, 1200, 1190, 1210, 1230], dtype=np.float64)
    average = (open_ + close) / 2.0
    trade_count = np.arange(10, 22, dtype=np.int64)

    return pd.DataFrame(
        {
            "date": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "average": average,
            "barCount": trade_count,
        }
    )


def _make_feature_config() -> SimpleNamespace:
    return SimpleNamespace(
        variance_ema_length=3,
        fast_ema_length=2,
        slow_ema_length=4,
        atr_length=3,
        srvi_length=3,
        epsilon=1e-8,
    )


def _load_bars(monkeypatch) -> pd.DataFrame:
    database_bars = _make_database_bars()
    load_calls: list[tuple[object, object, object]] = []

    def fake_load(self, root, start_date, end_date):
        load_calls.append((root, start_date, end_date))
        return database_bars

    monkeypatch.setattr(data_module.Database, "load", fake_load)

    data_config = SimpleNamespace(root="unit-test-root", start_date="2024-01-02", end_date="2024-01-03")
    loaded_bars = data_module.load_raw_bars_dataframe(data_config)
    assert load_calls == [("unit-test-root", "2024-01-02", "2024-01-03")]
    return loaded_bars


def test_load_raw_bars_dataframe_normalizes_schema_and_prints_top_10(monkeypatch) -> None:
    loaded_bars = _load_bars(monkeypatch)

    assert list(loaded_bars.columns) == list(BAR_COLUMNS)
    assert pd.api.types.is_datetime64_dtype(loaded_bars["timestamp"])
    assert loaded_bars["trade_count"].dtype == np.int64
    assert loaded_bars["day_id"].tolist() == [0] * 6 + [1] * 6
    assert loaded_bars["bar_in_day"].tolist() == list(range(6)) + list(range(6))
    assert loaded_bars["bars_in_day"].tolist() == [6] * 12
    np.testing.assert_allclose(
        loaded_bars["tau"].to_numpy(dtype=np.float32),
        np.tile(np.linspace(0.0, 1.0, 6, dtype=np.float32), 2),
    )
    assert loaded_bars["is_session_end"].tolist() == [False] * 5 + [True] + [False] * 5 + [True]

    print(loaded_bars.head(10).to_string(index=False))


def test_build_preprocessed_arrays_returns_expected_shapes_and_features(monkeypatch) -> None:
    loaded_bars = _load_bars(monkeypatch)

    preprocessed = data_module.build_preprocessed_arrays(loaded_bars, _make_feature_config())
    close = loaded_bars["close"].to_numpy(dtype=np.float64)
    expected_log_returns = np.zeros_like(close, dtype=np.float32)
    expected_log_returns[1:] = np.log(close[1:] / close[:-1]).astype(np.float32)

    assert preprocessed.ohlc.shape == (12, 4)
    assert preprocessed.static_features.shape == (12, 7)
    assert preprocessed.atr.shape == (12,)
    assert preprocessed.sigma_price.shape == (12,)
    assert preprocessed.day_ids.shape == (12,)
    assert preprocessed.bar_in_day.shape == (12,)
    assert preprocessed.session_end_mask.shape == (12,)

    assert preprocessed.ohlc.dtype == np.float32
    assert preprocessed.static_features.dtype == np.float32
    assert preprocessed.atr.dtype == np.float32
    assert preprocessed.sigma_price.dtype == np.float32
    assert preprocessed.day_ids.dtype == np.int32
    assert preprocessed.bar_in_day.dtype == np.int32
    assert preprocessed.session_end_mask.dtype == np.bool_

    np.testing.assert_allclose(
        preprocessed.ohlc[0],
        loaded_bars.loc[0, ["open", "high", "low", "close"]].to_numpy(dtype=np.float32),
    )
    np.testing.assert_allclose(preprocessed.static_features[:, 0], loaded_bars["tau"].to_numpy(dtype=np.float32))
    np.testing.assert_allclose(preprocessed.static_features[:, 1], expected_log_returns, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(preprocessed.day_ids, loaded_bars["day_id"].to_numpy(dtype=np.int32))
    np.testing.assert_array_equal(preprocessed.bar_in_day, loaded_bars["bar_in_day"].to_numpy(dtype=np.int32))
    np.testing.assert_array_equal(
        preprocessed.session_end_mask,
        loaded_bars["is_session_end"].to_numpy(dtype=bool),
    )
    assert np.isfinite(preprocessed.static_features).all()
    assert np.isfinite(preprocessed.ohlc).all()
    assert np.isfinite(preprocessed.atr).all()
    assert np.isfinite(preprocessed.sigma_price).all()


def test_build_walk_forward_folds_creates_expected_windows(monkeypatch) -> None:
    loaded_bars = _load_bars(monkeypatch)
    preprocessed = data_module.build_preprocessed_arrays(loaded_bars, _make_feature_config())

    folds = data_module.build_walk_forward_folds(
        preprocessed_arrays=preprocessed,
        train_window_bars=6,
        inference_window_bars=3,
        fold_stride_bars=2,
        episode_length=3,
        episode_stride=2,
    )

    assert len(folds) == 2

    first_fold, second_fold = folds
    assert (first_fold.fold_id, first_fold.train_start, first_fold.train_end, first_fold.inference_start, first_fold.inference_end) == (
        0,
        0,
        6,
        6,
        9,
    )
    assert (second_fold.fold_id, second_fold.train_start, second_fold.train_end, second_fold.inference_start, second_fold.inference_end) == (
        1,
        2,
        8,
        8,
        11,
    )
    np.testing.assert_array_equal(first_fold.episode_start_indices, np.array([0, 2], dtype=np.int32))
    np.testing.assert_array_equal(second_fold.episode_start_indices, np.array([0, 2], dtype=np.int32))


def test_build_walk_forward_folds_returns_empty_when_history_is_too_short(monkeypatch) -> None:
    loaded_bars = _load_bars(monkeypatch)
    preprocessed = data_module.build_preprocessed_arrays(loaded_bars, _make_feature_config())

    folds = data_module.build_walk_forward_folds(
        preprocessed_arrays=preprocessed,
        train_window_bars=10,
        inference_window_bars=4,
        fold_stride_bars=1,
        episode_length=3,
        episode_stride=1,
    )

    assert folds == []
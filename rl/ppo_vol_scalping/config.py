from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def make_grid_axis(start: float, stop: float, step: float) -> tuple[float, ...]:
    num_steps = int(round((stop - start) / step))
    return tuple(round(start + step * index, 10) for index in range(num_steps))


def make_grid_axis_inclusive(start: float, stop: float, step: float) -> tuple[float, ...]:
    num_steps = int(round((stop - start) / step)) + 1
    return tuple(round(start + step * index, 10) for index in range(num_steps))


@dataclass(frozen=True)
class DataConfig:
    root: str = "NVDA"
    start_date: int = 20200101
    end_date: int = 20251231
    train_window_bars: int = 80_000
    inference_window_bars: int = 40_000
    fold_stride_bars: int = 40_000


@dataclass(frozen=True)
class FeatureConfig:
    atr_over_ema_length: int = 10
    atr_length: int = 8
    directional_atr_ema_length: int = 5
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if self.atr_length <= 0:
            raise ValueError("Feature atr_length must be positive")
        if self.atr_over_ema_length <= 0:
            raise ValueError("Feature atr_over_ema_length must be positive")
        if self.directional_atr_ema_length <= 0:
            raise ValueError("Feature directional_atr_ema_length must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("Feature epsilon must be positive")


@dataclass(frozen=True)
class EnvironmentConfig:
    quote_size_shares: float = 100.0
    ibkr_monthly_volume_shares: float = 0.0
    ibkr_commission_min_per_order: float = 0.35
    ibkr_commission_max_trade_value_ratio: float = 0.01
    slippage_atr_multiple: float = 0.0
    spread_atr_multiple: float = 0.0
    max_entry_atr_over_ema: float = 0.005
    min_entry_bar_in_day: int = 5

    def __post_init__(self) -> None:
        if self.quote_size_shares <= 0.0:
            raise ValueError("Environment quote_size_shares must be positive")
        if self.ibkr_monthly_volume_shares < 0.0:
            raise ValueError("Environment ibkr_monthly_volume_shares must be non-negative")
        if self.ibkr_commission_min_per_order < 0.0:
            raise ValueError("Environment ibkr_commission_min_per_order must be non-negative")
        if not 0.0 <= self.ibkr_commission_max_trade_value_ratio <= 1.0:
            raise ValueError(
                "Environment ibkr_commission_max_trade_value_ratio must be between 0 and 1"
            )
        if self.slippage_atr_multiple < 0.0:
            raise ValueError("Environment slippage_atr_multiple must be non-negative")
        if self.spread_atr_multiple < 0.0:
            raise ValueError("Environment spread_atr_multiple must be non-negative")
        if self.max_entry_atr_over_ema < 0.0:
            raise ValueError("Environment max_entry_atr_over_ema must be non-negative")
        if self.min_entry_bar_in_day < 0:
            raise ValueError("Environment min_entry_bar_in_day must be non-negative")


@dataclass(frozen=True)
class SearchConfig:
    k_init_atr_multiples: tuple[float, ...] = field(
        default_factory=lambda: make_grid_axis_inclusive(0, 2, 0.1)
    )
    a_tp_atr_multiples: tuple[float, ...] = field(
        default_factory=lambda: make_grid_axis_inclusive(0, 2, 0.1)
    )
    k_act_atr_multiples: tuple[float, ...] = field(
        default_factory=lambda: make_grid_axis_inclusive(0., 0., 0.1)
    )
    entry_short_thresholds: tuple[float, ...] = field(
        default_factory=lambda: make_grid_axis_inclusive(0.5, 1.2, 0.05)
    )
    entry_long_thresholds: tuple[float, ...] = field(
        default_factory=lambda: make_grid_axis_inclusive(-1.2, -0.5, 0.05)
    )

    def __post_init__(self) -> None:
        if not self.k_init_atr_multiples:
            raise ValueError("Search k_init_atr_multiples cannot be empty")
        if not self.a_tp_atr_multiples:
            raise ValueError("Search a_tp_atr_multiples cannot be empty")
        if not self.k_act_atr_multiples:
            raise ValueError("Search k_act_atr_multiples cannot be empty")
        if not self.entry_short_thresholds:
            raise ValueError("Search entry_short_thresholds cannot be empty")
        if not self.entry_long_thresholds:
            raise ValueError("Search entry_long_thresholds cannot be empty")
        if any(value < 0.0 for value in self.k_init_atr_multiples):
            raise ValueError("Search k_init_atr_multiples must be non-negative")
        if any(value < 0.0 for value in self.a_tp_atr_multiples):
            raise ValueError("Search a_tp_atr_multiples must be non-negative")
        if any(value < 0.0 for value in self.k_act_atr_multiples):
            raise ValueError("Search k_act_atr_multiples must be non-negative")
        if max(self.entry_long_thresholds) >= min(self.entry_short_thresholds):
            raise ValueError("Search entry_long_thresholds must stay below entry_short_thresholds")


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: Path = field(default_factory=lambda: Path("logs") / "ppo_vol_scalping")
    json_path: str = "grid_search_summary.json"
    evaluation_annualization_factor: int = 252 * 390
    emit_debug_print: bool = True

    def __post_init__(self) -> None:
        if self.evaluation_annualization_factor <= 0:
            raise ValueError("Logging evaluation_annualization_factor must be positive")


@dataclass(frozen=True)
class PPOVolScalpingConfig:
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("Config seed must be non-negative")


def make_default_config() -> PPOVolScalpingConfig:
    return PPOVolScalpingConfig()
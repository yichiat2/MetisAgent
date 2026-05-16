from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    root: str = "NVDA"
    start_date: int = 20200101
    end_date: int = 20251231
    train_window_bars: int = 80_000
    inference_window_bars: int = 40_000
    fold_stride_bars: int = 10_000


@dataclass(frozen=True)
class FeatureConfig:
    fast_ema_length: int = 8
    slow_ema_length: int = 30
    atr_length: int = 8
    directional_atr_ema_length: int = 5
    srvi_length: int = 9
    epsilon: float = 1e-8


@dataclass(frozen=True)
class EnvironmentConfig:
    episode_length: int = 128
    episode_stride: int = 128
    max_inventory: int = 1000
    flatten_at_session_end: bool = True
    quote_size_shares: float = 1000.0
    ibkr_monthly_volume_shares: float = 0.0
    ibkr_commission_min_per_order: float = 0.35
    ibkr_commission_max_trade_value_ratio: float = 0.01
    action_low: tuple[float, ...] = (-3.0, 0.0)
    action_high: tuple[float, ...] = (3.0, 6.0)

    def __post_init__(self) -> None:
        if len(self.action_low) != len(self.action_high):
            raise ValueError("Environment action_low and action_high lengths must match")
        if any(high <= low for low, high in zip(self.action_low, self.action_high, strict=True)):
            raise ValueError("Each environment action_high entry must exceed the corresponding action_low entry")
        if self.ibkr_monthly_volume_shares < 0.0:
            raise ValueError("Environment ibkr_monthly_volume_shares cannot be negative")
        if self.ibkr_commission_min_per_order < 0.0:
            raise ValueError("Environment ibkr_commission_min_per_order cannot be negative")
        if self.ibkr_commission_max_trade_value_ratio < 0.0:
            raise ValueError("Environment ibkr_commission_max_trade_value_ratio cannot be negative")


@dataclass(frozen=True)
class RewardConfig:
    damped_pnl_eta: float = 0.25
    inventory_penalty_eta: float = 0.01
    reward_epsilon: float = 1e-8


@dataclass(frozen=True)
class PPOConfig:
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 1e-5
    min_learning_rate: float = 1e-6
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    minibatch_size: int = 128
    num_env: int = 128
    num_update: int = 100
    epochs: int = 10

    def __post_init__(self) -> None:
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("PPO learning rates must be positive")
        if self.min_learning_rate <= 0.0:
            raise ValueError("PPO min_learning_rate must be positive")
        if self.min_learning_rate > self.actor_learning_rate:
            raise ValueError("PPO min_learning_rate cannot exceed actor_learning_rate")
        if self.min_learning_rate > self.critic_learning_rate:
            raise ValueError("PPO min_learning_rate cannot exceed critic_learning_rate")


@dataclass(frozen=True)
class ModelConfig:
    hidden_sizes: tuple[int, ...] = (8, 8)
    action_dim: int = 2

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("Model action_dim must be positive")


@dataclass(frozen=True)
class CheckpointConfig:
    file_dir: Path = field(default_factory=lambda: Path("checkpoints") / "ppo_vol_scalping")
    file_name: str = "latest_fold.pkl"

    @property
    def file_path(self) -> Path:
        return self.file_dir / self.file_name


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: Path = field(default_factory=lambda: Path("logs") / "ppo_vol_scalping")
    json_path: str = "metrics.json"
    print_every_epochs: int = 1
    evaluation_annualization_factor: int = 252 * 390
    emit_debug_print: bool = True


@dataclass(frozen=True)
class PPOVolScalpingConfig:
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        if self.model.action_dim != len(self.environment.action_low):
            raise ValueError("Model action_dim must match the number of environment action bounds")


def make_default_config() -> PPOVolScalpingConfig:
    return PPOVolScalpingConfig()
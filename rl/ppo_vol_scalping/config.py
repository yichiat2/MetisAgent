from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import jax.numpy as jnp


@dataclass(frozen=True)
class DataConfig:
    root: str = "QQQ"
    start_date: int = 20200101
    end_date: int = 20251231
    train_window_bars: int = 80_000
    inference_window_bars: int = 20_000
    fold_stride_bars: int = 40_000


@dataclass(frozen=True)
class FeatureConfig:
    variance_ema_length: int = 9
    fast_ema_length: int = 8
    slow_ema_length: int = 30
    atr_length: int = 14
    srvi_length: int = 9
    epsilon: float = 1e-8


@dataclass(frozen=True)
class EnvironmentConfig:
    episode_length: int = 128
    episode_stride: int = 32
    max_inventory: int = 100
    max_quote_size: int = 100
    flatten_at_session_end: bool = True


@dataclass(frozen=True)
class RewardConfig:
    damped_pnl_eta: float = 0.5
    inventory_penalty_eta: float = 0.1
    reward_epsilon: float = 1e-8


@dataclass(frozen=True)
class PPOConfig:
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 1e-3
    actor_l1: float = 0.0
    critic_l1: float = 0.0
    minibatch_size: int = 64
    num_env: int = 32
    num_update: int = 10
    epochs: int = 10


@dataclass(frozen=True)
class ModelConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    action_dim: int = 3
    action_scale: jnp.ndarray = field(default_factory=lambda: jnp.array((2.0, 2.0, 1.0), dtype=jnp.float32))


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
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def make_default_config() -> PPOVolScalpingConfig:
    return PPOVolScalpingConfig()
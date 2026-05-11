from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    root: str = "NVDA"
    start_date: int = 20200101
    end_date: int = 20251231
    train_window_bars: int = 150_000
    inference_window_bars: int = 50_000
    fold_stride_bars: int = 50_000


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
    episode_length: int = 32
    episode_stride: int = 32
    max_inventory: int = 100
    flatten_at_session_end: bool = True
    price_action_levels: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
    quote_size_shares: float = 100.0


@dataclass(frozen=True)
class RewardConfig:
    damped_pnl_eta: float = 0.15
    inventory_penalty_eta: float = 0.0
    reward_epsilon: float = 1e-8


@dataclass(frozen=True)
class PPOConfig:
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 1e-5
    discount: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coefficient: float = 0.01
    actor_l1: float = 0.
    critic_l1: float = 0.
    minibatch_size: int = 64
    num_env: int = 32
    num_update: int = 20
    epochs: int = 5


@dataclass(frozen=True)
class ModelConfig:
    hidden_sizes: tuple[int, ...] = (8, 8)
    action_dim: int = 2
    action_cardinality: int = 4


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


def make_default_config() -> PPOVolScalpingConfig:
    return PPOVolScalpingConfig()
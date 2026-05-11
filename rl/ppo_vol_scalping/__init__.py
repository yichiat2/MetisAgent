"""Stepwise PPO volatility scalping package.

This package currently implements the Step 1-5 surface around the data,
model, and one-step environment functions used by the PPO volatility
scalping design.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ACTOR_STATE_DIM": ("contracts", "ACTOR_STATE_DIM"),
    "Actor": ("model", "Actor"),
    "BAR_COLUMNS": ("contracts", "BAR_COLUMNS"),
    "CRITIC_STATE_DIM": ("contracts", "CRITIC_STATE_DIM"),
    "Critic": ("model", "Critic"),
    "DataConfig": ("config", "DataConfig"),
    "EnvParam": ("env", "EnvParam"),
    "EnvState": ("env", "EnvState"),
    "EnvironmentConfig": ("config", "EnvironmentConfig"),
    "FeatureConfig": ("config", "FeatureConfig"),
    "Fold": ("contracts", "Fold"),
    "LoggingConfig": ("config", "LoggingConfig"),
    "ModelConfig": ("config", "ModelConfig"),
    "Observation": ("contracts", "Observation"),
    "PPOConfig": ("config", "PPOConfig"),
    "PPOVolScalpingConfig": ("config", "PPOVolScalpingConfig"),
    "PreprocessedArrays": ("contracts", "PreprocessedArrays"),
    "RewardConfig": ("config", "RewardConfig"),
    "STATE_DIM": ("contracts", "STATE_DIM"),
    "STATIC_FEATURE_DIM": ("contracts", "STATIC_FEATURE_DIM"),
    "build_critic_observation": ("env", "build_critic_observation"),
    "build_env_param": ("env", "build_env_param"),
    "build_observation": ("env", "build_observation"),
    "build_observations": ("env", "build_observations"),
    "build_preprocessed_arrays": ("data", "build_preprocessed_arrays"),
    "build_walk_forward_folds": ("data", "build_walk_forward_folds"),
    "create_train_states": ("model", "create_train_states"),
    "deterministic_action": ("model", "deterministic_action"),
    "env_reset": ("env", "env_reset"),
    "env_step": ("env", "env_step"),
    "get_entropy": ("model", "get_entropy"),
    "get_log_prob": ("model", "get_log_prob"),
    "load_raw_bars_dataframe": ("data", "load_raw_bars_dataframe"),
    "make_default_config": ("config", "make_default_config"),
    "sample_and_log_prob": ("model", "sample_and_log_prob"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "ACTOR_STATE_DIM",
    "BAR_COLUMNS",
    "Actor",
    "CRITIC_STATE_DIM",
    "Critic",
    "DataConfig",
    "EnvParam",
    "EnvState",
    "EnvironmentConfig",
    "FeatureConfig",
    "Fold",
    "LoggingConfig",
    "ModelConfig",
    "Observation",
    "PPOConfig",
    "PPOVolScalpingConfig",
    "PreprocessedArrays",
    "RewardConfig",
    "STATE_DIM",
    "STATIC_FEATURE_DIM",
    "build_critic_observation",
    "build_env_param",
    "build_preprocessed_arrays",
    "build_observation",
    "build_observations",
    "build_walk_forward_folds",
    "create_train_states",
    "deterministic_action",
    "env_reset",
    "env_step",
    "get_entropy",
    "get_log_prob",
    "load_raw_bars_dataframe",
    "make_default_config",
    "sample_and_log_prob",
]
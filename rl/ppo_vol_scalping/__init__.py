"""ATR-imbalance volatility scalping package.

This package runs deterministic walk-forward backtests with vmapped grid
search over trailing-stop and ATR-imbalance entry thresholds.
"""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "BAR_COLUMNS": ("contracts", "BAR_COLUMNS"),
    "DataConfig": ("config", "DataConfig"),
    "EnvParam": ("env", "EnvParam"),
    "EnvState": ("env", "EnvState"),
    "EnvironmentConfig": ("config", "EnvironmentConfig"),
    "FeatureConfig": ("config", "FeatureConfig"),
    "Fold": ("contracts", "Fold"),
    "LoggingConfig": ("config", "LoggingConfig"),
    "PPOVolScalpingConfig": ("config", "PPOVolScalpingConfig"),
    "PreprocessedArrays": ("contracts", "PreprocessedArrays"),
    "SearchConfig": ("config", "SearchConfig"),
    "STATIC_FEATURE_NAMES": ("contracts", "STATIC_FEATURE_NAMES"),
    "build_env_param": ("env", "build_env_param"),
    "build_preprocessed_arrays": ("data", "build_preprocessed_arrays"),
    "build_strategy_grid": ("driver", "build_strategy_grid"),
    "build_walk_forward_folds": ("data", "build_walk_forward_folds"),
    "env_reset": ("env", "env_reset"),
    "env_step": ("env", "env_step"),
    "load_raw_bars_dataframe": ("data", "load_raw_bars_dataframe"),
    "make_default_config": ("config", "make_default_config"),
    "run_grid_search": ("driver", "run_grid_search"),
    "run_strategy_summary": ("driver", "run_strategy_summary"),
    "run_strategy_trajectory": ("driver", "run_strategy_trajectory"),
    "summarize_episode": ("driver", "summarize_episode"),
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
    "BAR_COLUMNS",
    "DataConfig",
    "EnvParam",
    "EnvState",
    "EnvironmentConfig",
    "FeatureConfig",
    "Fold",
    "LoggingConfig",
    "PPOVolScalpingConfig",
    "PreprocessedArrays",
    "SearchConfig",
    "STATIC_FEATURE_NAMES",
    "build_env_param",
    "build_preprocessed_arrays",
    "build_strategy_grid",
    "build_walk_forward_folds",
    "env_reset",
    "env_step",
    "load_raw_bars_dataframe",
    "make_default_config",
    "run_grid_search",
    "run_strategy_summary",
    "run_strategy_trajectory",
    "summarize_episode",
]
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from .config import PPOVolScalpingConfig, make_default_config
from .contracts import Fold, PreprocessedArrays, StrategyGrid
from .data import build_preprocessed_arrays, build_walk_forward_folds, load_raw_bars_dataframe
from .env import build_env_param, env_reset, env_step
from .visual import prepare_inference_payload, push_inference_metrics, start_dashboard_server


@struct.dataclass
class SummaryAccumulator:
    initial_portfolio_value: jnp.ndarray
    last_portfolio_value: jnp.ndarray
    cumulative_pnl: jnp.ndarray
    peak_portfolio_value: jnp.ndarray
    max_drawdown: jnp.ndarray
    return_sum: jnp.ndarray
    return_sq_sum: jnp.ndarray
    downside_sq_sum: jnp.ndarray


def build_strategy_grid(config: PPOVolScalpingConfig) -> StrategyGrid:
    k_init = jnp.asarray(config.search.k_init_atr_multiples, dtype=jnp.float32)
    a_tp = jnp.asarray(config.search.a_tp_atr_multiples, dtype=jnp.float32)
    k_act = jnp.asarray(config.search.k_act_atr_multiples, dtype=jnp.float32)
    entry_short = jnp.asarray(config.search.entry_short_thresholds, dtype=jnp.float32)
    entry_long = jnp.asarray(config.search.entry_long_thresholds, dtype=jnp.float32)
    k_init_mesh, a_tp_mesh, k_act_mesh, short_mesh, long_mesh = jnp.meshgrid(
        k_init,
        a_tp,
        k_act,
        entry_short,
        entry_long,
        indexing="ij",
    )
    return StrategyGrid(
        k_init=k_init_mesh.reshape(-1),
        a_tp=a_tp_mesh.reshape(-1),
        k_act=k_act_mesh.reshape(-1),
        entry_short=short_mesh.reshape(-1),
        entry_long=long_mesh.reshape(-1),
    )


def summarize_episode(
    initial_portfolio_value: jax.Array,
    annualization_factor: jax.Array,
    step_info: Mapping[str, jax.Array],
    k_init: jax.Array,
    a_tp: jax.Array,
    k_act: jax.Array,
    entry_short: jax.Array,
    entry_long: jax.Array,
) -> dict[str, jax.Array]:
    epsilon = jnp.asarray(1e-8, dtype=jnp.float32)
    annualization = jnp.sqrt(jnp.asarray(annualization_factor, dtype=jnp.float32))
    step_returns = step_info["return"]
    finite_returns = jnp.where(jnp.isfinite(step_returns), step_returns, 0.0)
    num_steps = jnp.asarray(step_returns.shape[0], dtype=jnp.float32)
    mean_return = jnp.sum(finite_returns) / num_steps
    return_sq_mean = jnp.sum(jnp.square(finite_returns)) / num_steps
    return_std = jnp.sqrt(jnp.maximum(return_sq_mean - jnp.square(mean_return), 0.0))
    downside_deviation = jnp.sqrt(
        jnp.sum(jnp.square(jnp.minimum(finite_returns, 0.0))) / num_steps
    )
    cumulative_pnl = jnp.cumsum(step_info["pnl"])
    cumulative_return = step_info["portfolio_value_after"] / initial_portfolio_value - 1.0
    portfolio_path = jnp.concatenate(
        [jnp.asarray([initial_portfolio_value], dtype=jnp.float32), step_info["portfolio_value_after"]]
    )
    running_peak = jax.lax.associative_scan(jnp.maximum, portfolio_path)
    drawdown = 1.0 - portfolio_path / jnp.maximum(running_peak, epsilon)
    bankruptcy = jnp.any(step_info["is_bankrupt"] > 0.0)
    sharpe_ratio = jnp.where(
        bankruptcy,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        annualization * mean_return / (return_std + epsilon),
    )
    sortino_ratio = jnp.where(
        bankruptcy,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        annualization * mean_return / (downside_deviation + epsilon),
    )
    trade_count = step_info["trade_count"][-1]
    winning_trade_count = step_info["winning_trade_count"][-1]
    realized_trade_pnl_sum = jnp.sum(step_info["realized_trade_pnl"])
    win_rate = jnp.where(
        trade_count > 0,
        winning_trade_count.astype(jnp.float32) / trade_count.astype(jnp.float32),
        0.0,
    )
    expectation_per_trade = jnp.where(
        trade_count > 0,
        realized_trade_pnl_sum / trade_count.astype(jnp.float32),
        0.0,
    )
    return {
        "bankruptcy": bankruptcy,
        "cumulative_pnl": cumulative_pnl,
        "cumulative_return": cumulative_return,
        "expectation_per_trade": expectation_per_trade,
        "final_cumulative_return": cumulative_return[-1],
        "max_drawdown": jnp.max(drawdown),
        "sharpe_ratio": sharpe_ratio,
        "sortino_objective": jnp.where(bankruptcy, -jnp.inf, sortino_ratio),
        "sortino_ratio": sortino_ratio,
        "total_pnl": cumulative_pnl[-1],
        "trade_count": trade_count,
        "win_rate": win_rate,
        "winning_trade_count": winning_trade_count,
        "k_init": jnp.asarray(k_init, dtype=jnp.float32),
        "a_tp": jnp.asarray(a_tp, dtype=jnp.float32),
        "k_act": jnp.asarray(k_act, dtype=jnp.float32),
        "entry_short": jnp.asarray(entry_short, dtype=jnp.float32),
        "entry_long": jnp.asarray(entry_long, dtype=jnp.float32),
    }


@jax.jit
def run_strategy_trajectory(
    env_param,
    annualization_factor: jax.Array,
    k_init: jax.Array,
    a_tp: jax.Array,
    k_act: jax.Array,
    entry_short: jax.Array,
    entry_long: jax.Array,
):
    initial_state = env_reset(
        env_param,
        k_init=k_init,
        a_tp=a_tp,
        k_act=k_act,
        entry_short=entry_short,
        entry_long=entry_long,
    )
    initial_portfolio_value = initial_state.cash

    def scan_step(state, _):
        next_state, info = env_step(state, env_param)
        return next_state, info

    _, step_info = jax.lax.scan(
        scan_step,
        initial_state,
        None,
        length=env_param.ohlc.shape[0] - 1,
    )
    summary_metrics = summarize_episode(
        initial_portfolio_value=initial_portfolio_value,
        annualization_factor=annualization_factor,
        step_info=step_info,
        k_init=k_init,
        a_tp=a_tp,
        k_act=k_act,
        entry_short=entry_short,
        entry_long=entry_long,
    )
    return {
        **step_info,
        **summary_metrics,
    }


@jax.jit
def run_strategy_summary(
    env_param,
    annualization_factor: jax.Array,
    k_init: jax.Array,
    a_tp: jax.Array,
    k_act: jax.Array,
    entry_short: jax.Array,
    entry_long: jax.Array,
):
    initial_state = env_reset(
        env_param,
        k_init=k_init,
        a_tp=a_tp,
        k_act=k_act,
        entry_short=entry_short,
        entry_long=entry_long,
    )
    initial_portfolio_value = initial_state.cash
    accumulator = SummaryAccumulator(
        initial_portfolio_value=initial_portfolio_value,
        last_portfolio_value=initial_portfolio_value,
        cumulative_pnl=jnp.asarray(0.0, dtype=jnp.float32),
        peak_portfolio_value=initial_portfolio_value,
        max_drawdown=jnp.asarray(0.0, dtype=jnp.float32),
        return_sum=jnp.asarray(0.0, dtype=jnp.float32),
        return_sq_sum=jnp.asarray(0.0, dtype=jnp.float32),
        downside_sq_sum=jnp.asarray(0.0, dtype=jnp.float32),
    )
    epsilon = jnp.asarray(1e-8, dtype=jnp.float32)

    def scan_step(carry, _):
        state, summary = carry
        next_state, info = env_step(state, env_param)
        finite_return = jnp.where(jnp.isfinite(info["return"]), info["return"], 0.0)
        next_peak = jnp.maximum(summary.peak_portfolio_value, info["portfolio_value_after"])
        next_drawdown = 1.0 - info["portfolio_value_after"] / jnp.maximum(next_peak, epsilon)
        next_summary = SummaryAccumulator(
            initial_portfolio_value=summary.initial_portfolio_value,
            last_portfolio_value=info["portfolio_value_after"],
            cumulative_pnl=summary.cumulative_pnl + info["pnl"],
            peak_portfolio_value=next_peak,
            max_drawdown=jnp.maximum(summary.max_drawdown, next_drawdown),
            return_sum=summary.return_sum + finite_return,
            return_sq_sum=summary.return_sq_sum + jnp.square(finite_return),
            downside_sq_sum=summary.downside_sq_sum + jnp.square(jnp.minimum(finite_return, 0.0)),
        )
        return (next_state, next_summary), None

    (final_state, final_summary), _ = jax.lax.scan(
        scan_step,
        (initial_state, accumulator),
        None,
        length=env_param.ohlc.shape[0] - 1,
    )
    num_steps = jnp.asarray(env_param.ohlc.shape[0] - 1, dtype=jnp.float32)
    mean_return = final_summary.return_sum / num_steps
    return_sq_mean = final_summary.return_sq_sum / num_steps
    return_std = jnp.sqrt(jnp.maximum(return_sq_mean - jnp.square(mean_return), 0.0))
    downside_deviation = jnp.sqrt(final_summary.downside_sq_sum / num_steps)
    annualization = jnp.sqrt(jnp.asarray(annualization_factor, dtype=jnp.float32))
    bankruptcy = final_state.is_bankrupt > 0.0
    trade_count = final_state.trade_count
    expectation_per_trade = jnp.where(
        trade_count > 0,
        final_state.realized_trade_pnl_sum / trade_count.astype(jnp.float32),
        0.0,
    )
    return {
        "bankruptcy": bankruptcy,
        "expectation_per_trade": expectation_per_trade,
        "final_cumulative_return": final_summary.last_portfolio_value / final_summary.initial_portfolio_value - 1.0,
        "max_drawdown": final_summary.max_drawdown,
        "sharpe_ratio": jnp.where(
            bankruptcy,
            jnp.asarray(jnp.nan, dtype=jnp.float32),
            annualization * mean_return / (return_std + 1e-8),
        ),
        "sortino_objective": jnp.where(
            bankruptcy,
            -jnp.inf,
            annualization * mean_return / (downside_deviation + 1e-8),
        ),
        "sortino_ratio": jnp.where(
            bankruptcy,
            jnp.asarray(jnp.nan, dtype=jnp.float32),
            annualization * mean_return / (downside_deviation + 1e-8),
        ),
        "total_pnl": final_summary.cumulative_pnl,
        "trade_count": trade_count,
        "win_rate": jnp.where(
            trade_count > 0,
            final_state.winning_trade_count.astype(jnp.float32) / trade_count.astype(jnp.float32),
            0.0,
        ),
        "winning_trade_count": final_state.winning_trade_count,
        "k_init": jnp.asarray(k_init, dtype=jnp.float32),
        "a_tp": jnp.asarray(a_tp, dtype=jnp.float32),
        "k_act": jnp.asarray(k_act, dtype=jnp.float32),
        "entry_short": jnp.asarray(entry_short, dtype=jnp.float32),
        "entry_long": jnp.asarray(entry_long, dtype=jnp.float32),
    }


@jax.jit
def run_grid_search(
    env_param,
    strategy_grid: StrategyGrid,
    annualization_factor: jax.Array,
):
    return jax.vmap(
        lambda k_init, a_tp, k_act, entry_short, entry_long: run_strategy_summary(
            env_param,
            annualization_factor,
            k_init,
            a_tp,
            k_act,
            entry_short,
            entry_long,
        )
    )(
        strategy_grid.k_init,
        strategy_grid.a_tp,
        strategy_grid.k_act,
        strategy_grid.entry_short,
        strategy_grid.entry_long,
    )


def _build_metric_candidate(
    grid_metrics: Mapping[str, np.ndarray],
    best_index: int,
) -> dict[str, Any]:
    return {
        "index": best_index,
        "k_init": float(np.asarray(grid_metrics["k_init"])[best_index]),
        "a_tp": float(np.asarray(grid_metrics["a_tp"])[best_index]),
        "k_act": float(np.asarray(grid_metrics["k_act"])[best_index]),
        "entry_short": float(np.asarray(grid_metrics["entry_short"])[best_index]),
        "entry_long": float(np.asarray(grid_metrics["entry_long"])[best_index]),
        "sharpe_ratio": float(np.asarray(grid_metrics["sharpe_ratio"])[best_index]),
        "sortino_ratio": float(np.asarray(grid_metrics["sortino_ratio"])[best_index]),
        "final_cumulative_return": float(np.asarray(grid_metrics["final_cumulative_return"])[best_index]),
        "max_drawdown": float(np.asarray(grid_metrics["max_drawdown"])[best_index]),
        "total_pnl": float(np.asarray(grid_metrics["total_pnl"])[best_index]),
        "trade_count": int(np.asarray(grid_metrics["trade_count"])[best_index]),
        "win_rate": float(np.asarray(grid_metrics["win_rate"])[best_index]),
        "expectation_per_trade": float(np.asarray(grid_metrics["expectation_per_trade"])[best_index]),
        "bankruptcy": bool(np.asarray(grid_metrics["bankruptcy"])[best_index]),
    }


def select_metric_candidate(
    grid_metrics: Mapping[str, np.ndarray],
    metric_name: str,
    maximize: bool,
) -> dict[str, Any]:
    metric_values = np.asarray(grid_metrics[metric_name], dtype=np.float64)
    fill_value = -np.inf if maximize else np.inf
    metric_values = np.where(np.isfinite(metric_values), metric_values, fill_value)
    best_index = int(np.argmax(metric_values) if maximize else np.argmin(metric_values))
    return _build_metric_candidate(grid_metrics, best_index)


def select_expectation_candidate(
    grid_metrics: Mapping[str, np.ndarray],
    min_trade_count: int,
) -> dict[str, Any]:
    if min_trade_count < 0:
        raise ValueError("min_trade_count must be non-negative")

    trade_counts = np.asarray(grid_metrics["trade_count"], dtype=np.int64)
    expectation_values = np.asarray(grid_metrics["expectation_per_trade"], dtype=np.float64)
    eligible_mask = trade_counts >= min_trade_count
    finite_eligible_mask = eligible_mask & np.isfinite(expectation_values)
    if not np.any(finite_eligible_mask):
        raise ValueError(
            "No finite expectation_per_trade candidate satisfies min_trade_count"
        )

    constrained_values = np.where(finite_eligible_mask, expectation_values, -np.inf)
    best_index = int(np.argmax(constrained_values))
    return _build_metric_candidate(grid_metrics, best_index)


def write_summary_file(config: PPOVolScalpingConfig, fold_summaries: list[dict[str, Any]]) -> None:
    log_dir = config.logging.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = log_dir / config.logging.json_path
    summary_path.write_text(json.dumps(fold_summaries, indent=2), encoding="utf-8")
    print(f"Wrote grid-search summary to {summary_path}")


def train(
    folds: list[Fold],
    preprocessed_arrays: PreprocessedArrays,
    config: PPOVolScalpingConfig,
) -> list[dict[str, Any]]:
    annualization_factor = jnp.asarray(config.logging.evaluation_annualization_factor, dtype=jnp.float32)
    strategy_grid = build_strategy_grid(config)
    fold_summaries: list[dict[str, Any]] = []
    dashboard_ready = False

    for fold in folds:
        train_arrays = preprocessed_arrays[fold.train_start:fold.train_end]
        infer_arrays = preprocessed_arrays[fold.inference_start:fold.inference_end]
        train_env_param = build_env_param(config, train_arrays)
        infer_env_param = build_env_param(config, infer_arrays)

        train_grid_metrics = jax.device_get(
            run_grid_search(train_env_param, strategy_grid, annualization_factor)
        )
        best_sharpe = select_metric_candidate(train_grid_metrics, "sharpe_ratio", maximize=True)
        best_sortino = select_metric_candidate(train_grid_metrics, "sortino_objective", maximize=True)
        best_expectation = select_expectation_candidate(
            train_grid_metrics,
            min_trade_count=500,
        )
        best_cumulative_return = select_metric_candidate(
            train_grid_metrics,
            "final_cumulative_return",
            maximize=True,
        )
        lowest_max_drawdown = select_metric_candidate(train_grid_metrics, "max_drawdown", maximize=False)
        inference_metrics = jax.device_get(
            run_strategy_trajectory(
                infer_env_param,
                annualization_factor,
                k_init=jnp.asarray(best_sharpe["k_init"], dtype=jnp.float32),
                a_tp=jnp.asarray(best_sharpe["a_tp"], dtype=jnp.float32),
                k_act=jnp.asarray(best_sharpe["k_act"], dtype=jnp.float32),
                entry_short=jnp.asarray(best_sharpe["entry_short"], dtype=jnp.float32),
                entry_long=jnp.asarray(best_sharpe["entry_long"], dtype=jnp.float32),
            )
        )

        if not dashboard_ready:
            try:
                start_dashboard_server()
                dashboard_ready = True
            except ImportError as exc:
                print(f"[Dashboard] {exc}")
        if dashboard_ready:
            push_inference_metrics(
                prepare_inference_payload(
                    metrics=inference_metrics,
                    ohlc=infer_arrays.ohlc,
                    atr=infer_arrays.atr,
                    atr_up=infer_arrays.atr_up,
                    atr_down=infer_arrays.atr_down,
                    atr_imbalance=infer_arrays.atr_imbalance,
                    fold_id=fold.fold_id,
                    dataset_label="inference",
                )
            )

        fold_summary = {
            "fold_id": fold.fold_id,
            "train_window": {
                "start": fold.train_start,
                "end": fold.train_end,
                "num_bars": fold.train_length,
            },
            "inference_window": {
                "start": fold.inference_start,
                "end": fold.inference_end,
                "num_bars": fold.inference_length,
            },
            "best_train_sharpe": best_sharpe,
            "best_train_sortino": best_sortino,
            "best_train_expectation": best_expectation,
            "best_train_cumulative_return": best_cumulative_return,
            "lowest_train_max_drawdown": lowest_max_drawdown,
            "inference": {
                "k_init": float(np.asarray(inference_metrics["k_init"])),
                "a_tp": float(np.asarray(inference_metrics["a_tp"])),
                "k_act": float(np.asarray(inference_metrics["k_act"])),
                "entry_short": float(np.asarray(inference_metrics["entry_short"])),
                "entry_long": float(np.asarray(inference_metrics["entry_long"])),
                "sharpe_ratio": float(np.asarray(inference_metrics["sharpe_ratio"])),
                "sortino_ratio": float(np.asarray(inference_metrics["sortino_ratio"])),
                "final_cumulative_return": float(np.asarray(inference_metrics["final_cumulative_return"])),
                "max_drawdown": float(np.asarray(inference_metrics["max_drawdown"])),
                "total_pnl": float(np.asarray(inference_metrics["total_pnl"])),
                "trade_count": int(np.asarray(inference_metrics["trade_count"])),
                "win_rate": float(np.asarray(inference_metrics["win_rate"])),
                "expectation_per_trade": float(np.asarray(inference_metrics["expectation_per_trade"])),
                "bankruptcy": bool(np.asarray(inference_metrics["bankruptcy"])),
            },
        }
        fold_summaries.append(fold_summary)

        if config.logging.emit_debug_print:
            print(
                f"Fold {fold.fold_id}: best train sortino k_init={best_sortino['k_init']:.2f}, "
                f"a_tp={best_sortino['a_tp']:.2f}, k_act={best_sortino['k_act']:.2f}, "
                f"entry_short={best_sortino['entry_short']:.2f}, entry_long={best_sortino['entry_long']:.2f}, "
                f"sortino={best_sortino['sortino_ratio']:.6f}, sharpe={best_sortino['sharpe_ratio']:.6f}, "
                f"cumret={best_sortino['final_cumulative_return']:.6f}, maxdd={best_sortino['max_drawdown']:.6f}."
            )
            print(
                f"Fold {fold.fold_id}: inference gives "
                f"sortino={fold_summary['inference']['sortino_ratio']:.6f}, "
                f"sharpe={fold_summary['inference']['sharpe_ratio']:.6f}, "
                f"cumret={fold_summary['inference']['final_cumulative_return']:.6f}, "
                f"maxdd={fold_summary['inference']['max_drawdown']:.6f}, "
                f"trades={fold_summary['inference']['trade_count']}."
            )
        input("Press Enter to continue to the next fold...")
    write_summary_file(config, fold_summaries)
    return fold_summaries


def main() -> None:
    config = make_default_config()
    raw_bars_df = load_raw_bars_dataframe(config.data)
    preprocessed_arrays = build_preprocessed_arrays(raw_bars_df, config.features)
    folds = build_walk_forward_folds(
        preprocessed_arrays=preprocessed_arrays,
        train_window_bars=config.data.train_window_bars,
        inference_window_bars=config.data.inference_window_bars,
        fold_stride_bars=config.data.fold_stride_bars,
    )
    train(folds=folds, preprocessed_arrays=preprocessed_arrays, config=config)


if __name__ == "__main__":
    main()
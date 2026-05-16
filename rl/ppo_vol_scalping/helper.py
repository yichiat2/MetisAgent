from __future__ import annotations

import pickle

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from .config import PPOVolScalpingConfig


def _summarize_inference(
    config: PPOVolScalpingConfig,
    step_pnl: jax.Array,
    step_rewards: jax.Array,
    step_returns: jax.Array,
    is_bankrupt: jax.Array,
    bid_fill: jax.Array,
    ask_fill: jax.Array,
) -> dict[str, jax.Array]:
    epsilon = jnp.asarray(config.reward.reward_epsilon, dtype=jnp.float32)
    annualization = jnp.sqrt(
        jnp.asarray(config.logging.evaluation_annualization_factor, dtype=jnp.float32)
    )
    cumulative_pnl = jnp.cumsum(step_pnl)
    total_reward = jnp.sum(step_rewards)
    num_steps = jnp.asarray(step_rewards.shape[0], dtype=jnp.float32)
    normalized_episode_reward = total_reward * config.environment.episode_length / num_steps
    bankruptcy_mask = is_bankrupt > 0.0
    path_returns = jnp.where(bankruptcy_mask, -1.0, step_returns)
    wealth_curve = jnp.concatenate(
        [jnp.ones((1,), dtype=step_returns.dtype), jnp.cumprod(1.0 + path_returns)]
    )
    cumulative_return = wealth_curve[1:] - 1.0
    running_peak = jax.lax.associative_scan(jnp.maximum, wealth_curve)
    drawdown = 1.0 - wealth_curve / (running_peak + epsilon)
    mean_return = jnp.mean(step_returns)
    return_std = jnp.std(step_returns)
    downside_deviation = jnp.sqrt(jnp.mean(jnp.square(jnp.minimum(step_returns, 0.0))))
    bankrupt_any = jnp.any(bankruptcy_mask)
    sharpe_ratio = jnp.where(
        bankrupt_any,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        annualization * mean_return / (return_std + epsilon),
    )
    sortino_ratio = jnp.where(
        bankrupt_any,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        annualization * mean_return / (downside_deviation + epsilon),
    )
    max_drawdown = jnp.max(drawdown)
    final_cumulative_return = cumulative_return[-1]
    bid_counts = jnp.asarray(bid_fill, dtype=jnp.int32)
    ask_counts = jnp.asarray(ask_fill, dtype=jnp.int32)
    transaction_count = jnp.sum(bid_counts + ask_counts, dtype=jnp.int32)
    return {
        "cumulative_pnl": cumulative_pnl,
        "normalized_episode_reward": normalized_episode_reward,
        "total_pnl": cumulative_pnl[-1],
        "total_reward": total_reward,
        "cumulative_return": cumulative_return,
        "final_cumulative_return": final_cumulative_return,
        "max_drawdown": max_drawdown,
        "bankruptcy": bankrupt_any,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "transaction_count": transaction_count,
    }


def _compute_episode_max_drawdown(
    portfolio_value_before: jax.Array,
    portfolio_value_after: jax.Array,
    epsilon: float,
) -> jax.Array:
    starting_portfolio_value = portfolio_value_before[:1]
    portfolio_path = jnp.concatenate([starting_portfolio_value, portfolio_value_after], axis=0)
    running_peak = jax.lax.associative_scan(jnp.maximum, portfolio_path)
    drawdown = 1.0 - portfolio_path / (running_peak + epsilon)
    return jnp.max(drawdown, axis=0)


def _reset_optimizer_state(train_state: TrainState) -> TrainState:
    step_dtype = jnp.asarray(train_state.step).dtype
    return train_state.replace(
        step=jnp.asarray(0, dtype=step_dtype),
        opt_state=train_state.tx.init(train_state.params),
    )


def _save_fold_checkpoint(
    config: PPOVolScalpingConfig,
    actor_state: TrainState,
    critic_state: TrainState,
    rng: jax.Array,
    completed_fold_id: int,
    next_fold_index: int,
) -> None:
    checkpoint_path = config.checkpoint.file_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "actor_opt_state": jax.device_get(actor_state.opt_state),
        "actor_params": jax.device_get(actor_state.params),
        "actor_step": int(jax.device_get(actor_state.step)),
        "completed_fold_id": completed_fold_id,
        "critic_opt_state": jax.device_get(critic_state.opt_state),
        "critic_params": jax.device_get(critic_state.params),
        "critic_step": int(jax.device_get(critic_state.step)),
        "next_fold_index": next_fold_index,
        "rng": jax.device_get(rng),
    }
    with checkpoint_path.open("wb") as checkpoint_file:
        pickle.dump(checkpoint_payload, checkpoint_file)
    print(f"Saved checkpoint to {checkpoint_path}")


def _load_fold_checkpoint(
    config: PPOVolScalpingConfig,
    actor_state: TrainState,
    critic_state: TrainState,
    rng: jax.Array,
) -> tuple[TrainState, TrainState, jax.Array, int, int | None]:
    checkpoint_path = config.checkpoint.file_path
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}; starting from fold 0")
        return actor_state, critic_state, rng, 0, None

    with checkpoint_path.open("rb") as checkpoint_file:
        checkpoint_payload = pickle.load(checkpoint_file)

    actor_step_dtype = jnp.asarray(actor_state.step).dtype
    critic_step_dtype = jnp.asarray(critic_state.step).dtype
    actor_state = actor_state.replace(
        step=jnp.asarray(checkpoint_payload["actor_step"], dtype=actor_step_dtype),
        params=jax.tree_util.tree_map(jnp.asarray, checkpoint_payload["actor_params"]),
        opt_state=jax.tree_util.tree_map(jnp.asarray, checkpoint_payload["actor_opt_state"]),
    )
    critic_state = critic_state.replace(
        step=jnp.asarray(checkpoint_payload["critic_step"], dtype=critic_step_dtype),
        params=jax.tree_util.tree_map(jnp.asarray, checkpoint_payload["critic_params"]),
        opt_state=jax.tree_util.tree_map(jnp.asarray, checkpoint_payload["critic_opt_state"]),
    )
    rng = jnp.asarray(checkpoint_payload["rng"])
    next_fold_index = int(checkpoint_payload.get("next_fold_index", 0))
    completed_fold_id = checkpoint_payload.get("completed_fold_id")
    print(f"Loaded checkpoint from {checkpoint_path}")
    return actor_state, critic_state, rng, next_fold_index, completed_fold_id
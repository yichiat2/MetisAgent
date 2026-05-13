from __future__ import annotations

from flax import struct
import jax.numpy as jnp
from .config import PPOVolScalpingConfig, RewardConfig
from .contracts import Observation, PreprocessedArrays

OPEN_INDEX = 0
HIGH_INDEX = 1
LOW_INDEX = 2
CLOSE_INDEX = 3


@struct.dataclass
class EnvState:
    step_index: jnp.ndarray
    global_index: jnp.ndarray
    inventory: jnp.ndarray
    cash: jnp.ndarray
    return_baseline: jnp.ndarray
    is_bankrupt: jnp.ndarray


@struct.dataclass
class EnvParam:
    episode_length: jnp.ndarray
    max_inventory: float
    flatten_at_session_end: bool
    quote_size_shares: jnp.ndarray
    action_low: jnp.ndarray
    action_high: jnp.ndarray
    ohlc: jnp.ndarray
    static_features: jnp.ndarray
    atr: jnp.ndarray
    day_ids: jnp.ndarray
    bar_in_day: jnp.ndarray
    session_end_mask: jnp.ndarray
    


def build_env_param(config: PPOVolScalpingConfig, train_arrays: PreprocessedArrays) -> EnvParam:
    return EnvParam(
        episode_length=jnp.asarray(config.environment.episode_length, dtype=jnp.float32),
        max_inventory=float(config.environment.max_inventory),
        flatten_at_session_end=config.environment.flatten_at_session_end,
        quote_size_shares=jnp.asarray(config.environment.quote_size_shares, dtype=jnp.float32),
        action_low=jnp.asarray(config.environment.action_low, dtype=jnp.float32),
        action_high=jnp.asarray(config.environment.action_high, dtype=jnp.float32),
        ohlc=jnp.asarray(train_arrays.ohlc),
        static_features=jnp.asarray(train_arrays.static_features),
        atr=jnp.asarray(train_arrays.atr),
        day_ids=jnp.asarray(train_arrays.day_ids),
        bar_in_day=jnp.asarray(train_arrays.bar_in_day),
        session_end_mask=jnp.asarray(train_arrays.session_end_mask),
    )


def _scale_action(action: jnp.ndarray, env_param: EnvParam) -> tuple[jnp.ndarray, jnp.ndarray]:
    normalized_action = jnp.clip(jnp.asarray(action, dtype=jnp.float32), 0.0, 1.0)
    action_scale = env_param.action_high - env_param.action_low
    scaled_action = env_param.action_low + action_scale * normalized_action
    return normalized_action, scaled_action

def build_observation(static_features: jnp.ndarray, inventory: float, max_inventory: float) -> jnp.ndarray:
    inventory_feature = inventory / max_inventory
    return jnp.concatenate([static_features, jnp.asarray(inventory_feature, dtype=jnp.float32)[None]], axis=-1)


def build_critic_observation(
    actor_observation: jnp.ndarray,
    step_index: jnp.ndarray | float,
    episode_length: jnp.ndarray,
) -> jnp.ndarray:
    remaining_steps = jnp.maximum(
        episode_length - jnp.asarray(step_index, dtype=jnp.float32),
        0.0,
    )
    remaining_fraction = remaining_steps / episode_length
    return jnp.concatenate(
        [actor_observation, jnp.asarray(remaining_fraction, dtype=jnp.float32)[None]],
        axis=-1,
    )


def build_observations(
    static_features: jnp.ndarray,
    inventory: float,
    max_inventory: float,
    step_index: jnp.ndarray | float,
    episode_length: jnp.ndarray,
) -> Observation:
    actor_observation = build_observation(static_features, inventory, max_inventory)
    critic_observation = build_critic_observation(
        actor_observation,
        step_index,
        episode_length,
    )
    return Observation(actor=actor_observation, critic=critic_observation)


def env_reset(
    env_param: EnvParam,
    global_index: int,
    initial_inventory: float = 0.0,
    initial_cash: float = 0.0,
) -> tuple[Observation, EnvState]:
    first_close = env_param.ohlc[global_index, CLOSE_INDEX]
    return_baseline = env_param.quote_size_shares * first_close
    state = EnvState(
        step_index=0,
        global_index=global_index,
        inventory=initial_inventory,
        cash=initial_cash,
        return_baseline=return_baseline,
        is_bankrupt=jnp.asarray(0.0, dtype=jnp.float32),
    )
    static_features_0 = env_param.static_features[global_index]
    observation = build_observations(
        static_features_0,
        state.inventory,
        env_param.max_inventory,
        state.step_index,
        env_param.episode_length,
    )
    return observation, state


def env_step(
    state: EnvState,
    action: jnp.ndarray,
    env_param: EnvParam,
    reward_config: RewardConfig,
) -> tuple[Observation, EnvState, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    # step_index tracks rollout progress; global_index selects the fold-local arrays.
    next_step_index = state.step_index + 1
    next_global_index = state.global_index + 1
    session_end = env_param.session_end_mask[state.global_index]
    should_flatten = env_param.flatten_at_session_end & session_end

    valid_transition = jnp.logical_not(session_end)
    inventory_before = state.inventory
    cash_before = state.cash
    current_bar = env_param.ohlc[state.global_index]
    next_bar = env_param.ohlc[next_global_index]
    next_open = next_bar[OPEN_INDEX]
    current_close = current_bar[CLOSE_INDEX]
    next_close = next_bar[CLOSE_INDEX]
    current_atr = env_param.atr[state.global_index]

    normalized_action, scaled_action = _scale_action(action, env_param)
    reservation_price = current_close - scaled_action[0] * current_atr
    spread = scaled_action[1] * current_atr
    bid_price = reservation_price - 0.5 * spread
    ask_price = reservation_price + 0.5 * spread

    max_inventory = env_param.max_inventory
    raw_quote_size = env_param.quote_size_shares
    bid_size = jnp.minimum(
        raw_quote_size,
        jnp.maximum(max_inventory - inventory_before, 0),
    )
    ask_size = jnp.minimum(
        raw_quote_size,
        jnp.maximum(max_inventory + inventory_before, 0),
    )

    bid_fill = jnp.where(
        valid_transition & (next_bar[LOW_INDEX] <= bid_price),
        1.0,
        0.0,
    )
    ask_fill = jnp.where(
        valid_transition & (next_bar[HIGH_INDEX] >= ask_price),
        1.0,
        0.0,
    )
    bid_execution_price = jnp.minimum(bid_price, next_open)
    ask_execution_price = jnp.maximum(ask_price, next_open)
    inventory_after_trade = inventory_before + bid_fill * bid_size - ask_fill * ask_size

    cash_after_trade = (
        cash_before
        - bid_fill * bid_size * bid_execution_price
        + ask_fill * ask_size * ask_execution_price
    )
    mark_close = jnp.where(valid_transition, next_close, current_close)
    equity_before = cash_before + inventory_before * current_close
    equity_after_mark = cash_after_trade + inventory_after_trade * mark_close
    portfolio_value_before = equity_before + state.return_baseline
    portfolio_value_after_mark = equity_after_mark + state.return_baseline
    pnl = equity_after_mark - equity_before
    damped_pnl_eta = reward_config.damped_pnl_eta
    inventory_penalty_eta = reward_config.inventory_penalty_eta
    reward_epsilon = reward_config.reward_epsilon
    current_is_bankrupt = (portfolio_value_before <= 0.0).astype(jnp.float32)
    is_bankrupt = jnp.maximum(state.is_bankrupt, current_is_bankrupt)
    step_return = jnp.where(
        is_bankrupt > 0.0,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        pnl / portfolio_value_before,
    )

    log_ret = jnp.log(portfolio_value_after_mark / portfolio_value_before) * 100
    damped_log_ret = log_ret - damped_pnl_eta * jnp.maximum(0.0, log_ret)

    # damped_log_ret = log_ret - damped_pnl_eta * jnp.square(jnp.minimum(0.0, log_ret))


    damped_pnl = pnl - damped_pnl_eta * jnp.square(jnp.maximum(0.0, pnl))
    flattened_cash = cash_after_trade + inventory_after_trade * current_close
    next_cash = jnp.where(should_flatten, flattened_cash, cash_after_trade)
    next_inventory = jnp.where(should_flatten, 0., inventory_after_trade)
    inventory_feature = next_inventory / env_param.max_inventory
    inventory_penalty = inventory_penalty_eta * jnp.square(inventory_feature)
    reward = damped_log_ret # - inventory_penalty
        
    next_state = EnvState(
        step_index=next_step_index,
        global_index=next_global_index,
        inventory=next_inventory,
        cash=next_cash,
        return_baseline=state.return_baseline,
        is_bankrupt=is_bankrupt,
    )
    next_observation = build_observations(
        env_param.static_features[next_global_index],
        next_inventory,
        env_param.max_inventory,
        next_step_index,
        env_param.episode_length,
    )
    done = 0.0
    portfolio_value_after = jnp.where(
        should_flatten,
        next_cash + state.return_baseline,
        portfolio_value_after_mark,
    )

    info = {
        "action": scaled_action,
        "ask_fill": ask_fill,
        "ask_execution_price": ask_execution_price,
        "ask_price": ask_price,
        "ask_size": ask_size,
        "bid_fill": bid_fill,
        "bid_execution_price": bid_execution_price,
        "bid_price": bid_price,
        "bid_size": bid_size,
        "cash_after": next_cash,
        "cash_after_trade": cash_after_trade,
        "cash_before": cash_before,
        "current_global_index": state.global_index,
        "current_step_index": state.step_index,
        "damped_pnl": damped_pnl,
        "done": done,
        "inventory_after": next_inventory,
        "inventory_after_trade": inventory_after_trade,
        "inventory_before": inventory_before,
        "inventory_penalty": inventory_penalty,
        "is_bankrupt": is_bankrupt,
        "next_global_index": next_global_index,
        "next_index": next_global_index,
        "next_step_index": next_step_index,
        "normalized_action": normalized_action,
        "pnl": pnl,
        "portfolio_value_after": portfolio_value_after,
        "portfolio_value_before": portfolio_value_before,
        "raw_quote_size": raw_quote_size,
        "reservation_price": reservation_price,
        "return": step_return,
        "session_end": session_end,
        "spread": spread,
        "valid_transition": valid_transition,
    }
    return next_observation, next_state, reward, done, info


__all__ = [
    "CLOSE_INDEX",
    "EnvParam",
    "EnvState",
    "HIGH_INDEX",
    "LOW_INDEX",
    "OPEN_INDEX",
    "build_observation",
    "build_critic_observation",
    "build_observations",
    "env_reset",
    "env_step",
]
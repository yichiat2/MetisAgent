from __future__ import annotations

from flax import struct
import jax.numpy as jnp

from .config import EnvironmentConfig, RewardConfig
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


@struct.dataclass
class EnvParam:
    max_inventory: float
    max_quote_size: float
    flatten_at_session_end: bool
    ohlc: jnp.ndarray
    static_features: jnp.ndarray
    atr: jnp.ndarray
    sigma_price: jnp.ndarray
    day_ids: jnp.ndarray
    bar_in_day: jnp.ndarray
    session_end_mask: jnp.ndarray
    

def build_observation(static_features: jnp.ndarray, inventory: float, max_inventory: float) -> jnp.ndarray:
    inventory_feature = inventory / max_inventory
    return jnp.concatenate([static_features, jnp.asarray(inventory_feature, dtype=jnp.float32)[None]], axis=-1)


def env_reset(
    env_param: EnvParam,
    global_index: int,
    initial_inventory: float = 0.0,
    initial_cash: float = 0.0,
) -> tuple[jnp.ndarray, EnvState]:
    state = EnvState(step_index=0, global_index=global_index, inventory=initial_inventory, cash=initial_cash)
    static_features_0 = env_param.static_features[global_index]
    observation = build_observation(static_features_0, state.inventory, env_param.max_inventory)
    return observation, state


def env_step(
    state: EnvState,
    action: jnp.ndarray,  # [3]
    env_param: EnvParam,
    reward_config: RewardConfig,
) -> tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
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
    current_close = current_bar[CLOSE_INDEX]
    next_close = next_bar[CLOSE_INDEX]
    current_atr = env_param.atr[state.global_index]
    current_sigma_price = env_param.sigma_price[state.global_index]
    bid_price = current_close - action[0] * current_sigma_price
    ask_price = current_close + action[1] * current_sigma_price
    max_quote_size = env_param.max_quote_size
    max_inventory = env_param.max_inventory
    raw_quote_size = jnp.rint(action[2] * max_quote_size).astype(jnp.float32)
    bid_size = jnp.minimum(raw_quote_size, jnp.maximum(max_inventory - inventory_before, 0))
    ask_size = jnp.minimum(raw_quote_size, jnp.maximum(max_inventory + inventory_before, 0))

    bid_fill = jnp.where(valid_transition & (next_bar[LOW_INDEX] <= bid_price), 1.0, 0.0)
    ask_fill = jnp.where(valid_transition & (next_bar[HIGH_INDEX] >= ask_price), 1.0, 0.0)
    inventory_after_trade = inventory_before + bid_fill * bid_size - ask_fill * ask_size

    cash_after_trade = (
        cash_before
        - bid_fill * bid_size * bid_price
        + ask_fill * ask_size * ask_price
    )
    mark_close = jnp.where(valid_transition, next_close, current_close)
    portfolio_value_before = cash_before + inventory_before * current_close
    portfolio_value_after_mark = cash_after_trade + inventory_after_trade * mark_close
    pnl = portfolio_value_after_mark - portfolio_value_before
    damped_pnl_eta = reward_config.damped_pnl_eta
    inventory_penalty_eta = reward_config.inventory_penalty_eta
    reward_epsilon = reward_config.reward_epsilon
    damped_pnl = pnl - jnp.maximum(0.0, damped_pnl_eta * pnl)

    trading_pnl = (
        bid_fill * bid_size * (next_close - bid_price)
        + ask_fill * ask_size * (ask_price - next_close)
    )
    trading_pnl = jnp.where(valid_transition, trading_pnl, 0.0)
    
    flattened_cash = cash_after_trade + inventory_after_trade * current_close
    next_cash = jnp.where(should_flatten, flattened_cash, cash_after_trade)
    next_inventory = jnp.where(should_flatten, 0., inventory_after_trade)
    inventory_feature = next_inventory / env_param.max_inventory
    inventory_penalty = inventory_penalty_eta * jnp.square(inventory_feature)
    reward_scale = current_atr + reward_epsilon
    reward = (damped_pnl + trading_pnl) / reward_scale - inventory_penalty
        
    next_state = EnvState(
        step_index=next_step_index,
        global_index=next_global_index,
        inventory=next_inventory,
        cash=next_cash,
    )
    next_observation = build_observation(
        env_param.static_features[next_global_index],
        next_inventory,
        env_param.max_inventory,
    )
    global_done = 0.0
    episode_done = 0.0
    done = 0.0
    portfolio_value_after = jnp.where(should_flatten, next_cash, portfolio_value_after_mark)

    info = {
        "action": action,
        "ask_fill": ask_fill,
        "ask_price": ask_price,
        "ask_size": ask_size,
        "bid_fill": bid_fill,
        "bid_price": bid_price,
        "bid_size": bid_size,
        "cash_after": next_cash,
        "cash_after_trade": cash_after_trade,
        "cash_before": cash_before,
        "current_global_index": state.global_index,
        "current_step_index": state.step_index,
        "damped_pnl": damped_pnl,
        "done": done,
        "episode_done": episode_done,
        "global_done": global_done,
        "inventory_after": next_inventory,
        "inventory_after_trade": inventory_after_trade,
        "inventory_before": inventory_before,
        "inventory_penalty": inventory_penalty,
        "next_global_index": next_global_index,
        "next_index": next_global_index,
        "next_step_index": next_step_index,
        "pnl": pnl,
        "portfolio_value_after": portfolio_value_after,
        "portfolio_value_before": portfolio_value_before,
        "raw_quote_size": raw_quote_size,
        "reward_scale": reward_scale,
        "session_end": session_end,
        "trading_pnl": trading_pnl,
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
    "env_reset",
    "env_step",
]
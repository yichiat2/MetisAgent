from __future__ import annotations

from flax import struct
import jax.numpy as jnp
from .config import PPOVolScalpingConfig, RewardConfig
from .contracts import PreprocessedArrays

OPEN_INDEX = 0
HIGH_INDEX = 1
LOW_INDEX = 2
CLOSE_INDEX = 3

PRICE_ACTION_LEVELS = jnp.array((0., 1.0, 2.0, 3.0), dtype=jnp.float32)
QUOTE_SIZE_SHARES = 100.0


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
    max_inventory: float
    flatten_at_session_end: bool
    ohlc: jnp.ndarray
    static_features: jnp.ndarray
    atr: jnp.ndarray
    day_ids: jnp.ndarray
    bar_in_day: jnp.ndarray
    session_end_mask: jnp.ndarray
    


def build_env_param(config: PPOVolScalpingConfig, train_arrays: PreprocessedArrays) -> EnvParam:
    return EnvParam(
        max_inventory=float(config.environment.max_inventory),
        flatten_at_session_end=config.environment.flatten_at_session_end,
        ohlc=jnp.asarray(train_arrays.ohlc),
        static_features=jnp.asarray(train_arrays.static_features),
        atr=jnp.asarray(train_arrays.atr),
        day_ids=jnp.asarray(train_arrays.day_ids),
        bar_in_day=jnp.asarray(train_arrays.bar_in_day),
        session_end_mask=jnp.asarray(train_arrays.session_end_mask),
    )

def build_observation(static_features: jnp.ndarray, inventory: float, max_inventory: float) -> jnp.ndarray:
    inventory_feature = inventory / max_inventory
    return jnp.concatenate([static_features, jnp.asarray(inventory_feature, dtype=jnp.float32)[None]], axis=-1)


def env_reset(
    env_param: EnvParam,
    global_index: int,
    initial_inventory: float = 0.0,
    initial_cash: float = 0.0,
) -> tuple[jnp.ndarray, EnvState]:
    first_close = env_param.ohlc[global_index, CLOSE_INDEX]
    return_baseline = jnp.asarray(QUOTE_SIZE_SHARES, dtype=jnp.float32) * first_close
    state = EnvState(
        step_index=0,
        global_index=global_index,
        inventory=initial_inventory,
        cash=initial_cash,
        return_baseline=return_baseline,
        is_bankrupt=jnp.asarray(0.0, dtype=jnp.float32),
    )
    static_features_0 = env_param.static_features[global_index]
    observation = build_observation(static_features_0, state.inventory, env_param.max_inventory)
    return observation, state


def env_step(
    state: EnvState,
    action: jnp.ndarray,  # [2] categorical ids
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
    next_open = next_bar[OPEN_INDEX]
    current_close = current_bar[CLOSE_INDEX]
    next_close = next_bar[CLOSE_INDEX]
    current_atr = env_param.atr[state.global_index]

    action_index = jnp.clip(action, 0, 3)
    bid_level = PRICE_ACTION_LEVELS[action_index[0]]
    ask_level = PRICE_ACTION_LEVELS[action_index[1]]

    no_quotes = (action_index[0] == 0) & (action_index[1] == 0)
    buy_market = (action_index[0] == 0) & (action_index[1] > 0)
    sell_market = (action_index[1] == 0) & (action_index[0] > 0)

    bid_limit_price = current_close - bid_level * current_atr
    ask_limit_price = current_close + ask_level * current_atr
    bid_price = jnp.where(buy_market, next_open, bid_limit_price)
    ask_price = jnp.where(sell_market, next_open, ask_limit_price)

    max_inventory = env_param.max_inventory
    raw_quote_size = jnp.asarray(QUOTE_SIZE_SHARES, dtype=jnp.float32)
    bid_size = jnp.where(
        no_quotes,
        0.0,
        jnp.minimum(raw_quote_size, jnp.maximum(max_inventory - inventory_before, 0)),
    )
    ask_size = jnp.where(
        no_quotes,
        0.0,
        jnp.minimum(raw_quote_size, jnp.maximum(max_inventory + inventory_before, 0)),
    )

    bid_fill = jnp.where(
        no_quotes,
        0.0,
        jnp.where(
            buy_market,
            jnp.where(valid_transition, 1.0, 0.0),
            jnp.where(valid_transition & (next_bar[LOW_INDEX] <= bid_limit_price), 1.0, 0.0),
        ),
    )
    ask_fill = jnp.where(
        no_quotes,
        0.0,
        jnp.where(
            sell_market,
            jnp.where(valid_transition, 1.0, 0.0),
            jnp.where(valid_transition & (next_bar[HIGH_INDEX] >= ask_limit_price), 1.0, 0.0),
        ),
    )
    inventory_after_trade = inventory_before + bid_fill * bid_size - ask_fill * ask_size

    cash_after_trade = (
        cash_before
        - bid_fill * bid_size * bid_price
        + ask_fill * ask_size * ask_price
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
    damped_log_ret = log_ret - jnp.maximum(0.0, damped_pnl_eta * log_ret)

    damped_pnl = pnl - jnp.maximum(0.0, damped_pnl_eta * pnl)
    flattened_cash = cash_after_trade + inventory_after_trade * current_close
    next_cash = jnp.where(should_flatten, flattened_cash, cash_after_trade)
    next_inventory = jnp.where(should_flatten, 0., inventory_after_trade)
    inventory_feature = next_inventory / env_param.max_inventory
    inventory_penalty = inventory_penalty_eta * jnp.square(inventory_feature)
    reward = damped_log_ret
        
    next_state = EnvState(
        step_index=next_step_index,
        global_index=next_global_index,
        inventory=next_inventory,
        cash=next_cash,
        return_baseline=state.return_baseline,
        is_bankrupt=is_bankrupt,
    )
    next_observation = build_observation(
        env_param.static_features[next_global_index],
        next_inventory,
        env_param.max_inventory,
    )
    done = 0.0
    portfolio_value_after = jnp.where(
        should_flatten,
        next_cash + state.return_baseline,
        portfolio_value_after_mark,
    )

    info = {
        "action": action_index,
        "action_rescaled": jnp.asarray((bid_level, ask_level), dtype=jnp.float32),
        "ask_fill": ask_fill,
        "ask_price": ask_price,
        "ask_size": ask_size,
        "bid_fill": bid_fill,
        "bid_price": bid_price,
        "bid_size": bid_size,
        "buy_market": buy_market.astype(jnp.float32),
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
        "no_quotes": no_quotes.astype(jnp.float32),
        "pnl": pnl,
        "pnl_vol_scaled": pnl / (current_atr + reward_epsilon),
        "portfolio_value_after": portfolio_value_after,
        "portfolio_value_before": portfolio_value_before,
        "raw_quote_size": raw_quote_size,
        "return": step_return,
        "sell_market": sell_market.astype(jnp.float32),
        "session_end": session_end,
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
    "QUOTE_SIZE_SHARES",
    "build_observation",
    "env_reset",
    "env_step",
]
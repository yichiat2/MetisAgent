from __future__ import annotations

from flax import struct
import jax.numpy as jnp
from .config import PPOVolScalpingConfig
from .contracts import PreprocessedArrays

OPEN_INDEX = 0
HIGH_INDEX = 1
LOW_INDEX = 2
CLOSE_INDEX = 3


@struct.dataclass
class EnvState:
    bar_index: jnp.ndarray
    cash: jnp.ndarray
    shares: jnp.ndarray
    position: jnp.ndarray
    entry_price: jnp.ndarray
    entry_atr: jnp.ndarray
    entry_equity: jnp.ndarray
    stop_price: jnp.ndarray
    trailing_anchor: jnp.ndarray
    k_init: jnp.ndarray
    a_tp: jnp.ndarray
    k_act: jnp.ndarray
    entry_short: jnp.ndarray
    entry_long: jnp.ndarray
    trade_count: jnp.ndarray
    winning_trade_count: jnp.ndarray
    realized_trade_pnl_sum: jnp.ndarray
    is_bankrupt: jnp.ndarray


@struct.dataclass
class EnvParam:
    quote_size_shares: jnp.ndarray
    ibkr_monthly_volume_shares: jnp.ndarray
    ibkr_commission_min_per_order: jnp.ndarray
    ibkr_commission_max_trade_value_ratio: jnp.ndarray
    slippage_atr_multiple: jnp.ndarray
    spread_atr_multiple: jnp.ndarray
    max_entry_atr_over_ema: jnp.ndarray
    min_entry_bar_in_day: jnp.ndarray
    ohlc: jnp.ndarray
    atr: jnp.ndarray
    atr_imbalance: jnp.ndarray
    atr_over_ema: jnp.ndarray
    bar_in_day: jnp.ndarray
    session_end_mask: jnp.ndarray


def build_env_param(config: PPOVolScalpingConfig, train_arrays: PreprocessedArrays) -> EnvParam:
    return EnvParam(
        quote_size_shares=jnp.asarray(config.environment.quote_size_shares, dtype=jnp.float32),
        ibkr_monthly_volume_shares=jnp.asarray(
            config.environment.ibkr_monthly_volume_shares,
            dtype=jnp.float32,
        ),
        ibkr_commission_min_per_order=jnp.asarray(
            config.environment.ibkr_commission_min_per_order,
            dtype=jnp.float32,
        ),
        ibkr_commission_max_trade_value_ratio=jnp.asarray(
            config.environment.ibkr_commission_max_trade_value_ratio,
            dtype=jnp.float32,
        ),
        slippage_atr_multiple=jnp.asarray(config.environment.slippage_atr_multiple, dtype=jnp.float32),
        spread_atr_multiple=jnp.asarray(config.environment.spread_atr_multiple, dtype=jnp.float32),
        max_entry_atr_over_ema=jnp.asarray(config.environment.max_entry_atr_over_ema, dtype=jnp.float32),
        min_entry_bar_in_day=jnp.asarray(config.environment.min_entry_bar_in_day, dtype=jnp.int32),
        ohlc=jnp.asarray(train_arrays.ohlc),
        atr=jnp.asarray(train_arrays.atr),
        atr_imbalance=jnp.asarray(train_arrays.atr_imbalance),
        atr_over_ema=jnp.asarray(train_arrays.atr_over_ema),
        bar_in_day=jnp.asarray(train_arrays.bar_in_day),
        session_end_mask=jnp.asarray(train_arrays.session_end_mask),
    )


def _ibkr_tiered_share_rate(monthly_volume_shares: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(
        monthly_volume_shares <= 300_000.0,
        0.0035,
        jnp.where(
            monthly_volume_shares <= 3_000_000.0,
            0.0020,
            jnp.where(
                monthly_volume_shares <= 20_000_000.0,
                0.0015,
                jnp.where(monthly_volume_shares <= 100_000_000.0, 0.0010, 0.0005),
            ),
        ),
    )


def _ibkr_tiered_commission(
    share_quantity: jnp.ndarray,
    trade_price: jnp.ndarray,
    env_param: EnvParam,
) -> jnp.ndarray:
    share_count = jnp.abs(share_quantity)
    trade_value = share_count * jnp.maximum(trade_price, 0.0)
    commission_rate = _ibkr_tiered_share_rate(env_param.ibkr_monthly_volume_shares)
    uncapped_commission = jnp.maximum(
        commission_rate * share_count,
        env_param.ibkr_commission_min_per_order,
    )
    capped_commission = jnp.minimum(
        uncapped_commission,
        env_param.ibkr_commission_max_trade_value_ratio * trade_value,
    )
    return jnp.where(share_count > 0.0, capped_commission, 0.0)


def env_reset(
    env_param: EnvParam,
    k_init: float,
    a_tp: float,
    k_act: float,
    entry_short: float,
    entry_long: float,
) -> EnvState:
    initial_cash = env_param.quote_size_shares * env_param.ohlc[0, OPEN_INDEX]
    nan_value = jnp.asarray(jnp.nan, dtype=jnp.float32)
    return EnvState(
        bar_index=jnp.asarray(0, dtype=jnp.int32),
        cash=initial_cash,
        shares=jnp.asarray(0.0, dtype=jnp.float32),
        position=jnp.asarray(0.0, dtype=jnp.float32),
        entry_price=nan_value,
        entry_atr=nan_value,
        entry_equity=initial_cash,
        stop_price=nan_value,
        trailing_anchor=nan_value,
        k_init=jnp.asarray(k_init, dtype=jnp.float32),
        a_tp=jnp.asarray(a_tp, dtype=jnp.float32),
        k_act=jnp.asarray(k_act, dtype=jnp.float32),
        entry_short=jnp.asarray(entry_short, dtype=jnp.float32),
        entry_long=jnp.asarray(entry_long, dtype=jnp.float32),
        trade_count=jnp.asarray(0, dtype=jnp.int32),
        winning_trade_count=jnp.asarray(0, dtype=jnp.int32),
        realized_trade_pnl_sum=jnp.asarray(0.0, dtype=jnp.float32),
        is_bankrupt=jnp.asarray(0.0, dtype=jnp.float32),
    )


def env_step(
    state: EnvState,
    env_param: EnvParam,
 ) -> tuple[EnvState, dict[str, jnp.ndarray]]:
    epsilon = jnp.asarray(1e-8, dtype=jnp.float32)
    nan_value = jnp.asarray(jnp.nan, dtype=jnp.float32)
    next_index = state.bar_index + 1
    last_index = jnp.asarray(env_param.ohlc.shape[0] - 1, dtype=jnp.int32)
    current_close = env_param.ohlc[state.bar_index, CLOSE_INDEX]
    next_bar = env_param.ohlc[next_index]
    next_open = next_bar[OPEN_INDEX]
    next_high = next_bar[HIGH_INDEX]
    next_low = next_bar[LOW_INDEX]
    next_close = next_bar[CLOSE_INDEX]
    portfolio_value_before = state.cash + state.shares * current_close

    next_bar_in_day = env_param.bar_in_day[next_index]
    next_is_session_end = env_param.session_end_mask[next_index]
    current_atr_over_ema = env_param.atr_over_ema[state.bar_index]
    entry_window_open = (next_bar_in_day >= env_param.min_entry_bar_in_day) & jnp.logical_not(next_is_session_end)
    can_open_trade = (
        (state.position == 0.0)
        & (portfolio_value_before > 0.0)
        & (current_atr_over_ema <= env_param.max_entry_atr_over_ema)
        & entry_window_open
    )
    current_imbalance = env_param.atr_imbalance[state.bar_index]
    current_atr = env_param.atr[state.bar_index]
    half_spread = 0.5 * env_param.spread_atr_multiple * current_atr
    slippage = env_param.slippage_atr_multiple * current_atr
    one_way_price_impact = half_spread + slippage
    long_signal = can_open_trade & (current_imbalance <= state.entry_long)
    short_signal = can_open_trade & (current_imbalance >= state.entry_short)
    requested_entry_direction = jnp.where(long_signal, 1.0, jnp.where(short_signal, -1.0, 0.0))
    max_whole_shares = jnp.floor(portfolio_value_before / jnp.maximum(next_open, epsilon))
    entry_direction = jnp.where(max_whole_shares >= 1.0, requested_entry_direction, 0.0)
    long_entry = entry_direction > 0.0
    short_entry = entry_direction < 0.0
    entry_size = jnp.where(
        entry_direction != 0.0,
        entry_direction * max_whole_shares,
        state.shares,
    )
    entry_execution_price = jnp.where(
        long_entry,
        next_open + one_way_price_impact,
        jnp.where(short_entry, jnp.maximum(next_open - one_way_price_impact, epsilon), 0.0),
    )
    entry_share_count = jnp.where(entry_direction != 0.0, jnp.abs(entry_size), 0.0)
    entry_commission = _ibkr_tiered_commission(entry_size, entry_execution_price, env_param)
    entry_slippage_cost = entry_share_count * slippage
    entry_spread_cost = entry_share_count * half_spread
    shares_after_entry = jnp.where(entry_direction != 0.0, entry_size, state.shares)
    cash_after_entry = jnp.where(
        entry_direction != 0.0,
        state.cash - shares_after_entry * entry_execution_price - entry_commission,
        state.cash,
    )
    position_after_entry = jnp.where(entry_direction != 0.0, entry_direction, state.position)
    entry_price_after = jnp.where(entry_direction != 0.0, entry_execution_price, state.entry_price)
    entry_atr_after = jnp.where(entry_direction != 0.0, current_atr, state.entry_atr)
    entry_equity_after = jnp.where(entry_direction != 0.0, portfolio_value_before, state.entry_equity)
    trailing_anchor_after_entry = jnp.where(entry_direction != 0.0, nan_value, state.trailing_anchor)
    initial_stop_distance = state.k_init * entry_atr_after
    activation_distance = state.a_tp * entry_atr_after
    trailing_stop_distance = state.k_act * entry_atr_after
    stop_after_entry = jnp.where(
        long_entry,
        entry_execution_price - initial_stop_distance,
        jnp.where(short_entry, entry_execution_price + initial_stop_distance, state.stop_price),
    )

    active_long = position_after_entry > 0.0
    active_short = position_after_entry < 0.0
    activation_price = jnp.where(
        active_long,
        entry_price_after + activation_distance,
        jnp.where(active_short, entry_price_after - activation_distance, nan_value),
    )
    long_gap_exit = active_long & (next_open <= stop_after_entry)
    short_gap_exit = active_short & (next_open >= stop_after_entry)
    long_stop_exit = active_long & jnp.logical_or(long_gap_exit, next_low <= stop_after_entry)
    short_stop_exit = active_short & jnp.logical_or(short_gap_exit, next_high >= stop_after_entry)
    stop_exit = long_stop_exit | short_stop_exit
    forced_exit = (position_after_entry != 0.0) & (next_index == last_index) & jnp.logical_not(stop_exit)
    exit_market_price = jnp.where(
        long_stop_exit,
        jnp.where(long_gap_exit, next_open, stop_after_entry),
        jnp.where(
            short_stop_exit,
            jnp.where(short_gap_exit, next_open, stop_after_entry),
            jnp.where(forced_exit, next_close, nan_value),
        ),
    )
    exit_execution_price = jnp.where(
        active_long & (stop_exit | forced_exit),
        jnp.maximum(exit_market_price - one_way_price_impact, epsilon),
        jnp.where(
            active_short & (stop_exit | forced_exit),
            exit_market_price + one_way_price_impact,
            0.0,
        ),
    )
    exit_share_count = jnp.where(stop_exit | forced_exit, jnp.abs(shares_after_entry), 0.0)
    exit_commission = _ibkr_tiered_commission(shares_after_entry, exit_execution_price, env_param)
    exit_slippage_cost = exit_share_count * slippage
    exit_spread_cost = exit_share_count * half_spread

    cash_after_exit = jnp.where(
        stop_exit | forced_exit,
        cash_after_entry + shares_after_entry * exit_execution_price - exit_commission,
        cash_after_entry,
    )
    realized_trade_pnl = jnp.where(
        stop_exit | forced_exit,
        cash_after_exit - entry_equity_after,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    trailing_active_before = jnp.isfinite(trailing_anchor_after_entry)
    long_activation_hit = active_long & (next_high >= activation_price)
    short_activation_hit = active_short & (next_low <= activation_price)
    trailing_active_next = trailing_active_before | long_activation_hit | short_activation_hit
    trailing_anchor_candidate = jnp.where(
        active_long,
        jnp.where(trailing_active_before, jnp.maximum(trailing_anchor_after_entry, next_high), next_high),
        jnp.where(
            active_short,
            jnp.where(trailing_active_before, jnp.minimum(trailing_anchor_after_entry, next_low), next_low),
            nan_value,
        ),
    )
    updated_anchor = jnp.where(trailing_active_next, trailing_anchor_candidate, trailing_anchor_after_entry)
    trailing_stop_candidate = jnp.where(
        active_long,
        updated_anchor - trailing_stop_distance,
        jnp.where(active_short, updated_anchor + trailing_stop_distance, stop_after_entry),
    )
    updated_stop = jnp.where(
        trailing_active_next & active_long,
        jnp.maximum(stop_after_entry, trailing_stop_candidate),
        jnp.where(
            trailing_active_next & active_short,
            jnp.minimum(stop_after_entry, trailing_stop_candidate),
            stop_after_entry,
        ),
    )

    shares_after = jnp.where(stop_exit | forced_exit, 0.0, shares_after_entry)
    position_after = jnp.where(stop_exit | forced_exit, 0.0, position_after_entry)
    cash_after = jnp.where(stop_exit | forced_exit, cash_after_exit, cash_after_entry)
    stop_price_after = jnp.where(stop_exit | forced_exit, nan_value, updated_stop)
    trailing_anchor_after = jnp.where(stop_exit | forced_exit, nan_value, updated_anchor)
    entry_price_next = jnp.where(stop_exit | forced_exit, nan_value, entry_price_after)
    entry_atr_next = jnp.where(stop_exit | forced_exit, nan_value, entry_atr_after)
    entry_equity_next = jnp.where(stop_exit | forced_exit, cash_after, entry_equity_after)
    portfolio_value_after = jnp.where(
        position_after == 0.0,
        cash_after,
        cash_after + shares_after * next_close,
    )
    pnl = portfolio_value_after - portfolio_value_before
    bankrupt_now = jnp.logical_or(portfolio_value_before <= 0.0, portfolio_value_after <= 0.0)
    is_bankrupt = jnp.maximum(state.is_bankrupt, bankrupt_now.astype(jnp.float32))
    step_return = jnp.where(
        is_bankrupt > 0.0,
        nan_value,
        pnl / jnp.maximum(portfolio_value_before, epsilon),
    )

    next_state = EnvState(
        bar_index=next_index,
        cash=cash_after,
        shares=shares_after,
        position=position_after,
        entry_price=entry_price_next,
        entry_atr=entry_atr_next,
        entry_equity=entry_equity_next,
        stop_price=stop_price_after,
        trailing_anchor=trailing_anchor_after,
        k_init=state.k_init,
        a_tp=state.a_tp,
        k_act=state.k_act,
        entry_short=state.entry_short,
        entry_long=state.entry_long,
        trade_count=state.trade_count + (stop_exit | forced_exit).astype(jnp.int32),
        winning_trade_count=state.winning_trade_count + (realized_trade_pnl > 0.0).astype(jnp.int32),
        realized_trade_pnl_sum=state.realized_trade_pnl_sum + realized_trade_pnl,
        is_bankrupt=is_bankrupt,
    )
    info = {
        "cash_after": cash_after,
        "commission_cost": entry_commission + exit_commission,
        "entry_long_price": jnp.where(long_entry, entry_execution_price, nan_value),
        "entry_short_price": jnp.where(short_entry, entry_execution_price, nan_value),
        "entry_size": jnp.where(entry_direction != 0.0, shares_after_entry, 0.0),
        "exit_long_price": jnp.where((stop_exit | forced_exit) & active_long, exit_execution_price, nan_value),
        "exit_short_price": jnp.where((stop_exit | forced_exit) & active_short, exit_execution_price, nan_value),
        "forced_exit": forced_exit.astype(jnp.float32),
        "is_bankrupt": is_bankrupt,
        "pnl": pnl,
        "portfolio_value_after": portfolio_value_after,
        "portfolio_value_before": portfolio_value_before,
        "position_after": position_after,
        "realized_trade_pnl": realized_trade_pnl,
        "return": step_return,
        "shares_after": shares_after,
        "slippage_cost": entry_slippage_cost + exit_slippage_cost,
        "spread_cost": entry_spread_cost + exit_spread_cost,
        "stop_exit": stop_exit.astype(jnp.float32),
        "stop_price_after": stop_price_after,
        "trade_count": next_state.trade_count,
        "transaction_cost": (
            entry_commission
            + exit_commission
            + entry_slippage_cost
            + exit_slippage_cost
            + entry_spread_cost
            + exit_spread_cost
        ),
        "winning_trade_count": next_state.winning_trade_count,
    }
    return next_state, info


__all__ = [
    "CLOSE_INDEX",
    "EnvParam",
    "EnvState",
    "HIGH_INDEX",
    "LOW_INDEX",
    "OPEN_INDEX",
    "build_env_param",
    "env_reset",
    "env_step",
]
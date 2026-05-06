from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PACKAGE_NAME = "_test_ppo_vol_scalping"
PACKAGE_DIR = REPO_ROOT / "rl" / "ppo_vol_scalping"

package_module = ModuleType(PACKAGE_NAME)
package_module.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package_module)

config_module = _load_module(f"{PACKAGE_NAME}.config", PACKAGE_DIR / "config.py")
contracts_module = _load_module(f"{PACKAGE_NAME}.contracts", PACKAGE_DIR / "contracts.py")
_load_module(f"{PACKAGE_NAME}.data", PACKAGE_DIR / "data.py")
env_module = _load_module(f"{PACKAGE_NAME}.env", PACKAGE_DIR / "env.py")
model_module = _load_module(f"{PACKAGE_NAME}.model", PACKAGE_DIR / "model.py")
visual_module = _load_module(f"{PACKAGE_NAME}.visual", PACKAGE_DIR / "visual.py")
driver_module = _load_module(f"{PACKAGE_NAME}.driver", PACKAGE_DIR / "driver.py")

PreprocessedArrays = contracts_module.PreprocessedArrays
STATE_DIM = contracts_module.STATE_DIM
make_default_config = config_module.make_default_config
build_env_param = env_module.build_env_param
env_reset = env_module.env_reset
env_step = env_module.env_step
create_train_states = model_module.create_train_states
QUOTE_SIZE_SHARES = env_module.QUOTE_SIZE_SHARES
prepare_inference_payload = visual_module.prepare_inference_payload
run_fold_inference = driver_module.run_fold_inference
summarize_inference = driver_module._summarize_inference


def test_prepare_inference_payload_aligns_ohlc_to_t_plus_one() -> None:
    metrics = {
        "actions": np.asarray([[0, 1], [3, 2], [1, 0]], dtype=np.int32),
        "ask_price": np.asarray([101.0, 102.0, 103.0], dtype=np.float32),
        "bid_price": np.asarray([99.0, 98.0, 97.0], dtype=np.float32),
        "portfolio_value": np.asarray([1000.0, 999.75, 1000.5], dtype=np.float32),
        "inventory": np.asarray([10.0, -5.0, 0.0], dtype=np.float32),
        "pnl": np.asarray([0.5, -0.25, 0.75], dtype=np.float32),
        "cumulative_pnl": np.asarray([0.5, 0.25, 1.0], dtype=np.float32),
        "transaction_count": np.asarray(4, dtype=np.int32),
        "total_pnl": np.asarray(1.0, dtype=np.float32),
        "final_cumulative_return": np.asarray(0.02, dtype=np.float32),
        "sharpe_ratio": np.asarray(1.25, dtype=np.float32),
        "sortino_ratio": np.asarray(1.75, dtype=np.float32),
        "max_drawdown": np.asarray(0.1, dtype=np.float32),
        "bankruptcy": np.asarray(False),
    }
    ohlc = np.asarray(
        [
            [100.0, 101.0, 99.0, 100.0],
            [101.0, 102.0, 100.0, 101.0],
            [102.0, 103.0, 101.0, 102.0],
            [103.0, 104.0, 102.0, 103.0],
        ],
        dtype=np.float32,
    )

    payload = prepare_inference_payload(
        metrics=metrics,
        ohlc=ohlc,
        fold_id=2,
        update_step=5,
        num_updates=10,
    )

    assert payload["fold_id"] == 2
    assert payload["update_step"] == 5
    assert payload["num_updates"] == 10
    assert payload["transaction_count"] == 4
    np.testing.assert_allclose(
        payload["metrics"]["portfolio_value"],
        metrics["portfolio_value"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(payload["ohlc"], ohlc[1:], rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(payload["actions"], metrics["actions"])


def test_summarize_inference_matches_expected_formula() -> None:
    config = make_default_config()
    step_pnl = jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float32)
    step_returns = jnp.asarray([0.01, -0.02, 0.03], dtype=jnp.float32)
    is_bankrupt = jnp.zeros_like(step_returns)
    bid_fill = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)
    ask_fill = jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float32)

    summary_metrics = summarize_inference(
        config=config,
        step_pnl=step_pnl,
        step_returns=step_returns,
        is_bankrupt=is_bankrupt,
        bid_fill=bid_fill,
        ask_fill=ask_fill,
    )

    returns_np = np.asarray(step_returns)
    pnl_np = np.asarray(step_pnl)
    annualization = np.sqrt(config.logging.evaluation_annualization_factor)
    wealth_curve = np.concatenate(
        [np.ones((1,), dtype=np.float32), np.cumprod(1.0 + returns_np)]
    )
    downside = np.sqrt(np.mean(np.square(np.minimum(returns_np, 0.0))))
    running_peak = np.maximum.accumulate(wealth_curve)
    expected_cumulative_return = wealth_curve[1:] - 1.0
    expected_drawdown = 1.0 - wealth_curve / (running_peak + config.reward.reward_epsilon)

    np.testing.assert_allclose(
        np.asarray(summary_metrics["cumulative_pnl"]),
        np.cumsum(pnl_np),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(summary_metrics["total_pnl"]),
        np.cumsum(pnl_np)[-1],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(summary_metrics["cumulative_return"]),
        expected_cumulative_return,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(summary_metrics["final_cumulative_return"]),
        expected_cumulative_return[-1],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(summary_metrics["sharpe_ratio"]),
        annualization * returns_np.mean() / (returns_np.std() + config.reward.reward_epsilon),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(summary_metrics["sortino_ratio"]),
        annualization * returns_np.mean() / (downside + config.reward.reward_epsilon),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(summary_metrics["max_drawdown"]),
        expected_drawdown.max(),
        rtol=1e-6,
        atol=1e-6,
    )
    assert int(np.asarray(summary_metrics["transaction_count"])) == 4
    assert not bool(np.asarray(summary_metrics["bankruptcy"]))


def test_run_fold_inference_collects_zero_trade_rollout_metrics() -> None:
    config = make_default_config()
    preprocessed_arrays = PreprocessedArrays(
        ohlc=np.asarray(
            [
                [100.0, 101.0, 99.0, 100.0],
                [100.0, 102.0, 99.0, 101.0],
                [101.0, 103.0, 100.0, 102.0],
                [102.0, 104.0, 101.0, 103.0],
            ],
            dtype=np.float32,
        ),
        static_features=np.zeros((4, STATE_DIM - 1), dtype=np.float32),
        atr=np.ones((4,), dtype=np.float32),
        day_ids=np.zeros((4,), dtype=np.int32),
        bar_in_day=np.arange(4, dtype=np.int32),
        session_end_mask=np.zeros((4,), dtype=bool),
    )
    env_param = build_env_param(config, preprocessed_arrays)

    actor_state, _ = create_train_states(config=config, rng=jax.random.PRNGKey(0))
    actor_state = actor_state.replace(
        params=jax.tree_util.tree_map(jnp.zeros_like, actor_state.params)
    )

    metrics = run_fold_inference(
        config=config,
        actor_state=actor_state,
        env_param=env_param,
    )

    expected_num_steps = preprocessed_arrays.ohlc.shape[0] - 1
    expected_portfolio_value = (
        QUOTE_SIZE_SHARES * preprocessed_arrays.ohlc[0, 3]
    )

    assert metrics["actions"].shape == (expected_num_steps, config.model.action_dim)
    np.testing.assert_array_equal(np.asarray(metrics["actions"]), 0)
    np.testing.assert_allclose(np.asarray(metrics["pnl"]), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(metrics["cumulative_pnl"]),
        0.0,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(metrics["cash"]), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(metrics["inventory"]), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(metrics["bid_fill"]), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(metrics["ask_fill"]), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(metrics["bid_size"]), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(metrics["ask_size"]), 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(metrics["portfolio_value"]),
        expected_portfolio_value,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(metrics["cumulative_return"]),
        0.0,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(metrics["final_cumulative_return"]),
        0.0,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(metrics["sharpe_ratio"]),
        0.0,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(metrics["sortino_ratio"]),
        0.0,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(metrics["max_drawdown"]),
        0.0,
        rtol=1e-6,
        atol=1e-6,
    )
    assert int(np.asarray(metrics["transaction_count"])) == 0
    assert not bool(np.asarray(metrics["bankruptcy"]))


def test_env_step_reads_only_current_and_next_bar() -> None:
    config = make_default_config()
    static_features = np.arange(5 * (STATE_DIM - 1), dtype=np.float32).reshape(5, STATE_DIM - 1)
    base_arrays = PreprocessedArrays(
        ohlc=np.asarray(
            [
                [100.0, 101.0, 99.0, 100.0],
                [101.0, 103.0, 100.0, 102.0],
                [102.0, 104.0, 101.0, 103.0],
                [103.0, 105.0, 102.0, 104.0],
                [104.0, 106.0, 103.0, 105.0],
            ],
            dtype=np.float32,
        ),
        static_features=static_features,
        atr=np.ones((5,), dtype=np.float32),
        day_ids=np.zeros((5,), dtype=np.int32),
        bar_in_day=np.arange(5, dtype=np.int32),
        session_end_mask=np.zeros((5,), dtype=bool),
    )

    _, state = env_reset(build_env_param(config, base_arrays), global_index=1)
    action = jnp.asarray([1, 2], dtype=jnp.int32)

    base_step = env_step(state, action, build_env_param(config, base_arrays), config.reward)

    future_mutated_arrays = PreprocessedArrays(
        ohlc=base_arrays.ohlc.copy(),
        static_features=base_arrays.static_features.copy(),
        atr=base_arrays.atr.copy(),
        day_ids=base_arrays.day_ids.copy(),
        bar_in_day=base_arrays.bar_in_day.copy(),
        session_end_mask=base_arrays.session_end_mask.copy(),
    )
    future_mutated_arrays.ohlc[4] = np.asarray([999.0, 999.0, 999.0, 999.0], dtype=np.float32)
    future_mutated_arrays.static_features[4] = 999.0

    future_step = env_step(
        state,
        action,
        build_env_param(config, future_mutated_arrays),
        config.reward,
    )

    np.testing.assert_allclose(np.asarray(base_step[0]), np.asarray(future_step[0]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(base_step[2]), np.asarray(future_step[2]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(base_step[4]["pnl"]),
        np.asarray(future_step[4]["pnl"]),
        rtol=1e-6,
        atol=1e-6,
    )

    next_bar_mutated_arrays = PreprocessedArrays(
        ohlc=base_arrays.ohlc.copy(),
        static_features=base_arrays.static_features.copy(),
        atr=base_arrays.atr.copy(),
        day_ids=base_arrays.day_ids.copy(),
        bar_in_day=base_arrays.bar_in_day.copy(),
        session_end_mask=base_arrays.session_end_mask.copy(),
    )
    next_bar_mutated_arrays.ohlc[2] = np.asarray([102.0, 103.0, 102.0, 102.0], dtype=np.float32)

    next_bar_step = env_step(
        state,
        action,
        build_env_param(config, next_bar_mutated_arrays),
        config.reward,
    )

    assert not np.allclose(np.asarray(base_step[2]), np.asarray(next_bar_step[2]))
    assert not np.allclose(
        np.asarray(base_step[4]["portfolio_value_after"]),
        np.asarray(next_bar_step[4]["portfolio_value_after"]),
    )
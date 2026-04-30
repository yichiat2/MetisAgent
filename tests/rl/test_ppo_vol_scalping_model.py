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


PACKAGE_NAME = "_test_ppo_vol_scalping"
PACKAGE_DIR = Path(__file__).resolve().parents[2] / "rl" / "ppo_vol_scalping"

package_module = ModuleType(PACKAGE_NAME)
package_module.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault(PACKAGE_NAME, package_module)

_load_module(f"{PACKAGE_NAME}.config", PACKAGE_DIR / "config.py")
_load_module(f"{PACKAGE_NAME}.contracts", PACKAGE_DIR / "contracts.py")
model_module = _load_module(f"{PACKAGE_NAME}.model", PACKAGE_DIR / "model.py")

get_log_prob = model_module.get_log_prob
sample_and_log_prob = model_module.sample_and_log_prob


def test_get_log_prob_normalizes_sigmoid_squashed_density_in_1d() -> None:
    mean = jnp.array([0.0], dtype=jnp.float32)
    log_std = jnp.array([0.0], dtype=jnp.float32)
    scale = jnp.array([2.0], dtype=jnp.float32)
    actions = jnp.linspace(1e-5, float(scale[0]) - 1e-5, 200_000, dtype=jnp.float32)[:, None]

    log_probs = jax.vmap(lambda action: get_log_prob(mean, log_std, action, scale))(actions)
    integral = jnp.trapezoid(jnp.exp(log_probs), actions[:, 0])

    np.testing.assert_allclose(np.asarray(integral), 1.0, rtol=1e-4, atol=1e-4)


def test_sample_and_log_prob_matches_recomputed_log_prob() -> None:
    mean = jnp.array([[0.3, -0.4, 0.1]], dtype=jnp.float32)
    log_std = jnp.array([[0.1, -0.2, -0.7]], dtype=jnp.float32)
    scale = jnp.array([2.0, 2.0, 1.0], dtype=jnp.float32)

    actions, sampled_log_prob = sample_and_log_prob(jax.random.PRNGKey(7), mean, log_std, scale)
    recomputed_log_prob = get_log_prob(mean, log_std, actions, scale)

    np.testing.assert_allclose(
        np.asarray(sampled_log_prob),
        np.asarray(recomputed_log_prob),
        rtol=1e-6,
        atol=1e-6,
    )
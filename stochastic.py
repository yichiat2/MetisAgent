from __future__ import annotations

from abc import abstractmethod
from typing import NamedTuple
from evosax.algorithms import SNES as ES
import jax
import jax.numpy as jnp
from flax.struct import dataclass
import chex
import numpy as np
import optax

from constants import _MINS_PER_DAY, _DT_MIN, _DT_OVERNIGHT, _DT_OVERWEEKEND


def make_dt_seq(num_days: int) -> np.ndarray:
  
    num_steps = _MINS_PER_DAY * num_days
    dt_out    = np.full(num_steps, _DT_MIN, dtype=np.float32)
    for day in range(1, num_days):
        index = day * _MINS_PER_DAY
        if index < num_steps:
            dt_out[index] = _DT_OVERNIGHT if day % 5 != 0 else _DT_OVERWEEKEND
    return dt_out


@dataclass
class Setting:
    popsize: int
    num_generations: int
    num_dims: int
    sigma_init: float
    dt: float
    num_particles: int
    rho_cpm: float

@dataclass
class DynSetting:
    S: chex.Array
    initial_guess: chex.Array
    dt_seq: chex.Array
    noises: chex.Array
    rs_seq: chex.Array  # RS variance observations; shape (T,) or empty (0,) if unused


# ── Parameter transform primitives ───────────────────────────────────────

def _sigmoid(x):
    return jax.nn.sigmoid(x)

def _inv_sigmoid(y):
    return jnp.log(y / (1.0 - y))

def _softplus(x):
    return jax.nn.softplus(x)

def _inv_softplus(y):
    return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(y)))


class StochasticProcessBase:
    """Base class for stochastic process models.

    Subclasses must define:
        PARAM_NAMES      : list[str]  – ordered parameter names
        PARAM_TRANSFORMS : dict       – {name: (transform, lo, hi)}

    Supported transform types:
        "sigmoid_ab"  – maps ℝ → (lo, hi) via sigmoid
        "tanh"        – maps ℝ → (lo, hi) via tanh, clipped to [lo, hi]
        "softplus"    – maps ℝ → (0, ∞), clipped to [lo, hi]
        "identity"    – no transform, clipped to [lo, hi]
    """

    PARAM_NAMES: list = []
    PARAM_TRANSFORMS: dict = {}

    def __init__(self,
                 popsize: int,
                 num_generations: int,
                 sigma_init: float,
                 dt: float,
                 num_particles: int,
                 S: np.ndarray,
                 rho_cpm: float,
                 dt_seq: np.ndarray | None = None):
        self.popsize = popsize
        self.num_generations = num_generations
        self.sigma_init = sigma_init
        self.dt = dt
        self.num_particles = num_particles
        self.rho_cpm = rho_cpm
        self.S = jnp.array(S, dtype=jnp.float32)
        self.dt_seq = np.asarray(dt_seq, dtype=np.float32) if dt_seq is not None else None

    def params_to_unconstrained(self, params: dict) -> jnp.ndarray:
        """Map constrained parameter dict → unconstrained vector."""
        vals = []
        for name in self.PARAM_NAMES:
            v = params[name]
            tfm, lo, hi = self.PARAM_TRANSFORMS[name]
            if tfm == "softplus":
                vals.append(_inv_softplus(jnp.clip(jnp.float32(v), lo + 1e-8, hi)))
            elif tfm == "tanh":
                vals.append(jnp.arctanh(jnp.clip(jnp.float32(v), lo, hi)))
            elif tfm == "sigmoid_ab":
                t = (jnp.float32(v) - lo) / (hi - lo)
                t = jnp.clip(t, 1e-6, 1.0 - 1e-6)
                vals.append(_inv_sigmoid(t))
            else:  # identity
                vals.append(jnp.float32(v))
        return jnp.array(vals)

    def unconstrained_to_params(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Map unconstrained vector → constrained parameter array."""
        vals = []
        for i, name in enumerate(self.PARAM_NAMES):
            tfm, lo, hi = self.PARAM_TRANSFORMS[name]
            v = theta[i]
            if tfm == "softplus":
                vals.append(jnp.clip(_softplus(v), lo, hi))
            elif tfm == "tanh":
                vals.append(jnp.clip(jnp.tanh(v), lo, hi))
            elif tfm == "sigmoid_ab":
                vals.append(jnp.float32(lo) + (jnp.float32(hi) - jnp.float32(lo)) * _sigmoid(v))
            else:  # identity
                vals.append(jnp.clip(v, lo, hi))
        return jnp.stack(vals)

    def fitness_function(self, x: chex.Array, setting: Setting, dsetting: DynSetting):
        carry, _ = self.loglikelihood(x, setting, dsetting)
        logllk = carry[-1]
        return -logllk

    @abstractmethod
    def loglikelihood(self, x: chex.Array, setting: Setting, dsetting: DynSetting):
        raise NotImplementedError

    @abstractmethod
    def get_noises(self, key: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError

    @abstractmethod
    def get_default_param(self, key: chex.PRNGKey):
        raise NotImplementedError

    @staticmethod
    def generator(seed: int, S0: float, length: int, dt: float, x: chex.Array):
        raise NotImplementedError

    def calibrate(self, key, setting: Setting, dsetting: DynSetting):
        init_mean = jnp.asarray(dsetting.initial_guess, dtype=jnp.float32)
        optimizer = optax.adam(learning_rate=0.01)
        strategy = ES(population_size=setting.popsize, solution=init_mean, optimizer=optimizer)
        strategy_params = strategy.default_params.replace(std_init=jnp.float32(setting.sigma_init))

        key, init_key, scan_key = jax.random.split(key, 3)
        state = strategy.init(init_key, init_mean, strategy_params)

        _rho      = jnp.float32(setting.rho_cpm)
        _sqrt1mr2 = jnp.float32(np.sqrt(1.0 - setting.rho_cpm ** 2))

        def _run_generation(carry, rng_step):
            state, noises = carry
            ask_key, tell_key, noise_key = jax.random.split(rng_step, 3)

            # CPM AR(1) noise update: xi drawn with same structure as noises
            # so that uniform columns stay statistically compatible with U[0,1).
            xi        = self.get_noises(noise_key)
            noises    = _rho * noises + _sqrt1mr2 * xi

            ds = DynSetting(
                S=dsetting.S,
                initial_guess=dsetting.initial_guess,
                dt_seq=dsetting.dt_seq,
                noises=noises,
                rs_seq=dsetting.rs_seq,
            )

            candidates, state = strategy.ask(ask_key, state, strategy_params)
            constrained_candidates = jax.vmap(self.unconstrained_to_params)(candidates)

            fitness = self.fitness_function(constrained_candidates, setting, ds)
            fitness = jnp.where(jnp.isfinite(fitness), fitness, jnp.float32(1e12))
            state, _ = strategy.tell(tell_key, candidates, fitness, state, strategy_params)

            generation = state.generation_counter - 1
            bic = 2 * state.best_fitness + setting.num_dims * jnp.log(jnp.maximum(dsetting.S.shape[0] - 1, 1))
            jax.debug.print("Generation: {x}, Fitness: {y:.2f}, Best: {z:.2f}, BIC: {a:.2f}", x=generation, y=fitness.mean(),
                            z=state.best_fitness, a=bic)

            best_param = self.unconstrained_to_params(state.best_solution)
            jax.debug.print("Parameter: {x}, Sigma: {y}", x=best_param, y=state.std)
            return (state, noises), None

        rng_steps = jax.random.split(scan_key, setting.num_generations)

        @jax.jit
        def _run_scan(state_init, noises_init, rng_keys):
            return jax.lax.scan(_run_generation, (state_init, noises_init), rng_keys)

        (state, _), _ = _run_scan(state, dsetting.noises, rng_steps)
        final_state = state
        bic = 2 * final_state.best_fitness + setting.num_dims * jnp.log(jnp.maximum(dsetting.S.shape[0] - 1, 1))
        return final_state.best_solution, bic

    def predict(self, key, x: chex.Array, h0: chex.Array, logv0: chex.Array, y0: chex.Array, setting: Setting, dsetting: DynSetting):
        pass

    def _log_prior_jax(self, theta: jnp.ndarray) -> jnp.ndarray:
        """JAX-differentiable log-prior of unconstrained vector ``theta``.

        Returns the sum of log-Jacobians of the parameter transforms,
        implementing a uniform prior on the constrained support.
        """
        log_p = jnp.float32(0.0)
        for i, name in enumerate(self.PARAM_NAMES):
            tfm, lo, hi = self.PARAM_TRANSFORMS[name]
            t = theta[i]
            if tfm == "sigmoid_ab":
                s = jax.nn.sigmoid(t)
                log_jac = (
                    jnp.log(jnp.float32(hi - lo))
                    + jnp.log(s + jnp.float32(1e-30))
                    + jnp.log(jnp.float32(1.0) - s + jnp.float32(1e-30))
                )
            elif tfm == "tanh":
                log_jac = jnp.log(jnp.float32(1.0) - jnp.tanh(t) ** 2 + jnp.float32(1e-30))
            elif tfm == "softplus":
                log_jac = -jnp.log1p(jnp.exp(-t))
            else:  # identity
                log_jac = jnp.float32(0.0)
            log_p = log_p + log_jac
        return log_p



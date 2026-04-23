"""Quadratic Rough Heston+ (QRH+) process in the SV-MM architecture.

Parameters (7-dim vector, ordered):
    x[0]  a0        – quadratic variance floor        a0 ∈ [0, 0.1)
    x[1]  a1        – quadratic coefficient           a1 ∈ (0.01, 2.0)
    x[2]  a2        – quadratic center                a2 ∈ (-2, 2)
    x[3]  rho       – spot-vol correlation            rho ∈ (-0.999, 0.999)
    x[4]  alpha     – roughness exponent              alpha ∈ (-0.499, -0.01)
    x[5]  sigma_obs – observation noise std           sigma_obs ∈ (1e-6, 0.01)
    x[6]  lambda_ov – overnight truncation factor     lambda_ov ∈ (0.1, 0.5)

Variance model:
    V(x) = a0 + a1 * (x - a2)²
    a0 = V_min
    a1 = (theta - V_min) / a2²  
    E[V] = theta + a1 * Var(X)  (Var(X) depends on alpha and the BM kernel coefficients)

State (Rømer multi-factor scheme):
    particles: (N, m+1)  — columns 0..m-1 = OU factors U^j, column m = X

Data layout (same as InhomoHestonProcess):
    Each trading day contributes (390 + 1) = 391 steps.
    Steps 0–389 within each day: dt = 1 min  (intraday)
    Step  390 within each day:   dt = 1050 min (overnight gap)
    Total steps per num_days: (390+1)*num_days

Overnight effective dt:
    n_sub = round(lambda_ov * 1050)
    dt_eff = n_sub * _DT_MIN
    (One Euler step of length dt_eff approximates the overnight period.)
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import chex
import numpy as np

from stochastic import StochasticProcessBase, Setting, DynSetting, make_dt_seq
from helper import (
    FilterInfo,
    _gaussian_logpdf,
    _systematic_resample,
    _effective_sample_size,
)
from bm_kernel import precompute_bm_table, get_nearest_bm, singular_consts_np
from constants import _MINS_PER_DAY, _DT_MIN, _DT_OVERNIGHT, _OVERNIGHT_MINS

_NUM_FACTORS_DEFAULT = 8
_BM_ALPHA_MIN        = -0.499
_BM_ALPHA_MAX        = -0.01
_BM_ALPHA_STEP       = 0.001
_BM_T_MAX_DAYS       = 60          # kernel fit horizon: 60 trading days
_BM_N_GRID           = 25000       # grid points per alpha for the BM fit
_DEFAULT_BM_CACHE_DIR = os.path.join(os.path.dirname(__file__), "logs", "bm_tables")

# ---------------------------------------------------------------------------
# JAX-pure QRH+ model functions
# ---------------------------------------------------------------------------

def _variance_fn(
    x: jnp.ndarray,
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
) -> jnp.ndarray:
    """V(x) = a0 + a1*(x-a2)².  Floored at 1e-8 for numerical safety."""
    return jnp.maximum(a0 + a1 * (x - a2) ** 2, jnp.float32(1e-8))


def _singular_increment(
    dW: jnp.ndarray,
    eps_eta: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: jnp.ndarray,
) -> jnp.ndarray:
    """Rømer hybrid singular increment  δX = beta*δW + std_eta*ε_η.

    beta    = dt^α / Γ(2+α)
    std_eta = sqrt(Var(δX) - Cov(δX,δW)²/dt)

    All computations are JAX-traced, compatible with jit and vmap.
    """
    G1 = jnp.exp(jsp.gammaln(jnp.float32(1.0) + alpha))
    G2 = jnp.exp(jsp.gammaln(jnp.float32(2.0) + alpha))
    beta    = dt ** alpha / G2
    var_eta = (
        dt ** (jnp.float32(2.0) * alpha + jnp.float32(1.0))
        / ((jnp.float32(2.0) * alpha + jnp.float32(1.0)) * G1 ** 2)
        - dt ** (jnp.float32(2.0) * alpha + jnp.float32(1.0)) / G2 ** 2
    )
    std_eta = jnp.sqrt(jnp.maximum(var_eta, jnp.float32(0.0)))
    return beta * dW + std_eta * eps_eta


def _predict_obs(
    u: jnp.ndarray,       # (m,) OU factors at time k-1
    x: jnp.ndarray,       # ()   X state at time k-1
    c: jnp.ndarray,       # (m,)
    gamma: jnp.ndarray,   # (m,)
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
    sigma_obs: jnp.ndarray,
    dt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """APF first-stage: predictive mean and variance via mean transition (dW=0).

    Mean transition sets dW=0 ⟹ delta_x=0 ⟹ x_hat = smooth part only.

    Returns
    -------
    mu        : predictive mean  = -¼(V_prev+V_hat)*dt
    sigma2    : predictive var   = V_prev*dt + sigma_obs²  (marginal, not conditional)
    """
    V_prev = _variance_fn(x, a0, a1, a2)
    x_hat  = jnp.sum(c * jnp.exp(-gamma * dt) * u)          # (,) smooth transition
    V_hat  = _variance_fn(x_hat, a0, a1, a2)
    mu     = -jnp.float32(0.25) * (V_prev + V_hat) * dt
    sigma2 = V_prev * dt + sigma_obs ** 2                    # marginal variance
    return mu, sigma2


def _propagate_one_det(
    u: jnp.ndarray,          # (m,) OU factors
    x: jnp.ndarray,          # ()   X state
    dW_raw: jnp.ndarray,     # ()   N(0,1) to be scaled by sqrt(dt)
    eps_eta_raw: jnp.ndarray,# ()   N(0,1) for singular increment
    c: jnp.ndarray,
    gamma: jnp.ndarray,
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
    rho: jnp.ndarray,
    sigma_obs: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Deterministic propagation using pre-drawn normals."""
    dW      = dW_raw * jnp.sqrt(dt)
    eps_eta = eps_eta_raw

    V_prev = _variance_fn(x, a0, a1, a2)
    sqrt_V = jnp.sqrt(V_prev)

    u_new   = (u + sqrt_V * dW) / (jnp.float32(1.0) + gamma * dt)
    delta_x = _singular_increment(dW, eps_eta, alpha, dt)
    smooth  = jnp.sum(c * jnp.exp(-gamma * dt) * u)
    x_new   = smooth + sqrt_V * delta_x

    V_new = _variance_fn(x_new, a0, a1, a2)
    obs_mean = -jnp.float32(0.25) * (V_prev + V_new) * dt + rho * sqrt_V * dW
    obs_var  = (jnp.float32(1.0) - rho ** 2) * V_prev * dt + sigma_obs ** 2
    return u_new, x_new, obs_mean, obs_var


def _propagate_one(
    u: jnp.ndarray,       # (m,) OU factors
    x: jnp.ndarray,       # ()   X state
    rng: jax.Array,       # single PRNG key
    c: jnp.ndarray,       # (m,)
    gamma: jnp.ndarray,   # (m,)
    a0: jnp.ndarray,
    a1: jnp.ndarray,
    a2: jnp.ndarray,
    rho: jnp.ndarray,
    sigma_obs: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Propagate one particle via the Rømer hybrid scheme.

    Returns
    -------
    u_new    : (m,)  updated OU factors
    x_new    : ()    updated X state
    obs_mean : ()    conditional obs mean   -(V_prev+V_new)/4*dt + rho*sqrt(V_prev)*dW
    obs_var  : ()    conditional obs var    (1-rho²)*V_prev*dt + sigma_obs²
    """
    rng_w, rng_eta = jax.random.split(rng)
    dW      = jax.random.normal(rng_w)   * jnp.sqrt(dt)
    eps_eta = jax.random.normal(rng_eta)

    V_prev = _variance_fn(x, a0, a1, a2)
    sqrt_V = jnp.sqrt(V_prev)

    # OU factor update: U^j_k = (U^j_{k-1} + sqrt(V_{k-1})*dW) / (1 + gamma_j*dt)
    u_new = (u + sqrt_V * dW) / (jnp.float32(1.0) + gamma * dt)

    # X update via Rømer hybrid scheme
    delta_x = _singular_increment(dW, eps_eta, alpha, dt)
    smooth  = jnp.sum(c * jnp.exp(-gamma * dt) * u)
    x_new   = smooth + sqrt_V * delta_x

    V_new = _variance_fn(x_new, a0, a1, a2)

    # Conditional observation mean and variance (sigma_obs included)
    obs_mean = -jnp.float32(0.25) * (V_prev + V_new) * dt + rho * sqrt_V * dW
    obs_var  = (jnp.float32(1.0) - rho ** 2) * V_prev * dt + sigma_obs ** 2

    return u_new, x_new, obs_mean, obs_var


# ---------------------------------------------------------------------------
# QRHProcess
# ---------------------------------------------------------------------------

class QRHProcess(StochasticProcessBase):
    """Quadratic Rough Heston+ process with time-inhomogeneous overnight gap.

    Parameters : [a0, a1, a2, rho, alpha, sigma_obs, lambda_ov]  (7-dim)
    State       : particles (N, m+1)  — OU factors + X state
    """

    PARAM_NAMES = ["a0", "a1", "a2", "rho", "alpha", "sigma_obs", "lambda_ov"]
    PARAM_TRANSFORMS = {
        "a0":        ("sigmoid_ab",   0.0,    0.1),
        "a1":        ("sigmoid_ab",   0.01,   2.0),
        "a2":        ("sigmoid_ab", -2.0,   2.0),
        "rho":       ("tanh",      -0.999,  0.999),
        "alpha":     ("sigmoid_ab", -0.499, -0.01),
        "sigma_obs": ("softplus",   1e-6,   0.01),
        "lambda_ov": ("sigmoid_ab", 0.1,    0.5),
    }

    def __init__(
        self,
        popsize: int,
        num_generations: int,
        sigma_init: float,
        dt: float,
        num_particles: int,
        S: np.ndarray,
        num_factors: int = _NUM_FACTORS_DEFAULT,
        bm_cache_dir: str | None = None,
        bm_num_workers: int | None = None,
    ):
        super().__init__(popsize, num_generations, sigma_init, dt, num_particles, S)
        self.m = num_factors

        cache_dir  = bm_cache_dir or _DEFAULT_BM_CACHE_DIR
        n_workers  = bm_num_workers if bm_num_workers is not None else (os.cpu_count() or 4)
        T_max      = _BM_T_MAX_DAYS / 252.0   # 60 trading days in year-fractions

        print(
            f"[QRHProcess] Precomputing Beylkin-Monzón table "
            f"(m={num_factors}, delta={_DT_MIN:.3e}, T={T_max:.4f} yr, "
            f"n_grid={_BM_N_GRID}, workers={n_workers}) …"
        )
        self.alpha_grid, self.c_table, self.gamma_table = precompute_bm_table(
            m=num_factors,
            delta=float(_DT_MIN),
            T=T_max,
            alpha_min=_BM_ALPHA_MIN,
            alpha_max=_BM_ALPHA_MAX,
            step=_BM_ALPHA_STEP,
            n_grid=_BM_N_GRID,
            num_workers=n_workers,
            cache_dir=cache_dir,
        )
        print("[QRHProcess] BM table ready.")

    def get_noises(self, key: chex.PRNGKey) -> chex.Array:
        """Pre-generate all particle filter noise for Common Random Numbers.

        Returns an array of shape (T, 2N+2) where:
            columns   0..N-1  : N(0,1) Brownian increment noise dW_raw
            columns   N..2N-1 : N(0,1) singular increment noise eps_eta
            column   2N       : Uniform[0,1) for first-stage systematic resampling
            column   2N+1     : Uniform[0,1) for second-stage systematic resampling
        """
        T = self.S.shape[0] - 1
        N = self.num_particles
        key, key_dw, key_eta, key_u1, key_u2 = jax.random.split(key, 5)
        dW_raw  = jax.random.normal(key_dw,  shape=(T, N))
        eps_eta = jax.random.normal(key_eta, shape=(T, N))
        u1      = jax.random.uniform(key_u1, shape=(T, 1))
        u2      = jax.random.uniform(key_u2, shape=(T, 1))
        return jnp.concatenate([dW_raw, eps_eta, u1, u2], axis=1)  # (T, 2N+2)

    # ------------------------------------------------------------------
    # APF log-likelihood
    # ------------------------------------------------------------------

    def loglikelihood(self, x: chex.Array, setting: Setting, dsetting: DynSetting):
        # x: (P, D) — P candidates in constrained parameter space.
        P = x.shape[0]
        N = setting.num_particles
        m = self.m
        PN = P * N

        a0        = x[:, 0]
        a1        = x[:, 1]
        a2        = x[:, 2]
        rho       = x[:, 3]
        alpha     = x[:, 4]
        sigma_obs = x[:, 5]
        lambda_ov = x[:, 6]

        # Per-candidate BM table lookup: c_p (P, m), gamma_p (P, m).
        c_p, gamma_p = jax.vmap(
            lambda al: get_nearest_bm(al, self.alpha_grid, self.c_table, self.gamma_table)
        )(alpha)

        # Flatten per-candidate params to (P*N,) / (P*N, m) for per-particle vmap.
        a0_pn        = jnp.repeat(a0,        N)
        a1_pn        = jnp.repeat(a1,        N)
        a2_pn        = jnp.repeat(a2,        N)
        rho_pn       = jnp.repeat(rho,       N)
        alpha_pn     = jnp.repeat(alpha,     N)
        sigma_obs_pn = jnp.repeat(sigma_obs, N)
        c_pn         = jnp.repeat(c_p,     N, axis=0)   # (P*N, m)
        gamma_pn     = jnp.repeat(gamma_p, N, axis=0)   # (P*N, m)

        log_prices  = jnp.log(dsetting.S)
        log_returns = log_prices[1:] - log_prices[:-1]  # (T,)
        dt_seq      = dsetting.dt_seq                    # (T,)
        noises_seq  = dsetting.noises                    # (T, 2N+2)  CRN

        base_dt = jnp.float32(_DT_MIN)
        ov_mins = jnp.float32(_OVERNIGHT_MINS)

        # Initial state: particles (P*N, m+1), weights (P, N), total_loglik (P,).
        particles    = jnp.zeros((PN, m + 1), dtype=jnp.float32)
        log_weights  = jnp.full((P, N), -jnp.log(jnp.float32(N)))
        total_loglik = jnp.zeros((P,), dtype=jnp.float32)

        def _apf_step(carry, xs):
            obs, dt_i, noises_t = xs
            particles_pn, log_weights, total_loglik = carry  # (P*N, m+1), (P, N), (P,)

            dW_raw_t  = noises_t[:N]       # (N,)  CRN
            eps_eta_t = noises_t[N:2*N]    # (N,)  CRN
            u1_t      = noises_t[2*N]      # scalar
            u2_t      = noises_t[2*N + 1]  # scalar

            dW_raw_pn  = jnp.tile(dW_raw_t,  P)   # (P*N,)
            eps_eta_pn = jnp.tile(eps_eta_t, P)    # (P*N,)

            # Per-candidate effective dt.
            is_overnight = dt_i > jnp.float32(1.5) * base_dt
            n_sub_p      = jnp.maximum(jnp.round(ov_mins * lambda_ov), jnp.float32(1.0))  # (P,)
            dt_eff_p     = jnp.where(is_overnight, n_sub_p * base_dt, base_dt)             # (P,)
            dt_eff_pn    = jnp.repeat(dt_eff_p, N)                                         # (P*N,)

            u_all_pn = particles_pn[:, :m]   # (P*N, m)
            x_all_pn = particles_pn[:, m]    # (P*N,)

            # ── Stage 1: pilot — vmap over P*N ───────────────────────────
            mu_hats_pn, sigma2_hats_pn = jax.vmap(
                lambda ui, xi, ci, gi, a0i, a1i, a2i, soi, dti:
                    _predict_obs(ui, xi, ci, gi, a0i, a1i, a2i, soi, dti)
            )(u_all_pn, x_all_pn, c_pn, gamma_pn, a0_pn, a1_pn, a2_pn, sigma_obs_pn, dt_eff_pn)

            log_g_pn = _gaussian_logpdf(obs, mu_hats_pn, sigma2_hats_pn)   # (P*N,)
            log_g    = log_g_pn.reshape(P, N)                               # (P, N)

            # ── Stage 2: first-stage aggregation — vmap over P ───────────
            log_xi     = log_weights + log_g                                                         # (P, N)
            log_Z1     = jax.vmap(jax.nn.logsumexp)(log_xi) - jnp.log(jnp.float32(N))              # (P,)
            ancestors1 = jax.vmap(lambda lw: _systematic_resample(lw, u1_t))(log_xi)               # (P, N)

            particles_3d   = particles_pn.reshape(P, N, m + 1)
            mu_hats_2d     = mu_hats_pn.reshape(P, N)
            sigma2_hats_2d = sigma2_hats_pn.reshape(P, N)

            parts_anc = jax.vmap(lambda par, a: par[a])(particles_3d, ancestors1)   # (P, N, m+1)
            mu_anc    = jax.vmap(lambda mh,  a: mh[a])(mu_hats_2d,   ancestors1)   # (P, N)
            s2_anc    = jax.vmap(lambda s2,  a: s2[a])(sigma2_hats_2d, ancestors1) # (P, N)

            u_par_pn  = parts_anc[:, :, :m].reshape(PN, m)   # (P*N, m)
            x_par_pn  = parts_anc[:, :,  m].reshape(PN)       # (P*N,)
            mu_anc_pn = mu_anc.reshape(PN)
            s2_anc_pn = s2_anc.reshape(PN)

            # ── Stage 3: propagation — vmap over P*N ─────────────────────
            new_u_pn, new_x_pn, exact_means_pn, exact_vars_pn = jax.vmap(
                lambda ui, xi, dwi, eei, ci, gi, a0i, a1i, a2i, rhi, soi, ali, dti:
                    _propagate_one_det(ui, xi, dwi, eei, ci, gi, a0i, a1i, a2i, rhi, soi, ali, dti)
            )(u_par_pn, x_par_pn, dW_raw_pn, eps_eta_pn,
              c_pn, gamma_pn, a0_pn, a1_pn, a2_pn, rho_pn, sigma_obs_pn, alpha_pn, dt_eff_pn)

            new_particles_pn = jnp.concatenate([new_u_pn, new_x_pn[:, None]], axis=-1)   # (P*N, m+1)

            # ── Stage 4: second-stage aggregation — vmap over P ──────────
            log_num_pn   = _gaussian_logpdf(obs, exact_means_pn, exact_vars_pn)   # (P*N,)
            log_den_pn   = _gaussian_logpdf(obs, mu_anc_pn, s2_anc_pn)            # (P*N,)
            log_alpha_pn = (log_num_pn - log_den_pn).reshape(P, N)                # (P, N)

            log_Z2 = jax.vmap(
                lambda la: jax.nn.logsumexp(la) - jnp.log(jnp.float32(N))
            )(log_alpha_pn)                                                                        # (P,)

            log_increment = log_Z1 + log_Z2
            total_loglik  = total_loglik + log_increment

            # ── Diagnostics: mean over P ──────────────────────────────────
            w_norm   = jax.nn.softmax(log_alpha_pn, axis=-1)                                      # (P, N)
            v_new_pn = jax.vmap(
                lambda xi, a0i, a1i, a2i: _variance_fn(xi, a0i, a1i, a2i)
            )(new_x_pn, a0_pn, a1_pn, a2_pn)                                                      # (P*N,)
            v_new     = v_new_pn.reshape(P, N)
            filt_mean = jnp.sum(w_norm * v_new, axis=-1)                                          # (P,)
            filt_std  = jnp.sqrt(
                jnp.maximum(jnp.sum(w_norm * (v_new - filt_mean[:, None]) ** 2, axis=-1), jnp.float32(0.0))
            )
            ess = jax.vmap(_effective_sample_size)(log_alpha_pn)                                  # (P,)

            # ── Second resample: vmap over P ──────────────────────────────
            new_parts_3d = new_particles_pn.reshape(P, N, m + 1)
            ancestors2   = jax.vmap(lambda la: _systematic_resample(la, u2_t))(log_alpha_pn)      # (P, N)
            parts_new    = jax.vmap(lambda par, a: par[a])(new_parts_3d, ancestors2)               # (P, N, m+1)
            log_weights_new = jnp.full((P, N), -jnp.log(jnp.float32(N)))

            # ── One-step-ahead prediction: mean over P*N ──────────────────
            x_res_pn = parts_new[:, :, m].reshape(PN)
            u_res_pn = parts_new[:, :, :m].reshape(PN, m)
            pred_mus_pn, pred_sigma2s_pn = jax.vmap(
                lambda ui, xi, ci, gi, a0i, a1i, a2i, soi:
                    _predict_obs(ui, xi, ci, gi, a0i, a1i, a2i, soi, base_dt)
            )(u_res_pn, x_res_pn, c_pn, gamma_pn, a0_pn, a1_pn, a2_pn, sigma_obs_pn)
            pred_lr_mean = jnp.mean(pred_mus_pn)
            pred_lr_std  = jnp.sqrt(
                jnp.mean(pred_sigma2s_pn)
                + jnp.mean((pred_mus_pn - pred_lr_mean) ** 2)
            )

            new_carry = (parts_new.reshape(PN, m + 1), log_weights_new, total_loglik)
            return new_carry, (
                filt_mean.mean(), filt_std.mean(), ess.mean(),
                log_increment.mean(), pred_lr_mean, pred_lr_std,
            )

        init_carry = (particles, log_weights, total_loglik)
        final_carry, (filt_means, filt_stds, ess_seq, loglik_incs, pred_means, pred_stds) = \
            jax.lax.scan(_apf_step, init_carry, (log_returns, dt_seq, noises_seq))

        return final_carry, FilterInfo(
            filtered_mean=filt_means,
            filtered_std=filt_stds,
            ess=ess_seq,
            loglik_increments=loglik_incs,
            pred_log_return_mean=pred_means,
            pred_log_return_std=pred_stds,
        )

    # ------------------------------------------------------------------
    # Default parameters
    # ------------------------------------------------------------------

    def get_default_param(self, key: chex.PRNGKey):
        initial_guess = {
            'a0':        0.01,
            'a1':        0.3,
            'a2':        0.1,
            'rho':       0.0,
            'alpha':     -0.1,
            'sigma_obs': 1e-4,
            'lambda_ov': 0.3,
        }

        num_dims      = len(initial_guess)
        initial_guess_unconstrained = self.params_to_unconstrained(initial_guess)

        dt_seq = jnp.array(
            make_dt_seq((self.S.shape[0] - 1) // _MINS_PER_DAY), dtype=jnp.float32
        )
        noises = self.get_noises(key)

        dsetting = DynSetting(
            S=self.S,
            initial_guess=initial_guess_unconstrained,
            dt_seq=dt_seq,
            noises=noises,
            rs_seq=jnp.empty((0,), dtype=jnp.float32),
        )
        setting = Setting(
            popsize=self.popsize,
            num_generations=self.num_generations,
            num_dims=num_dims,
            sigma_init=self.sigma_init,
            dt=self.dt,
            num_particles=self.num_particles,
        )
        return setting, dsetting

    # ------------------------------------------------------------------
    # Synthetic data generator
    # ------------------------------------------------------------------

    @staticmethod
    def generator(
        seed: int,
        S0: float,
        num_days: int,
        params: np.ndarray,
        num_factors: int = _NUM_FACTORS_DEFAULT,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic QRH+ path and return observations and variance.

        Data layout: (390 + 1) * num_days steps in total.  Each day consists
        of 390 intraday steps at dt = 1 min followed by 1 overnight step with
        effective dt = round(lambda_ov * 1050) * _DT_MIN.

        Parameters
        ----------
        seed        : RNG seed
        S0          : initial price (> 0)
        num_days    : number of trading days to simulate
        params      : (7,) array [a0, a1, a2, rho, alpha, sigma_obs, lambda_ov]
        num_factors : number of Beylkin-Monzón OU factors

        Returns
        -------
        log_returns : (num_days * 391,) float32
        variances   : (num_days * 391,) float32   true variance V(X_k)
        """
        if num_days < 1:
            raise ValueError("num_days must be >= 1")
        if S0 <= 0.0:
            raise ValueError("S0 must be positive")

        params = np.asarray(params, dtype=np.float64)
        if params.shape != (7,):
            raise ValueError(
                "params must have shape (7,): [a0, a1, a2, rho, alpha, sigma_obs, lambda_ov]"
            )

        a0, a1, a2, rho, alpha, sigma_obs, lambda_ov = params
        sqrt_one_minus_rho_sq = np.sqrt(max(1.0 - rho ** 2, 1e-8))
        rng = np.random.default_rng(seed)
        m = num_factors

        base_dt    = float(_DT_MIN)
        T_max      = _BM_T_MAX_DAYS / 252.0
        n_sub_ov   = max(1, round(float(_OVERNIGHT_MINS) * lambda_ov))
        dt_ov_eff  = n_sub_ov * base_dt

        # Precompute kernel coefficients and singular increment constants
        from bm_kernel import fit_exponential_sum
        c_np, gamma_np = fit_exponential_sum(
            float(alpha), m, delta=base_dt, T=T_max, n_grid=500
        )
        c_np     = np.asarray(c_np,     dtype=np.float64)
        gamma_np = np.asarray(gamma_np, dtype=np.float64)

        sc_intra = singular_consts_np(float(alpha), base_dt)
        sc_ov    = singular_consts_np(float(alpha), dt_ov_eff)

        length      = (_MINS_PER_DAY + 1) * num_days
        u_arr       = np.zeros((length + 1, m),  dtype=np.float64)
        X_arr       = np.zeros(length + 1,       dtype=np.float64)
        V_arr       = np.zeros(length + 1,       dtype=np.float64)
        log_returns = np.zeros(length,           dtype=np.float64)
        variances   = np.zeros(length,           dtype=np.float64)

        # Initial state: X_0 = 0, u_0 = 0
        X_arr[0] = np.sum(c_np * u_arr[0])
        V_arr[0] = max(a0 + a1 * (X_arr[0] - a2) ** 2, 1e-8)

        for step in range(length):
            in_block     = step % (_MINS_PER_DAY + 1)
            is_overnight = (in_block == _MINS_PER_DAY)

            dt  = dt_ov_eff if is_overnight else base_dt
            sc  = sc_ov     if is_overnight else sc_intra

            V_prev  = max(V_arr[step], 1e-8)
            sqrt_V  = np.sqrt(V_prev)

            dW      = rng.normal() * np.sqrt(dt)
            eps_eta = rng.normal()
            dW_perp = rng.normal() * np.sqrt(dt)

            # OU factor update
            u_arr[step + 1] = (u_arr[step] + sqrt_V * dW) / (1.0 + gamma_np * dt)

            # X update via Rømer hybrid scheme
            delta_x     = sc["beta"] * dW + sc["std_eta"] * eps_eta
            smooth      = np.sum(c_np * np.exp(-gamma_np * dt) * u_arr[step])
            X_arr[step + 1] = smooth + sqrt_V * delta_x
            V_arr[step + 1] = max(a0 + a1 * (X_arr[step + 1] - a2) ** 2, 1e-8)

            # Log-return with observation noise
            lr = (
                -0.25 * (V_prev + V_arr[step + 1]) * dt
                + sqrt_V * (rho * dW + sqrt_one_minus_rho_sq * dW_perp)
            )
            lr += rng.normal() * sigma_obs

            log_returns[step] = lr
            variances[step]   = V_arr[step + 1]

        return log_returns.astype(np.float32), variances.astype(np.float32)

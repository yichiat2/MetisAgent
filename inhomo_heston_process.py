"""Inhomogeneous Heston process.

Identical to the standard Heston APF except for the overnight interval
(dt = 1050 minutes expressed in year-fraction).  During that interval the
variance state is propagated using ``round(1050 * lambda_ov)`` equal
sub-steps so that the overnight vol-of-vol effect can be modulated
independently from the intraday dynamics.

Parameters (7-dim vector):
    x[0]  v0        – initial variance
    x[1]  rho       – price-vol diffusion correlation
    x[2]  kappa     – mean-reversion speed
    x[3]  theta     – long-run variance
    x[4]  sigma     – vol of vol
    x[5]  r         – risk-free drift
    x[6]  lambda_ov – overnight sub-step intensity   (0.01 < lambda < 0.5)
    x[7]  alpha_rs  – Rogers-Satchell shape parameter (0.5 < alpha < 20.0)

Data layout:
    Each trading day produces (390 + 1) steps.  Steps 0-389 within each day
    have dt = 1 minute; step 390 has dt = 1050 minutes (overnight gap).
    Generator call: length = (390 + 1) * num_days.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex
import numpy as np

from constants import _MINS_PER_DAY, _DT_MIN, _DT_OVERNIGHT, _OVERNIGHT_MINS
from stochastic import StochasticProcessBase, Setting, DynSetting, make_dt_seq
from helper import (
    VARIANCE_FLOOR,
    FilterInfo,
    _positive_variance,
    _gaussian_logpdf,
    _gamma_logpdf,
    _systematic_resample,
    _effective_sample_size,
)

class InhomoHestonProcess(StochasticProcessBase):
    """Heston model with sub-stepped overnight variance propagation."""

    PARAM_NAMES = ["v0", "rho", "kappa", "theta", "sigma", "r", "lambda_ov", "alpha_rs"]
    PARAM_TRANSFORMS = {
        "v0":        ("sigmoid_ab", 0.01,  0.1),
        "rho":       ("tanh",      -0.99,  0.99),
        "kappa":     ("sigmoid_ab", 0.1,  10.0),
        "theta":     ("sigmoid_ab", 0.01,  1.0),
        "sigma":     ("sigmoid_ab", 0.1,  10.0),
        "r":         ("sigmoid_ab", -0.00001, 0.00001),
        "lambda_ov": ("sigmoid_ab", 0.01,  0.5),
        "alpha_rs":  ("sigmoid_ab", 0.5,  20.0),
    }

    def __init__(
        self,
        popsize: int,
        num_generations: int,
        sigma_init: float,
        dt: float,
        num_particles: int,
        S: np.ndarray,
        rho_cpm: float = 0.0,
        rs_data: np.ndarray | None = None,
    ):
        super().__init__(popsize, num_generations, sigma_init, dt, num_particles, S, rho_cpm)
        # rs_data: shape (T,) Rogers-Satchell variance observations aligned with S[1:].
        # None is replaced by a zeros placeholder so the likelihood is dominated by
        # the log-return observation (Gamma scale becomes degenerate; the term still
        # evaluates but has near-zero contribution relative to the return term).
        T = S.shape[0] - 1
        if rs_data is not None:
            self._rs_data = jnp.array(rs_data, dtype=jnp.float32)
        else:
            self._rs_data = jnp.full((T,), fill_value=1e-6, dtype=jnp.float32)

    # ------------------------------------------------------------------
    # APF log-likelihood
    # ------------------------------------------------------------------

    def loglikelihood(self, x: chex.Array, setting: Setting, dsetting: DynSetting):
        # x: (P, D) — P candidates in constrained parameter space.
        P = x.shape[0]
        N = setting.num_particles
        PN = P * N

        v0        = x[:, 0]
        rho       = x[:, 1]
        kappa     = x[:, 2]
        theta     = x[:, 3]
        sigma     = x[:, 4]
        r_val     = x[:, 5]
        lambda_ov = x[:, 6]
        alpha_rs  = x[:, 7]

        # Flatten per-candidate params to (P*N,) for per-particle vmap.
        kappa_pn = jnp.repeat(kappa, N)
        theta_pn = jnp.repeat(theta, N)
        sigma_pn = jnp.repeat(sigma, N)
        rho_pn   = jnp.repeat(rho,   N)
        r_pn     = jnp.repeat(r_val, N)
        alpha_rs_pn = jnp.repeat(alpha_rs, N)    # (P*N,)

        log_prices  = jnp.log(dsetting.S)
        log_returns = log_prices[1:] - log_prices[:-1]  # (T,)
        dt_seq      = dsetting.dt_seq                    # (T,)
        noises_seq  = dsetting.noises                    # (T, N+2)  CRN
        rs_seq      = dsetting.rs_seq                    # (T,)

        base_dt  = jnp.float32(_DT_MIN)
        ov_mins  = jnp.float32(_OVERNIGHT_MINS)
        

        # Initial state: particles (P*N,), total_loglik (P,).
        # log_weights is omitted from carry: after every step they are reset to
        # uniform (-log N), so there is no information to propagate.
        particles    = jnp.repeat(v0, N)
        total_loglik = jnp.zeros((P,), dtype=jnp.float32)
        log_N        = jnp.log(jnp.float32(N))

        def _apf_step(carry, xs):
            obs, dt_i, noises_t, rs_obs = xs
            particles_pn, total_loglik = carry

            eps_v_t = noises_t[:N]       # (N,)  CRN
            u1_t    = noises_t[N]        # scalar
            u2_t    = noises_t[N + 1]    # scalar
            eps_pn  = jnp.tile(eps_v_t, P)  # (P*N,)

            # Per-candidate effective dt (depends on lambda_ov).
            is_overnight = dt_i > jnp.float32(1.5) * base_dt
            n_sub_p      = jnp.maximum(jnp.round(ov_mins * lambda_ov), jnp.float32(1.0))  # (P,)
            dt_eff_p     = jnp.where(is_overnight, n_sub_p * base_dt, dt_i)                # (P,)
            dt_eff_pn    = jnp.repeat(dt_eff_p, N)                                         # (P*N,)

            # ── Pilot: Gaussian (return) + Gamma (RS) ────────────────────
            def _pilot_one(v, k, th, r, dt_e, a_rs):
                vp         = _positive_variance(v + k * (th - v) * dt_e)
                log_g_ret  = _gaussian_logpdf(obs, (r - jnp.float32(0.5) * vp) * dt_e, vp * dt_e)
                rs_scale   = _positive_variance(vp * dt_e / a_rs)
                log_g_rs   = _gamma_logpdf(rs_obs, a_rs, rs_scale)
                return log_g_ret + log_g_rs

            log_g_pn = jax.vmap(_pilot_one)(particles_pn, kappa_pn, theta_pn, r_pn, dt_eff_pn, alpha_rs_pn)
            log_g    = log_g_pn.reshape(P, N)

            # ── First-stage: vmap over P ──────────────────────────────────
            log_Z1     = jax.vmap(jax.nn.logsumexp)(log_g) - log_N                                                         # (P,)
            ancestors1 = jax.vmap(lambda w: _systematic_resample(w, u1_t))(log_g)             # (P, N)

            particles_2d = particles_pn.reshape(P, N)
            v_par_2d  = jax.vmap(lambda p, a: p[a])(particles_2d, ancestors1)                 # (P, N)
            log_g_sel = jax.vmap(lambda lg, a: lg[a])(log_g, ancestors1)                      # (P, N)

            # ── Propagation: vmap over P*N ────────────────────────────────
            def _propagate_one(v, k, th, s, e, dt_e):
                return _positive_variance(
                    v + k * (th - v) * dt_e
                    + s * jnp.sqrt(_positive_variance(v)) * jnp.sqrt(dt_e) * e
                )

            v_next_pn = jax.vmap(_propagate_one)(
                v_par_2d.reshape(PN), kappa_pn, theta_pn, sigma_pn, eps_pn, dt_eff_pn
            )                                                                                   # (P*N,)
            v_next = v_next_pn.reshape(P, N)                                                   # (P, N)

            # ── Conditional likelihood: Gaussian (return) + Gamma (RS) ───
            def _log_p_one(v_n, r, rh, e, dt_e, a_rs):
                mu_c   = (r - jnp.float32(0.5) * v_n) * dt_e + jnp.sqrt(_positive_variance(v_n * dt_e)) * rh * e
                sig2_c = _positive_variance(v_n * (jnp.float32(1.0) - rh ** 2) * dt_e)
                log_p_ret = _gaussian_logpdf(obs, mu_c, sig2_c)
                rs_scale  = _positive_variance(v_n * dt_e / a_rs)
                log_p_rs  = _gamma_logpdf(rs_obs, a_rs, rs_scale)
                return log_p_ret + log_p_rs

            log_p_pn = jax.vmap(_log_p_one)(v_next_pn, r_pn, rho_pn, eps_pn, dt_eff_pn, alpha_rs_pn)  # (P*N,)
            log_p    = log_p_pn.reshape(P, N)                                                  # (P, N)

            log_alpha = log_p - log_g_sel                                                      # (P, N)

            # ── Second-stage: vmap over P ─────────────────────────────────
            lse_alpha     = jax.vmap(jax.nn.logsumexp)(log_alpha)                         # (P,) – single logsumexp over P×N
            log_Z2        = lse_alpha - log_N                           # (P,)

            log_increment = log_Z1 + log_Z2
            total_loglik  = total_loglik + log_increment

            # ── Diagnostics: mean over P ──────────────────────────────────
            w_alpha   = jnp.exp(log_alpha - lse_alpha[:, None])                           # (P, N) – reuse lse_alpha, no extra softmax
            filt_mean = jnp.sum(w_alpha * v_next, axis=-1)                                # (P,)
            filt_std  = jnp.sqrt(jnp.sum(w_alpha * (v_next - filt_mean[:, None]) ** 2, axis=-1))
            log_alpha_norm = log_alpha - lse_alpha[:, None]                               # (P, N) – already normalized
            ess       = jnp.exp(-jax.vmap(jax.nn.logsumexp)(jnp.float32(2.0) * log_alpha_norm))  # (P,) – one logsumexp, skips re-normalization

            # ── Second resample: vmap over P ──────────────────────────────
            ancestors2      = jax.vmap(lambda w: _systematic_resample(w, u2_t))(log_alpha)      # (P, N)
            particles_new   = jax.vmap(lambda vn, a: vn[a])(v_next, ancestors2)               # (P, N)

            # ── One-step-ahead prediction: mean over P*N ──────────────────
            def _pred_one(v, k, th, r, dt_e):
                vp = _positive_variance(v + k * (th - v) * dt_e)
                return (r - jnp.float32(0.5) * vp) * dt_e, vp * dt_e

            pred_mean_pn, pred_var_pn = jax.vmap(_pred_one)(
                particles_new.reshape(PN), kappa_pn, theta_pn, r_pn, dt_eff_pn
            )
            pred_lr_mean = jnp.mean(pred_mean_pn)
            pred_lr_std  = jnp.sqrt(
                jnp.mean(pred_var_pn) + jnp.mean((pred_mean_pn - pred_lr_mean) ** 2)
            )

            new_carry = (particles_new.reshape(PN), total_loglik)
            return new_carry, (
                filt_mean.mean(), filt_std.mean(), ess.mean(),
                log_increment.mean(), pred_lr_mean, pred_lr_std,
            )

        init_carry = (particles, total_loglik)
        final_carry, (filt_means, filt_stds, ess_seq, loglik_incs, pred_means, pred_stds) = \
            jax.lax.scan(_apf_step, init_carry, (log_returns, dt_seq, noises_seq, rs_seq))

        return final_carry, FilterInfo(
            filtered_mean=filt_means,
            filtered_std=filt_stds,
            ess=ess_seq,
            loglik_increments=loglik_incs,
            pred_log_return_mean=pred_means,
            pred_log_return_std=pred_stds,
        )

    # ------------------------------------------------------------------
    # Noise pre-generation (Common Random Numbers)
    # ------------------------------------------------------------------

    def get_noises(self, key: chex.PRNGKey) -> chex.Array:
        """Pre-generate all particle filter noise for Common Random Numbers.

        Returns an array of shape (T, N+2) where:
            columns  0..N-1 : N(0,1) variance-propagation noise eps_v
            column   N      : Uniform[0,1) for first-stage systematic resampling
            column   N+1    : Uniform[0,1) for second-stage systematic resampling

        Using the same noise array across all parameter evaluations during
        CMA-ES calibration eliminates Monte Carlo noise as a source of fitness
        variance, leaving only the true parameter-driven signal.
        """
        T = self.S.shape[0] - 1
        N = self.num_particles
        key, key_eps, key_u1, key_u2 = jax.random.split(key, 4)
        eps_v = jax.random.normal(key_eps, shape=(T, N))          # (T, N)
        u1    = jax.random.uniform(key_u1, shape=(T, 1))          # (T, 1)
        u2    = jax.random.uniform(key_u2, shape=(T, 1))          # (T, 1)
        return jnp.concatenate([eps_v, u1, u2], axis=1)           # (T, N+2)

    # ------------------------------------------------------------------
    # Default parameters
    # ------------------------------------------------------------------

    def get_default_param(self, key: chex.PRNGKey):
        initial_guess = {
            'v0':        0.04,
            'rho':       0.0,
            'kappa':     2.0,
            'theta':     0.04,
            'sigma':     0.2,
            'r':         0.0,
            'lambda_ov': 0.1,
            'alpha_rs':  2.0,
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
            rs_seq=self._rs_data,
        )
        setting = Setting(
            popsize=self.popsize,
            num_generations=self.num_generations,
            num_dims=num_dims,
            sigma_init=self.sigma_init,
            dt=self.dt,
            num_particles=self.num_particles,
            rho_cpm=self.rho_cpm,
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
    ):
        """Generate synthetic OHLC prices, log-returns, variance paths, and
        Rogers-Satchell intrabar variance estimates.

        Data layout: (390 + 1) * num_days steps.
        Each day has 390 intraday steps at dt = 1 minute followed by
        1 overnight step at dt = 1050 minutes.

        Intrabar O, H, L are simulated via an exact Brownian-bridge
        maximum/minimum distribution (Revuz & Yor reflection principle),
        conditioning on the realised close log-return and constant intrabar
        variance v_next.  Open = previous close, Close = Open * exp(log_return).
        High = Open * exp(h),  Low = Open * exp(l) where:
            h = (c + sqrt(c² − 2·v·dt·ln U_H)) / 2,  U_H ~ Uniform(0,1)
            l = (c − sqrt(c² − 2·v·dt·ln U_L)) / 2,  U_L ~ Uniform(0,1)
        with c = log_return, v = v_next, dt as the bar width.

        The Rogers-Satchell estimator for each bar is:
            RS = (h−c)·h + (l−c)·l
        where h = ln(H/O), l = ln(L/O), c = ln(C/O).
        RS is an unbiased estimator of v·dt.

        Args:
            seed:     RNG seed.
            S0:       Initial price (> 0).
            num_days: Number of trading days.
            params:   Shape (7,) array ordered as
                      [v0, rho, kappa, theta, sigma, r, lambda_ov].

        Returns:
            Tuple (log_returns, variances, opens, highs, lows, rs_variances)
            each of shape ``((390 + 1) * num_days,)`` and dtype float32.
        """
        if num_days < 1:
            raise ValueError("num_days must be >= 1")
        if S0 <= 0.0:
            raise ValueError("S0 must be positive")

        params = np.asarray(params, dtype=np.float64)
        if params.shape != (8,):
            raise ValueError(
                "params must have shape (8,) ordered as "
                "[v0, rho, kappa, theta, sigma, r, lambda_ov, rs_alpha]"
            )

        v0, rho, kappa, theta, sigma, r, lambda_ov, rs_alpha = params
        sqrt_one_minus_rho_sq = np.sqrt(max(1.0 - rho ** 2, 1e-8))
        rng = np.random.default_rng(seed)

        length          = (_MINS_PER_DAY + 1) * num_days
        variances       = np.zeros(length, dtype=np.float64)
        log_returns     = np.zeros(length, dtype=np.float64)
        opens           = np.zeros(length, dtype=np.float64)
        highs           = np.zeros(length, dtype=np.float64)
        lows            = np.zeros(length, dtype=np.float64)
        rs_variances    = np.zeros(length, dtype=np.float64)
        variance_prev   = max(v0, VARIANCE_FLOOR)
        price_prev      = float(S0)

        n_sub   = max(1.0, round(_OVERNIGHT_MINS * lambda_ov))
        base_dt = float(_DT_MIN)

        for step in range(length):
            in_block     = step % (_MINS_PER_DAY + 1)
            is_overnight = (in_block == _MINS_PER_DAY)

            dt = n_sub * base_dt if is_overnight else base_dt

            # ── Variance propagation ─────────────────────────────────────────
            eps_v  = rng.normal()
            v_next = max(
                variance_prev
                + kappa * (theta - variance_prev) * dt
                + sigma * np.sqrt(variance_prev) * np.sqrt(dt) * eps_v,
                VARIANCE_FLOOR,
            )

            # ── Correlated price shock ───────────────────────────────────────
            eps_orthogonal   = rng.normal()
            correlated_shock = rho * eps_v + sqrt_one_minus_rho_sq * eps_orthogonal
            log_return = (
                (r - 0.5 * v_next) * dt
                + np.sqrt(v_next * dt) * correlated_shock
            )

            # ── Brownian-bridge OHLC ─────────────────────────────────────────
            # c = log(C/O) = log_return; bridge from 0 → c with variance v_next*dt
            c        = log_return
            var_dt   = max(v_next * dt, 1e-30)
            u_h      = rng.uniform()
            u_l      = rng.uniform()

            disc_h   = c * c - 2.0 * var_dt * np.log(u_h)
            disc_l   = c * c - 2.0 * var_dt * np.log(u_l)
            h        = (c + np.sqrt(max(disc_h, 0.0))) / 2.0   # log(H/O)
            l        = (c - np.sqrt(max(disc_l, 0.0))) / 2.0   # log(L/O)

            # Clamp to valid range (numerical safety)
            h = max(h, max(0.0, c))
            l = min(l, min(0.0, c))

            # ── Rogers-Satchell estimator: E[RS] = v_next * dt ──────────────
            rs = (h - c) * h + (l - c) * l

            # ── Record ───────────────────────────────────────────────────────
            variances[step]    = v_next
            log_returns[step]  = log_return
            opens[step]        = price_prev
            highs[step]        = price_prev * np.exp(h)
            lows[step]         = price_prev * np.exp(l)
            rs_variances[step] = max(rs, 1e-30)

            price_prev    = price_prev * np.exp(log_return)
            variance_prev = v_next

        return (
            log_returns.astype(np.float32),
            variances.astype(np.float32),
            opens.astype(np.float32),
            highs.astype(np.float32),
            lows.astype(np.float32),
            rs_variances.astype(np.float32),
        )

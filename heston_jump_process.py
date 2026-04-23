"""Heston-Jump process.

.. deprecated::
    ``HestonJumpProcess`` has been retired.  Use
    :class:`svlogv_jump_process.StochasticVolatilityJumpProcess` instead,
    which models the latent log-variance via an OU process with additive
    Gaussian jumps and uses a fully-adapted RB-APF for inference
    (no second resampling stage, exact particle weights).

Standard Heston stochastic volatility augmented with a correlated compound-
jump process (Bernoulli arrivals, log-normal variance jumps, Gaussian log-
return jumps).  Overnight price gaps are captured by the jump component.
The time step is provided via ``dt_seq`` and may vary per step (e.g. a
longer overnight interval).

Risk-neutral drift correction
------------------------------
The per-step compensator ``comp^t`` ensures ``E[exp(r_t)] = exp(r·dt_t)``
at every step regardless of step size::

    m_r    = exp(μ_Jr + ½σ_Jr²)       # constant — precomputed outside scan
    p_J^t  = λ_J · Δt_t               # varies per step
    comp^t = log[(1 - p_J^t) + p_J^t · m_r]  # recomputed inside scan
    drift_term = (r - ½V_t) · dt_t - comp^t

``m_r`` and ``m_V`` (the jump MGF constants) depend only on model
parameters and are precomputed outside the scan.  ``p_J^t`` and ``comp^t``
depend on the current step's ``dt_t`` and are computed inside the scan.

Parameters (12-dim vector)
---------------------------
    x[0]   v0        initial variance
    x[1]   rho       price–vol diffusion correlation
    x[2]   kappa     mean-reversion speed
    x[3]   theta     long-run variance
    x[4]   sigma     vol of vol
    x[5]   r         risk-free drift
    x[6]   lambda_J  jump intensity (jumps per year)
    x[7]   mu_Jr     mean log-return jump size
    x[8]   sigma_Jr  std dev of log-return jump
    x[9]   mu_JV     mean of log-variance multiplier (log-normal)
    x[10]  sigma_JV  std dev of log-variance multiplier
    x[11]  rho_J     correlation between log-variance jump and log-return jump

Data layout
-----------
Each trading day produces 390 steps at dt = _DT_MIN (1 min).
Total steps = 390 × num_days.  There are no special overnight dt values;
overnight price gaps are modelled by the jump process.

CRN noise layout  (T, 3N+2)
----------------------------
    cols  0 .. N-1   : N(0,1)  variance-propagation noise ε_V
    cols  N .. 2N-1  : N(0,1)  shared jump-factor noise Z₁
    cols 2N .. 3N-1  : U[0,1)  jump-indicator noise u_jump
    col   3N         : U[0,1)  first-stage  systematic-resampling uniform
    col   3N+1       : U[0,1)  second-stage systematic-resampling uniform
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex
import numpy as np

from constants import _MINS_PER_DAY, _DT_MIN
from stochastic import StochasticProcessBase, Setting, DynSetting, make_dt_seq
from helper import (
    VARIANCE_FLOOR,
    FilterInfo,
    _positive_variance,
    _gaussian_logpdf,
    _systematic_resample,
)


class HestonJumpProcess(StochasticProcessBase):
    """Heston model with correlated compound-jump process.

    A shared Bernoulli indicator ``I_t`` drives both a multiplicative
    variance jump (log-normal factor) and an additive log-return jump
    (Gaussian).  The two jump sizes share a common Gaussian factor ``Z₁``
    so their correlation ``ρ_J`` is exactly parameterised.

    Risk-neutrality is maintained via the per-step compensator::

        comp = log[(1 - λ_J·Δt) + λ_J·Δt · exp(μ_Jr + ½σ_Jr²)]

    The Auxiliary Particle Filter (APF) pilot uses a two-branch Gaussian
    mixture to pre-weight particles towards the observation.  No Rogers-
    Satchell or overnight sub-stepping terms are included.
    """

    PARAM_NAMES = [
        "v0", "rho", "kappa", "theta", "sigma", "r",
        "lambda_J", "mu_Jr", "sigma_Jr", "mu_JV", "sigma_JV", "rho_J",
    ]
    PARAM_TRANSFORMS = {
        "v0":       ("sigmoid_ab",  0.01,     0.1),
        "rho":      ("tanh",       -0.99,     0.99),
        "kappa":    ("sigmoid_ab",  0.1,     10.0),
        "theta":    ("sigmoid_ab",  0.01,     1.0),
        "sigma":    ("sigmoid_ab",  0.1,     1.0),
        "r":        ("sigmoid_ab", -0.00001,  0.00001),
        "lambda_J": ("sigmoid_ab",  1.0,   500.0),
        "mu_Jr":    ("sigmoid_ab", -0.2,      0.2),
        "sigma_Jr": ("sigmoid_ab",  1e-3,     0.3),
        "mu_JV":    ("sigmoid_ab", -0.2,      0.2),
        "sigma_JV": ("sigmoid_ab",  1e-3,     1.0),
        "rho_J":    ("tanh",       -0.99,     0.99),
    }

    # ------------------------------------------------------------------
    # APF log-likelihood
    # ------------------------------------------------------------------

    def loglikelihood(self, x: chex.Array, setting: Setting, dsetting: DynSetting):
        """Two-stage APF with jump-mixture pilot.

        Args:
            x: (P, 12) constrained parameter candidates.

        Returns:
            (carry, FilterInfo) where carry[-1] is (P,) total log-likelihoods.
        """
        P = x.shape[0]
        N = setting.num_particles
        PN = P * N

        v0       = x[:, 0]   # (P,)
        rho      = x[:, 1]
        kappa    = x[:, 2]
        theta    = x[:, 3]
        sigma    = x[:, 4]
        r_val    = x[:, 5]
        lambda_J = x[:, 6]
        mu_Jr    = x[:, 7]
        sigma_Jr = x[:, 8]
        mu_JV    = x[:, 9]
        sigma_JV = x[:, 10]
        rho_J    = x[:, 11]

        # ── Per-candidate constants broadcast to (P*N,) ──────────────────
        kappa_pn    = jnp.repeat(kappa,    N)
        theta_pn    = jnp.repeat(theta,    N)
        sigma_pn    = jnp.repeat(sigma,    N)
        rho_pn      = jnp.repeat(rho,      N)
        r_pn        = jnp.repeat(r_val,    N)
        lambda_J_pn = jnp.repeat(lambda_J, N)
        mu_Jr_pn    = jnp.repeat(mu_Jr,    N)
        sigma_Jr_pn = jnp.repeat(sigma_Jr, N)
        mu_JV_pn    = jnp.repeat(mu_JV,    N)
        sigma_JV_pn = jnp.repeat(sigma_JV, N)
        rho_J_pn    = jnp.repeat(rho_J,    N)

        # Jump MGF constants: no dt dependence → precomputed once.
        #   m_V  = E[exp(J_V)] = exp(μ_JV + ½σ_JV²)
        #   m_r  = E[exp(J_r)] = exp(μ_Jr + ½σ_Jr²)
        #   p_J  = λ_J · Δt_t  (step-dependent) → computed inside scan
        #   comp = log[(1-p_J) + p_J·m_r]       → computed inside scan
        m_V    = jnp.exp(mu_JV + jnp.float32(0.5) * sigma_JV ** 2)   # (P,)
        m_r    = jnp.exp(mu_Jr + jnp.float32(0.5) * sigma_Jr ** 2)   # (P,)
        m_V_pn = jnp.repeat(m_V, N)   # (P*N,)
        m_r_pn = jnp.repeat(m_r, N)   # (P*N,)

        log_prices  = jnp.log(dsetting.S)
        log_returns = log_prices[1:] - log_prices[:-1]   # (T,)
        dt_seq      = dsetting.dt_seq                     # (T,)
        noises_seq  = dsetting.noises                     # (T, 3N+2)

        # Initial state: particles (P*N,), total_loglik (P,).
        particles    = jnp.repeat(v0, N)
        total_loglik = jnp.zeros((P,), dtype=jnp.float32)
        log_N        = jnp.log(jnp.float32(N))

        def _apf_step(carry, xs):
            obs, dt_i, noises_t = xs
            particles_pn, total_loglik = carry

            # Split CRN noise columns.
            eps_v_t  = noises_t[:N]            # (N,) ε_V
            z1_t     = noises_t[N: 2 * N]      # (N,) Z₁
            u_jump_t = noises_t[2 * N: 3 * N]  # (N,) U for jump indicator
            u1_t     = noises_t[3 * N]          # scalar — first-stage resample
            u2_t     = noises_t[3 * N + 1]      # scalar — second-stage resample

            # Tile per-particle CRN draws to (P*N,).
            eps_pn  = jnp.tile(eps_v_t, P)
            z1_pn_t = jnp.tile(z1_t, P)
            u_j_pn  = jnp.tile(u_jump_t, P)

            # Per-step jump probability and compensator (both depend on dt_i).
            #   p_J^t  = λ_J · Δt_t   ensures Bernoulli rate scales with step size
            #   comp^t = log[(1-p_J^t) + p_J^t · m_r]  preserves risk-neutrality
            p_J_pn      = lambda_J_pn * dt_i
            comp_pn     = jnp.log(
                jnp.maximum(jnp.float32(1.0) - p_J_pn, jnp.float32(1e-30))
                + p_J_pn * m_r_pn
            )
            log_pJ_pn   = jnp.log(jnp.maximum(p_J_pn, jnp.float32(1e-30)))
            log_1mpJ_pn = jnp.log(jnp.maximum(jnp.float32(1.0) - p_J_pn, jnp.float32(1e-30)))

            # ── Pilot: two-branch Gaussian mixture ───────────────────────
            # Branch 0 (no jump): uses drift-only (ε_V → 0) variance.
            # Branch 1 (jump):    pilot variance inflated by m_V to capture
            #                     expected post-jump diffusion.
            def _pilot_one(v, k, th, r, m_v, p_j, mu_jr, sigma_jr,
                           comp_val, log_pj, log_1mpj):
                vp0    = _positive_variance(v + k * (th - v) * dt_i)
                vp1    = vp0 * m_v
                mu0    = (r - jnp.float32(0.5) * vp0) * dt_i - comp_val
                mu1    = (r - jnp.float32(0.5) * vp1) * dt_i - comp_val + mu_jr
                sig2_0 = vp0 * dt_i
                sig2_1 = vp1 * dt_i + sigma_jr ** 2
                return jnp.logaddexp(
                    log_1mpj + _gaussian_logpdf(obs, mu0, sig2_0),
                    log_pj   + _gaussian_logpdf(obs, mu1, sig2_1),
                )

            log_g_pn = jax.vmap(_pilot_one)(
                particles_pn, kappa_pn, theta_pn, r_pn,
                m_V_pn, p_J_pn, mu_Jr_pn, sigma_Jr_pn,
                comp_pn, log_pJ_pn, log_1mpJ_pn,
            )   # (P*N,)
            log_g = log_g_pn.reshape(P, N)

            # ── First-stage resampling ────────────────────────────────────
            log_Z1     = jax.vmap(jax.nn.logsumexp)(log_g) - log_N      # (P,)
            ancestors1 = jax.vmap(
                lambda w: _systematic_resample(w, u1_t)
            )(log_g)   # (P, N)

            particles_2d = particles_pn.reshape(P, N)
            v_par_2d  = jax.vmap(lambda p, a: p[a])(particles_2d, ancestors1)  # (P, N)
            log_g_sel = jax.vmap(lambda lg, a: lg[a])(log_g, ancestors1)       # (P, N)

            # ── Propagation + conditional log-likelihood ──────────────────
            # The return is conditioned on the realised ε_V and Z₁ so that
            # the Z₂ (independent jump component) is analytically marginalised.
            def _prop_logp_one(v, k, th, s, rh, r,
                               m_v, p_j, mu_jv, sigma_jv, rho_j,
                               mu_jr, sigma_jr, comp_val,
                               e, z1, u_j):
                # Variance propagation: floor diffusion, then apply exp jump.
                J_V    = mu_jv + sigma_jv * z1
                v_cont = _positive_variance(
                    v + k * (th - v) * dt_i
                    + s * jnp.sqrt(_positive_variance(v)) * jnp.sqrt(dt_i) * e
                )
                I_t    = (u_j < p_j).astype(jnp.float32)
                v_next = v_cont * jnp.exp(I_t * J_V)

                # Conditional mean: drift + correlated diffusion + jump mean.
                mu_cond = (
                    (r - jnp.float32(0.5) * v_next) * dt_i
                    - comp_val
                    + rh * jnp.sqrt(_positive_variance(v_next * dt_i)) * e
                    + I_t * (mu_jr + sigma_jr * rho_j * z1)
                )
                # Conditional variance: residual diffusion + independent jump.
                sig2_cond = _positive_variance(
                    v_next * (jnp.float32(1.0) - rh ** 2) * dt_i
                    + I_t * sigma_jr ** 2 * (jnp.float32(1.0) - rho_j ** 2)
                )
                log_p = _gaussian_logpdf(obs, mu_cond, sig2_cond)
                return v_next, log_p

            v_next_pn, log_p_pn = jax.vmap(_prop_logp_one)(
                v_par_2d.reshape(PN),
                kappa_pn, theta_pn, sigma_pn, rho_pn, r_pn,
                m_V_pn, p_J_pn, mu_JV_pn, sigma_JV_pn, rho_J_pn,
                mu_Jr_pn, sigma_Jr_pn, comp_pn,
                eps_pn, z1_pn_t, u_j_pn,
            )   # each (P*N,)
            v_next = v_next_pn.reshape(P, N)
            log_p  = log_p_pn.reshape(P, N)

            log_alpha = log_p - log_g_sel   # (P, N)

            # ── Second-stage ──────────────────────────────────────────────
            lse_alpha     = jax.vmap(jax.nn.logsumexp)(log_alpha)   # (P,)
            log_Z2        = lse_alpha - log_N
            log_increment = log_Z1 + log_Z2
            total_loglik  = total_loglik + log_increment

            # ── Diagnostics ───────────────────────────────────────────────
            w_alpha        = jnp.exp(log_alpha - lse_alpha[:, None])      # (P, N)
            filt_mean      = jnp.sum(w_alpha * v_next, axis=-1)           # (P,)
            filt_std       = jnp.sqrt(
                jnp.sum(w_alpha * (v_next - filt_mean[:, None]) ** 2, axis=-1)
            )
            log_alpha_norm = log_alpha - lse_alpha[:, None]
            ess            = jnp.exp(
                -jax.vmap(jax.nn.logsumexp)(jnp.float32(2.0) * log_alpha_norm)
            )

            # ── Second-stage resampling ───────────────────────────────────
            ancestors2    = jax.vmap(
                lambda w: _systematic_resample(w, u2_t)
            )(log_alpha)   # (P, N)
            particles_new = jax.vmap(lambda vn, a: vn[a])(v_next, ancestors2)

            # ── One-step-ahead prediction (jump-mixture approximation) ────
            def _pred_one(v, k, th, r, m_v, p_j, mu_jr, sigma_jr, comp_val):
                vp0 = _positive_variance(v + k * (th - v) * dt_i)
                vp1 = vp0 * m_v
                mu0 = (r - jnp.float32(0.5) * vp0) * dt_i - comp_val
                mu1 = (r - jnp.float32(0.5) * vp1) * dt_i - comp_val + mu_jr
                # Law of total expectation.
                bar_r = (jnp.float32(1.0) - p_j) * mu0 + p_j * mu1
                # Law of total variance.
                var_pred = (
                    (jnp.float32(1.0) - p_j) * vp0 * dt_i
                    + p_j * (vp1 * dt_i + sigma_jr ** 2)
                    + p_j * (jnp.float32(1.0) - p_j) * (mu1 - mu0) ** 2
                )
                return bar_r, var_pred

            pred_mean_pn, pred_var_pn = jax.vmap(_pred_one)(
                particles_new.reshape(PN),
                kappa_pn, theta_pn, r_pn,
                m_V_pn, p_J_pn, mu_Jr_pn, sigma_Jr_pn, comp_pn,
            )
            pred_lr_mean = jnp.mean(pred_mean_pn)
            pred_lr_std  = jnp.sqrt(
                jnp.mean(pred_var_pn)
                + jnp.mean((pred_mean_pn - pred_lr_mean) ** 2)
            )

            new_carry = (particles_new.reshape(PN), total_loglik)
            return new_carry, (
                filt_mean.mean(), filt_std.mean(), ess.mean(),
                log_increment.mean(), pred_lr_mean, pred_lr_std,
            )

        init_carry = (particles, total_loglik)
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
    # CRN noise pre-generation
    # ------------------------------------------------------------------

    def get_noises(self, key: chex.PRNGKey) -> chex.Array:
        """Pre-generate all particle-filter noise for Common Random Numbers.

        Returns shape ``(T, 3*N + 2)``::

            cols  0 .. N-1   N(0,1)  ε_V  variance-propagation noise
            cols  N .. 2N-1  N(0,1)  Z₁   shared jump-factor noise
            cols 2N .. 3N-1  U[0,1)  u    jump-indicator noise
            col   3N         U[0,1)       first-stage  resampling uniform
            col   3N+1       U[0,1)       second-stage resampling uniform
        """
        T = self.S.shape[0] - 1
        N = self.num_particles
        key, key_eps, key_z1, key_uj, key_u1, key_u2 = jax.random.split(key, 6)
        eps_v  = jax.random.normal(key_eps, shape=(T, N))
        z1     = jax.random.normal(key_z1,  shape=(T, N))
        u_jump = jax.random.uniform(key_uj, shape=(T, N))
        u1     = jax.random.uniform(key_u1, shape=(T, 1))
        u2     = jax.random.uniform(key_u2, shape=(T, 1))
        return jnp.concatenate([eps_v, z1, u_jump, u1, u2], axis=1)   # (T, 3N+2)

    # ------------------------------------------------------------------
    # Default parameters / DynSetting factory
    # ------------------------------------------------------------------

    def get_default_param(self, key: chex.PRNGKey):
        initial_guess = {
            "v0":       0.02,
            "rho":      -0.2,
            "kappa":    2.0,
            "theta":    0.06,
            "sigma":    0.1,
            "r":        0.0,
            "lambda_J": 300,
            "mu_Jr":    0.0,
            "sigma_Jr": 0.02,
            "mu_JV":    0.0,
            "sigma_JV": 0.05,
            "rho_J":    0.0,
        }
        num_dims = len(initial_guess)
        initial_guess_unconstrained = self.params_to_unconstrained(initial_guess)

        T        = self.S.shape[0] - 1
        num_days = T // _MINS_PER_DAY
        dt_seq   = jnp.array(make_dt_seq(num_days), dtype=jnp.float32)
        noises = self.get_noises(key)
        rs_placeholder = jnp.zeros((T,), dtype=jnp.float32)

        dsetting = DynSetting(
            S=self.S,
            initial_guess=initial_guess_unconstrained,
            dt_seq=dt_seq,
            noises=noises,
            rs_seq=rs_placeholder,
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
        """Generate a synthetic log-return and variance path.

        Data layout: ``390 × num_days`` steps — 390 intraday 1-min steps. Gap dt varies:
        ``_DT_OVERNIGHT`` Mon–Thu, ``_DT_OVERWEEKEND`` Friday (first day is
        assumed Monday).  Overnight/weekend price gaps emerge from the jump
        process scaled by the larger dt.

        The discretisation follows the Euler–Maruyama scheme with:

        * Variance floor applied to the diffusion part only.
        * Jump multiplier ``exp(I_t · J_V)`` applied after the floor.
        * Risk-neutral drift correction ``comp`` subtracted from the log-
          return drift at every step.

        Args:
            seed:      RNG seed.
            S0:        Initial price (> 0).
            num_days:  Number of trading days (390 intraday steps each).
            params:    Shape-(12,) array ordered as
                       ``[v0, rho, kappa, theta, sigma, r,
                          lambda_J, mu_Jr, sigma_Jr, mu_JV, sigma_JV, rho_J]``.

        Returns:
            Tuple ``(log_returns, variances)`` each of shape
            ``(390 * num_days,)`` and dtype ``float32``.
        """
        if num_days < 1:
            raise ValueError("num_days must be >= 1")
        if S0 <= 0.0:
            raise ValueError("S0 must be positive")
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (12,):
            raise ValueError(
                "params must have shape (12,) ordered as "
                "[v0, rho, kappa, theta, sigma, r, "
                "lambda_J, mu_Jr, sigma_Jr, mu_JV, sigma_JV, rho_J]"
            )

        (v0, rho, kappa, theta, sigma, r,
         lambda_J, mu_Jr, sigma_Jr, mu_JV, sigma_JV, rho_J) = params

        sqrt_1m_rho2  = np.sqrt(max(1.0 - rho ** 2,   1e-8))
        sqrt_1m_rhoJ2 = np.sqrt(max(1.0 - rho_J ** 2, 1e-8))

        # m_r = E[exp(J_r)] depends only on params; p_J and comp are computed
        # per step inside the loop because dt may vary across steps.
        m_r = np.exp(mu_Jr + 0.5 * sigma_Jr ** 2)

        rng      = np.random.default_rng(seed)
        dt_arr   = make_dt_seq(num_days)          # (390 * num_days,)
        length   = len(dt_arr)

        variances   = np.zeros(length, dtype=np.float64)
        log_returns = np.zeros(length, dtype=np.float64)
        v_prev      = max(v0, VARIANCE_FLOOR)

        for step in range(length):
            dt   = float(dt_arr[step])
            # Per-step jump probability and risk-neutral compensator.
            p_J  = lambda_J * dt
            comp = np.log(max((1.0 - p_J) + p_J * m_r, 1e-30))

            # Draw independent noise components.
            eps_v      = rng.normal()
            z1         = rng.normal()   # shared jump factor
            z2         = rng.normal()   # independent jump component
            u          = rng.uniform()  # jump indicator
            eps_s_orth = rng.normal()   # orthogonal price shock

            # Jump indicator and sizes.
            I_t = 1.0 if u < p_J else 0.0
            J_V = mu_JV + sigma_JV * z1
            J_r = mu_Jr + sigma_Jr * (rho_J * z1 + sqrt_1m_rhoJ2 * z2)

            # Variance propagation: floor diffusion, then multiply exp-jump.
            v_cont = max(
                v_prev
                + kappa * (theta - v_prev) * dt
                + sigma * np.sqrt(v_prev) * np.sqrt(dt) * eps_v,
                VARIANCE_FLOOR,
            )
            v_next = v_cont * np.exp(I_t * J_V)

            # Log-return with risk-neutral correction.
            # Correlated price shock: ρ·ε_V + √(1-ρ²)·ε_S⊥
            corr_shock = rho * eps_v + sqrt_1m_rho2 * eps_s_orth
            log_return = (
                (r - 0.5 * v_next) * dt
                - comp
                + np.sqrt(v_next * dt) * corr_shock
                + I_t * J_r
            )

            variances[step]   = v_next
            log_returns[step] = log_return
            v_prev = v_next

        return (
            log_returns.astype(np.float32),
            variances.astype(np.float32),
        )

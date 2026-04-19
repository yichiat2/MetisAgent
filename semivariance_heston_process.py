"""Semivariance Heston process with two correlated CIR variance components.

Upside variance v+ and downside variance v- follow independent CIR processes
whose innovations are correlated.  The active variance for each bar is
selected by the sign of the observed log-return:

    sigma_t^2 = v_t^+   if  r_t >= 0   (up-bar)
    sigma_t^2 = v_t^-   if  r_t <  0   (down-bar)

Both processes are propagated at every step.  Only the active one appears in
the emission likelihood.

Correlation structure (three BM pairs):
    <dW^S,   dW^{v+}> = rho_p  dt
    <dW^S,   dW^{v-}> = rho_m  dt
    <dW^{v+},dW^{v-}> = rho_pm dt

Innovation decomposition used inside the APF:

    eps_vp  ~ N(0,1)                                            [pre-generated CRN]
    eta_vm  ~ N(0,1)  independent of eps_vp                    [pre-generated CRN]
    eps_vm   = rho_pm * eps_vp + sqrt(1 - rho_pm^2) * eta_vm   [derived per-candidate]

This enforces Cov(eps_vp, eps_vm) = rho_pm exactly.  The price–vol correlations
rho_p and rho_m enter only in the second-stage (correction) weight, analogously
to the standard Heston APF.

Parameters (13-dim vector):
    x[0]   v0p       – initial upside variance
    x[1]   dv0m      – v0m - v0p increment (>= 0);  v0m = v0p + dv0m
    x[2]   kappa_p   – mean-reversion speed  (v+)
    x[3]   kappa_m   – mean-reversion speed  (v-)
    x[4]   theta_p   – long-run variance      (v+)
    x[5]   dtheta_m  – theta_m - theta_p increment (>= 0);  theta_m = theta_p + dtheta_m
    x[6]   sigma_p   – vol of vol             (v+)
    x[7]   sigma_m   – vol of vol             (v-)
    x[8]   rho_p     – price-vol correlation  (v+)   typically in (-0.99, 0)
    x[9]   rho_m     – price-vol correlation  (v-)   typically in (-0.99, 0)
    x[10]  rho_pm    – v+/v- cross-correlation
    x[11]  r         – risk-free drift
    x[12]  lambda_ov – overnight sub-step intensity  (0.01 < lambda_ov < 0.5)

Data layout:  (390 + 1) * num_days steps.
    Steps 0–389 per day: intraday at dt = 1 min.
    Step      390 per day: overnight at dt = 1050 min, sub-stepped by
              round(1050 * lambda_ov) equal intraday steps.

Noise layout in get_noises():  shape (T, 2*N + 2)
    columns   0 .. N-1  :  eps_vp   N(0,1) CRN for v+ propagation
    columns   N .. 2N-1  :  eta_vm   N(0,1) CRN orthogonal component for v-
    column    2N          :  u1       Uniform[0,1) first-stage resample
    column    2N+1        :  u2       Uniform[0,1) second-stage resample
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import chex
import numpy as np
from scipy.stats import norm

from constants import _MINS_PER_DAY, _DT_MIN, _OVERNIGHT_MINS
from stochastic import StochasticProcessBase, Setting, DynSetting
from helper import (
    VARIANCE_FLOOR,
    _positive_variance,
    _gaussian_logpdf,
    _systematic_resample,
)
from inhomo_heston_process import InhomoHestonProcess


# ---------------------------------------------------------------------------
# Filter output type
# ---------------------------------------------------------------------------

class SVFilterInfo(NamedTuple):
    """Per-step posterior summaries returned by SemivarianceHestonProcess.loglikelihood."""
    filtered_mean_p: chex.Array        # E[v+_t | Y_{1:t}]          shape (T,)
    filtered_std_p:  chex.Array        # std[v+_t | Y_{1:t}]         shape (T,)
    filtered_mean_m: chex.Array        # E[v-_t | Y_{1:t}]           shape (T,)
    filtered_std_m:  chex.Array        # std[v-_t | Y_{1:t}]         shape (T,)
    ess:             chex.Array        # effective sample size        shape (T,)
    loglik_increments: chex.Array      # per-step log Z_1 + log Z_2  shape (T,)
    pred_log_return_mean: chex.Array   # one-step-ahead mean          shape (T,)
    pred_log_return_std:  chex.Array   # one-step-ahead std           shape (T,)


# ---------------------------------------------------------------------------
# Process class
# ---------------------------------------------------------------------------

class SemivarianceHestonProcess(StochasticProcessBase):
    """Semivariance Heston APF: two correlated CIR components, sign-selected emission.

    Observation: close-to-close log-return  r_t = log(C_t / C_{t-1}).
    Latent state per particle: (v+, v-) in R_{>0}^2.
    """

    PARAM_NAMES = [
        "v0p", "dv0m",
        "kappa_p", "kappa_m",
        "theta_p", "dtheta_m",
        "sigma_p", "sigma_m",
        "rho_p", "rho_m", "rho_pm",
        "r",
        "lambda_ov",
    ]

    PARAM_TRANSFORMS = {
        "v0p":      ("sigmoid_ab", 0.001, 0.5),
        "dv0m":     ("sigmoid_ab", 0.0,   0.499),
        "kappa_p":  ("sigmoid_ab", 0.1,  20.0),
        "kappa_m":  ("sigmoid_ab", 0.1,  20.0),
        "theta_p":  ("sigmoid_ab", 0.001, 0.5),
        "dtheta_m": ("sigmoid_ab", 0.0,   0.499),
        "sigma_p":  ("sigmoid_ab", 0.05,  2.0),
        "sigma_m":  ("sigmoid_ab", 0.05,  2.0),
        "rho_p":    ("tanh",      -0.99,  0.99),
        "rho_m":    ("tanh",      -0.99,  0.99),
        "rho_pm":   ("tanh",      -0.99,  0.99),
        "r":        ("sigmoid_ab", -0.00001, 0.00001),
        "lambda_ov":("sigmoid_ab",  0.01,  0.5),
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
    ):
        super().__init__(popsize, num_generations, sigma_init, dt, num_particles, S, rho_cpm)

    # ------------------------------------------------------------------
    # APF log-likelihood
    # ------------------------------------------------------------------

    def loglikelihood(
        self,
        x: chex.Array,
        setting: Setting,
        dsetting: DynSetting,
    ):
        """Auxiliary particle filter log-likelihood for the semivariance model.

        Args:
            x: (P, 13) constrained parameter array  (P candidates).
            setting:  static hyper-parameters.
            dsetting: data + noise arrays.

        Returns:
            (carry, SVFilterInfo)
            carry = (particles_vp, particles_vm, total_loglik)
                     shapes (P*N,), (P*N,), (P,)
        """
        P  = x.shape[0]
        N  = setting.num_particles
        PN = P * N

        # ── Unpack parameters ──────────────────────────────────────────
        v0p      = x[:, 0]
        dv0m     = x[:, 1]
        v0m      = v0p + dv0m          # v0m >= v0p by construction
        kappa_p  = x[:, 2]
        kappa_m  = x[:, 3]
        theta_p  = x[:, 4]
        dtheta_m = x[:, 5]
        theta_m  = theta_p + dtheta_m  # theta_m >= theta_p by construction
        sigma_p  = x[:, 6]
        sigma_m  = x[:, 7]
        rho_p    = x[:, 8]
        rho_m    = x[:, 9]
        rho_pm   = x[:, 10]
        r_val    = x[:, 11]
        lambda_ov = x[:, 12]

        # Broadcast parameters to (P*N,) for per-particle vmap
        kappa_p_pn  = jnp.repeat(kappa_p,  N)
        kappa_m_pn  = jnp.repeat(kappa_m,  N)
        theta_p_pn  = jnp.repeat(theta_p,  N)
        theta_m_pn  = jnp.repeat(theta_m,  N)
        sigma_p_pn  = jnp.repeat(sigma_p,  N)
        sigma_m_pn  = jnp.repeat(sigma_m,  N)
        rho_p_pn    = jnp.repeat(rho_p,    N)
        rho_m_pn    = jnp.repeat(rho_m,    N)
        rho_pm_pn   = jnp.repeat(rho_pm,   N)
        r_pn        = jnp.repeat(r_val,    N)

        # ── Data ───────────────────────────────────────────────────────
        log_prices  = jnp.log(dsetting.S)
        log_returns = log_prices[1:] - log_prices[:-1]   # (T,)
        dt_seq      = dsetting.dt_seq                     # (T,)
        noises_seq  = dsetting.noises                     # (T, 2N+2)

        base_dt = jnp.float32(_DT_MIN)
        ov_mins = jnp.float32(_OVERNIGHT_MINS)

        # ── Initial carry ──────────────────────────────────────────────
        particles_vp = jnp.repeat(v0p, N)           # (P*N,)
        particles_vm = jnp.repeat(v0m, N)           # (P*N,)
        total_loglik = jnp.zeros((P,), dtype=jnp.float32)
        log_N        = jnp.log(jnp.float32(N))

        # ── APF step (scanned over T) ──────────────────────────────────
        def _apf_step(carry, xs):
            obs, dt_i, noises_t = xs
            prt_vp_pn, prt_vm_pn, total_loglik = carry

            # Split pre-generated CRN noise
            eps_vp_t = noises_t[:N]         # (N,)
            eta_vm_t = noises_t[N:2 * N]    # (N,)
            u1_t     = noises_t[2 * N]      # scalar
            u2_t     = noises_t[2 * N + 1]  # scalar

            # Tile CRN across P candidates (same noise → common random numbers)
            eps_vp_pn = jnp.tile(eps_vp_t, P)   # (P*N,)
            eta_vm_pn = jnp.tile(eta_vm_t, P)   # (P*N,)

            # Per-candidate effective dt (overnight sub-stepping)
            is_overnight = dt_i > jnp.float32(1.5) * base_dt
            n_sub_p      = jnp.maximum(jnp.round(ov_mins * lambda_ov), jnp.float32(1.0))  # (P,)
            dt_eff_p     = jnp.where(is_overnight, n_sub_p * base_dt, dt_i)               # (P,)
            dt_eff_pn    = jnp.repeat(dt_eff_p, N)                                         # (P*N,)

            # Direction flag: we evaluate this dynamically inside the weights now
            # using exact drift to match the generative process.
            
            # ── Pilot weights (first stage) ─────────────────────────────
            # Marginal emission p(r_t | v_{t-1}, theta) using deterministic
            # pilot propagation (no stochastic term → variance = v*dt).
            def _pilot_one(vp, vm, kp, km, tp, tm, r, dt_e):
                vp_pilot = _positive_variance(vp + kp * (tp - vp) * dt_e)
                vm_pilot = _positive_variance(vm + km * (tm - vm) * dt_e)
                
                # drift = r * dt_e - jnp.log(
                #     jnp.exp(jnp.float32(0.5) * vm_pilot * dt_e) * jss.ndtr(-jnp.sqrt(vm_pilot * dt_e)) +
                #     jnp.exp(jnp.float32(0.5) * vp_pilot * dt_e) * jss.ndtr(jnp.sqrt(vp_pilot * dt_e))
                # )
                
                mean_noise = (jnp.sqrt(vp_pilot) - jnp.sqrt(vm_pilot)) / jnp.sqrt(jnp.float32(2.0 * jnp.pi)) * jnp.sqrt(dt_e)
                var_noise = (vp_pilot + vm_pilot) / jnp.float32(2.0) * dt_e - mean_noise**2
                drift = r * dt_e - mean_noise - jnp.float32(0.5) * var_noise

                is_up_local = obs >= drift
                
                log_up = _gaussian_logpdf(obs, drift, vp_pilot * dt_e)
                log_dn = _gaussian_logpdf(obs, drift, vm_pilot * dt_e)
                return jnp.where(is_up_local, log_up, log_dn)

            log_g_pn = jax.vmap(_pilot_one)(
                prt_vp_pn, prt_vm_pn,
                kappa_p_pn, kappa_m_pn,
                theta_p_pn, theta_m_pn,
                r_pn, dt_eff_pn,
            )  # (P*N,)
            log_g = log_g_pn.reshape(P, N)

            # ── First resample ──────────────────────────────────────────
            log_Z1     = jax.vmap(jax.nn.logsumexp)(log_g) - log_N              # (P,)
            ancestors1 = jax.vmap(lambda w: _systematic_resample(w, u1_t))(log_g)  # (P, N)

            prt_vp_2d  = prt_vp_pn.reshape(P, N)
            prt_vm_2d  = prt_vm_pn.reshape(P, N)
            vp_sel     = jax.vmap(lambda p, a: p[a])(prt_vp_2d, ancestors1)    # (P, N)
            vm_sel     = jax.vmap(lambda p, a: p[a])(prt_vm_2d, ancestors1)    # (P, N)
            log_g_sel  = jax.vmap(lambda lg, a: lg[a])(log_g, ancestors1)      # (P, N)

            # ── Propagate both CIR processes ────────────────────────────
            # eps_vm = rho_pm * eps_vp + sqrt(1 - rho_pm^2) * eta_vm
            # enforces Cov(eps_vp, eps_vm) = rho_pm exactly.
            def _propagate_pair(vp, vm, kp, km, tp, tm, sp, sm, rho_pm_, e_vp, eta_vm, dt_e):
                eps_vm = (
                    rho_pm_ * e_vp
                    + jnp.sqrt(_positive_variance(jnp.float32(1.0) - rho_pm_ ** 2))
                    * eta_vm
                )
                vp_new = _positive_variance(
                    vp + kp * (tp - vp) * dt_e
                    + sp * jnp.sqrt(_positive_variance(vp) * dt_e) * e_vp
                )
                vm_new = _positive_variance(
                    vm + km * (tm - vm) * dt_e
                    + sm * jnp.sqrt(_positive_variance(vm) * dt_e) * eps_vm
                )
                return vp_new, vm_new

            vp_next_pn, vm_next_pn = jax.vmap(_propagate_pair)(
                vp_sel.reshape(PN), vm_sel.reshape(PN),
                kappa_p_pn, kappa_m_pn,
                theta_p_pn, theta_m_pn,
                sigma_p_pn, sigma_m_pn,
                rho_pm_pn,
                eps_vp_pn, eta_vm_pn,
                dt_eff_pn,
            )  # each (P*N,)

            vp_next = vp_next_pn.reshape(P, N)
            vm_next = vm_next_pn.reshape(P, N)

            # ── Second-stage (correction) weights ───────────────────────
            # Conditional likelihood of r_t given the propagated variance noise.
            # Up-bar:   r | v+_n, eps_vp  ~ N(mu_vp + rho_p*sqrt(v+*dt)*eps_vp,
            #                                  (1-rho_p^2)*v+*dt)
            # Down-bar: r | v-_n, eps_vm  ~ N(mu_vm + rho_m*sqrt(v-*dt)*eps_vm,
            #                                  (1-rho_m^2)*v-*dt)
            def _log_p_one(vp_n, vm_n, rp, rm, rho_pm_, e_vp, eta_vm, r, dt_e):
                eps_vm = (
                    rho_pm_ * e_vp
                    + jnp.sqrt(_positive_variance(jnp.float32(1.0) - rho_pm_ ** 2))
                    * eta_vm
                )
           
                vp_n = _positive_variance(vp_n)
                vm_n = _positive_variance(vm_n)
                
                mean_noise = (jnp.sqrt(vp_n) - jnp.sqrt(vm_n)) / jnp.sqrt(jnp.float32(2.0 * jnp.pi)) * jnp.sqrt(dt_e)
                var_noise = (vp_n + vm_n) / jnp.float32(2.0) * dt_e - mean_noise**2
                drift = r * dt_e - mean_noise - jnp.float32(0.5) * var_noise
               
                is_up_local = obs >= drift
               
                # Correct observation weighting by exact Gaussian conditioning of eps_S on BOTH state noises.
                # eps_vm = rho_pm_ * e_vp + sqrt(1 - rho_pm_**2) * eta_vm
                # eps_S conditionally expects a contribution from both independent noises
                beta_denom = jnp.sqrt(_positive_variance(jnp.float32(1.0) - rho_pm_ ** 2))
                beta = (rm - rp * rho_pm_) / _positive_variance(beta_denom)
                eps_S_cond_mean = rp * e_vp + beta * eta_vm
                cond_var_scale = _positive_variance(jnp.float32(1.0) - rp**2 - beta**2)
                
                # Up-bar branch
                mu_up  = drift + jnp.sqrt(vp_n * dt_e) * eps_S_cond_mean
                s2_up  = _positive_variance(vp_n * dt_e * cond_var_scale)
                lp_up  = _gaussian_logpdf(obs, mu_up, s2_up)
                # Down-bar branch
                mu_dn  = drift + jnp.sqrt(vm_n * dt_e) * eps_S_cond_mean
                s2_dn  = _positive_variance(vm_n * dt_e * cond_var_scale)
                lp_dn  = _gaussian_logpdf(obs, mu_dn, s2_dn)
                return jnp.where(is_up_local, lp_up, lp_dn)

            log_p_pn = jax.vmap(_log_p_one)(
                vp_next_pn, vm_next_pn,
                rho_p_pn, rho_m_pn, rho_pm_pn,
                eps_vp_pn, eta_vm_pn,
                r_pn, dt_eff_pn,
            )  # (P*N,)
            log_p     = log_p_pn.reshape(P, N)
            log_alpha = log_p - log_g_sel        # (P, N)

            # ── Log-likelihood increment ────────────────────────────────
            lse_alpha    = jax.vmap(jax.nn.logsumexp)(log_alpha)   # (P,)
            log_Z2       = lse_alpha - log_N                        # (P,)
            log_increment = log_Z1 + log_Z2
            total_loglik  = total_loglik + log_increment            # (P,)

            # ── Posterior summaries (averaged over P for diagnostics) ───
            w_alpha  = jnp.exp(log_alpha - lse_alpha[:, None])     # (P, N)
            mean_vp  = jnp.sum(w_alpha * vp_next, axis=-1)         # (P,)
            std_vp   = jnp.sqrt(
                jnp.sum(w_alpha * (vp_next - mean_vp[:, None]) ** 2, axis=-1)
            )
            mean_vm  = jnp.sum(w_alpha * vm_next, axis=-1)         # (P,)
            std_vm   = jnp.sqrt(
                jnp.sum(w_alpha * (vm_next - mean_vm[:, None]) ** 2, axis=-1)
            )
            log_alpha_norm = log_alpha - lse_alpha[:, None]
            ess = jnp.exp(
                -jax.vmap(jax.nn.logsumexp)(jnp.float32(2.0) * log_alpha_norm)
            )   # (P,)

            # ── Second resample ─────────────────────────────────────────
            ancestors2   = jax.vmap(lambda w: _systematic_resample(w, u2_t))(log_alpha)  # (P, N)
            prt_vp_new   = jax.vmap(lambda vn, a: vn[a])(vp_next, ancestors2)            # (P, N)
            prt_vm_new   = jax.vmap(lambda vn, a: vn[a])(vm_next, ancestors2)            # (P, N)

            # ── One-step-ahead predictive stats ─────────────────────────
            def _pred_one(vp_n, vm_n, kp, km, tp, tm, r, dt_e):
                vp_pilot = _positive_variance(vp_n + kp * (tp - vp_n) * dt_e)
                vm_pilot = _positive_variance(vm_n + km * (tm - vm_n) * dt_e)
                
                mean_noise = (jnp.sqrt(vp_pilot) - jnp.sqrt(vm_pilot)) / jnp.sqrt(jnp.float32(2.0 * jnp.pi)) * jnp.sqrt(dt_e)
                var_noise = (vp_pilot + vm_pilot) / jnp.float32(2.0) * dt_e - mean_noise**2
                drift = r * dt_e - mean_noise - jnp.float32(0.5) * var_noise
                
                is_up_local = obs >= drift
                
                sigma2   = jnp.where(is_up_local, vp_pilot, vm_pilot)
                return drift, sigma2 * dt_e

            pred_mu_pn, pred_var_pn = jax.vmap(_pred_one)(
                prt_vp_new.reshape(PN), prt_vm_new.reshape(PN),
                kappa_p_pn, kappa_m_pn,
                theta_p_pn, theta_m_pn,
                r_pn, dt_eff_pn,
            )
            pred_lr_mean = jnp.mean(pred_mu_pn)
            pred_lr_std  = jnp.sqrt(
                jnp.mean(pred_var_pn)
                + jnp.mean((pred_mu_pn - pred_lr_mean) ** 2)
            )

            new_carry = (
                prt_vp_new.reshape(PN),
                prt_vm_new.reshape(PN),
                total_loglik,
            )
            return new_carry, (
                mean_vp.mean(), std_vp.mean(),
                mean_vm.mean(), std_vm.mean(),
                ess.mean(), log_increment.mean(),
                pred_lr_mean, pred_lr_std,
            )

        init_carry = (particles_vp, particles_vm, total_loglik)
        final_carry, outs = jax.lax.scan(
            _apf_step,
            init_carry,
            (log_returns, dt_seq, noises_seq),
        )
        (
            filt_mean_p, filt_std_p,
            filt_mean_m, filt_std_m,
            ess_seq, loglik_incs,
            pred_means, pred_stds,
        ) = outs

        return final_carry, SVFilterInfo(
            filtered_mean_p=filt_mean_p,
            filtered_std_p=filt_std_p,
            filtered_mean_m=filt_mean_m,
            filtered_std_m=filt_std_m,
            ess=ess_seq,
            loglik_increments=loglik_incs,
            pred_log_return_mean=pred_means,
            pred_log_return_std=pred_stds,
        )

    # ------------------------------------------------------------------
    # Noise pre-generation  (Common Random Numbers)
    # ------------------------------------------------------------------

    def get_noises(self, key: chex.PRNGKey) -> chex.Array:
        """Pre-generate all APF noise for Common Random Numbers.

        Returns shape (T, 2*N + 2):
            columns   0 .. N-1  : eps_vp  N(0,1) for v+ propagation
            columns   N .. 2N-1 : eta_vm  N(0,1) orthogonal component for v-
            column    2N        : u1  Uniform[0,1) first-stage resample
            column    2N+1      : u2  Uniform[0,1) second-stage resample
        """
        T = self.S.shape[0] - 1
        N = self.num_particles
        key, k_vp, k_vm, k_u1, k_u2 = jax.random.split(key, 5)
        eps_vp = jax.random.normal(k_vp,  shape=(T, N))
        eta_vm = jax.random.normal(k_vm,  shape=(T, N))
        u1     = jax.random.uniform(k_u1, shape=(T, 1))
        u2     = jax.random.uniform(k_u2, shape=(T, 1))
        return jnp.concatenate([eps_vp, eta_vm, u1, u2], axis=1)   # (T, 2N+2)

    # ------------------------------------------------------------------
    # Default parameters / DynSetting
    # ------------------------------------------------------------------

    def get_default_param(self, key: chex.PRNGKey):
        initial_guess = {
            "v0p":      0.02,
            "dv0m":     0.02,
            "kappa_p":  2.0,
            "kappa_m":  2.0,
            "theta_p":  0.02,
            "dtheta_m": 0.02,
            "sigma_p":  0.2,
            "sigma_m":  0.2,
            "rho_p":    -0.2,
            "rho_m":    -0.3,
            "rho_pm":    0.2,
            "r":         0.0,
            "lambda_ov": 0.1,
        }
        num_dims = len(initial_guess)
        initial_guess_unc = self.params_to_unconstrained(initial_guess)

        dt_seq = jnp.array(
            InhomoHestonProcess.make_dt_seq(self.S.shape[0] - 1),
            dtype=jnp.float32,
        )
        noises = self.get_noises(key)

        dsetting = DynSetting(
            S=self.S,
            initial_guess=initial_guess_unc,
            dt_seq=dt_seq,
            noises=noises,
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
        """Generate synthetic close-to-close log-returns and variance paths.

        Data layout: (390 + 1) * num_days steps.
        Each day: 390 intraday steps (dt = 1 min) + 1 overnight step
        (dt = 1050 min, sub-stepped by round(1050 * lambda_ov) intraday
        steps for variance propagation).

        The 3×3 innovation correlation matrix for (eps_S, eps_vp, eps_vm):
            [[1,      rho_p,  rho_m ],
             [rho_p,  1,      rho_pm],
             [rho_m,  rho_pm, 1     ]]
        is decomposed via Cholesky; correlated draws are produced at each step.

        Active variance is selected by sign(eps_S) (equivalent to sign of
        the return at vanishing drift), matching the APF's sign(r_t) rule.

        Args:
            seed:     RNG seed.
            S0:       Initial price (> 0).
            num_days: Number of trading days.
            params:   Shape (13,) array ordered as
                      [v0p, v0m, kappa_p, kappa_m, theta_p, theta_m,
                       sigma_p, sigma_m, rho_p, rho_m, rho_pm, r, lambda_ov].

        Returns:
            (log_returns, variances_p, variances_m) each shape
            ``((390 + 1) * num_days,)`` and dtype float32.
        """
        if num_days < 1:
            raise ValueError("num_days must be >= 1")
        if S0 <= 0.0:
            raise ValueError("S0 must be positive")

        params = np.asarray(params, dtype=np.float64)
        if params.shape != (13,):
            raise ValueError(
                "params must have shape (13,) ordered as "
                "[v0p, v0m, kappa_p, kappa_m, theta_p, theta_m, "
                "sigma_p, sigma_m, rho_p, rho_m, rho_pm, r, lambda_ov]"
            )

        (
            v0p, v0m,
            kappa_p, kappa_m,
            theta_p, theta_m,
            sigma_p, sigma_m,
            rho_p, rho_m, rho_pm,
            r_drift, lambda_ov,
        ) = params

        # ── Cholesky decomposition of the 3×3 correlation matrix ──────
        # Rows/columns: (eps_S, eps_vp, eps_vm)
        corr = np.array(
            [
                [1.0,    rho_p,  rho_m ],
                [rho_p,  1.0,    rho_pm],
                [rho_m,  rho_pm, 1.0   ],
            ],
            dtype=np.float64,
        )
        # Ensure positive-definiteness (safety clamp for marginal params)
        eigmin = np.linalg.eigvalsh(corr).min()
        if eigmin < 1e-8:
            corr += (1e-6 - eigmin) * np.eye(3)
        L = np.linalg.cholesky(corr)   # lower-triangular (3, 3)

        rng    = np.random.default_rng(seed)
        length = (_MINS_PER_DAY + 1) * num_days

        log_returns = np.zeros(length, dtype=np.float64)
        variances_p = np.zeros(length, dtype=np.float64)
        variances_m = np.zeros(length, dtype=np.float64)

        vp       = max(float(v0p), VARIANCE_FLOOR)
        vm       = max(float(v0m), VARIANCE_FLOOR)
        n_sub    = max(1.0, round(float(_OVERNIGHT_MINS) * lambda_ov))
        base_dt  = float(_DT_MIN)

        for step in range(length):
            in_block     = step % (_MINS_PER_DAY + 1)
            is_overnight = in_block == _MINS_PER_DAY
            dt           = n_sub * base_dt if is_overnight else base_dt

            z = rng.standard_normal(3)
            eps_S, eps_vp_raw, eps_vm_raw = L @ z   # correlated innovations

            vp_new = max(
                vp + kappa_p * (theta_p - vp) * dt
                + sigma_p * np.sqrt(max(vp, VARIANCE_FLOOR) * dt) * eps_vp_raw,
                VARIANCE_FLOOR,
            )
            vm_new = max(
                vm + kappa_m * (theta_m - vm) * dt
                + sigma_m * np.sqrt(max(vm, VARIANCE_FLOOR) * dt) * eps_vm_raw,
                VARIANCE_FLOOR,
            )

            # Select active variance (diffusion) by the sign of the price shock
            sigma2     = vp_new if eps_S >= 0.0 else vm_new
            mean_noise = (np.sqrt(vp_new) - np.sqrt(vm_new)) / np.sqrt(2.0 * np.pi) * np.sqrt(dt)
            var_noise = (vp_new + vm_new) / 2.0 * dt - mean_noise**2
            drift = r_drift * dt - mean_noise - 0.5 * var_noise
            
            log_return = drift + np.sqrt(sigma2 * dt) * eps_S

            log_returns[step] = log_return
            variances_p[step] = vp_new
            variances_m[step] = vm_new
            vp, vm = vp_new, vm_new

        return (
            log_returns.astype(np.float32),
            variances_p.astype(np.float32),
            variances_m.astype(np.float32),
        )

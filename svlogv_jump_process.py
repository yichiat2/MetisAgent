"""Stochastic Volatility Model with CIR Variance Process and Correlated Gaussian Jumps.

Design document: docs/svlogv_jump_apf_design.md

Model
-----
Latent state: V_t (variance), driven by a CIR process with additive Gaussian jumps.
Observation:  y_t = log(S_t / S_{t-1}), driven by V_t (the *current* variance).

Discretised SDEs (Euler-Maruyama):

    y_t = (r - 0.5*V_t)*dt - c_t + sqrt(V_t*dt)*epsilon_t + I_t*J_t^S
    V_t = V_{t-1} + kappa*(theta - V_{t-1})*dt
            + sigma_v*sqrt(V_{t-1}*dt)*eta_t + I_t*J_t^V

where:
    V_t >= VARIANCE_FLOOR                     (clipped to stay positive)
    epsilon_t   = z1_t                        standard normal
    eta_t       = rho*z1_t + sqrt(1-rho^2)*z2_t
    I_t         ~ Bernoulli(lambda_J * dt_t)  shared jump indicator
    J_t^S       = mu_JS + sigma_JS * z_j1
    J_t^V       = mu_JV + sigma_JV*(rho_J*z_j1 + sqrt(1-rho_J^2)*z_j2)
    c_t         = log[(1-p_t) + p_t * exp(mu_JS + 0.5*sigma_JS^2)]
    p_t         = lambda_J * dt_t             (dt_t from dt_seq)

Note: diffusive terms use dt = _DT_MIN (1 min); jump arrival uses dt_t.

Inference: Auxiliary Particle Filter (APF) with Rao-Blackwellised pilot stage.

  Pilot stage:
    hat_V^(i) = max(V_{t-1}^(i) + kappa*(theta - V_{t-1}^(i))*dt, VARIANCE_FLOOR)
    mu_y^(i)    = (r - 0.5*hat_V^(i))*dt - c_t
    g_t^(i)     = (1-p_t)*N(y_t; mu_y^(i), hat_V^(i)*dt)
                 +   p_t *N(y_t; mu_y^(i)+mu_JS, hat_V^(i)*dt + sigma_JS^2)

  Propagation stage (vanilla APF, no Kalman):
    1. Compute posterior jump probability pi^(j) from Bayes on I_t.
    2. Sample b^(j) ~ Bernoulli(pi^(j)) via pre-drawn u_mix.
    3. Sample J_t^V ~ N(mu_JV, sigma_JV^2) from marginal prior.
    4. Advance V_t with CIR diffusion + b^(j)*J_t^V, clip to VARIANCE_FLOOR.

  Correction weights (Rao-Blackwellised over J^S | J^V, b):
    J^S | J^V ~ N(mu_JS|V, sigma2_JS|V)
      mu_JS|V    = mu_JS + (sigma_JS * rho_J / sigma_JV) * (J_V - mu_JV)
      sigma2_JS|V = sigma_JS^2 * (1 - rho_J^2)
    p(y_t | ell_t, b, J_V, z_eta):
      mu_y  = (r - 0.5*V)*dt - c_t + sqrt(V*dt)*rho*z_eta + b * mu_JS|V
      sig2  = V*dt*(1 - rho^2) + b * sigma2_JS|V
    w^(j) = N(y_t; mu_y^(j), sig2^(j)) / g_t^(a_j)
    Second systematic resample (independent u_res2) with these weights.

CRN noise layout: (T, 3*N + 2)
    cols  0 .. N-1     N(0,1)  z_eta    OU diffusion / leverage noise
    cols  N .. 2N-1    N(0,1)  z_jv     J_V marginal sample
    col   2N           U[0,1)  u_res    first systematic resample
    col   2N+1         U[0,1)  u_res2   second systematic resample
    cols  2N+2 .. 3N+1 U[0,1)  u_mix    per-particle Bernoulli threshold

Parameters (12-dimensional):
    x[0]  v0        initial variance
    x[1]  kappa     CIR mean-reversion speed (yr^-1)
    x[2]  theta     CIR long-run variance level
    x[3]  sigma_v   CIR vol-of-variance coefficient (yr^-0.5)
    x[4]  rho       diffusion correlation
    x[5]  r         risk-free rate (yr^-1)
    x[6]  lambda_J  jump intensity (jumps yr^-1)
    x[7]  mu_JS     mean log-return jump size
    x[8]  sigma_JS  std of log-return jump
    x[9]  mu_JV     mean log-variance jump size
    x[10] sigma_JV  std of log-variance jump
    x[11] rho_J     correlation between J^S and J^V
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
    EPS,
    SVJFilterInfo,
    _gaussian_logpdf,
    _systematic_resample,
)


class StochasticVolatilityJumpProcess(StochasticProcessBase):
    """CIR variance model with correlated bivariate Gaussian jumps.

    See docs/svlogv_jump_apf_design.md for full mathematical derivation.
    """

    PARAM_NAMES = [
        "v0", "kappa", "theta", "sigma_v", "rho",
        "lambda_J", "mu_JS", "sigma_JS", "mu_JV", "sigma_JV", "rho_J",
    ]
    PARAM_TRANSFORMS = {
        "v0":       ("sigmoid_ab",  1e-4,  1.0),
        "kappa":    ("softplus",    0.01,  30.0),
        "theta":    ("sigmoid_ab",  1e-4,  1.0),
        "sigma_v":  ("softplus",    0.01,  30.0),
        "rho":      ("tanh",       -0.99,  0.99),
        "lambda_J": ("softplus",    0.1,  200.0),
        "mu_JS":    ("sigmoid_ab", -0.2,   0.2),
        "sigma_JS": ("softplus",    1e-3,  0.3),
        "mu_JV":    ("sigmoid_ab", -0.3,   0.3),
        "sigma_JV": ("softplus",    1e-3,  5.0),
        "rho_J":    ("tanh",       -0.99,  0.99),
    }

    # ------------------------------------------------------------------
    # APF log-likelihood
    # ------------------------------------------------------------------

    def loglikelihood(self, x: chex.Array, setting: Setting, dsetting: DynSetting):
        """Auxiliary Particle Filter log-likelihood with Rao-Blackwellised pilot.

        Args:
            x:        (P, 12) constrained parameter candidates.
            setting:  filter / ES hyper-parameters.
            dsetting: data and pre-generated CRN noise.

        Returns:
            (carry, FilterInfo) where carry[-1] is a (P,) array of total
            log-likelihoods, one per candidate.
        """
        P  = x.shape[0]
        N  = setting.num_particles
        PN = P * N
        dt = jnp.float32(_DT_MIN)

        # ── Unpack constrained parameters ──────────────────────────────
        v0        = x[:, 0]
        kappa     = x[:, 1]
        theta     = x[:, 2]
        sigma_v   = x[:, 3]
        rho       = x[:, 4]
        lambda_J  = x[:, 5]
        mu_JS     = x[:, 6]
        sigma_JS  = x[:, 7]
        mu_JV     = x[:, 8]
        sigma_JV  = x[:, 9]
        rho_J     = x[:, 10]

        # ── Broadcast parameters to (P*N,) ─────────────────────────────
        kappa_pn    = jnp.repeat(kappa,    N)
        theta_pn    = jnp.repeat(theta,    N)
        sigma_v_pn  = jnp.repeat(sigma_v,  N)
        rho_pn      = jnp.repeat(rho,      N)
        lambda_J_pn = jnp.repeat(lambda_J, N)
        mu_JS_pn    = jnp.repeat(mu_JS,    N)
        sigma_JS_pn = jnp.repeat(sigma_JS, N)
        mu_JV_pn    = jnp.repeat(mu_JV,    N)
        sigma_JV_pn = jnp.repeat(sigma_JV, N)
        rho_J_pn    = jnp.repeat(rho_J,    N)

        # Jump MGF: m_r = E[exp(J^S)] = exp(mu_JS + 0.5*sigma_JS^2).
        # Parameter-dependent, not dt-dependent -> precomputed outside scan.
        m_r    = jnp.exp(mu_JS + 0.5 * sigma_JS ** 2)  # (P,)
        m_r_pn = jnp.repeat(m_r, N)                     # (P*N,)

        log_returns = jnp.log(dsetting.S[1:]) - jnp.log(dsetting.S[:-1])  # (T,)
        dt_seq      = dsetting.dt_seq                                        # (T,)
        dt_next_seq = jnp.concatenate([dt_seq[1:], dt_seq[-1:]])             # (T,) next-step dt
        noises_seq  = dsetting.noises                                        # (T, 6N+2)

        particles    = jnp.repeat(v0, N)               # (P*N,)
        total_loglik = jnp.zeros((P,), dtype=jnp.float32)
        log_N        = jnp.log(jnp.float32(N))

        def _apf_step(carry, xs):
            obs, dt_t, dt_next_t, noises_t = xs
            particles_pn, total_loglik = carry

            # ── Split CRN noise ─────────────────────────────────────────
            z_eta_t       = noises_t[0:           N]          # (N,)
            z_jv_t        = noises_t[N:       2 * N]          # (N,)
            u_res_t       = noises_t[2 * N]                   # scalar
            u_res2_t      = noises_t[2 * N + 1]               # scalar
            u_mix_t       = noises_t[2 * N + 2: 3 * N + 2]   # (N,)
            z_eta_pred_t  = noises_t[3 * N + 2: 4 * N + 2]   # (N,) prediction noise
            z_jv_pred_t   = noises_t[4 * N + 2: 5 * N + 2]   # (N,) prediction J_V noise
            u_mix_pred_t  = noises_t[5 * N + 2: 6 * N + 2]   # (N,) prediction Bernoulli

            # ── Per-step jump probability and compensator ────────────────
            p_J_pn      = 1.0 - jnp.exp(-lambda_J_pn * dt_t)
            comp_pn     = jnp.log(jnp.maximum(1.0 - p_J_pn + p_J_pn * m_r_pn, EPS))
            log_pJ_pn   = jnp.log(jnp.maximum(p_J_pn,       EPS))
            log_1mpJ_pn = jnp.log(jnp.maximum(1.0 - p_J_pn, EPS))

            # ── Pilot: RB marginal likelihood g_t^(i) ───────────────────
            def _pilot_one(v, mu_js, sigma_js, log_pj, log_1mpj, k, th):
                hat_V = jnp.maximum(v + k * (th - v) * dt, VARIANCE_FLOOR)
                mu_y    = jnp.float32(0.0)
                sig2_0  = hat_V * dt
                log_f0  = _gaussian_logpdf(obs, mu_y,         sig2_0)
                sig2_1  = sig2_0 + sigma_js ** 2
                log_f1  = _gaussian_logpdf(obs, mu_y + mu_js, sig2_1)
                log_g   = jnp.logaddexp(log_1mpj + log_f0, log_pj + log_f1)
                return log_g, log_f0, log_f1

            log_g_pn, log_f0_pn, log_f1_pn = jax.vmap(_pilot_one)(
                particles_pn, mu_JS_pn, sigma_JS_pn,
                log_pJ_pn, log_1mpJ_pn, kappa_pn, theta_pn,
            )   # each (P*N,)

            log_g = log_g_pn.reshape(P, N)

            # ── Log-likelihood increment ─────────────────────────────────
            log_Z        = jax.vmap(jax.nn.logsumexp)(log_g) - log_N  # (P,)
            total_loglik = total_loglik + log_Z

            # ── Ancestor resampling (pilot weights) ──────────────────────
            ancestors  = jax.vmap(
                lambda w: _systematic_resample(w, u_res_t)
            )(log_g)                                                    # (P, N)

            particles_2d = particles_pn.reshape(P, N)
            v_anc        = jax.vmap(lambda p, a: p[a])(particles_2d, ancestors)  # (P, N)

            # Gather pilot log-densities at ancestor indices.
            log_f0_sel = jax.vmap(lambda lf, a: lf[a])(
                log_f0_pn.reshape(P, N), ancestors)                    # (P, N)
            log_f1_sel = jax.vmap(lambda lf, a: lf[a])(
                log_f1_pn.reshape(P, N), ancestors)                    # (P, N)
            log_g_sel  = jax.vmap(lambda lf, a: lf[a])(
                log_g, ancestors)                                       # (P, N)

            # ── Tile noise across P populations ──────────────────────────
            z_eta_pn = jnp.tile(z_eta_t, P)   # (P*N,)
            z_jv_pn  = jnp.tile(z_jv_t,  P)   # (P*N,)
            u_mix_pn = jnp.tile(u_mix_t,  P)   # (P*N,)

            # ── Propagation: sample I_t, J_t^V, advance V_t ─────────────
            # J_S is not sampled here; it will be Rao-Blackwellised out in
            # the correction weight via J_S | J_V ~ N(mu_JS|V, sigma2_JS|V).
            def _propagate_one(v, log_pj, log_1mpj, log_f0_, log_f1_,
                               sv, k, th, mu_jv, sigma_jv,
                               z_eta, z_jv, u_mix):
                # Posterior jump probability via Bayes
                logit_pi = (log_pj - log_1mpj) + log_f1_ - log_f0_
                pi       = jax.nn.sigmoid(logit_pi)
                b   = (u_mix < pi).astype(jnp.float32)
                J_V = mu_jv + sigma_jv * z_jv
                v_eff = jnp.maximum(v, VARIANCE_FLOOR)
                v_new = (v + k * (th - v) * dt
                         + sv * jnp.sqrt(v_eff * dt) * z_eta
                         + b * J_V)
                return jnp.clip(v_new, VARIANCE_FLOOR, 5.0), b, J_V, pi

            v_new_pn, b_pn, J_V_pn, pi_pn = jax.vmap(_propagate_one)(
                v_anc.reshape(PN),
                log_pJ_pn, log_1mpJ_pn,
                log_f0_sel.reshape(PN), log_f1_sel.reshape(PN),
                sigma_v_pn, kappa_pn, theta_pn,
                mu_JV_pn, sigma_JV_pn,
                z_eta_pn, z_jv_pn, u_mix_pn,
            )   # each (P*N,)

            # ── Correction weights: w^(j) = p(y_t|ell_t^(j), b^(j), J_V^(j), z_eta^(j)) / g_t^(a_j) ─
            # Marginalise J_S | J_V analytically (Rao-Blackwellised):
            #   J_S | J_V ~ N(mu_JS|V, sigma2_JS|V)
            #   mu_JS|V    = mu_JS + (sigma_JS * rho_J / sigma_JV) * (J_V - mu_JV)
            #   sigma2_JS|V = sigma_JS^2 * (1 - rho_J^2)
            # Conditioning on (z_eta, b, J_V) gives a single Gaussian:
            #   mu_y  = (r - 0.5V)*dt - c_t + sqrt(V*dt)*rho*z_eta + b * mu_JS|V
            #   sig2  = V*dt*(1 - rho^2) + b * sigma2_JS|V
            def _obs_loglik_one(v, rho, z_eta, b, J_V, mu_js, sigma_js, mu_jv, sigma_jv, rho_j):
                V_eff            = jnp.maximum(v, VARIANCE_FLOOR)
                mu_JS_given_JV   = mu_js + (sigma_js * rho_j / sigma_jv) * (J_V - mu_jv)
                sig2_JS_given_JV = sigma_js ** 2 * (1.0 - rho_j ** 2)
                mu_y = jnp.sqrt(V_eff * dt) * rho * z_eta + b * mu_JS_given_JV
                sig2 = V_eff * dt * (1.0 - rho ** 2) + b * sig2_JS_given_JV
                return _gaussian_logpdf(obs, mu_y, sig2)

            log_py_pn = jax.vmap(_obs_loglik_one)(
                v_new_pn, rho_pn,
                z_eta_pn, b_pn, J_V_pn,
                mu_JS_pn, sigma_JS_pn, mu_JV_pn, sigma_JV_pn, rho_J_pn,
            )   # (P*N,)

            log_w   = (log_py_pn - log_g_sel.reshape(PN)).reshape(P, N)

            # Second systematic resample with correction weights 
            ancestors_w = jax.vmap(
                lambda w: _systematic_resample(w, u_res2_t)
            )(log_w)                                                   # (P, N)
            v_final   = jax.vmap(lambda p, a: p[a])(
                v_new_pn.reshape(P, N), ancestors_w
            )                                                          # (P, N)

            # ── Diagnostics ──────────────────────────────────────────────
            V_final   = v_final
            filt_mean = jnp.mean(V_final, axis=-1)                    # (P,)
            filt_q05  = jnp.percentile(V_final,  5.0, axis=-1)        # (P,)
            filt_q25  = jnp.percentile(V_final, 25.0, axis=-1)        # (P,)
            filt_q50  = jnp.percentile(V_final, 50.0, axis=-1)        # (P,)
            filt_q75  = jnp.percentile(V_final, 75.0, axis=-1)        # (P,)
            filt_q95  = jnp.percentile(V_final, 95.0, axis=-1)        # (P,)
            ess_val = jnp.exp(
                2.0 * jax.vmap(jax.nn.logsumexp)(log_w)
                - jax.vmap(lambda w: jax.nn.logsumexp(2.0 * w))(log_w)
            )   # (P,)

            # ── One-step-ahead predictive moments ────────────────────────
            # Propagate v_final one step forward using dt_next_t, sample jump
            # indicator and jump variance, then compute p(r_{t+1}) = 1/N sum N(mu_i, var_i)
            # where each component mirrors _obs_loglik_one conditioned on z_eta:
            #   mu_r  = sqrt(V_next * dt_next_t) * rho * z_eta_p + b * mu_JS|V
            #   var_r = V_next * dt_next_t * (1 - rho^2) + b * sigma2_JS|V
            p_J_pn_next = 1.0 - jnp.exp(-lambda_J_pn * dt_next_t)           # (P*N,)

            z_eta_pred_pn = jnp.tile(z_eta_pred_t, P)   # (P*N,)
            z_jv_pred_pn  = jnp.tile(z_jv_pred_t,  P)   # (P*N,)
            u_mix_pred_pn = jnp.tile(u_mix_pred_t,  P)   # (P*N,)

            def _pred_one(v, sv, k, th, p_j, rho,
                          mu_js, sigma_js, mu_jv, sigma_jv, rho_j,
                          z_eta_p, z_jv_p, u_mix_p):
                # CIR diffusion step (next interval)
                v_eff  = jnp.maximum(v, VARIANCE_FLOOR)
                v_prop = (v + k * (th - v) * dt + sv * jnp.sqrt(v_eff * dt) * z_eta_p)
                # Jump at next step
                b_pred   = (u_mix_p < p_j).astype(jnp.float32)
                J_V_pred = mu_jv + sigma_jv * z_jv_p
                V_next   = jnp.clip(v_prop + b_pred * J_V_pred, VARIANCE_FLOOR, 5.0)
                # Predictive return: conditioned on z_eta_p (leverage) and b_pred
                mu_JS_given_JV   = mu_js + (sigma_js * rho_j / sigma_jv) * (J_V_pred - mu_jv)
                sig2_JS_given_JV = sigma_js ** 2 * (1.0 - rho_j ** 2)
                mu_r  = jnp.sqrt(V_next * dt) * rho * z_eta_p + b_pred * mu_JS_given_JV
                var_r = V_next * dt * (1.0 - rho ** 2) + b_pred * sig2_JS_given_JV
                return mu_r, var_r, V_next

            pred_mu_pn, pred_var_pn, pred_V_next_pn = jax.vmap(_pred_one)(
                v_final.reshape(PN),
                sigma_v_pn, kappa_pn, theta_pn, p_J_pn_next, rho_pn,
                mu_JS_pn, sigma_JS_pn, mu_JV_pn, sigma_JV_pn, rho_J_pn,
                z_eta_pred_pn, z_jv_pred_pn, u_mix_pred_pn,
            )
            pred_lr_mean  = jnp.mean(pred_mu_pn)
            _delta        = pred_mu_pn - pred_lr_mean
            _m2           = jnp.mean(pred_var_pn + _delta ** 2)
            _m3           = jnp.mean(_delta ** 3 + 3.0 * _delta * pred_var_pn)
            _m4           = jnp.mean(_delta ** 4 + 6.0 * _delta ** 2 * pred_var_pn + 3.0 * pred_var_pn ** 2)
            pred_lr_std   = jnp.sqrt(jnp.maximum(_m2, EPS))
            pred_lr_skew  = _m3 / jnp.maximum(_m2, EPS) ** 1.5
            pred_lr_kurt  = _m4 / jnp.maximum(_m2, EPS) ** 2 - 3.0
            pred_var_mean = jnp.mean(pred_V_next_pn)
            pred_var_q05  = jnp.percentile(pred_V_next_pn,  5.0)
            pred_var_q25  = jnp.percentile(pred_V_next_pn, 25.0)
            pred_var_q50  = jnp.percentile(pred_V_next_pn, 50.0)
            pred_var_q75  = jnp.percentile(pred_V_next_pn, 75.0)
            pred_var_q95  = jnp.percentile(pred_V_next_pn, 95.0)
            jump_prob_t = jnp.mean(pi_pn)   # mean posterior jump probability

            new_carry = (v_final.reshape(PN), total_loglik)
            return new_carry, (
                filt_mean.mean(), filt_q05.mean(), filt_q25.mean(),
                filt_q50.mean(), filt_q75.mean(), filt_q95.mean(),
                ess_val.mean(), log_Z.mean(),
                pred_lr_mean, pred_lr_std, pred_lr_skew, pred_lr_kurt,
                pred_var_mean, pred_var_q05, pred_var_q25, pred_var_q50, pred_var_q75, pred_var_q95,
                jump_prob_t,
            )

        init_carry = (particles, total_loglik)
        final_carry, (
            filt_means, filt_q05s, filt_q25s, filt_q50s, filt_q75s, filt_q95s,
            ess_seq, loglik_incs, pred_means, pred_stds, pred_skews, pred_kurts,
            pred_var_means, pred_var_q05s, pred_var_q25s, pred_var_q50s, pred_var_q75s, pred_var_q95s,
            jump_probs,
        ) = jax.lax.scan(_apf_step, init_carry, (log_returns, dt_seq, dt_next_seq, noises_seq))

        return final_carry, SVJFilterInfo(
            filtered_mean=filt_means,
            filtered_q05=filt_q05s,
            filtered_q25=filt_q25s,
            filtered_q50=filt_q50s,
            filtered_q75=filt_q75s,
            filtered_q95=filt_q95s,
            ess=ess_seq,
            loglik_increments=loglik_incs,
            pred_log_return_mean=pred_means,
            pred_log_return_std=pred_stds,
            pred_log_return_skew=pred_skews,
            pred_log_return_kurt=pred_kurts,
            pred_var_mean=pred_var_means,
            pred_var_q05=pred_var_q05s,
            pred_var_q25=pred_var_q25s,
            pred_var_q50=pred_var_q50s,
            pred_var_q75=pred_var_q75s,
            pred_var_q95=pred_var_q95s,
            jump_prob=jump_probs,
        )

    # ------------------------------------------------------------------
    # CRN noise pre-generation
    # ------------------------------------------------------------------

    def get_noises(self, key: chex.PRNGKey) -> chex.Array:
        """Pre-generate all particle-filter noise for Common Random Numbers.

        Returns shape ``(T, 6*N + 2)``:

            cols  0   .. N-1    N(0,1)  z_eta       OU diffusion / leverage noise
            cols  N   .. 2N-1   N(0,1)  z_jv        J_V marginal sample
            col   2N           U[0,1)  u_res       first systematic resample
            col   2N+1         U[0,1)  u_res2      second systematic resample
            cols  2N+2 .. 3N+1 U[0,1)  u_mix       per-particle Bernoulli threshold
            cols  3N+2 .. 4N+1 N(0,1)  z_eta_pred  CIR diffusion for predictive step
            cols  4N+2 .. 5N+1 N(0,1)  z_jv_pred   J_V sample for predictive step
            cols  5N+2 .. 6N+1 U[0,1)  u_mix_pred  jump indicator for predictive step
        """
        T = self.S.shape[0] - 1
        N = self.num_particles
        key, k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 9)
        z_eta       = jax.random.normal(k1, shape=(T, N))
        z_jv        = jax.random.normal(k2, shape=(T, N))
        u_res       = jax.random.uniform(k3, shape=(T, 1))
        u_res2      = jax.random.uniform(k4, shape=(T, 1))
        u_mix       = jax.random.uniform(k5, shape=(T, N))
        z_eta_pred  = jax.random.normal(k6, shape=(T, N))
        z_jv_pred   = jax.random.normal(k7, shape=(T, N))
        u_mix_pred  = jax.random.uniform(k8, shape=(T, N))
        return jnp.concatenate(
            [z_eta, z_jv, u_res, u_res2, u_mix, z_eta_pred, z_jv_pred, u_mix_pred], axis=1
        )  # (T, 6N+2)

    # ------------------------------------------------------------------
    # Default parameters / DynSetting factory
    # ------------------------------------------------------------------

    def get_default_param(self, key: chex.PRNGKey):
        import math
        initial_guess = {
            "v0":       0.06,
            "kappa":    2.0,
            "theta":    0.04,
            "sigma_v":  0.5,
            "rho":      -0.5,
            "lambda_J": 100.0,
            "mu_JS":    0,
            "sigma_JS": 0.05,
            "mu_JV":    0,
            "sigma_JV": 0.1,
            "rho_J":    -0.3,
        }
        num_dims = len(initial_guess)
        initial_guess_unconstrained = self.params_to_unconstrained(initial_guess)

        T = self.S.shape[0] - 1
        if self.dt_seq is not None:
            dt_seq = jnp.array(self.dt_seq, dtype=jnp.float32)
        else:
            num_days = T // _MINS_PER_DAY
            dt_seq   = jnp.array(make_dt_seq(num_days), dtype=jnp.float32)
        noises   = self.get_noises(key)
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
        """Generate a synthetic price path and variance path.

        Discretisation order per step (V_t computed before y_t so that
        y_t uses the *current* variance V_t):

            1. Sample z1, z2, z_j1, z_j2 ~ N(0,1) and u ~ U[0,1).
            2. Advance variance (CIR):
               V_t = max(V_{t-1} + kappa*(theta-V_{t-1})*dt_min
                         + sigma_v*sqrt(V_{t-1}*dt_min)*eta + I_t*J_t^V,
                         VARIANCE_FLOOR)
            3. Compute risk-neutral log-return:
               y_t = (r - 0.5*V_t)*dt_min - c_t + sqrt(V_t*dt_min)*z1 + I_t*J_t^S

        Args:
            seed:      RNG seed.
            S0:        Initial price (> 0).
            num_days:  Number of trading days (390 intraday steps each).
            params:    Shape-(12,) array in PARAM_NAMES order.

        Returns:
            Tuple (log_returns, variances), each shape (390*num_days,),
            dtype float32.
        """
        if num_days < 1:
            raise ValueError("num_days must be >= 1")
        if S0 <= 0.0:
            raise ValueError("S0 must be positive")
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (11,):
            raise ValueError("params must have shape (11,)")

        (v0, kappa, theta, sigma_v, rho,
         lambda_J, mu_JS, sigma_JS, mu_JV, sigma_JV, rho_J) = params

        sqrt_1m_rho2  = np.sqrt(max(1.0 - rho   ** 2, 1e-8))
        sqrt_1m_rhoJ2 = np.sqrt(max(1.0 - rho_J ** 2, 1e-8))
        m_r           = np.exp(mu_JS + 0.5 * sigma_JS ** 2)

        rng    = np.random.default_rng(seed)
        dt_arr = make_dt_seq(num_days)
        length = len(dt_arr)

        variances   = np.zeros(length, dtype=np.float64)
        log_returns = np.zeros(length, dtype=np.float64)
        v_prev      = float(v0)
        print(f"dt Min: {dt_arr.min()} Max: {dt_arr.max()}")
        p_J_min = lambda_J * dt_arr.min()
        p_J_max = lambda_J * dt_arr.max()
        print(f"pJ Min: {p_J_min:.6f} Max: {p_J_max:.6f}")

        for step in range(length):
            dt_t = float(dt_arr[step])
            p_J  = 1.0 - np.exp(-lambda_J * dt_t)
            comp = np.log(max((1.0 - p_J) + p_J * m_r, 1e-30))

            z1   = rng.normal()
            z2   = rng.normal()
            z_j1 = rng.normal()
            z_j2 = rng.normal()
            u    = rng.uniform()

            eta = rho * z1 + sqrt_1m_rho2 * z2
            I_t = 1.0 if u < p_J else 0.0
            J_S = mu_JS + sigma_JS * z_j1
            J_V = mu_JV + sigma_JV * (rho_J * z_j1 + sqrt_1m_rhoJ2 * z_j2)

            # Step 1: advance variance (CIR with floor)
            v_t = (
                v_prev
                + kappa * (theta - v_prev) * _DT_MIN
                + sigma_v * np.sqrt(max(v_prev, VARIANCE_FLOOR) * _DT_MIN) * eta
                + I_t * J_V
            )
            V_t = max(v_t, VARIANCE_FLOOR)

            # Step 2: log-return using current V_t
            y_t = (
                np.sqrt(V_t * _DT_MIN) * z1
                + I_t * J_S
            )

            variances[step]   = V_t
            log_returns[step] = y_t
            v_prev = V_t

        return (
            log_returns.astype(np.float32),
            variances.astype(np.float32),
        )

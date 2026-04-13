from stochastic import StochasticProcessBase, Setting, DynSetting
from helper import (
    VARIANCE_FLOOR,
    FilterInfo,
    _positive_variance,
    _gaussian_logpdf,
    _systematic_resample,
    _effective_sample_size,
)

import jax
import jax.numpy as jnp
import chex
import numpy as np

class HestonProcess(StochasticProcessBase):

    PARAM_NAMES = ["v0", "rho", "kappa", "theta", "sigma", "r"]
    PARAM_TRANSFORMS = {
        "v0":    ("sigmoid_ab", 0.01,  1.0),
        "rho":   ("tanh",      -0.99,  0.99),
        "kappa": ("sigmoid_ab", 0.1,  10.0),
        "theta": ("sigmoid_ab", 0.01,  1.0),
        "sigma": ("sigmoid_ab", 0.01,  1.0),
        "r":     ("sigmoid_ab", -0.05, 0.05),
    }

    def __init__(self,
                 popsize: int,
                 num_generations: int,
                 sigma_init: float,
                 dt: float,
                 num_particles: int,
                 S: np.ndarray):
        super().__init__(popsize, num_generations, sigma_init, dt, num_particles, S)

    def get_noises(self, key: chex.PRNGKey) -> chex.Array:
        """Pre-generate all particle filter noise for Common Random Numbers.

        Returns an array of shape (T, N+2) where:
            columns  0..N-1 : N(0,1) variance-propagation noise eps_v
            column   N      : Uniform[0,1) for first-stage systematic resampling
            column   N+1    : Uniform[0,1) for second-stage systematic resampling
        """
        T = self.S.shape[0] - 1
        N = self.num_particles
        key, key_eps, key_u1, key_u2 = jax.random.split(key, 4)
        eps_v = jax.random.normal(key_eps, shape=(T, N))
        u1    = jax.random.uniform(key_u1, shape=(T, 1))
        u2    = jax.random.uniform(key_u2, shape=(T, 1))
        return jnp.concatenate([eps_v, u1, u2], axis=1)           # (T, N+2)

    def loglikelihood(self, x: chex.Array, setting: Setting, dsetting: DynSetting):
        # x: (P, D) — P candidates in constrained parameter space.
        P = x.shape[0]
        N = setting.num_particles
        PN = P * N

        v0    = x[:, 0]
        rho   = x[:, 1]
        kappa = x[:, 2]
        theta = x[:, 3]
        sigma = x[:, 4]
        r_val = x[:, 5]

        # Flatten per-candidate params to (P*N,) for per-particle vmap.
        kappa_pn = jnp.repeat(kappa, N)
        theta_pn = jnp.repeat(theta, N)
        sigma_pn = jnp.repeat(sigma, N)
        rho_pn   = jnp.repeat(rho,   N)
        r_pn     = jnp.repeat(r_val, N)

        log_prices  = jnp.log(dsetting.S)
        log_returns = log_prices[1:] - log_prices[:-1]  # (T,)
        dt_seq      = dsetting.dt_seq                    # (T,)
        noises_seq  = dsetting.noises                    # (T, N+2)  CRN

        # Initial state: particles (P*N,), weights (P, N), total_loglik (P,).
        particles    = jnp.repeat(v0, N)
        log_weights  = jnp.full((P, N), -jnp.log(jnp.float32(N)))
        total_loglik = jnp.zeros((P,), dtype=jnp.float32)

        def _apf_step(carry, xs):
            obs, dt_i, noises_t = xs
            particles_pn, log_weights, total_loglik = carry

            eps_v_t = noises_t[:N]       # (N,)  CRN variance noise
            u1_t    = noises_t[N]        # scalar
            u2_t    = noises_t[N + 1]    # scalar
            # Tile CRN noise across all P candidates.
            eps_pn = jnp.tile(eps_v_t, P)  # (P*N,)

            # ── Pilot: vmap over P*N ─────────────────────────────────────
            def _pilot_one(v, k, th, r):
                vp = _positive_variance(v + k * (th - v) * dt_i)
                return _gaussian_logpdf(obs, (r - jnp.float32(0.5) * vp) * dt_i, vp * dt_i)

            log_g_pn = jax.vmap(_pilot_one)(particles_pn, kappa_pn, theta_pn, r_pn)  # (P*N,)
            log_g    = log_g_pn.reshape(P, N)                                          # (P, N)

            # ── First-stage: vmap over P ──────────────────────────────────
            log_xi     = log_weights + log_g                                                   # (P, N)
            log_Z1     = jax.vmap(jax.nn.logsumexp)(log_xi)                                   # (P,)
            ancestors1 = jax.vmap(lambda lw: _systematic_resample(lw, u1_t))(log_xi)          # (P, N)

            particles_2d = particles_pn.reshape(P, N)
            v_par_2d  = jax.vmap(lambda p, a: p[a])(particles_2d, ancestors1)                 # (P, N)
            log_g_sel = jax.vmap(lambda lg, a: lg[a])(log_g, ancestors1)                      # (P, N)

            # ── Propagation: vmap over P*N ────────────────────────────────
            def _propagate_one(v, k, th, s, e):
                return _positive_variance(
                    v + k * (th - v) * dt_i
                    + s * jnp.sqrt(_positive_variance(v)) * jnp.sqrt(dt_i) * e
                )

            v_next_pn = jax.vmap(_propagate_one)(
                v_par_2d.reshape(PN), kappa_pn, theta_pn, sigma_pn, eps_pn
            )                                                                                   # (P*N,)
            v_next = v_next_pn.reshape(P, N)                                                   # (P, N)

            # ── Conditional likelihood: vmap over P*N ────────────────────
            def _log_p_one(v_n, r, rh, e):
                mu_c   = (r - jnp.float32(0.5) * v_n) * dt_i + jnp.sqrt(_positive_variance(v_n * dt_i)) * rh * e
                sig2_c = _positive_variance(v_n * (jnp.float32(1.0) - rh ** 2) * dt_i)
                return _gaussian_logpdf(obs, mu_c, sig2_c)

            log_p_pn = jax.vmap(_log_p_one)(v_next_pn, r_pn, rho_pn, eps_pn)                 # (P*N,)
            log_p    = log_p_pn.reshape(P, N)                                                  # (P, N)

            log_alpha = log_p - log_g_sel                                                      # (P, N)

            # ── Second-stage: vmap over P ─────────────────────────────────
            log_Z2 = jax.vmap(
                lambda la: jax.nn.logsumexp(la) - jnp.log(jnp.float32(N))
            )(log_alpha)                                                                        # (P,)

            log_increment = log_Z1 + log_Z2
            total_loglik  = total_loglik + log_increment

            # ── Diagnostics: mean over P ──────────────────────────────────
            w_norm    = jax.nn.softmax(log_alpha, axis=-1)                                     # (P, N)
            filt_mean = jnp.sum(w_norm * v_next, axis=-1)                                      # (P,)
            filt_std  = jnp.sqrt(jnp.sum(w_norm * (v_next - filt_mean[:, None]) ** 2, axis=-1))
            ess       = jax.vmap(_effective_sample_size)(log_alpha)                            # (P,)

            # ── Second resample: vmap over P ──────────────────────────────
            ancestors2      = jax.vmap(lambda la: _systematic_resample(la, u2_t))(log_alpha)  # (P, N)
            particles_new   = jax.vmap(lambda vn, a: vn[a])(v_next, ancestors2)               # (P, N)
            log_weights_new = jnp.full((P, N), -jnp.log(jnp.float32(N)))

            # ── One-step-ahead prediction: mean over P*N ──────────────────
            def _pred_one(v, k, th, r):
                vp = _positive_variance(v + k * (th - v) * dt_i)
                return (r - jnp.float32(0.5) * vp) * dt_i, vp * dt_i

            pred_mean_pn, pred_var_pn = jax.vmap(_pred_one)(
                particles_new.reshape(PN), kappa_pn, theta_pn, r_pn
            )
            pred_lr_mean = jnp.mean(pred_mean_pn)
            pred_lr_std  = jnp.sqrt(
                jnp.mean(pred_var_pn) + jnp.mean((pred_mean_pn - pred_lr_mean) ** 2)
            )

            new_carry = (particles_new.reshape(PN), log_weights_new, total_loglik)
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

    def get_default_param(self, key: chex.PRNGKey):
        initial_guess = {
            'v0':    0.02,
            'rho':   0.0,
            'kappa': 2.0,
            'theta': 0.02,
            'sigma': 0.1,
            'r':     0.0,
        }
        num_dims = len(initial_guess)
        initial_guess_unconstrained = self.params_to_unconstrained(initial_guess)

        dt_seq = jnp.full((self.S.shape[0] - 1,), self.dt, dtype=jnp.float32)
        noises = self.get_noises(key)
        dsetting = DynSetting(S=self.S, initial_guess=initial_guess_unconstrained, dt_seq=dt_seq, noises=noises)
        setting = Setting(popsize=self.popsize, num_generations=self.num_generations, num_dims=num_dims,
                          sigma_init=self.sigma_init, dt=self.dt, num_particles=self.num_particles)
        return setting, dsetting

    @staticmethod
    def generator(seed: int, S0: float, length: int, dt: float, params: np.ndarray, dt_seq=None):
        if length < 0:
            raise ValueError("length must be non-negative")
        if S0 <= 0.0:
            raise ValueError("S0 must be positive")

        params = np.asarray(params, dtype=np.float64)
        if params.shape != (6,):
            raise ValueError("params must have shape (6,) ordered as [v0, rho, kappa, theta, sigma, r]")

        v0, rho, kappa, theta, sigma, r = params
        sqrt_one_minus_rho_sq = np.sqrt(max(1.0 - rho ** 2, 1e-8))
        rng = np.random.default_rng(seed)

        variances = np.zeros(length, dtype=np.float64)
        log_returns = np.zeros(length, dtype=np.float64)
        variance_prev = max(v0, VARIANCE_FLOOR)

        for step in range(length):
            dt_step = float(dt_seq[step]) if dt_seq is not None else dt
            sqrt_dt_step = np.sqrt(dt_step)
            eps_variance = rng.normal()
            eps_orthogonal = rng.normal()

            variance_next = (
                variance_prev
                + kappa * (theta - variance_prev) * dt_step
                + sigma * np.sqrt(variance_prev) * sqrt_dt_step * eps_variance
            )
            variance_next = max(variance_next, VARIANCE_FLOOR)

            correlated_shock = rho * eps_variance + sqrt_one_minus_rho_sq * eps_orthogonal
            log_return = (r - 0.5 * variance_next) * dt_step + np.sqrt(variance_next * dt_step) * correlated_shock

            variances[step] = variance_next
            log_returns[step] = log_return
            variance_prev = variance_next

        return log_returns.astype(np.float32), variances.astype(np.float32)
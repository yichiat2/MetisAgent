import jax
import jax.numpy as jnp
import chex
import numpy as np

from stochastic import StochasticProcessBase, Setting, DynSetting
from helper import (
    VARIANCE_FLOOR,
    FilterInfo,
    _positive_variance,
    _gaussian_logpdf,
    _systematic_resample,
    _effective_sample_size,
)


class HestonJumpProcess(StochasticProcessBase):
    """Heston model with correlated Gaussian jumps in log-return and multiplicative
    log-normal jumps in variance.  A single Bernoulli indicator is shared between the
    two jump components; overnight / weekend intervals (dt_i > 1.5 * base_dt) force
    p_J = 1.

    Parameters (12-dim vector):
        x[0]  v0        – initial variance
        x[1]  rho       – price-vol diffusion correlation
        x[2]  kappa     – mean-reversion speed
        x[3]  theta     – long-run variance
        x[4]  sigma     – vol of vol
        x[5]  r         – risk-free drift
        x[6]  p_J       – intraday jump probability per dt
        x[7]  mu_Jr     – mean log-return jump size
        x[8]  sigma_Jr  – std dev log-return jump
        x[9]  mu_JV     – mean of log-variance multiplier
        x[10] sigma_JV  – std dev of log-variance multiplier
        x[11] rho_J     – correlation between J_V and J_r (through shared Z1)
    """

    PARAM_NAMES = [
        "v0", "rho", "kappa", "theta", "sigma", "r",
        "p_J", "mu_Jr", "sigma_Jr", "mu_JV", "sigma_JV", "rho_J",
    ]
    PARAM_TRANSFORMS = {
        "v0":       ("sigmoid_ab", 0.01,  1.0),
        "rho":      ("tanh",      -0.99,  0.99),
        "kappa":    ("sigmoid_ab", 0.1,  10.0),
        "theta":    ("sigmoid_ab", 0.01,  1.0),
        "sigma":    ("sigmoid_ab", 0.01,  1.0),
        "r":        ("sigmoid_ab", -0.05, 0.05),
        "p_J":      ("sigmoid_ab", 0.01,  1.0),
        "mu_Jr":    ("sigmoid_ab", -0.1,  0.1),
        "sigma_Jr": ("sigmoid_ab", 1e-3,  0.3),
        "mu_JV":    ("sigmoid_ab", -1.0,  1.0),
        "sigma_JV": ("sigmoid_ab", 1e-3,  0.69),
        "rho_J":    ("tanh",      -0.99,  0.99),
    }

    def __init__(
        self,
        popsize: int,
        num_generations: int,
        sigma_init: float,
        dt: float,
        num_particles: int,
        S: np.ndarray,
        dt_seq_np: np.ndarray | None = None,
    ):
        super().__init__(popsize, num_generations, sigma_init, dt, num_particles, S)
        self._dt_seq = jnp.array(dt_seq_np, dtype=jnp.float32) if dt_seq_np is not None else jnp.full((self.S.shape[0] - 1,), self.dt, dtype=jnp.float32)

    def get_noises(self, key: chex.PRNGKey) -> chex.Array:
        """Pre-generate all particle filter noise for Common Random Numbers.

        Returns an array of shape (T, 3N+2) where:
            columns   0..N-1  : N(0,1) variance-propagation noise eps_v
            columns   N..2N-1 : N(0,1) jump-variance shock Z1
            columns  2N..3N-1 : Uniform[0,1) for Bernoulli jump indicator
            column   3N       : Uniform[0,1) for first-stage systematic resampling
            column   3N+1     : Uniform[0,1) for second-stage systematic resampling
        """
        T = self.S.shape[0] - 1
        N = self.num_particles
        key, key_eps, key_z1, key_u_jump, key_u1, key_u2 = jax.random.split(key, 6)
        eps_v  = jax.random.normal(key_eps,  shape=(T, N))
        Z1     = jax.random.normal(key_z1,   shape=(T, N))
        u_jump = jax.random.uniform(key_u_jump, shape=(T, N))
        u1     = jax.random.uniform(key_u1,  shape=(T, 1))
        u2     = jax.random.uniform(key_u2,  shape=(T, 1))
        return jnp.concatenate([eps_v, Z1, u_jump, u1, u2], axis=1)  # (T, 3N+2)

    # ------------------------------------------------------------------
    # APF log-likelihood
    # ------------------------------------------------------------------

    def loglikelihood(self, x: chex.Array, setting: Setting, dsetting: DynSetting):
        # x: (P, D) — P candidates in constrained parameter space.
        P = x.shape[0]
        N = setting.num_particles
        PN = P * N

        v0       = x[:, 0]
        rho      = x[:, 1]
        kappa    = x[:, 2]
        theta    = x[:, 3]
        sigma    = x[:, 4]
        r_val    = x[:, 5]
        p_J      = x[:, 6]
        mu_Jr    = x[:, 7]
        sigma_Jr = x[:, 8]
        mu_JV    = x[:, 9]
        sigma_JV = x[:, 10]
        rho_J    = x[:, 11]

        base_dt = jnp.float32(setting.dt)

        # Flatten per-candidate params to (P*N,) for per-particle vmap.
        kappa_pn    = jnp.repeat(kappa,    N)
        theta_pn    = jnp.repeat(theta,    N)
        sigma_pn    = jnp.repeat(sigma,    N)
        rho_pn      = jnp.repeat(rho,      N)
        r_pn        = jnp.repeat(r_val,    N)
        mu_Jr_pn    = jnp.repeat(mu_Jr,    N)
        sigma_Jr_pn = jnp.repeat(sigma_Jr, N)
        mu_JV_pn    = jnp.repeat(mu_JV,    N)
        sigma_JV_pn = jnp.repeat(sigma_JV, N)
        rho_J_pn    = jnp.repeat(rho_J,    N)

        m_V_pn = jnp.exp(mu_JV_pn + jnp.float32(0.5) * sigma_JV_pn ** 2)  # (P*N,)

        log_prices  = jnp.log(dsetting.S)
        log_returns = log_prices[1:] - log_prices[:-1]  # (T,)
        dt_seq      = dsetting.dt_seq                    # (T,)
        noises_seq  = dsetting.noises                    # (T, 3N+2)  CRN

        # Initial state: particles (P*N,), weights (P, N), total_loglik (P,).
        particles    = jnp.repeat(v0, N)
        log_weights  = jnp.full((P, N), -jnp.log(jnp.float32(N)))
        total_loglik = jnp.zeros((P,), dtype=jnp.float32)

        def _apf_step(carry, xs):
            obs, dt_i, noises_t = xs
            particles_pn, log_weights, total_loglik = carry

            eps_v_t  = noises_t[:N]          # (N,)
            Z1_t     = noises_t[N:2*N]       # (N,)
            u_jump_t = noises_t[2*N:3*N]     # (N,)
            u1_t     = noises_t[3*N]         # scalar
            u2_t     = noises_t[3*N + 1]     # scalar

            eps_pn    = jnp.tile(eps_v_t,  P)   # (P*N,)
            Z1_pn     = jnp.tile(Z1_t,     P)   # (P*N,)
            u_jump_pn = jnp.tile(u_jump_t, P)   # (P*N,)

            # Per-candidate jump probability (forced to 0 intraday, p_J overnight).
            p_t_p  = jnp.where(dt_i > jnp.float32(1.5) * base_dt, p_J, jnp.float32(0.0))  # (P,)
            p_t_pn = jnp.repeat(p_t_p, N)                                                    # (P*N,)

            # ── Pilot: vmap over P*N ─────────────────────────────────────
            def _pilot_one(v, k, th, r, p_t, m_v, mu_jr, sig_jr):
                v0_ = _positive_variance(v + k * (th - v) * base_dt)
                v1_ = v0_ * m_v
                lg0 = _gaussian_logpdf(obs, (r - jnp.float32(0.5) * v0_) * base_dt, v0_ * base_dt)
                lg1 = _gaussian_logpdf(obs, (r - jnp.float32(0.5) * v1_) * base_dt + mu_jr, v1_ * base_dt + sig_jr ** 2)
                return jnp.logaddexp(
                    jnp.log(jnp.float32(1.0) - p_t) + lg0,
                    jnp.log(p_t) + lg1,
                )

            log_g_pn = jax.vmap(_pilot_one)(
                particles_pn, kappa_pn, theta_pn, r_pn, p_t_pn, m_V_pn, mu_Jr_pn, sigma_Jr_pn
            )                                                                                  # (P*N,)
            log_g = log_g_pn.reshape(P, N)

            # ── First-stage: vmap over P ──────────────────────────────────
            log_xi     = log_weights + log_g                                                   # (P, N)
            log_Z1     = jax.vmap(jax.nn.logsumexp)(log_xi)                                   # (P,)
            ancestors1 = jax.vmap(lambda lw: _systematic_resample(lw, u1_t))(log_xi)          # (P, N)

            particles_2d = particles_pn.reshape(P, N)
            v_par_2d  = jax.vmap(lambda p, a: p[a])(particles_2d, ancestors1)                 # (P, N)
            log_g_sel = jax.vmap(lambda lg, a: lg[a])(log_g, ancestors1)                      # (P, N)

            # ── Propagation: vmap over P*N ────────────────────────────────
            def _propagate_one(v, k, th, s, e, Z1, u_jump, mu_jv, sig_jv, p_t):
                I_t    = (u_jump < p_t).astype(jnp.float32)
                J_V    = mu_jv + sig_jv * Z1
                v_cont = _positive_variance(
                    v + k * (th - v) * base_dt
                    + s * jnp.sqrt(_positive_variance(v) * base_dt) * e
                )
                return v_cont * jnp.exp(I_t * J_V)

            v_next_pn = jax.vmap(_propagate_one)(
                v_par_2d.reshape(PN), kappa_pn, theta_pn, sigma_pn,
                eps_pn, Z1_pn, u_jump_pn, mu_JV_pn, sigma_JV_pn, p_t_pn
            )                                                                                  # (P*N,)
            v_next = v_next_pn.reshape(P, N)

            # ── Conditional likelihood: vmap over P*N ────────────────────
            def _log_p_one(v_n, r, rh, e, Z1, u_jump, mu_jr, sig_jr, rh_j, p_t):
                I_t  = (u_jump < p_t).astype(jnp.float32)
                mu_c = (
                    (r - jnp.float32(0.5) * v_n) * base_dt
                    + rh * jnp.sqrt(_positive_variance(v_n) * base_dt) * e
                    + I_t * (mu_jr + sig_jr * rh_j * Z1)
                )
                sig2_c = _positive_variance(
                    v_n * (jnp.float32(1.0) - rh ** 2) * base_dt
                    + I_t * sig_jr ** 2 * (jnp.float32(1.0) - rh_j ** 2)
                )
                return _gaussian_logpdf(obs, mu_c, sig2_c)

            log_p_pn = jax.vmap(_log_p_one)(
                v_next_pn, r_pn, rho_pn, eps_pn, Z1_pn, u_jump_pn,
                mu_Jr_pn, sigma_Jr_pn, rho_J_pn, p_t_pn
            )                                                                                  # (P*N,)
            log_p    = log_p_pn.reshape(P, N)

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
            def _pred_one(v, k, th, r, p_t, m_v, mu_jr, sig_jr):
                vp0 = _positive_variance(v + k * (th - v) * base_dt)
                vp1 = vp0 * m_v
                mu0 = (r - jnp.float32(0.5) * vp0) * base_dt
                mu1 = (r - jnp.float32(0.5) * vp1) * base_dt + mu_jr
                r_mean = (jnp.float32(1.0) - p_t) * mu0 + p_t * mu1
                r_var  = (
                    (jnp.float32(1.0) - p_t) * vp0 * base_dt
                    + p_t * (vp1 * base_dt + sig_jr ** 2)
                    + p_t * (jnp.float32(1.0) - p_t) * (mu1 - mu0) ** 2
                )
                return r_mean, r_var

            pred_mean_pn, pred_var_pn = jax.vmap(_pred_one)(
                particles_new.reshape(PN), kappa_pn, theta_pn, r_pn,
                p_t_pn, m_V_pn, mu_Jr_pn, sigma_Jr_pn
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

    # ------------------------------------------------------------------
    # Default parameters (annualised, dt in fraction of year)
    # ------------------------------------------------------------------

    def get_default_param(self, key: chex.PRNGKey):
        initial_guess = {
            'v0':       0.04,
            'rho':      0.0,
            'kappa':    2.0,
            'theta':    0.04,
            'sigma':    0.3,
            'r':        0.0,
            'p_J':      0.1,
            'mu_Jr':    0.0,
            'sigma_Jr': 0.1,
            'mu_JV':    0.0,
            'sigma_JV': 0.1,
            'rho_J':    0,
        }

        num_dims = len(initial_guess)
        initial_guess_unconstrained = self.params_to_unconstrained(initial_guess)
        noises = self.get_noises(key)

        dsetting = DynSetting(
            S=self.S,
            initial_guess=initial_guess_unconstrained,
            dt_seq=self._dt_seq,
            noises=noises,
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
        length: int,
        dt: float,
        params: np.ndarray,
        dt_seq: np.ndarray | None = None,
    ):
        """Generate synthetic log-returns and variance paths under the jump model.

        Args:
            seed:   RNG seed.
            S0:     Initial price (> 0).
            length: Number of time steps.
            dt:     Nominal (base) time increment — used as the reference for the
                    overnight gate condition (dt_step > 1.5 * dt).
            params: Shape (12,) array ordered as
                    [v0, rho, kappa, theta, sigma, r,
                     p_J, mu_Jr, sigma_Jr, mu_JV, sigma_JV, rho_J].
            dt_seq: Optional length-``length`` array of per-step time increments.
                    When None every step uses ``dt``.

        Returns:
            (log_returns, variances) each of shape (length,) and dtype float32.
        """
        if length < 0:
            raise ValueError("length must be non-negative")
        if S0 <= 0.0:
            raise ValueError("S0 must be positive")

        params = np.asarray(params, dtype=np.float64)
        if params.shape != (12,):
            raise ValueError(
                "params must have shape (12,) ordered as "
                "[v0, rho, kappa, theta, sigma, r, p_J, mu_Jr, sigma_Jr, mu_JV, sigma_JV, rho_J]"
            )

        v0, rho, kappa, theta, sigma, r, p_J, mu_Jr, sigma_Jr, mu_JV, sigma_JV, rho_J = params
        sqrt_one_minus_rho_sq   = np.sqrt(max(1.0 - rho   ** 2, 1e-8))
        sqrt_one_minus_rho_J_sq = np.sqrt(max(1.0 - rho_J ** 2, 1e-8))
        rng = np.random.default_rng(seed)

        variances   = np.zeros(length, dtype=np.float64)
        log_returns = np.zeros(length, dtype=np.float64)
        variance_prev = max(v0, VARIANCE_FLOOR)

        for step in range(length):
            dt_step = float(dt_seq[step]) if dt_seq is not None else dt
            p_t     = p_J if dt_step > 1.5 * dt else 0.0

            eps_v = rng.normal()
            eps_s = rng.normal()
            Z1    = rng.normal()
            Z2    = rng.normal()
            u     = rng.random()

            I_t = 1.0 if u < p_t else 0.0

            # Log-variance multiplier (eq. 4a)
            J_V = mu_JV + sigma_JV * Z1

            # Floor diffusion part, then apply multiplicative jump (eq. 1)
            v_cont = max(
                variance_prev
                + kappa * (theta - variance_prev) * dt_step
                + sigma * np.sqrt(variance_prev) * np.sqrt(dt_step) * eps_v,
                VARIANCE_FLOOR,
            )
            v_next = v_cont * np.exp(I_t * J_V)

            # Log-return jump (eq. 4b)
            J_r = mu_Jr + sigma_Jr * (rho_J * Z1 + sqrt_one_minus_rho_J_sq * Z2)

            # Log-return (eq. 2)
            correlated_shock = rho * eps_v + sqrt_one_minus_rho_sq * eps_s
            log_return = (
                (r - 0.5 * v_next) * dt_step
                + np.sqrt(v_next * dt_step) * correlated_shock
                + I_t * J_r
            )

            variances[step]   = v_next
            log_returns[step] = log_return
            variance_prev     = v_next

        return log_returns.astype(np.float32), variances.astype(np.float32)

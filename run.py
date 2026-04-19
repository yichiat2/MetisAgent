from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from inhomo_heston_process import InhomoHestonProcess
from qrh_process import QRHProcess
from semivariance_heston_process import SemivarianceHestonProcess
from constants import _MINS_PER_DAY, _DT_MIN, _DT_OVERNIGHT


PARAM_NAMES = ("v0", "rho", "kappa", "theta", "sigma", "r")
PARAM_NAMES_INHOMO = ("v0", "rho", "kappa", "theta", "sigma", "r", "lambda_ov")
PARAM_NAMES_QRH = ("a0", "a1", "a2", "rho", "alpha", "sigma_obs", "lambda_ov")
PARAM_NAMES_SV = (
    "v0p", "v0m",
    "kappa_p", "kappa_m",
    "theta_p", "theta_m",
    "sigma_p", "sigma_m",
    "rho_p", "rho_m", "rho_pm",
    "r", "lambda_ov",
)
PARAM_NAMES_JUMP = (
    "v0", "rho", "kappa", "theta", "sigma", "r",
    "p_J", "mu_Jr", "sigma_Jr", "mu_JV", "sigma_JV", "rho_J",
)


def _to_param_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES, values, strict=True)}


def _to_jump_param_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES_JUMP, values, strict=True)}


def _make_intraday_dt_seq(num_days: int) -> np.ndarray:
    """Build a dt_seq (fraction of year) for ``num_days`` trading days.

    Each day has 390 intraday steps of _DT_MIN followed by one overnight
    step of _DT_OVERNIGHT (1050 min), so the sequence has length
    ``num_days * 390`` with overnight entries at positions
    389, 779, 1169, …
    """
    length  = num_days * _MINS_PER_DAY
    indices = np.arange(length, dtype=np.int32)
    dt_seq  = np.where((indices + 1) % _MINS_PER_DAY == 0, _DT_OVERNIGHT, _DT_MIN)
    return dt_seq.astype(np.float32)


def _prices_from_log_returns(S0: float, log_returns: np.ndarray) -> np.ndarray:
    cumulative_log_returns = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(log_returns, dtype=np.float64)]
    )
    prices = S0 * np.exp(cumulative_log_returns)
    return prices.astype(np.float32)


def _save_diagnostic_plot(
    prices: np.ndarray,
    log_returns: np.ndarray,
    true_variance: np.ndarray,
    filtered_variance: np.ndarray,
    filtered_std: np.ndarray,
    plot_path: str | Path,
    true_param_filtered_variance: np.ndarray | None = None,
) -> str:
    plot_path = Path(plot_path)

    timeline = np.arange(log_returns.shape[0])
    price_timeline = np.arange(prices.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)

    axes[0].plot(price_timeline, prices, linewidth=1.5)
    axes[0].set_title("Synthetic Heston Price Path")
    axes[0].set_ylabel("Price")

    axes[1].plot(timeline, log_returns, linewidth=1.0)
    axes[1].set_title("Observed Log Returns")
    axes[1].set_ylabel("Log return")

    lower = np.maximum(filtered_variance - 2.0 * filtered_std, 0.0)
    upper = filtered_variance + 2.0 * filtered_std
    axes[2].plot(timeline, true_variance, label="true variance", linewidth=1.5)
    axes[2].plot(timeline, filtered_variance, label="filtered variance (fitted)", linewidth=1.5)
    axes[2].fill_between(timeline, lower, upper, alpha=0.2, label="filtered +/- 2 std")
    if true_param_filtered_variance is not None:
        axes[2].plot(timeline, true_param_filtered_variance,
                     label="filtered variance (true params)", linewidth=1.5, linestyle="--")
    axes[2].set_title("Variance Filtering")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Variance")
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


def run_heston(
    seed: int = 7,
    S0: float = 100.0,
    length: int = 5000,
    dt: float = _DT_MIN,
    true_params: np.ndarray | None = None,
    popsize: int = 256,
    num_generations: int = 100,
    sigma_init: float = 0.1,
    num_particles: int = 2048,
    plot_path: str | Path | None = None,
) -> dict[str, object]:
    if true_params is None:
        true_params = np.array([0.04, -0.7, 4.0, 0.04, 0.2, 0.0], dtype=np.float32)
    else:
        true_params = np.asarray(true_params, dtype=np.float32)

    log_returns, true_variance = HestonProcess.generator(
        seed=seed,
        S0=S0,
        length=length,
        dt=dt,
        params=true_params,
    )
    prices = _prices_from_log_returns(S0, log_returns)

    process = HestonProcess(
        popsize=popsize,
        num_generations=num_generations,
        sigma_init=sigma_init,
        dt=dt,
        num_particles=num_particles,
        S=prices,
    )
    setting, dsetting = process.get_default_param(jax.random.PRNGKey(seed))

    fit_key = jax.random.PRNGKey(seed + 1)
    best_member, bic = process.calibrate(fit_key, setting, dsetting)
    fitted_params = np.asarray(
        jax.device_get(process.unconstrained_to_params(best_member)),
        dtype=np.float32,
    )

    filter_carry, filter_info = process.loglikelihood(
        jnp.asarray(fitted_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    filtered_variance = np.asarray(jax.device_get(filter_info.filtered_mean), dtype=np.float32)
    filtered_std = np.asarray(jax.device_get(filter_info.filtered_std), dtype=np.float32)
    ess = np.asarray(jax.device_get(filter_info.ess), dtype=np.float32)
    loglik_increments = np.asarray(jax.device_get(filter_info.loglik_increments), dtype=np.float32)
    total_loglik = float(jax.device_get(filter_carry[-1][0]))
    bic = float(jax.device_get(bic))

    result: dict[str, object] = {
        "prices": prices,
        "log_returns": log_returns,
        "true_variance": true_variance,
        "filtered_variance": filtered_variance,
        "filtered_std": filtered_std,
        "ess": ess,
        "loglik_increments": loglik_increments,
        "true_params": _to_param_dict(true_params),
        "fitted_params": _to_param_dict(fitted_params),
        "best_loglik": total_loglik,
        "bic": bic,
        "variance_rmse": float(np.sqrt(np.mean((filtered_variance - true_variance) ** 2))),
        "mean_ess": float(np.mean(ess)),
    }

    if plot_path is not None:
        result["plot_path"] = _save_diagnostic_plot(
            prices=prices,
            log_returns=log_returns,
            true_variance=true_variance,
            filtered_variance=filtered_variance,
            filtered_std=filtered_std,
            plot_path=plot_path,
        )

    return result


def run_heston_jump(
    seed: int = 7,
    S0: float = 100.0,
    num_days: int = 40,
    true_params: np.ndarray | None = None,
    popsize: int = 256,
    num_generations: int = 100,
    sigma_init: float = 0.5,
    num_particles: int = 1024,
    plot_path: str | Path | None = None,
) -> dict[str, object]:
    """Run synthetic-data generation, CMA-ES calibration, and particle-filter
    evaluation for the Heston-with-jumps model at 1-minute resolution.

    The dt_seq encodes 390 intraday 1-min steps followed by one 1050-min
    overnight gap, repeating for every trading day.
    """
    if true_params is None:
        true_params = np.array(
            # v0,   rho,   kappa, theta, sigma, r
            [0.04, -0.7,   4.0,  0.04,  0.2,   0.0,
            # p_J,   mu_Jr, sigma_Jr, mu_JV, sigma_JV, rho_J
             0.50,  0.0,   0.15,     0.0,   0.05,     -0.5],
            dtype=np.float32,
        )
    else:
        true_params = np.asarray(true_params, dtype=np.float32)

    dt_seq_np = _make_intraday_dt_seq(num_days)
    length    = len(dt_seq_np)

    log_returns, true_variance = HestonJumpProcess.generator(
        seed=seed,
        S0=S0,
        length=length,
        dt=_DT_MIN,
        params=true_params,
        dt_seq=dt_seq_np,
    )
    prices = _prices_from_log_returns(S0, log_returns)
    # plot the prices and true variance to sanity check the data generation
    # before running the (more expensive) CMA-ES calibration and particle filter
    if plot_path is not None:
        _save_diagnostic_plot(
            prices=prices,
            log_returns=log_returns,
            true_variance=true_variance,
            filtered_variance=true_variance, # just plot the true variance as a sanity check
            filtered_std=np.zeros_like(true_variance), # no uncertainty bands for the true variance, so set std to zero
            plot_path=f"data_generation_{plot_path}",
        )
    process = HestonJumpProcess(
        popsize=popsize,
        num_generations=num_generations,
        sigma_init=sigma_init,
        dt=_DT_MIN,
        num_particles=num_particles,
        S=prices,
        dt_seq_np=dt_seq_np,
    )
    setting, dsetting = process.get_default_param(jax.random.PRNGKey(seed))

    fit_key = jax.random.PRNGKey(seed + 1)
    best_member, bic = process.calibrate(fit_key, setting, dsetting)
    fitted_params = np.asarray(
        jax.device_get(process.unconstrained_to_params(best_member)),
        dtype=np.float32,
    )

    filter_carry, filter_info = process.loglikelihood(
        jnp.asarray(fitted_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    filtered_variance  = np.asarray(jax.device_get(filter_info.filtered_mean),    dtype=np.float32)
    filtered_std       = np.asarray(jax.device_get(filter_info.filtered_std),     dtype=np.float32)
    ess                = np.asarray(jax.device_get(filter_info.ess),              dtype=np.float32)
    loglik_increments  = np.asarray(jax.device_get(filter_info.loglik_increments),dtype=np.float32)
    total_loglik       = float(jax.device_get(filter_carry[-1][0]))
    bic                = float(jax.device_get(bic))

    result: dict[str, object] = {
        "prices":            prices,
        "log_returns":       log_returns,
        "true_variance":     true_variance,
        "filtered_variance": filtered_variance,
        "filtered_std":      filtered_std,
        "ess":               ess,
        "loglik_increments": loglik_increments,
        "true_params":       _to_jump_param_dict(true_params),
        "fitted_params":     _to_jump_param_dict(fitted_params),
        "best_loglik":       total_loglik,
        "bic":               bic,
        "variance_rmse":     float(np.sqrt(np.mean((filtered_variance - true_variance) ** 2))),
        "mean_ess":          float(np.mean(ess)),
    }

    if plot_path is not None:
        result["plot_path"] = _save_diagnostic_plot(
            prices=prices,
            log_returns=log_returns,
            true_variance=true_variance,
            filtered_variance=filtered_variance,
            filtered_std=filtered_std,
            plot_path=plot_path,
        )

    return result


def _to_inhomo_param_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES_INHOMO, values, strict=True)}


def _to_qrh_param_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES_QRH, values, strict=True)}


def _to_sv_param_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES_SV, values, strict=True)}


def _save_sv_diagnostic_plot(
    prices: np.ndarray,
    log_returns: np.ndarray,
    true_variance_p: np.ndarray,
    true_variance_m: np.ndarray,
    filtered_mean_p: np.ndarray,
    filtered_std_p: np.ndarray,
    filtered_mean_m: np.ndarray,
    filtered_std_m: np.ndarray,
    plot_path: str | Path,
    true_param_filtered_mean_p: np.ndarray | None = None,
    true_param_filtered_mean_m: np.ndarray | None = None,
) -> str:
    """Four-panel diagnostic plot for the Semivariance Heston model.

    Panel 1: Price path.
    Panel 2: Log returns (up-bars green, down-bars red).
    Panel 3: Upside variance v+ — true vs filtered (fitted) vs filtered (true params).
    Panel 4: Downside variance v- — same layout.
    """
    plot_path = Path(plot_path)
    T = log_returns.shape[0]
    tl = np.arange(T)
    pl = np.arange(prices.shape[0])

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)

    # Panel 1 – price
    axes[0].plot(pl, prices, linewidth=1.2, color="steelblue")
    axes[0].set_title("Synthetic Semivariance Heston Price Path")
    axes[0].set_ylabel("Price")

    # Panel 2 – returns with direction colouring
    colors = np.where(log_returns >= 0, "green", "red")
    axes[1].bar(tl, log_returns, color=colors, width=1.0, linewidth=0)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("Observed Log Returns  (green = up-bar, red = down-bar)")
    axes[1].set_ylabel("Log return")

    # Panel 3 – upside variance v+
    ax = axes[2]
    ax.plot(tl, true_variance_p, label="true v+",     color="forestgreen", linewidth=1.5)
    ax.plot(tl, filtered_mean_p, label="filtered v+ (fitted)", color="limegreen",
            linewidth=1.2, linestyle="--")
    lower_p = np.maximum(filtered_mean_p - 2.0 * filtered_std_p, 0.0)
    upper_p = filtered_mean_p + 2.0 * filtered_std_p
    ax.fill_between(tl, lower_p, upper_p, alpha=0.15, color="limegreen")
    if true_param_filtered_mean_p is not None:
        ax.plot(tl, true_param_filtered_mean_p, label="filtered v+ (true params)",
                color="darkgreen", linewidth=1.0, linestyle=":")
    ax.set_title("Upside Variance v+")
    ax.set_ylabel("Variance")
    ax.legend(loc="upper right", fontsize=8)

    # Panel 4 – downside variance v-
    ax = axes[3]
    ax.plot(tl, true_variance_m, label="true v-",     color="firebrick", linewidth=1.5)
    ax.plot(tl, filtered_mean_m, label="filtered v- (fitted)", color="tomato",
            linewidth=1.2, linestyle="--")
    lower_m = np.maximum(filtered_mean_m - 2.0 * filtered_std_m, 0.0)
    upper_m = filtered_mean_m + 2.0 * filtered_std_m
    ax.fill_between(tl, lower_m, upper_m, alpha=0.15, color="tomato")
    if true_param_filtered_mean_m is not None:
        ax.plot(tl, true_param_filtered_mean_m, label="filtered v- (true params)",
                color="darkred", linewidth=1.0, linestyle=":")
    ax.set_title("Downside Variance v-")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Variance")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(plot_path)


def run_semivariance_heston(
    seed: int = 10,
    S0: float = 100.0,
    num_days: int = 60,
    true_params: np.ndarray | None = None,
    popsize: int = 256,
    num_generations: int = 200,
    sigma_init: float = 0.8,
    num_particles: int = 4096,
    plot_path: str | Path | None = None,
) -> dict[str, object]:
    """Run synthetic-data generation, CMA-ES calibration, and APF evaluation
    for the Semivariance Heston model.

    Data layout: (390 + 1) * num_days steps.  Steps 0–389 within each day
    have dt = 1 minute; step 390 has dt = 1050 minutes (overnight gap),
    sub-stepped by round(1050 * lambda_ov) intraday steps for variance
    propagation.

    Observation: close-to-close log-return  r_t = log(C_t / C_{t-1}).
    Latent state: (v+_t, v-_t) — upside and downside CIR variances.

    Args:
        seed:            RNG seed.
        S0:              Initial price.
        num_days:        Number of simulated trading days.
        true_params:     Shape-(13,) array
                         [v0p, v0m, kappa_p, kappa_m, theta_p, theta_m,
                          sigma_p, sigma_m, rho_p, rho_m, rho_pm, r, lambda_ov].
                         Defaults to a realistic asymmetric parameter set.
        popsize:         CMA-ES population size.
        num_generations: CMA-ES iteration budget.
        sigma_init:      CMA-ES initial step-size.
        num_particles:   APF particle count.
        plot_path:       If given, saves a four-panel diagnostic PNG at this path.

    Returns:
        Dictionary with keys:
            prices, log_returns,
            true_variance_p, true_variance_m,
            filtered_mean_p, filtered_std_p,
            filtered_mean_m, filtered_std_m,
            ess, loglik_increments,
            true_params, fitted_params,
            best_loglik, bic,
            variance_rmse_p, variance_rmse_m,
            mean_ess,
            [true_param_filtered_mean_p, true_param_filtered_mean_m,
             true_param_loglik,
             true_param_variance_rmse_p, true_param_variance_rmse_m],
            [plot_path].
    """
    if true_params is None:
        
        true_params = np.array(
            [
                0.04,  0.06,   # v0p, v0m
                4.0,   4.0,    # kappa_p, kappa_m
                0.04,  0.06,   # theta_p, theta_m
                0.3,   0.3,    # sigma_p, sigma_m
               -0.3,  -0.6,   # rho_p,   rho_m
                0.5,           # rho_pm
                0.0,           # r
                0.2,           # lambda_ov
            ],
            dtype=np.float32,
        )
    else:
        true_params = np.asarray(true_params, dtype=np.float32)

    log_returns, true_variance_p, true_variance_m = SemivarianceHestonProcess.generator(
        seed=seed,
        S0=S0,
        num_days=num_days,
        params=true_params,
    )
    prices = _prices_from_log_returns(S0, log_returns)

    # Data-generation sanity plot (true variance only, no filter yet)
    if plot_path is not None:
        _save_sv_diagnostic_plot(
            prices=prices,
            log_returns=log_returns,
            true_variance_p=true_variance_p,
            true_variance_m=true_variance_m,
            filtered_mean_p=true_variance_p,
            filtered_std_p=np.zeros_like(true_variance_p),
            filtered_mean_m=true_variance_m,
            filtered_std_m=np.zeros_like(true_variance_m),
            plot_path=f"data_generation_{plot_path}",
        )

    process = SemivarianceHestonProcess(
        popsize=popsize,
        num_generations=num_generations,
        sigma_init=sigma_init,
        dt=float(_DT_MIN),
        num_particles=num_particles,
        S=prices,
        rho_cpm=0.99,
    )
    setting, dsetting = process.get_default_param(jax.random.PRNGKey(seed))

    fit_key = jax.random.PRNGKey(seed + 1)
    best_member, bic = process.calibrate(fit_key, setting, dsetting)
    fitted_params = np.asarray(
        jax.device_get(process.unconstrained_to_params(best_member)),
        dtype=np.float32,
    )

    # Filter pass with fitted parameters
    filter_carry, filter_info = process.loglikelihood(
        jnp.asarray(fitted_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    # Filter pass with true parameters (upper-bound reference)
    true_params[1] = true_params[1] - true_params[0] # 1: v0m -> dv0m
    true_params[5] = true_params[5] - true_params[4] # 5: theta_m -> dtheta_m
    true_filter_carry, true_filter_info = process.loglikelihood(
        jnp.asarray(true_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    # ── Collect results ────────────────────────────────────────────────
    filtered_mean_p  = np.asarray(jax.device_get(filter_info.filtered_mean_p),  dtype=np.float32)
    filtered_std_p   = np.asarray(jax.device_get(filter_info.filtered_std_p),   dtype=np.float32)
    filtered_mean_m  = np.asarray(jax.device_get(filter_info.filtered_mean_m),  dtype=np.float32)
    filtered_std_m   = np.asarray(jax.device_get(filter_info.filtered_std_m),   dtype=np.float32)
    ess              = np.asarray(jax.device_get(filter_info.ess),               dtype=np.float32)
    loglik_increments = np.asarray(jax.device_get(filter_info.loglik_increments), dtype=np.float32)
    total_loglik     = float(jax.device_get(filter_carry[-1][0]))
    bic              = float(jax.device_get(bic))

    true_param_filtered_mean_p = np.asarray(
        jax.device_get(true_filter_info.filtered_mean_p), dtype=np.float32
    )
    true_param_filtered_mean_m = np.asarray(
        jax.device_get(true_filter_info.filtered_mean_m), dtype=np.float32
    )
    true_param_loglik = float(jax.device_get(true_filter_carry[-1][0]))

    result: dict[str, object] = {
        "prices":               prices,
        "log_returns":          log_returns,
        "true_variance_p":      true_variance_p,
        "true_variance_m":      true_variance_m,
        "filtered_mean_p":      filtered_mean_p,
        "filtered_std_p":       filtered_std_p,
        "filtered_mean_m":      filtered_mean_m,
        "filtered_std_m":       filtered_std_m,
        "ess":                  ess,
        "loglik_increments":    loglik_increments,
        "true_params":          _to_sv_param_dict(true_params),
        "fitted_params":        _to_sv_param_dict(fitted_params),
        "best_loglik":          total_loglik,
        "bic":                  bic,
        "variance_rmse_p":      float(np.sqrt(np.mean((filtered_mean_p - true_variance_p) ** 2))),
        "variance_rmse_m":      float(np.sqrt(np.mean((filtered_mean_m - true_variance_m) ** 2))),
        "mean_ess":             float(np.mean(ess)),
        "true_param_filtered_mean_p":  true_param_filtered_mean_p,
        "true_param_filtered_mean_m":  true_param_filtered_mean_m,
        "true_param_loglik":           true_param_loglik,
        "true_param_variance_rmse_p":  float(
            np.sqrt(np.mean((true_param_filtered_mean_p - true_variance_p) ** 2))
        ),
        "true_param_variance_rmse_m":  float(
            np.sqrt(np.mean((true_param_filtered_mean_m - true_variance_m) ** 2))
        ),
    }

    if plot_path is not None:
        result["plot_path"] = _save_sv_diagnostic_plot(
            prices=prices,
            log_returns=log_returns,
            true_variance_p=true_variance_p,
            true_variance_m=true_variance_m,
            filtered_mean_p=filtered_mean_p,
            filtered_std_p=filtered_std_p,
            filtered_mean_m=filtered_mean_m,
            filtered_std_m=filtered_std_m,
            plot_path=plot_path,
            true_param_filtered_mean_p=true_param_filtered_mean_p,
            true_param_filtered_mean_m=true_param_filtered_mean_m,
        )

    return result


def run_inhomo_heston(
    seed: int = 9,
    S0: float = 100.0,
    num_days: int = 40,
    true_params: np.ndarray | None = None,
    popsize: int = 256,
    num_generations: int = 100,
    sigma_init: float = 0.8,
    num_particles: int = 2048,
    plot_path: str | Path | None = None,
) -> dict[str, object]:
    """Run synthetic-data generation, CMA-ES calibration, and particle-filter
    evaluation for the Inhomogeneous Heston model.

    Data layout: (390 + 1) * num_days steps.  Steps 0-389 within each day
    have dt = 1 minute; step 390 has dt = 1050 minutes (overnight gap).
    Overnight variance is propagated using round(1050 * lambda_ov) sub-steps.

    Args:
        seed:            RNG seed.
        S0:              Initial price.
        num_days:        Number of simulated trading days.
        true_params:     Shape-(7,) array [v0, rho, kappa, theta, sigma, r, lambda_ov].
                         Defaults to a reasonable set if None.
        popsize:         CMA-ES population size.
        num_generations: CMA-ES iteration budget.
        sigma_init:      CMA-ES initial step-size.
        num_particles:   Particle-filter particle count.
        plot_path:       If given, saves a diagnostic PNG at this path.

    Returns:
        Dictionary with keys: prices, log_returns, true_variance,
        filtered_variance, filtered_std, ess, loglik_increments,
        true_params, fitted_params, best_loglik, bic, variance_rmse,
        mean_ess, [plot_path].
    """
    if true_params is None:
        true_params = np.array(
            [0.04, -0.7, 4.0, 0.04, 0.2, 0.0, 0.2], dtype=np.float32
        )
    else:
        true_params = np.asarray(true_params, dtype=np.float32)

    log_returns, true_variance = InhomoHestonProcess.generator(
        seed=seed,
        S0=S0,
        num_days=num_days,
        params=true_params,
    )
    prices = _prices_from_log_returns(S0, log_returns)

    if plot_path is not None:
        _save_diagnostic_plot(
            prices=prices,
            log_returns=log_returns,
            true_variance=true_variance,
            filtered_variance=true_variance,
            filtered_std=np.zeros_like(true_variance),
            plot_path=f"data_generation_{plot_path}",
        )

    process = InhomoHestonProcess(
        popsize=popsize,
        num_generations=num_generations,
        sigma_init=sigma_init,
        dt=float(_DT_MIN),
        num_particles=num_particles,
        S=prices,
        rho_cpm=0.99,
    )
    setting, dsetting = process.get_default_param(jax.random.PRNGKey(seed))

    fit_key = jax.random.PRNGKey(seed + 1)
    best_member, bic = process.calibrate(fit_key, setting, dsetting)
    fitted_params = np.asarray(
        jax.device_get(process.unconstrained_to_params(best_member)),
        dtype=np.float32,
    )

    filter_carry, filter_info = process.loglikelihood(
        jnp.asarray(fitted_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    # Second filter pass using true parameters
    true_filter_carry, true_filter_info = process.loglikelihood(
        jnp.asarray(true_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )
    true_param_filtered_variance = np.asarray(
        jax.device_get(true_filter_info.filtered_mean), dtype=np.float32
    )

    true_param_filtered_variance = np.asarray(
        jax.device_get(true_filter_info.filtered_mean), dtype=np.float32
    )
    true_param_loglik = float(jax.device_get(true_filter_carry[-1][0]))


    filtered_variance  = np.asarray(jax.device_get(filter_info.filtered_mean),     dtype=np.float32)
    filtered_std       = np.asarray(jax.device_get(filter_info.filtered_std),      dtype=np.float32)
    ess                = np.asarray(jax.device_get(filter_info.ess),               dtype=np.float32)
    loglik_increments  = np.asarray(jax.device_get(filter_info.loglik_increments), dtype=np.float32)
    total_loglik       = float(jax.device_get(filter_carry[-1][0]))
    bic                = float(jax.device_get(bic))

    result: dict[str, object] = {
        "prices":            prices,
        "log_returns":       log_returns,
        "true_variance":     true_variance,
        "filtered_variance": filtered_variance,
        "filtered_std":      filtered_std,
        "ess":               ess,
        "loglik_increments": loglik_increments,
        "true_params":       _to_inhomo_param_dict(true_params),
        "fitted_params":     _to_inhomo_param_dict(fitted_params),
        "best_loglik":       total_loglik,
        "bic":               bic,
        "variance_rmse":     float(np.sqrt(np.mean((filtered_variance - true_variance) ** 2))),
        "mean_ess":          float(np.mean(ess)),
        "true_param_filtered_variance": true_param_filtered_variance,
        "true_param_loglik":            true_param_loglik,
        "true_param_variance_rmse": float(
            np.sqrt(np.mean((true_param_filtered_variance - true_variance) ** 2))
        ),
    }

    if plot_path is not None:
        result["plot_path"] = _save_diagnostic_plot(
            prices=prices,
            log_returns=log_returns,
            true_variance=true_variance,
            filtered_variance=filtered_variance,
            filtered_std=filtered_std,
            plot_path=plot_path,
            true_param_filtered_variance=true_param_filtered_variance,
        )

    return result


def run_qrh(
    seed: int = 7,
    S0: float = 100.0,
    num_days: int = 15,
    true_params: np.ndarray | None = None,
    popsize: int = 1024,
    num_generations: int = 100,
    sigma_init: float = 0.5,
    num_particles: int = 1024,
    num_factors: int = 8,
    plot_path: str | Path | None = None,
) -> dict[str, object]:
    """Run synthetic-data generation, CMA-ES calibration, and particle-filter
    evaluation for the Quadratic Rough Heston+ (QRH+) model.

    Data layout: (390 + 1) * num_days steps.  Steps 0–389 within each day
    have dt = 1 minute; step 390 has dt = 1050 minutes (overnight gap).
    The overnight effective dt is round(lambda_ov * 1050) * _DT_MIN.

    Parameters [a0, a1, a2, rho, alpha, sigma_obs, lambda_ov]  (7-dim).
    """
    if true_params is None:
        true_params = np.array(
            [0.04,  0.6, 0.2, -0.7, -0.05, 1e-5, 0.2], dtype=np.float32
        )
        # a0=0 is on the lower bound; bump to small positive for numerical safety
        true_params[0] = np.float32(0.001)
    else:
        true_params = np.asarray(true_params, dtype=np.float32)

    log_returns, true_variance = QRHProcess.generator(
        seed=seed,
        S0=S0,
        num_days=num_days,
        params=true_params,
        num_factors=num_factors,
    )
    prices = _prices_from_log_returns(S0, log_returns)

    if plot_path is not None:
        _save_diagnostic_plot(
            prices=prices,
            log_returns=log_returns,
            true_variance=true_variance,
            filtered_variance=true_variance,
            filtered_std=np.zeros_like(true_variance),
            plot_path=f"data_generation_{plot_path}",
        )

    process = QRHProcess(
        popsize=popsize,
        num_generations=num_generations,
        sigma_init=sigma_init,
        dt=float(_DT_MIN),
        num_particles=num_particles,
        S=prices,
        num_factors=num_factors,
    )
    setting, dsetting = process.get_default_param(jax.random.PRNGKey(seed))

    fit_key = jax.random.PRNGKey(seed + 1)
    best_member, bic = process.calibrate(fit_key, setting, dsetting)
    fitted_params = np.asarray(
        jax.device_get(process.unconstrained_to_params(best_member)),
        dtype=np.float32,
    )

    filter_carry, filter_info = process.loglikelihood(
        jnp.asarray(fitted_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    # Second filter pass with true parameters for comparison
    true_filter_carry, true_filter_info = process.loglikelihood(
        jnp.asarray(true_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )
    true_param_filtered_variance = np.asarray(
        jax.device_get(true_filter_info.filtered_mean), dtype=np.float32
    )
    true_param_loglik = float(jax.device_get(true_filter_carry[-1][0]))

    filtered_variance  = np.asarray(jax.device_get(filter_info.filtered_mean),     dtype=np.float32)
    filtered_std       = np.asarray(jax.device_get(filter_info.filtered_std),      dtype=np.float32)
    ess                = np.asarray(jax.device_get(filter_info.ess),               dtype=np.float32)
    loglik_increments  = np.asarray(jax.device_get(filter_info.loglik_increments), dtype=np.float32)
    total_loglik       = float(jax.device_get(filter_carry[-1][0]))
    bic                = float(jax.device_get(bic))

    result: dict[str, object] = {
        "prices":            prices,
        "log_returns":       log_returns,
        "true_variance":     true_variance,
        "filtered_variance": filtered_variance,
        "filtered_std":      filtered_std,
        "ess":               ess,
        "loglik_increments": loglik_increments,
        "true_params":       _to_qrh_param_dict(true_params),
        "fitted_params":     _to_qrh_param_dict(fitted_params),
        "best_loglik":       total_loglik,
        "bic":               bic,
        "variance_rmse":     float(np.sqrt(np.mean((filtered_variance - true_variance) ** 2))),
        "mean_ess":          float(np.mean(ess)),
        "true_param_filtered_variance": true_param_filtered_variance,
        "true_param_loglik":            true_param_loglik,
        "true_param_variance_rmse": float(
            np.sqrt(np.mean((true_param_filtered_variance - true_variance) ** 2))
        ),
    }

    if plot_path is not None:
        result["plot_path"] = _save_diagnostic_plot(
            prices=prices,
            log_returns=log_returns,
            true_variance=true_variance,
            filtered_variance=filtered_variance,
            filtered_std=filtered_std,
            plot_path=plot_path,
            true_param_filtered_variance=true_param_filtered_variance,
        )

    return result


def main() -> None:
    # print("=== Heston (no jumps) ===")
    # result = run_heston(plot_path="heston_diagnostic_plot.png")
    # summary = {
    #     "true_params":   result["true_params"],
    #     "fitted_params": result["fitted_params"],
    #     "best_loglik":   result["best_loglik"],
    #     "bic":           result["bic"],
    #     "variance_rmse": result["variance_rmse"],
    #     "mean_ess":      result["mean_ess"],
    # }
    # print(json.dumps(summary, indent=2, sort_keys=True))

    # print("\n=== Heston-Jump (1-min intraday + 1050-min overnight) ===")
    # result_jump = run_heston_jump(plot_path="heston_jump_diagnostic_plot.png")
    # summary_jump = {
    #     "true_params":   result_jump["true_params"],
    #     "fitted_params": result_jump["fitted_params"],
    #     "best_loglik":   result_jump["best_loglik"],
    #     "bic":           result_jump["bic"],
    #     "variance_rmse": result_jump["variance_rmse"],
    #     "mean_ess":      result_jump["mean_ess"],
    # }
    # print(json.dumps(summary_jump, indent=2, sort_keys=True))

    # print("\n=== Inhomogeneous Heston (sub-stepped overnight variance) ===")
    # result_ih = run_inhomo_heston(plot_path="inhomo_heston_diagnostic_plot.png")
    # summary_ih = {
    #     "true_params":              result_ih["true_params"],
    #     "fitted_params":            result_ih["fitted_params"],
    #     "true_param_loglik":        result_ih["true_param_loglik"],
    #     "best_loglik":              result_ih["best_loglik"],
    #     "bic":                      result_ih["bic"],
    #     "variance_rmse":            result_ih["variance_rmse"],
    #     "true_param_variance_rmse": result_ih["true_param_variance_rmse"],
    #     "mean_ess":                 result_ih["mean_ess"],
    # }
    # print(json.dumps(summary_ih, indent=2, sort_keys=True))

    # print("\n=== Quadratic Rough Heston+ (QRH+) ===")
    # result_qrh = run_qrh(plot_path="qrh_diagnostic_plot.png")
    # summary_qrh = {
    #     "true_params":              result_qrh["true_params"],
    #     "fitted_params":            result_qrh["fitted_params"],
    #     "true_param_loglik":        result_qrh["true_param_loglik"],
    #     "best_loglik":              result_qrh["best_loglik"],
    #     "bic":                      result_qrh["bic"],
    #     "variance_rmse":            result_qrh["variance_rmse"],
    #     "true_param_variance_rmse": result_qrh["true_param_variance_rmse"],
    #     "mean_ess":                 result_qrh["mean_ess"],
    # }
    # print(json.dumps(summary_qrh, indent=2, sort_keys=True))

    print("\n=== Semivariance Heston (two correlated CIR processes) ===")
    result_sv = run_semivariance_heston(
        plot_path="semivariance_heston_diagnostic_plot.png"
    )
    summary_sv = {
        "true_params":               result_sv["true_params"],
        "fitted_params":             result_sv["fitted_params"],
        "true_param_loglik":         result_sv["true_param_loglik"],
        "best_loglik":               result_sv["best_loglik"],
        "bic":                       result_sv["bic"],
        "variance_rmse_p":           result_sv["variance_rmse_p"],
        "variance_rmse_m":           result_sv["variance_rmse_m"],
        "true_param_variance_rmse_p": result_sv["true_param_variance_rmse_p"],
        "true_param_variance_rmse_m": result_sv["true_param_variance_rmse_m"],
        "mean_ess":                  result_sv["mean_ess"],
    }
    print(json.dumps(summary_sv, indent=2, sort_keys=True))



if __name__ == "__main__":
    main()
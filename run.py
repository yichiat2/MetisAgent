from __future__ import annotations

from concurrent.futures import process
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from torch import seed
from heston_jump_process import HestonJumpProcess
from svlogv_jump_process import StochasticVolatilityJumpProcess
from inhomo_heston_process import InhomoHestonProcess
from qrh_process import QRHProcess
from semivariance_heston_process import SemivarianceHestonProcess
from stochastic import DynSetting
from constants import _MINS_PER_DAY, _DT_MIN, _DT_OVERNIGHT


PARAM_NAMES = ("v0", "rho", "kappa", "theta", "sigma", "r")
PARAM_NAMES_INHOMO = ("v0", "rho", "kappa", "theta", "sigma", "r", "lambda_ov", "alpha_rs")
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
PARAM_NAMES_SVJ = (
    "v0", "kappa", "theta", "sigma_v", "rho",
    "lambda_J", "mu_JS", "sigma_JS", "mu_JV", "sigma_JV", "rho_J",
)


def _to_param_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES, values, strict=True)}


def _to_jump_param_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES_JUMP, values, strict=True)}


def _to_svj_param_dict(values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(PARAM_NAMES_SVJ, values, strict=True)}


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


def _save_inhomo_diagnostic_plot(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    log_returns: np.ndarray,
    true_variance: np.ndarray,
    filtered_variance: np.ndarray,
    filtered_std: np.ndarray,
    plot_path: str | Path,
    true_param_filtered_variance: np.ndarray | None = None,
) -> str:
    """Three-panel Plotly diagnostic plot for the Inhomogeneous Heston model.

    Panel 1: OHLC candlestick.
    Panel 2: Log-return bar chart (green = up, red = down).
    Panel 3: Variance — true path, filtered mean +/- 2σ (fitted params),
             and optionally filtered mean using true params.

    Writes an interactive HTML alongside the PNG.  PNG is produced via
    kaleido when available, otherwise falls back to a Matplotlib render.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plot_path = Path(plot_path)
    T  = log_returns.shape[0]
    tl = np.arange(T)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "Synthetic InhomoHeston OHLC",
            "Log Returns",
            "Variance: true / filtered",
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.25, 0.35],
    )

    # ── Panel 1: candlestick ─────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=tl,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="OHLC",
            increasing_line_color="limegreen",
            decreasing_line_color="tomato",
        ),
        row=1, col=1,
    )
    fig.update_layout(xaxis_rangeslider_visible=False)

    # ── Panel 2: log-return bar chart ────────────────────────────────
    bar_colors = np.where(log_returns >= 0, "limegreen", "tomato").tolist()
    fig.add_trace(
        go.Bar(
            x=tl,
            y=log_returns,
            marker_color=bar_colors,
            name="Log return",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # ── Panel 3: variance ────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=tl, y=true_variance, mode="lines",
            line=dict(color="royalblue", width=1.5),
            name="true variance",
        ),
        row=3, col=1,
    )
    lower = np.maximum(filtered_variance - 2.0 * filtered_std, 0.0)
    upper = filtered_variance + 2.0 * filtered_std
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([tl, tl[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.15)",
            line=dict(color="rgba(255,165,0,0)"),
            name="filtered ±2σ (fitted)",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tl, y=filtered_variance, mode="lines",
            line=dict(color="darkorange", width=1.5),
            name="filtered mean (fitted)",
        ),
        row=3, col=1,
    )
    if true_param_filtered_variance is not None:
        fig.add_trace(
            go.Scatter(
                x=tl, y=true_param_filtered_variance, mode="lines",
                line=dict(color="purple", width=1.2, dash="dash"),
                name="filtered mean (true params)",
            ),
            row=3, col=1,
        )

    fig.update_layout(
        height=900,
        title_text="InhomoHeston OHLC Diagnostic",
        template="plotly_white",
        legend=dict(orientation="v", x=1.01, y=0.5),
    )
    fig.update_yaxes(title_text="Price",      row=1, col=1)
    fig.update_yaxes(title_text="Log return", row=2, col=1)
    fig.update_yaxes(title_text="Variance",   row=3, col=1)
    fig.update_xaxes(title_text="Time step",  row=3, col=1)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(plot_path.with_suffix(".html")))

    try:
        fig.write_image(str(plot_path), scale=2, engine="kaleido")
        return str(plot_path)
    except Exception:
        # Fallback: Matplotlib static render
        from matplotlib.patches import Rectangle

        x = np.arange(T)
        figm, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=False)

        ax0 = axs[0]
        width = 0.6
        for i in range(T):
            o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])
            color = "limegreen" if c >= o else "tomato"
            ax0.vlines(x[i], l, h, color=color, linewidth=0.6)
            rect_y = min(o, c)
            rect_h = max(abs(c - o), 1e-8)
            ax0.add_patch(Rectangle(
                (x[i] - width / 2.0, rect_y), width, rect_h,
                facecolor=color, edgecolor=color, linewidth=0.3,
            ))
        ax0.autoscale_view()
        ax0.set_title("Synthetic InhomoHeston OHLC")
        ax0.set_ylabel("Price")

        colors = np.where(log_returns >= 0, "limegreen", "tomato")
        axs[1].bar(x, log_returns, color=colors, width=1.0, linewidth=0)
        axs[1].axhline(0, color="black", linewidth=0.5)
        axs[1].set_title("Log Returns")
        axs[1].set_ylabel("Log return")

        axs[2].plot(x, true_variance, color="royalblue", linewidth=1.2, label="true variance")
        axs[2].fill_between(x, lower, upper, alpha=0.15, color="orange", label="filtered ±2σ (fitted)")
        axs[2].plot(x, filtered_variance, color="darkorange", linewidth=1.2, label="filtered mean (fitted)")
        if true_param_filtered_variance is not None:
            axs[2].plot(x, true_param_filtered_variance, color="purple", linewidth=1.0,
                        linestyle="--", label="filtered mean (true params)")
        axs[2].set_title("Variance: true / filtered")
        axs[2].set_ylabel("Variance")
        axs[2].set_xlabel("Time step")
        axs[2].legend(loc="upper right", fontsize=8)

        figm.tight_layout()
        figm.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(figm)
        return str(plot_path)



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
    seed: int = 15,
    S0: float = 100.0,
    num_days: int = 40,
    true_params: np.ndarray | None = None,
    popsize: int = 256,
    num_generations: int = 100,
    sigma_init: float = 0.8,
    num_particles: int = 2048,
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
    seed: int = 10,
    S0: float = 100.0,
    num_days: int = 30,
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
        true_params:     Shape-(8,) array [v0, rho, kappa, theta, sigma, r, lambda_ov, alpha_rs].
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
            [0.04, -0.7, 4.0, 0.04, 0.2, 0.0, 0.1, 2.0], dtype=np.float32
        )
    else:
        true_params = np.asarray(true_params, dtype=np.float32)

    log_returns, true_variance, opens, highs, lows, rs_variances = \
        InhomoHestonProcess.generator(
            seed=seed,
            S0=S0,
            num_days=num_days,
            params=true_params,
        )
    prices = _prices_from_log_returns(S0, log_returns)
    closes = prices[1:]  # close for each bar = price after the bar

    if plot_path is not None:
        _save_inhomo_diagnostic_plot(
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
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
        rs_data=rs_variances,
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
        "opens":             opens,
        "highs":             highs,
        "lows":              lows,
        "closes":            closes,
        "log_returns":       log_returns,
        "rs_variances":      rs_variances,
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
        result["plot_path"] = _save_inhomo_diagnostic_plot(
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            log_returns=log_returns,
            true_variance=true_variance,
            filtered_variance=filtered_variance,
            filtered_std=filtered_std,
            plot_path=plot_path,
            true_param_filtered_variance=true_param_filtered_variance,
        )

    return result


def run_real_inhomo_heston(
    root: str,
    num_steps: int = 20000,
    seed: int = 0,
    popsize: int = 256,
    num_generations: int = 100,
    sigma_init: float = 0.8,
    num_particles: int = 4096,
    plot_path: str | Path | None = None,
    db_name: str = "stock",
    db_username: str = "metis",
    db_password: str = "123456",
) -> dict[str, object]:
    """Calibrate InhomoHestonProcess on real 1-min OHLCV data from the database.

    Loads ``num_steps + 1`` rows (RTH 1-min bars) for ``root`` from MySQL,
    preprocesses the price series, Rogers-Satchell variance, and dt sequence,
    then runs CMA-ES calibration followed by an APF filter pass.

    Args:
        root:            Ticker symbol (e.g. ``"SPY"``).
        num_steps:       Number of log-return steps to use (default 20 000).
                         ``num_steps + 1`` rows are loaded from the database.
        seed:            JAX PRNG seed for noise pre-generation.
        popsize:         CMA-ES population size.
        num_generations: CMA-ES iteration budget.
        sigma_init:      CMA-ES initial step-size.
        num_particles:   APF particle count.
        plot_path:       If given, saves a Plotly diagnostic HTML/PNG at this path.
        db_name:         MySQL database name.
        db_username:     MySQL username.
        db_password:     MySQL password.

    Returns:
        Dictionary with keys: root, S, opens, highs, lows, closes,
        rs_data, rs_variance_proxy, dt_seq, log_returns,
        filtered_variance, filtered_std, ess, loglik_increments,
        fitted_params, best_loglik, bic, mean_ess, [plot_path].
    """
    import sys
    import os
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
    from data.database import Database

    # ── Load data ──────────────────────────────────────────────────────
    db = Database(db_name=db_name, db_username=db_username, db_password=db_password)
    df = db.load(root=root, start_date=20000101, end_date=21001231)
    if df is None or len(df) == 0:
        raise ValueError(f"No data found for root='{root}'")

    df = df.sort_values("date").reset_index(drop=True)
    need = num_steps + 1
    if len(df) < need:
        raise ValueError(
            f"Only {len(df)} rows available for root='{root}', need {need}"
        )
    df = df.iloc[-need:].reset_index(drop=True)   # shape (T+1, ...)

    T = num_steps

    # ── Price series S ─────────────────────────────────────────────────
    # S[i] = close of bar i; log_return[i] = log(S[i+1] / S[i])
    S = df["close"].values.astype(np.float32)    # (T+1,)

    # ── dt sequence ────────────────────────────────────────────────────
    # Transition i → i+1: overnight if the two bars fall on different
    # calendar days, otherwise one intraday minute.
    dates   = pd.to_datetime(df["date"])
    day_arr = dates.dt.date.values
    is_overnight = (day_arr[1:] != day_arr[:-1])  # (T,) bool
    dt_seq = np.where(is_overnight, _DT_OVERNIGHT, _DT_MIN).astype(np.float32)  # (T,)

    # ── Rogers-Satchell variance ───────────────────────────────────────
    # rs[i] measures within-bar variance for bar i+1 (the bar whose close
    # price becomes S[i+1]).  Uses log-OHLC relative to the bar's open.
    #   h = log(H/O),  l = log(L/O),  c = log(C/O)
    #   RS = h*(h-c) + l*(l-c)
    eps = np.float64(1e-10)
    open_  = df["open"].values[1:].astype(np.float64)
    high   = df["high"].values[1:].astype(np.float64)
    low    = df["low"].values[1:].astype(np.float64)
    close  = df["close"].values[1:].astype(np.float64)

    h = np.log(np.maximum(high,  eps) / np.maximum(open_, eps))
    l = np.log(np.maximum(low,   eps) / np.maximum(open_, eps))
    c = np.log(np.maximum(close, eps) / np.maximum(open_, eps))
    rs_data = np.maximum(h * (h - c) + l * (l - c), 1e-10).astype(np.float32)  # (T,)

    # RS-based annualised variance proxy for plotting (intraday bars only;
    # overnight bars have artificially large dt so we clip them out).
    dt_intra = np.where(is_overnight, _DT_MIN, dt_seq)
    rs_variance_proxy = (rs_data / dt_intra).astype(np.float32)                 # (T,)

    # ── OHLC price arrays for the candlestick panel ────────────────────
    opens  = df["open"].values[1:].astype(np.float32)
    highs  = df["high"].values[1:].astype(np.float32)
    lows   = df["low"].values[1:].astype(np.float32)
    closes = df["close"].values[1:].astype(np.float32)   # == S[1:]

    log_returns = np.log(S[1:] / S[:-1]).astype(np.float32)  # (T,)

    # ── Build process and override dt_seq / rs_seq ─────────────────────
    process = InhomoHestonProcess(
        popsize=popsize,
        num_generations=num_generations,
        sigma_init=sigma_init,
        dt=float(_DT_MIN),
        num_particles=num_particles,
        S=S,
        rho_cpm=0.99,
        rs_data=rs_data,
    )
    setting, dsetting = process.get_default_param(jax.random.PRNGKey(seed))
    # Replace the synthetic dt_seq with the real one.
    dsetting = dsetting.replace(dt_seq=jnp.array(dt_seq, dtype=jnp.float32))

    # ── CMA-ES calibration ─────────────────────────────────────────────
    fit_key = jax.random.PRNGKey(seed + 1)
    best_member, bic = process.calibrate(fit_key, setting, dsetting)
    fitted_params = np.asarray(
        jax.device_get(process.unconstrained_to_params(best_member)),
        dtype=np.float32,
    )

    # ── APF filter pass with fitted parameters ─────────────────────────
    filter_carry, filter_info = process.loglikelihood(
        jnp.asarray(fitted_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    filtered_variance  = np.asarray(jax.device_get(filter_info.filtered_mean),     dtype=np.float32)
    filtered_std       = np.asarray(jax.device_get(filter_info.filtered_std),      dtype=np.float32)
    ess                = np.asarray(jax.device_get(filter_info.ess),               dtype=np.float32)
    loglik_increments  = np.asarray(jax.device_get(filter_info.loglik_increments), dtype=np.float32)
    total_loglik       = float(jax.device_get(filter_carry[-1][0]))
    bic                = float(jax.device_get(bic))

    result: dict[str, object] = {
        "root":               root,
        "S":                  S,
        "opens":              opens,
        "highs":              highs,
        "lows":               lows,
        "closes":             closes,
        "log_returns":        log_returns,
        "rs_data":            rs_data,
        "rs_variance_proxy":  rs_variance_proxy,
        "dt_seq":             dt_seq,
        "filtered_variance":  filtered_variance,
        "filtered_std":       filtered_std,
        "ess":                ess,
        "loglik_increments":  loglik_increments,
        "fitted_params":      _to_inhomo_param_dict(fitted_params),
        "best_loglik":        total_loglik,
        "bic":                bic,
        "mean_ess":           float(np.mean(ess)),
    }

    if plot_path is not None:
        result["plot_path"] = _save_inhomo_diagnostic_plot(
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            log_returns=log_returns,
            true_variance=rs_variance_proxy,
            filtered_variance=filtered_variance,
            filtered_std=filtered_std,
            plot_path=plot_path,
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


def _save_jump_diagnostic_plot(
    prices: np.ndarray,
    true_variance: np.ndarray,
    filtered_mean: np.ndarray,
    filtered_q05: np.ndarray,
    filtered_q25: np.ndarray,
    filtered_q75: np.ndarray,
    filtered_q95: np.ndarray,
    plot_path: str | Path,
    opens: np.ndarray | None = None,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    closes: np.ndarray | None = None,
    jump_prob: np.ndarray | None = None,
    pred_log_return_mean: np.ndarray | None = None,
    pred_log_return_std: np.ndarray | None = None,
    true_param_filtered_variance: np.ndarray | None = None,
) -> str:
    """Three-panel Plotly diagnostic plot for the SVJ model.

    Panel 1: OHLC candlestick plus one-step-ahead predicted price band.
    Panel 2: Posterior jump probability (bar chart).
    Panel 3: Variance — true path, filtered median with quantile bands
             (25-75% deep orange, 5-25% and 75-95% light orange),
             and optionally filtered median using true params.

    Writes an interactive HTML alongside the PNG.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    plot_path = Path(plot_path)
    T  = true_variance.shape[0]
    tl = np.arange(T)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "SVJ Price Path",
            "Posterior Jump Probability",
            "Variance: true / filtered",
        ),
        vertical_spacing=0.08,
        row_heights=[0.35, 0.25, 0.40],
    )

    # ── Panel 1: OHLC candlestick + optional predicted price band ────
    # OHLC bar i spans step i (close = prices[i+1]).
    fig.add_trace(
        go.Candlestick(
            x=tl,
            open=opens, high=highs, low=lows, close=closes,
            increasing_line_color="limegreen",
            decreasing_line_color="tomato",
            name="OHLC",
        ),
        row=1, col=1,
    )
    # Predicted price band: prediction made at t targets close at t+1 → plot at x=t.
    if pred_log_return_mean is not None and pred_log_return_std is not None:
        pred_price = prices[1:] * np.exp(pred_log_return_mean)
        pred_upper = prices[1:] * np.exp(pred_log_return_mean + pred_log_return_std)
        pred_lower = prices[1:] * np.exp(pred_log_return_mean - pred_log_return_std)
        pred_price[1:] = pred_price[:-1]
        pred_upper[1:] = pred_upper[:-1]
        pred_lower[1:] = pred_lower[:-1]

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([tl, tl[::-1]]),
                y=np.concatenate([pred_upper, pred_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(34,139,34,0.12)",
                line=dict(color="rgba(34,139,34,0)"),
                name="pred price +/-1sigma",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=tl, y=pred_price, mode="lines",
                line=dict(color="forestgreen", width=1.0, dash="dot"),
                name="pred price mean",
            ),
            row=1, col=1,
        )

    # ── Panel 2: posterior jump probability ──────────────────────────
    _jump_vals = jump_prob if jump_prob is not None else np.zeros(T)
    fig.add_trace(
        go.Bar(
            x=tl, y=_jump_vals,
            marker_color="rgba(180,60,60,1.0)",
            marker_line_width=0,
            width=1.5,
            name="jump prob",
        ),
        row=2, col=1,
    )

    # ── Panel 3: variance with quantile bands ────────────────────────
    fig.add_trace(
        go.Scatter(
            x=tl, y=true_variance, mode="lines",
            line=dict(color="royalblue", width=1.5),
            name="true variance",
        ),
        row=3, col=1,
    )
    # Outer band: 5-95% (light orange)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([tl, tl[::-1]]),
            y=np.concatenate([filtered_q95, filtered_q05[::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.10)",
            line=dict(color="rgba(255,165,0,0)"),
            name="filtered 5-95% (fitted)",
        ),
        row=3, col=1,
    )
    # Inner band: 25-75% (deep orange)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([tl, tl[::-1]]),
            y=np.concatenate([filtered_q75, filtered_q25[::-1]]),
            fill="toself",
            fillcolor="rgba(255,140,0,0.25)",
            line=dict(color="rgba(255,140,0,0)"),
            name="filtered 25-75% (fitted)",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=tl, y=filtered_mean, mode="lines",
            line=dict(color="darkorange", width=1.5),
            name="filtered mean (fitted)",
        ),
        row=3, col=1,
    )
    if true_param_filtered_variance is not None:
        fig.add_trace(
            go.Scatter(
                x=tl, y=true_param_filtered_variance, mode="lines",
                line=dict(color="purple", width=1.2, dash="dash"),
                name="filtered mean (true params)",
            ),
            row=3, col=1,
        )

    fig.update_layout(
        height=900,
        title_text="SVJ Diagnostic",
        template="plotly_white",
        legend=dict(orientation="v", x=1.01, y=0.5),
    )
    fig.update_yaxes(title_text="Price",      row=1, col=1)
    fig.update_yaxes(title_text="Jump prob",  row=2, col=1)
    fig.update_yaxes(title_text="Variance",   row=3, col=1)
    fig.update_xaxes(title_text="Time step",  row=3, col=1)
    
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(plot_path.with_suffix(".html")))

    try:
        fig.write_image(str(plot_path), scale=2, engine="kaleido")
    except Exception:
        pass
    return str(plot_path)


def run_heston_jump(
    seed: int = 20,
    S0: float = 100.0,
    num_days: int = 30,
    true_params: np.ndarray | None = None,
    popsize: int = 16,
    num_generations: int = 100,
    sigma_init: float = 0.8,
    num_particles: int = 4096,
    plot_path: str | Path | None = None,
) -> dict[str, object]:
    """Run synthetic-data generation, CMA-ES calibration, and APF evaluation
    for the Heston-Jump model.

    Data layout: ``390 × num_days`` steps, all at ``dt = 1 minute``.
    There are no special overnight dt values; overnight price gaps are
    described by the correlated jump process.

    Risk-neutral property is maintained via the per-step compensator::

        comp = log[(1 - p_J) + p_J · exp(μ_Jr + ½σ_Jr²)]

    Args:
        seed:            RNG seed.
        S0:              Initial price.
        num_days:        Number of simulated trading days (390 steps each).
        true_params:     Shape-(12,) array
                         ``[v0, rho, kappa, theta, sigma, r,
                            p_J, mu_Jr, sigma_Jr, mu_JV, sigma_JV, rho_J]``.
                         Defaults to a realistic jump parameter set.
        popsize:         CMA-ES population size.
        num_generations: CMA-ES iteration budget.
        sigma_init:      CMA-ES initial step-size.
        num_particles:   APF particle count.
        plot_path:       If given, saves a diagnostic HTML/PNG at this path.

    Returns:
        Dictionary with keys:
            prices, log_returns, true_variance,
            filtered_variance, filtered_std, ess, loglik_increments,
            true_params, fitted_params, best_loglik, bic,
            variance_rmse, mean_ess,
            true_param_filtered_variance, true_param_loglik,
            true_param_variance_rmse, [plot_path].
    """
    if true_params is None:
        true_params = np.array(
            [
                0.04,   # v0
               -0.6,    # rho
                3.0,    # kappa
                0.04,   # theta
                0.3,    # sigma
                0.0,    # r
                100,   # lambda_J  
               -0.01,   # mu_Jr — small negative mean jump
                0.05,   # sigma_Jr
                0.01,    # mu_JV — variance jumps upward on average
                0.05,    # sigma_JV
               -0.5,    # rho_J — large variance jumps → negative return jumps
            ],
            dtype=np.float32,
        )
    else:
        true_params = np.asarray(true_params, dtype=np.float32)

    log_returns, true_variance = HestonJumpProcess.generator(
        seed=seed,
        S0=S0,
        num_days=num_days,
        params=true_params,
    )
    prices = _prices_from_log_returns(S0, log_returns)

    # Data-generation sanity plot (true variance, no filter pass yet).
    if plot_path is not None:
        _save_jump_diagnostic_plot(
            prices=prices,
            true_variance=true_variance,
            filtered_mean=true_variance,
            filtered_q05=true_variance,
            filtered_q25=true_variance,
            filtered_q75=true_variance,
            filtered_q95=true_variance,
            plot_path=f"data_generation_{plot_path}",
        )

    process = HestonJumpProcess(
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

    # APF filter pass with fitted parameters.
    filter_carry, filter_info = process.loglikelihood(
        jnp.asarray(fitted_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    # APF filter pass with true parameters (upper-bound reference).
    true_filter_carry, true_filter_info = process.loglikelihood(
        jnp.asarray(true_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    filtered_variance  = np.asarray(jax.device_get(filter_info.filtered_mean),      dtype=np.float32)
    filtered_std       = np.asarray(jax.device_get(filter_info.filtered_std),       dtype=np.float32)
    ess                = np.asarray(jax.device_get(filter_info.ess),                dtype=np.float32)
    loglik_increments  = np.asarray(jax.device_get(filter_info.loglik_increments),  dtype=np.float32)
    total_loglik       = float(jax.device_get(filter_carry[-1][0]))
    bic_val            = float(jax.device_get(bic))

    true_param_filtered_variance = np.asarray(
        jax.device_get(true_filter_info.filtered_mean), dtype=np.float32
    )
    true_param_loglik = float(jax.device_get(true_filter_carry[-1][0]))

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
        "bic":               bic_val,
        "variance_rmse":     float(np.sqrt(np.mean((filtered_variance - true_variance) ** 2))),
        "mean_ess":          float(np.mean(ess)),
        "true_param_filtered_variance": true_param_filtered_variance,
        "true_param_loglik":            true_param_loglik,
        "true_param_variance_rmse": float(
            np.sqrt(np.mean((true_param_filtered_variance - true_variance) ** 2))
        ),
    }

    if plot_path is not None:
        _fq05 = np.maximum(filtered_variance - 1.645 * filtered_std, 0.0)
        _fq25 = np.maximum(filtered_variance - 0.674 * filtered_std, 0.0)
        _fq75 = filtered_variance + 0.674 * filtered_std
        _fq95 = filtered_variance + 1.645 * filtered_std
        result["plot_path"] = _save_jump_diagnostic_plot(
            prices=prices,
            true_variance=true_variance,
            filtered_mean=filtered_variance,
            filtered_q05=_fq05,
            filtered_q25=_fq25,
            filtered_q75=_fq75,
            filtered_q95=_fq95,
            plot_path=plot_path,
            pred_log_return_mean=np.asarray(jax.device_get(filter_info.pred_log_return_mean), dtype=np.float32),
            pred_log_return_std=np.asarray(jax.device_get(filter_info.pred_log_return_std), dtype=np.float32),
            true_param_filtered_variance=true_param_filtered_variance,
        )

    return result


def run_svlogv_jump(
    seed: int = 20,
    S0: float = 100.0,
    num_days: int = 30,
    true_params: np.ndarray | None = None,
    popsize: int = 128,
    num_generations: int = 100,
    sigma_init: float = 0.8,
    num_particles: int = 4096,
    plot_path: str | Path | None = None,
) -> dict[str, object]:
    """Run synthetic-data generation, CMA-ES calibration, and RB-APF evaluation
    for the Stochastic Volatility Log-Variance Jump model.

    The latent state is ``ell_t = log(V_t)`` driven by an OU process with
    additive Gaussian jumps shared with the log-return.  Inference uses a
    fully-adapted RB-APF: pilot weights equal the exact marginal likelihood,
    propagation samples from the Kalman posterior Gaussian mixture, and all
    correction weights are exactly 1/N — no second resampling stage.

    Args:
        seed:            RNG seed.
        S0:              Initial price.
        num_days:        Number of simulated trading days (390 steps each).
        true_params:     Shape-(12,) array
                         ``[ell0, kappa, theta, sigma_v, rho, r,
                            lambda_J, mu_JS, sigma_JS, mu_JV, sigma_JV, rho_J]``.
                         Defaults to a realistic parameter set.
        popsize:         CMA-ES population size.
        num_generations: CMA-ES iteration budget.
        sigma_init:      CMA-ES initial step-size.
        num_particles:   APF particle count.
        plot_path:       If given, saves a diagnostic HTML/PNG at this path.

    Returns:
        Dictionary with keys:
            prices, log_returns, true_variance, true_log_variance,
            filtered_variance, filtered_std, ess, loglik_increments,
            true_params, fitted_params, best_loglik, bic,
            variance_rmse, mean_ess,
            true_param_filtered_variance, true_param_loglik,
            true_param_variance_rmse, [plot_path].
    """
    import math
    if true_params is None:
        true_params = np.array(
            [
                0.20,  # v0  — 20% annualised vol squared
                4.0,             # kappa — OU mean-reversion speed
                0.08,  # theta — long-run variance
                1.2,             # sigma_v
               -0.7,             # rho
                25.0,           # lambda_J — 50 jumps/year
               -0.01,            # mu_JS — small negative mean return jump
                0.08,            # sigma_JS
               -0.02,             # mu_JV — variance jumps tend upward in log-space
                0.08,             # sigma_JV — substantial log-variance jump uncertainty
               -0.5,             # rho_J — large vol jumps -> negative return jumps
            ],
            dtype=np.float32,
        )
    else:
        true_params = np.asarray(true_params, dtype=np.float32)
    # jax.config.update("jax_disable_jit", True)
    log_returns, true_variance = StochasticVolatilityJumpProcess.generator(
        seed=seed,
        S0=S0,
        num_days=num_days,
        params=true_params,
    )
    prices = _prices_from_log_returns(S0, log_returns)

    if plot_path is not None:
        _save_jump_diagnostic_plot(
            prices=prices,
            true_variance=true_variance,
            filtered_mean=true_variance,
            filtered_q05=true_variance,
            filtered_q25=true_variance,
            filtered_q75=true_variance,
            filtered_q95=true_variance,
            plot_path=f"data_generation_{plot_path}",
        )
    
    process = StochasticVolatilityJumpProcess(
        popsize=popsize,
        num_generations=num_generations,
        sigma_init=sigma_init,
        dt=float(_DT_MIN),
        num_particles=num_particles,
        S=prices,
        rho_cpm=0.5,
    )
    setting, dsetting = process.get_default_param(jax.random.PRNGKey(seed))

    fit_key = jax.random.PRNGKey(seed + 1)
    best_member, bic = process.calibrate(fit_key, setting, dsetting)
    fitted_params = np.asarray(
        jax.device_get(process.unconstrained_to_params(best_member)),
        dtype=np.float32,
    )

    # RB-APF filter pass with fitted parameters.
    filter_carry, filter_info = process.loglikelihood(
        jnp.asarray(fitted_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    # RB-APF filter pass with true parameters (upper-bound reference).
    true_filter_carry, true_filter_info = process.loglikelihood(
        jnp.asarray(true_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    filtered_variance  = np.asarray(jax.device_get(filter_info.filtered_mean),     dtype=np.float32)
    filtered_q05       = np.asarray(jax.device_get(filter_info.filtered_q05),      dtype=np.float32)
    filtered_q25       = np.asarray(jax.device_get(filter_info.filtered_q25),      dtype=np.float32)
    filtered_q50       = np.asarray(jax.device_get(filter_info.filtered_q50),      dtype=np.float32)
    filtered_q75       = np.asarray(jax.device_get(filter_info.filtered_q75),      dtype=np.float32)
    filtered_q95       = np.asarray(jax.device_get(filter_info.filtered_q95),      dtype=np.float32)
    jump_prob          = np.asarray(jax.device_get(filter_info.jump_prob),          dtype=np.float32)
    pred_lr_mean       = np.asarray(jax.device_get(filter_info.pred_log_return_mean), dtype=np.float32)
    pred_lr_std        = np.asarray(jax.device_get(filter_info.pred_log_return_std),  dtype=np.float32)
    ess                = np.asarray(jax.device_get(filter_info.ess),               dtype=np.float32)
    loglik_increments  = np.asarray(jax.device_get(filter_info.loglik_increments), dtype=np.float32)
    total_loglik       = float(jax.device_get(filter_carry[-1][0]))
    bic_val            = float(jax.device_get(bic))

    true_param_filtered_variance = np.asarray(
        jax.device_get(true_filter_info.filtered_mean), dtype=np.float32
    )
    true_param_loglik = float(jax.device_get(true_filter_carry[-1][0]))

    result: dict[str, object] = {
        "prices":            prices,
        "log_returns":       log_returns,
        "true_variance":     true_variance,
        "filtered_variance": filtered_variance,
        "filtered_q05":      filtered_q05,
        "filtered_q25":      filtered_q25,
        "filtered_q50":      filtered_q50,
        "filtered_q75":      filtered_q75,
        "filtered_q95":      filtered_q95,
        "jump_prob":         jump_prob,
        "ess":               ess,
        "loglik_increments": loglik_increments,
        "true_params":       _to_svj_param_dict(true_params),
        "fitted_params":     _to_svj_param_dict(fitted_params),
        "best_loglik":       total_loglik,
        "bic":               bic_val,
        "variance_rmse":     float(np.sqrt(np.mean((filtered_variance - true_variance) ** 2))),
        "mean_ess":          float(np.mean(ess)),
        "true_param_filtered_variance": true_param_filtered_variance,
        "true_param_loglik":            true_param_loglik,
        "true_param_variance_rmse": float(
            np.sqrt(np.mean((true_param_filtered_variance - true_variance) ** 2))
        ),
    }

    if plot_path is not None:
        result["plot_path"] = _save_jump_diagnostic_plot(
            prices=prices,
            true_variance=true_variance,
            filtered_mean=filtered_variance,
            filtered_q05=filtered_q05,
            filtered_q25=filtered_q25,
            filtered_q75=filtered_q75,
            filtered_q95=filtered_q95,
            plot_path=plot_path,
            jump_prob=jump_prob,
            pred_log_return_mean=pred_lr_mean,
            pred_log_return_std=pred_lr_std,
            true_param_filtered_variance=true_param_filtered_variance,
        )

    return result


def run_real_svlogv_jump(
    root: str,
    num_steps: int = 20000,
    seed: int = 0,
    popsize: int = 256,
    num_generations: int = 100,
    sigma_init: float = 0.8,
    num_particles: int = 2048,
    plot_path: str | Path | None = None,
    db_name: str = "stock",
    db_username: str = "metis",
    db_password: str = "123456",
) -> dict[str, object]:
    """Calibrate StochasticVolatilityJumpProcess on real 1-min OHLCV data.

    Loads ``num_steps + 1`` rows (RTH 1-min bars) for ``root`` from MySQL,
    preprocesses the close-to-close log-return series and dt sequence,
    then runs CMA-ES calibration followed by an RB-APF filter pass.
    The EWA of close-to-close squared returns is computed as a variance
    proxy for plotting only; it is not used in the likelihood.

    Args:
        root:            Ticker symbol (e.g. ``"SPY"``).
        num_steps:       Number of log-return steps to use (default 20 000).
                         ``num_steps + 1`` rows are loaded from the database.
        seed:            JAX PRNG seed for noise pre-generation.
        popsize:         CMA-ES population size.
        num_generations: CMA-ES iteration budget.
        sigma_init:      CMA-ES initial step-size.
        num_particles:   RB-APF particle count.
        plot_path:       If given, saves a Plotly diagnostic HTML/PNG at this path.
        db_name:         MySQL database name.
        db_username:     MySQL username.
        db_password:     MySQL password.

    Returns:
        Dictionary with keys: root, S, opens, highs, lows, closes,
        ewa_c2c_variance, dt_seq, log_returns,
        filtered_variance, filtered_std, ess, loglik_increments,
        fitted_params, best_loglik, bic, mean_ess, [plot_path].
    """
    import sys
    import os
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data"))
    from data.database import Database

    # ── Load data ──────────────────────────────────────────────────────
    db = Database(db_name=db_name, db_username=db_username, db_password=db_password)
    df = db.load(root=root, start_date=20230101, end_date=21001231)
    if df is None or len(df) == 0:
        raise ValueError(f"No data found for root='{root}'")

    df = df.sort_values("date").reset_index(drop=True)
    need = num_steps + 1
    if len(df) < need:
        raise ValueError(
            f"Only {len(df)} rows available for root='{root}', need {need}"
        )
    df = df.iloc[-need:].reset_index(drop=True)   # shape (T+1, ...)

    T = num_steps

    # ── Price series S ─────────────────────────────────────────────────
    # S[i] = close of bar i; log_return[i] = log(S[i+1] / S[i])
    S = df["close"].values.astype(np.float32)    # (T+1,)

    # ── dt sequence ────────────────────────────────────────────────────
    # Transition i → i+1: overnight if the two bars fall on different
    # calendar days, otherwise one intraday minute.
    dates   = pd.to_datetime(df["date"])
    day_arr = dates.dt.date.values
    is_overnight = (day_arr[1:] != day_arr[:-1])  # (T,) bool
    dt_seq = np.where(is_overnight, _DT_OVERNIGHT, _DT_MIN).astype(np.float32)  # (T,)

    # ── OHLC price arrays for the candlestick panel ────────────────────
    opens  = df["open"].values[1:].astype(np.float32)
    highs  = df["high"].values[1:].astype(np.float32)
    lows   = df["low"].values[1:].astype(np.float32)
    closes = df["close"].values[1:].astype(np.float32)   # == S[1:]

    log_returns = np.log(S[1:] / S[:-1]).astype(np.float32)  # (T,)

    # ── EWA close-to-close variance proxy (for plotting only) ──────────
    # Annualised per-step variance: c2c_raw[t] = r[t]² / dt[t]
    # Smoothed with an exponential moving average (half-life = 30 steps).
    _ewa_alpha = 1.0 - np.exp(-np.log(2.0) / 30.0)
    c2c_raw = (log_returns.astype(np.float64) ** 2 / dt_seq.astype(np.float64)).astype(np.float32)
    ewa_c2c_variance = np.empty_like(c2c_raw)
    ewa_c2c_variance[0] = c2c_raw[0]
    for i in range(1, len(c2c_raw)):
        ewa_c2c_variance[i] = (
            _ewa_alpha * c2c_raw[i] + (1.0 - _ewa_alpha) * ewa_c2c_variance[i - 1]
        )

    # ── Build process and override dt_seq ─────────────────────────────
    process = StochasticVolatilityJumpProcess(
        popsize=popsize,
        num_generations=num_generations,
        sigma_init=sigma_init,
        dt=float(_DT_MIN),
        num_particles=num_particles,
        S=S,
        rho_cpm=0.5,
        dt_seq=dt_seq,
    )
    setting, dsetting = process.get_default_param(jax.random.PRNGKey(seed))
    
    # ── CMA-ES calibration ─────────────────────────────────────────────
    fit_key = jax.random.PRNGKey(seed + 1)
    best_member, bic = process.calibrate(fit_key, setting, dsetting)
    fitted_params = np.asarray(
        jax.device_get(process.unconstrained_to_params(best_member)),
        dtype=np.float32,
    )

    process = StochasticVolatilityJumpProcess(
        popsize=1,
        num_generations=1,
        sigma_init=sigma_init,
        dt=float(_DT_MIN),
        num_particles=num_particles,
        S=S,
        rho_cpm=0.5,
        dt_seq=dt_seq,
    )
    setting, dsetting = process.get_default_param(jax.random.PRNGKey(seed))
    # ── RB-APF filter pass with fitted parameters ──────────────────────
    filter_carry, filter_info = process.loglikelihood(
        jnp.asarray(fitted_params, dtype=jnp.float32)[None, :],
        setting,
        dsetting,
    )

    filtered_variance  = np.asarray(jax.device_get(filter_info.filtered_mean),     dtype=np.float32)
    filtered_q05       = np.asarray(jax.device_get(filter_info.filtered_q05),      dtype=np.float32)
    filtered_q25       = np.asarray(jax.device_get(filter_info.filtered_q25),      dtype=np.float32)
    filtered_q50       = np.asarray(jax.device_get(filter_info.filtered_q50),      dtype=np.float32)
    filtered_q75       = np.asarray(jax.device_get(filter_info.filtered_q75),      dtype=np.float32)
    filtered_q95       = np.asarray(jax.device_get(filter_info.filtered_q95),      dtype=np.float32)
    jump_prob          = np.asarray(jax.device_get(filter_info.jump_prob),          dtype=np.float32)
    pred_lr_mean       = np.asarray(jax.device_get(filter_info.pred_log_return_mean), dtype=np.float32)
    pred_lr_std        = np.asarray(jax.device_get(filter_info.pred_log_return_std),  dtype=np.float32)
    ess                = np.asarray(jax.device_get(filter_info.ess),               dtype=np.float32)
    loglik_increments  = np.asarray(jax.device_get(filter_info.loglik_increments), dtype=np.float32)
    total_loglik       = float(jax.device_get(filter_carry[-1][0]))
    bic                = float(jax.device_get(bic))

    result: dict[str, object] = {
        "root":               root,
        "S":                  S,
        "opens":              opens,
        "highs":              highs,
        "lows":               lows,
        "closes":             closes,
        "log_returns":        log_returns,
        "ewa_c2c_variance":   ewa_c2c_variance,
        "dt_seq":             dt_seq,
        "filtered_variance":  filtered_variance,
        "filtered_q05":       filtered_q05,
        "filtered_q25":       filtered_q25,
        "filtered_q50":       filtered_q50,
        "filtered_q75":       filtered_q75,
        "filtered_q95":       filtered_q95,
        "jump_prob":          jump_prob,
        "ess":                ess,
        "loglik_increments":  loglik_increments,
        "fitted_params":      _to_svj_param_dict(fitted_params),
        "best_loglik":        total_loglik,
        "bic":                bic,
        "mean_ess":           float(np.mean(ess)),
    }

    if plot_path is not None:
        result["plot_path"] = _save_jump_diagnostic_plot(
            prices=S,
            true_variance=ewa_c2c_variance,
            filtered_mean=filtered_variance,
            filtered_q05=filtered_q05,
            filtered_q25=filtered_q25,
            filtered_q75=filtered_q75,
            filtered_q95=filtered_q95,
            plot_path=plot_path,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            jump_prob=jump_prob,
            pred_log_return_mean=pred_lr_mean,
            pred_log_return_std=pred_lr_std,
        )

    return result


def main() -> None:


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
    # run_real_inhomo_heston(
    #     root="QQQ",plot_path="real_inhomo_heston_diagnostic_plot.png")


    # print("\n=== SVLogV-Jump model (RB-APF) ===")
    # result_svj = run_svlogv_jump(plot_path="svlogv_jump_diagnostic_plot.png")
    # summary_svj = {
    #     "true_params":              result_svj["true_params"],
    #     "fitted_params":            result_svj["fitted_params"],
    #     "true_param_loglik":        result_svj["true_param_loglik"],
    #     "best_loglik":              result_svj["best_loglik"],
    #     "bic":                      result_svj["bic"],
    #     "variance_rmse":            result_svj["variance_rmse"],
    #     "true_param_variance_rmse": result_svj["true_param_variance_rmse"],
    #     "mean_ess":                 result_svj["mean_ess"],
    # }
    # print(json.dumps(summary_svj, indent=2, sort_keys=True))

    print("\n=== SVLogV-Jump model (RB-APF) ===")
    result_svj = run_real_svlogv_jump(root="QQQ", plot_path="real_svlogv_jump_diagnostic_plot.png")
    summary_svj = {
        "fitted_params":            result_svj["fitted_params"],
        "best_loglik":              result_svj["best_loglik"],
        "bic":                      result_svj["bic"],
        "mean_ess":                 result_svj["mean_ess"],
    }
    print(json.dumps(summary_svj, indent=2, sort_keys=True))

    # print("\n=== Semivariance Heston (two correlated CIR processes) ===")
    # result_sv = run_semivariance_heston(
    #     plot_path="semivariance_heston_diagnostic_plot.png"
    # )
    # summary_sv = {
    #     "true_params":               result_sv["true_params"],
    #     "fitted_params":             result_sv["fitted_params"],
    #     "true_param_loglik":         result_sv["true_param_loglik"],
    #     "best_loglik":               result_sv["best_loglik"],
    #     "bic":                       result_sv["bic"],
    #     "variance_rmse_p":           result_sv["variance_rmse_p"],
    #     "variance_rmse_m":           result_sv["variance_rmse_m"],
    #     "true_param_variance_rmse_p": result_sv["true_param_variance_rmse_p"],
    #     "true_param_variance_rmse_m": result_sv["true_param_variance_rmse_m"],
    #     "mean_ess":                  result_sv["mean_ess"],
    # }
    # print(json.dumps(summary_sv, indent=2, sort_keys=True))



if __name__ == "__main__":
    main()
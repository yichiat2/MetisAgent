"""bm_kernel.py
--------------
Beylkin-Monzón exponential-sum approximation of the power-law kernel

    K(t) = t^α / Γ(1+α),  α ∈ (-0.5, 0)

used by the Quadratic Rough Heston+ (QRH+) model.

Key public API
--------------
precompute_bm_table(m, delta, T, ...)  → (alpha_grid, c_table, gamma_table)
    Offline: fit c, gamma on an alpha grid; cache to disk.

get_nearest_bm(alpha, alpha_grid, c_table, gamma_table)  → (c, gamma)
    Online (JAX-traced): nearest-grid lookup. Compatible with jax.jit / vmap.

singular_consts_np(alpha, delta)  → {"beta": float, "std_eta": float}
    Host-side singular increment constants for the data generator.
"""

from __future__ import annotations

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import gamma as scipy_gamma

# ---------------------------------------------------------------------------
# Core least-squares solver
# ---------------------------------------------------------------------------

def fit_exponential_sum(
    alpha: float,
    m: int,
    delta: float = 1.0,
    T: float = 1000.0,
    n_grid: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an m-term exponential sum to K(t) = t^α / Γ(1+α) on [delta, T].

    Uses a two-stage approach: initialise log-spaced gammas, solve c via
    least-squares, then refine gammas jointly.

    Parameters
    ----------
    alpha  : roughness exponent ∈ (-0.5, 0)
    m      : number of exponential terms
    delta  : smallest time in the fit grid (usually one time step Δ)
    T      : largest time in the fit grid
    n_grid : number of log-spaced grid points for the fit

    Returns
    -------
    c     : (m,) coefficient array (float64)
    gamma : (m,) decay-rate array (float64, positive, ascending)
    """
    G = scipy_gamma(1.0 + alpha)
    t_grid = np.geomspace(delta, T, n_grid)
    target = t_grid ** alpha / G
    weights = 1.0 / (np.abs(target) + 1e-12)
    weights /= weights.sum()
    sqrt_w = np.sqrt(weights)

    gamma_lo = max(0.1 / T, 1e-4)
    gamma_hi = min(10.0 / delta, 1e6)
    log_gamma_init = np.linspace(np.log(gamma_lo), np.log(gamma_hi), m)

    def _solve_c(log_gammas):
        gammas = np.exp(np.clip(log_gammas, -10, 15))
        basis = np.exp(-gammas[None, :] * t_grid[:, None])
        wb = sqrt_w[:, None] * basis
        wt = sqrt_w * target
        cs, _, _, _ = np.linalg.lstsq(wb, wt, rcond=None)
        return cs

    def _residuals(log_gammas):
        gammas = np.exp(np.clip(log_gammas, -10, 15))
        cs = _solve_c(log_gammas)
        basis = np.exp(-gammas[None, :] * t_grid[:, None])
        pred = basis @ cs
        return np.sum(weights * (pred - target) ** 2)

    result = minimize(
        _residuals,
        log_gamma_init,
        method="L-BFGS-B",
        bounds=[(-10, 15)] * m,
        options={"maxiter": 5000, "ftol": 1e-15},
    )
    gammas = np.exp(np.clip(result.x, -10, 15))
    cs = _solve_c(result.x)

    order = np.argsort(gammas)
    return cs[order].astype(np.float64), gammas[order].astype(np.float64)


# ---------------------------------------------------------------------------
# Multi-core worker
# ---------------------------------------------------------------------------

def _fit_bm_worker(
    args: tuple[float, int, float, float, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Process-pool wrapper around fit_exponential_sum."""
    alpha, m, delta, T, n_grid = args
    c, gamma = fit_exponential_sum(
        float(alpha), int(m), float(delta), float(T), int(n_grid)
    )
    return float(alpha), c, gamma


# ---------------------------------------------------------------------------
# Precomputed alpha-grid lookup table
# ---------------------------------------------------------------------------

_TABLE_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def precompute_bm_table(
    m: int,
    delta: float,
    T: float,
    alpha_min: float = -0.499,
    alpha_max: float = -0.01,
    step: float = 0.001,
    n_grid: int = 25000,
    num_workers: int = 10,
    cache_dir: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Precompute (c, gamma) on a 1-D alpha grid for fast nearest lookup.

    Results are stored on disk (as an .npz file) and in a module-level
    in-memory cache.  Subsequent calls with the same parameters return
    immediately from one of the caches.

    Parameters
    ----------
    m          : number of exponential terms per alpha
    delta      : smallest time in the kernel fit (same units as T)
    T          : kernel fit horizon
    alpha_min  : inclusive lower bound of the alpha grid
    alpha_max  : inclusive upper bound of the alpha grid
    step       : uniform spacing of the alpha grid
    n_grid     : kernel fit grid points per alpha value
    num_workers: number of processes for parallel fitting
    cache_dir  : directory for the .npz cache file; None disables disk cache

    Returns
    -------
    alpha_grid  : (n_alpha,)     float32 JAX array
    c_table     : (n_alpha, m)   float32 JAX array
    gamma_table : (n_alpha, m)   float32 JAX array
    """
    n_steps = int(np.round((alpha_max - alpha_min) / step))
    alpha_grid_np = alpha_min + step * np.arange(n_steps + 1, dtype=np.float64)

    cache_key = (
        int(m),
        round(float(delta), 12),
        round(float(T), 12),
        round(float(alpha_min), 6),
        round(float(alpha_max), 6),
        round(float(step), 6),
        int(n_grid),
    )

    if cache_key in _TABLE_CACHE:
        ag, ct, gt = _TABLE_CACHE[cache_key]
        return (
            jnp.array(ag, dtype=jnp.float32),
            jnp.array(ct, dtype=jnp.float32),
            jnp.array(gt, dtype=jnp.float32),
        )

    cache_path: str | None = None
    if cache_dir is not None:
        fname = (
            f"bm_m{m}_d{delta:.12e}_T{T:.12e}_"
            f"amin{alpha_min:.3f}_amax{alpha_max:.3f}_"
            f"step{step:.4f}_ng{n_grid}.npz"
        )
        cache_path = os.path.join(cache_dir, fname)
        if os.path.exists(cache_path):
            data = np.load(cache_path)
            ag = np.asarray(data["alpha_grid"], dtype=np.float64)
            ct = np.asarray(data["c_table"],    dtype=np.float64)
            gt = np.asarray(data["gamma_table"], dtype=np.float64)
            _TABLE_CACHE[cache_key] = (ag, ct, gt)
            return (
                jnp.array(ag, dtype=jnp.float32),
                jnp.array(ct, dtype=jnp.float32),
                jnp.array(gt, dtype=jnp.float32),
            )

    tasks = [
        (float(a), int(m), float(delta), float(T), int(n_grid))
        for a in alpha_grid_np
    ]
    if num_workers == 1:
        results = [_fit_bm_worker(t) for t in tasks]
    else:
        chunksize = max(1, len(tasks) // (num_workers * 4))
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp_ctx) as ex:
            results = list(ex.map(_fit_bm_worker, tasks, chunksize=chunksize))

    results.sort(key=lambda r: r[0])
    ct = np.stack([r[1] for r in results])   # (n_alpha, m)
    gt = np.stack([r[2] for r in results])   # (n_alpha, m)

    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)  # type: ignore[arg-type]
        np.savez(cache_path, alpha_grid=alpha_grid_np, c_table=ct, gamma_table=gt)

    _TABLE_CACHE[cache_key] = (alpha_grid_np, ct, gt)
    return (
        jnp.array(alpha_grid_np, dtype=jnp.float32),
        jnp.array(ct,            dtype=jnp.float32),
        jnp.array(gt,            dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# JAX-traced nearest-grid lookup
# ---------------------------------------------------------------------------

def get_nearest_bm(
    alpha: jnp.ndarray,
    alpha_grid: jnp.ndarray,       # (n_alpha,)
    c_table: jnp.ndarray,          # (n_alpha, m)
    gamma_table: jnp.ndarray,      # (n_alpha, m)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the (c, gamma) row nearest to *alpha*.

    Compatible with jax.jit and jax.vmap: all operations are JAX primitives.
    """
    step = alpha_grid[1] - alpha_grid[0]
    idx = jnp.rint((alpha - alpha_grid[0]) / step).astype(jnp.int32)
    idx = jnp.clip(idx, 0, alpha_grid.shape[0] - 1)
    return c_table[idx], gamma_table[idx]   # each (m,)


# ---------------------------------------------------------------------------
# Host-side singular increment constants (for the data generator)
# ---------------------------------------------------------------------------

def singular_consts_np(alpha: float, delta: float) -> dict[str, float]:
    """Compute beta and std_eta for the Rømer hybrid singular increment.

    δX = beta * δW + std_eta * ε_η,  ε_η ∼ N(0,1) independent.

    Derived from the exact first two moments of the Riemann-Liouville
    singularity at lag zero:

        Cov(δX, δW) = Δ^{α+1} / Γ(α+2)
        Var(δX)     = Δ^{2α+1} / [(2α+1) Γ(1+α)²]

    Parameters
    ----------
    alpha : roughness exponent ∈ (-0.5, 0)
    delta : time step Δ (same units as T in precompute_bm_table)

    Returns
    -------
    dict with keys "beta" (float) and "std_eta" (float)
    """
    G1 = scipy_gamma(1.0 + alpha)
    G2 = scipy_gamma(2.0 + alpha)
    cov_dx_dw = delta ** (alpha + 1.0) / G2
    var_dx    = delta ** (2.0 * alpha + 1.0) / ((2.0 * alpha + 1.0) * G1 ** 2)
    var_eta   = max(var_dx - cov_dx_dw ** 2 / delta, 0.0)
    beta      = cov_dx_dw / delta           # = delta^alpha / G2
    return {"beta": float(beta), "std_eta": float(np.sqrt(var_eta))}

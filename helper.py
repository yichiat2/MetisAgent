from typing import NamedTuple

import jax
import jax.numpy as jnp
import chex
import numpy as np


VARIANCE_FLOOR = 1e-8
EPS = 1e-30

class FilterCarry(NamedTuple):
    particles: chex.Array
    log_weights: chex.Array
    key: chex.PRNGKey
    total_loglik: chex.Array


class FilterInfo(NamedTuple):
    filtered_mean: chex.Array
    filtered_std: chex.Array
    ess: chex.Array
    loglik_increments: chex.Array
    pred_log_return_mean: chex.Array
    pred_log_return_std: chex.Array


class SVJFilterInfo(NamedTuple):
    """FilterInfo variant for StochasticVolatilityJumpProcess.

    Uses per-step variance quantiles instead of std, and adds jump_prob.
    """
    filtered_mean: chex.Array          # (T,) particle mean of V
    filtered_q05:  chex.Array          # (T,) 5th-percentile of V
    filtered_q25:  chex.Array          # (T,) 25th-percentile of V
    filtered_q50:  chex.Array          # (T,) 50th-percentile (median) of V
    filtered_q75:  chex.Array          # (T,) 75th-percentile of V
    filtered_q95:  chex.Array          # (T,) 95th-percentile of V
    ess:                  chex.Array   # (T,) effective sample size
    loglik_increments:    chex.Array   # (T,) per-step log-likelihood
    pred_log_return_mean: chex.Array   # (T,) one-step-ahead predicted log-return mean
    pred_log_return_std:  chex.Array   # (T,) one-step-ahead predicted log-return std
    pred_log_return_skew: chex.Array   # (T,) one-step-ahead predicted log-return skewness
    pred_log_return_kurt: chex.Array   # (T,) one-step-ahead predicted log-return excess kurtosis
    pred_var_mean:        chex.Array   # (T,) one-step-ahead predicted variance mean
    pred_var_q05:         chex.Array   # (T,) 5th-percentile of predicted variance
    pred_var_q25:         chex.Array   # (T,) 25th-percentile of predicted variance
    pred_var_q50:         chex.Array   # (T,) 50th-percentile of predicted variance
    pred_var_q75:         chex.Array   # (T,) 75th-percentile of predicted variance
    pred_var_q95:         chex.Array   # (T,) 95th-percentile of predicted variance
    jump_prob:            chex.Array   # (T,) mean posterior jump probability


def _positive_variance(variance: chex.Array) -> chex.Array:
    return jnp.maximum(variance, jnp.float32(VARIANCE_FLOOR))


def _gaussian_logpdf(y: chex.Array, mean: chex.Array, variance: chex.Array) -> chex.Array:
    variance = _positive_variance(variance)
    return -0.5 * (jnp.log(2.0 * jnp.pi * variance) + (y - mean) ** 2 / variance)


def _systematic_resample(log_weights: chex.Array, u: chex.Array) -> chex.Array:
    """Systematic resampling given a pre-drawn uniform u ~ Uniform[0, 1)."""
    # num_particles = log_weights.shape[0]
    # weights = jax.nn.softmax(log_weights)
    # cumulative_weights = jnp.cumsum(weights)
    # positions = (u + jnp.arange(num_particles, dtype=jnp.float32)) / num_particles
    # ancestors = jnp.searchsorted(cumulative_weights, positions, side="right")
    # return jnp.clip(ancestors, 0, num_particles - 1)
    num_particles = log_weights.shape[0]
    weights = jax.nn.softmax(log_weights)
    cumulative_weights = jnp.cumsum(weights)
    
    # In systematic sampling: positions[i] = (u + i) / N
    # A particle i 'wins' a sample if positions[j] falls in [CDF[i-1], CDF[i]]
    # Trick: calculate the integer number of steps (1/N) that fit in each weight bin
    
    # Shifted CDF to find boundaries
    cdf_shifted = jnp.concatenate([jnp.zeros(1), cumulative_weights[:-1]])
    
    # Number of samples before each particle's start and end point
    counts_end = jnp.floor(num_particles * (cumulative_weights - u / num_particles)).astype(jnp.int32)
    counts_start = jnp.floor(num_particles * (cdf_shifted - u / num_particles)).astype(jnp.int32)
    
    # The 'histogram' of counts per particle
    particle_counts = jnp.maximum(0, counts_end - counts_start)
    
    # 2. Expand counts to get ancestor indices (e.g., using jnp.repeat)
    # This replaces searchsorted with a direct construction
    ancestors = jnp.repeat(jnp.arange(num_particles), particle_counts, total_repeat_length=num_particles)
    
    return ancestors


def _gamma_logpdf(x: chex.Array, alpha: float, scale: chex.Array) -> chex.Array:
    """Log-pdf of Gamma(shape=alpha, scale=scale).

    Parameterisation: mean = alpha * scale, variance = alpha * scale^2.
    Both x and scale are floored to VARIANCE_FLOOR for numerical safety.
    """
    x     = _positive_variance(x)
    scale = _positive_variance(scale)
    return (
        (alpha - 1.0) * jnp.log(x)
        - x / scale
        - alpha * jnp.log(scale)
        - jax.lax.lgamma(jnp.float32(alpha))
    )


def _effective_sample_size(log_weights: chex.Array) -> chex.Array:
    normalized_log_weights = log_weights - jax.nn.logsumexp(log_weights)
    return jnp.exp(-jax.nn.logsumexp(2.0 * normalized_log_weights))


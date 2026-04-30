from __future__ import annotations

from collections.abc import Callable

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import PPOVolScalpingConfig
from .contracts import STATE_DIM


def sample_and_log_prob(rng: jax.Array, mean: jnp.ndarray, log_std: jnp.ndarray, scale: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    std = jnp.exp(log_std)
    eps = jax.random.normal(rng, shape=mean.shape)
    z = mean + std * eps
    action = scale * jax.nn.sigmoid(z)
    action = jnp.clip(action, 1e-5, scale - 1e-5)  
    # log p(z) element-wise via reparameterisation
    log_prob_z = -0.5 * eps ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
    # For a = scale * sigmoid(z), log p(a) = log p(z) - log |da/dz|.
    log_abs_det = jnp.log(scale) - jax.nn.softplus(-z) - jax.nn.softplus(z)
    log_prob = jnp.sum(log_prob_z - log_abs_det, axis=-1)
    return action, log_prob

def deterministic_action(mean: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(scale * jax.nn.sigmoid(mean), 1e-5, scale - 1e-5)

def get_log_prob(mean: jnp.ndarray, log_std: jnp.ndarray, action: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    action = jnp.clip(action, 1e-5, scale - 1e-5)
    std = jnp.exp(log_std)
    # Invert the transform: z = logit(action / scale)
    u = action / scale
    z = jnp.log(u) - jnp.log1p(-u)
    log_prob_z = -0.5 * ((z - mean) / std) ** 2 - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
    log_abs_det = jnp.log(scale) - jax.nn.softplus(-z) - jax.nn.softplus(z)
    return jnp.sum(log_prob_z - log_abs_det, axis=-1)

def get_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
    # Differential entropy of the base Normal: 0.5 * (1 + log(2*pi) + 2*log_std)
    return jnp.sum(0.5 * (1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * log_std), axis=-1)


class Actor(nn.Module):
    hidden_sizes: tuple[int, ...] = (64, 64)
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    action_dim: int = 3
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> distrax.Distribution:
        activation = lambda x: nn.leaky_relu(x, negative_slope=0.01)
        hidden = state
        for width in self.hidden_sizes:
            hidden = nn.Dense(
                width,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(hidden)
            hidden = activation(hidden)
        mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),)(hidden)
        raw_log_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0),)(hidden)
        log_std = jnp.clip(raw_log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class Critic(nn.Module):
    hidden_sizes: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        activation = lambda x: nn.leaky_relu(x, negative_slope=0.01)
        hidden = state
        for width in self.hidden_sizes:
            hidden = nn.Dense(
                width,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(hidden)
            hidden = activation(hidden)

        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(hidden)
        return jnp.squeeze(value, axis=-1)


def create_train_states(
    config: PPOVolScalpingConfig,
    rng: jax.Array,
) -> tuple[TrainState, TrainState]:
    dummy_state = jnp.zeros((STATE_DIM,), dtype=jnp.float32)
    actor = Actor(
        hidden_sizes=config.model.hidden_sizes,
        log_std_min=config.model.log_std_min,
        log_std_max=config.model.log_std_max,
        action_dim=config.model.action_dim,
        action_scale=config.model.action_scale,
    )
    critic = Critic(hidden_sizes=config.model.hidden_sizes)

    actor_rng, critic_rng = jax.random.split(rng)
    actor_params = actor.init(actor_rng, dummy_state)["params"]
    critic_params = critic.init(critic_rng, dummy_state)["params"]

    actor_tx = optax.adam(learning_rate=config.ppo.actor_learning_rate)
    critic_tx = optax.adam(learning_rate=config.ppo.critic_learning_rate)

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=actor_tx,
    )
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=critic_tx,
    )
    return actor_state, critic_state




__all__ = [
    "Actor",
    "Critic",
    "create_train_states",
]
from __future__ import annotations

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import PPOVolScalpingConfig
from .contracts import ACTOR_STATE_DIM, CRITIC_STATE_DIM


def sample_and_log_prob(
    rng: jax.Array,
    dist: distrax.Beta,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    action = dist.sample(seed=rng)
    log_prob = get_log_prob(dist, action)
    return action, log_prob


def deterministic_action(dist: distrax.Beta) -> jnp.ndarray:
    return dist.mean()


def get_log_prob(dist: distrax.Beta, action: jnp.ndarray) -> jnp.ndarray:
    action = jnp.asarray(action)
    eps = jnp.asarray(1e-6, dtype=action.dtype)
    within_support = jnp.all((action >= 0.0) & (action <= 1.0), axis=-1)
    action = jnp.clip(action, eps, 1.0 - eps)
    log_prob = jnp.sum(dist.log_prob(action), axis=-1)
    return jnp.where(within_support, log_prob, -jnp.inf)


def get_entropy(dist: distrax.Beta) -> jnp.ndarray:
    return jnp.sum(dist.entropy(), axis=-1)


class Actor(nn.Module):
    hidden_sizes: tuple[int, ...] = (64, 64)
    action_dim: int = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> distrax.Beta:
        activation = lambda x: nn.leaky_relu(x, negative_slope=0.01)
        hidden = state
        for width in self.hidden_sizes:
            hidden = nn.Dense(
                width,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(hidden)
            hidden = activation(hidden)

        alpha_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)
        beta_logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)
        alpha = jax.nn.softplus(alpha_logits) + 1.0
        beta = jax.nn.softplus(beta_logits) + 1.0
        return distrax.Beta(alpha=alpha, beta=beta)


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
    dummy_actor_state = jnp.zeros((ACTOR_STATE_DIM,), dtype=jnp.float32)
    dummy_critic_state = jnp.zeros((CRITIC_STATE_DIM,), dtype=jnp.float32)
    actor = Actor(
        hidden_sizes=config.model.hidden_sizes,
        action_dim=config.model.action_dim,
    )
    critic = Critic(hidden_sizes=config.model.hidden_sizes)

    actor_rng, critic_rng = jax.random.split(rng)
    actor_params = actor.init(actor_rng, dummy_actor_state)["params"]
    critic_params = critic.init(critic_rng, dummy_critic_state)["params"]

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
    "deterministic_action",
    "get_entropy",
    "get_log_prob",
    "sample_and_log_prob",
]
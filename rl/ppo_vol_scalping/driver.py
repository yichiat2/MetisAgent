
import jax

from .config import PPOVolScalpingConfig, make_default_config
from .data import Fold, PreprocessedArrays
from .contracts import Transition
from .env import EnvParam, env_reset, env_step
from .model import Actor, Critic, create_train_states
import jax.numpy as jnp
from flax.training.train_state import TrainState

@jax.jit
def run_fold(config: PPOVolScalpingConfig,
             actor_state: TrainState,
             critic_state: TrainState,
             train_arrays: PreprocessedArrays, 
             eval_arrays: PreprocessedArrays, 
             episode_start_indices: jnp.ndarray,
             rng: jax.Array) -> tuple[TrainState, TrainState]:

    num_envs = config.ppo.num_env
    num_episodes = episode_start_indices.shape[0]
    num_runs = num_episodes // num_envs
    episode_length = config.environment.episode_length

    # Build a single EnvParam that covers the entire training window. Each
    # episode selects its slice via global_index offsets stored in EnvState.
    env_param = EnvParam(
        max_inventory=float(config.environment.max_inventory),
        max_quote_size=float(config.environment.max_quote_size),
        flatten_at_session_end=config.environment.flatten_at_session_end,
        ohlc=train_arrays.ohlc,
        static_features=train_arrays.static_features,
        atr=train_arrays.atr,
        sigma_price=train_arrays.sigma_price,
        day_ids=train_arrays.day_ids,
        bar_in_day=train_arrays.bar_in_day,
        session_end_mask=train_arrays.session_end_mask,
    )

    env_param_eval = EnvParam(
        max_inventory=float(config.environment.max_inventory),
        max_quote_size=float(config.environment.max_quote_size),
        flatten_at_session_end=config.environment.flatten_at_session_end,
        ohlc=eval_arrays.ohlc,
        static_features=eval_arrays.static_features,
        atr=eval_arrays.atr,
        sigma_price=eval_arrays.sigma_price,
        day_ids=eval_arrays.day_ids,
        bar_in_day=eval_arrays.bar_in_day,
        session_end_mask=eval_arrays.session_end_mask,
    )

    def _update_step(runner_state, unused):
        actor_state, critic_state, rng = runner_state

        # Shuffle episode_start_indices and drop the tail so shape is (num_runs, num_envs).
        rng, _rng = jax.random.split(rng)
        shuffled = jax.random.permutation(_rng, episode_start_indices)  # shuffle actual bar positions
        run_indices = shuffled[: num_runs * num_envs].reshape((num_runs, num_envs))  # (num_runs, num_envs)

        def _run_step(carry, start_indices):
            # start_indices: (num_envs,)  — the episode start bar for each env.
            actor_state, critic_state, rng = carry

            # Reset all envs in parallel.
            init_obs, init_states = jax.vmap(
                lambda idx: env_reset(env_param, global_index=idx)
            )(start_indices)  # obs: (num_envs, obs_dim), states: batched EnvState

            def _env_step(step_carry, unused):
                obs, env_states, rng = step_carry

                # Sample actions from actor.
                rng, _rng = jax.random.split(rng)

                dist = actor_state.apply_fn({"params": actor_state.params}, obs)
                actions, log_probs = dist.sample_and_log_prob(seed=_rng)  # (num_envs, 3), (num_envs,)

                # Critic values.
                values = critic_state.apply_fn({"params": critic_state.params}, obs)  # (num_envs,)

                # Step all envs.
                next_obs, next_states, rewards, dones, infos = jax.vmap(
                    lambda s, a: env_step(s, a, env_param, config.reward)
                )(env_states, actions)

                transition = Transition(
                    done=dones,
                    action=actions,
                    value=values,
                    reward=rewards,
                    log_prob=log_probs,
                    obs=obs,
                    info=infos,
                )
                return (next_obs, next_states, rng), transition

            (last_obs, last_states, rng), traj_batch = jax.lax.scan(
                _env_step,
                (init_obs, init_states, rng),
                None,
                length=episode_length,
            )
            # traj_batch fields: (episode_length, num_envs, ...)

            # Bootstrap value for the last observation.
            last_val = critic_state.apply_fn({"params": critic_state.params}, last_obs)  # (num_envs,)

            # ----------------------------------------------------------------
            # GAE
            # ----------------------------------------------------------------
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config.ppo.discount * next_value * (1.0 - done) - value
                    gae = delta + config.ppo.discount * config.ppo.gae_lambda * (1.0 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # ----------------------------------------------------------------
            # PPO update
            # ----------------------------------------------------------------
            def _update_epoch(update_state, unused):
                actor_state, critic_state, traj_batch, advantages, targets, rng = update_state

                rng, _rng = jax.random.split(rng)
                batch_size = episode_length * num_envs
                num_minibatches = batch_size // config.ppo.minibatch_size

                permutation = jax.random.permutation(_rng, batch_size)

                def _flatten(x):
                    return x.reshape((batch_size,) + x.shape[2:])

                batch = jax.tree_util.tree_map(_flatten, (traj_batch, advantages, targets))
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((num_minibatches, -1) + x.shape[1:]), shuffled_batch
                )
                mb_traj, mb_adv, mb_tgt = minibatches

                def _update_minbatch(states, minibatch):
                    actor_state, critic_state = states
                    traj, adv, tgt = minibatch

                    # Actor loss.
                    def _actor_loss(actor_params):
                        dist = actor_state.apply_fn({"params": actor_params}, traj.obs)
                        log_prob = dist.log_prob(traj.action)
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                        loss1 = ratio * norm_adv
                        loss2 = jnp.clip(ratio, 1.0 - config.ppo.clip_epsilon, 1.0 + config.ppo.clip_epsilon) * norm_adv
                        actor_loss = -jnp.minimum(loss1, loss2).mean()
                        entropy = dist.entropy().mean()
                        total = actor_loss - config.ppo.entropy_coefficient * entropy
                        return total, (actor_loss, entropy)

                    # Critic loss.
                    def _critic_loss(critic_params):
                        value = critic_state.apply_fn({"params": critic_params}, traj.obs)
                        value_clipped = traj.value + jnp.clip(
                            value - traj.value, -config.ppo.clip_epsilon, config.ppo.clip_epsilon
                        )
                        vf_loss = 0.5 * jnp.maximum(
                            jnp.square(value - tgt),
                            jnp.square(value_clipped - tgt),
                        ).mean()
                        return vf_loss, vf_loss

                    (actor_total, (actor_loss, entropy)), actor_grads = jax.value_and_grad(
                        _actor_loss, has_aux=True
                    )(actor_state.params)
                    (critic_total, _), critic_grads = jax.value_and_grad(
                        _critic_loss, has_aux=True
                    )(critic_state.params)

                    # L1 regularisation.
                    if config.ppo.actor_l1 > 0.0:
                        actor_grads = jax.tree_util.tree_map(
                            lambda g, p: g + config.ppo.actor_l1 * jnp.sign(p),
                            actor_grads, actor_state.params,
                        )
                    if config.ppo.critic_l1 > 0.0:
                        critic_grads = jax.tree_util.tree_map(
                            lambda g, p: g + config.ppo.critic_l1 * jnp.sign(p),
                            critic_grads, critic_state.params,
                        )

                    actor_state = actor_state.apply_gradients(grads=actor_grads)
                    critic_state = critic_state.apply_gradients(grads=critic_grads)
                    loss_info = (actor_total, critic_total, actor_loss, entropy)
                    return (actor_state, critic_state), loss_info

                (actor_state, critic_state), loss_info = jax.lax.scan(
                    _update_minbatch,
                    (actor_state, critic_state),
                    (mb_traj, mb_adv, mb_tgt),
                )
                update_state = (actor_state, critic_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (actor_state, critic_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, length=config.ppo.epochs
            )
            actor_state, critic_state, _, _, _, rng = update_state

            return (actor_state, critic_state, rng), loss_info

        # Scan over num_runs episodes.
        (actor_state, critic_state, rng), run_loss_info = jax.lax.scan(
            _run_step,
            (actor_state, critic_state, rng),
            run_indices,
        )

        # TODO: Perform a full run over eval_arrays and collect metrics based. 


        runner_state = (actor_state, critic_state, rng)
        return runner_state, run_loss_info

    runner_state = (actor_state, critic_state, rng)
    runner_state, metrics = jax.lax.scan(
        _update_step, runner_state, None, length=config.ppo.num_update
    )
    actor_state, critic_state, _ = runner_state
    return actor_state, critic_state, metrics

def run(folds: list[Fold], preprocessed_arrays: PreprocessedArrays) -> None:

    config = make_default_config()
    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)
    actor_state, critic_state = create_train_states(config=config, rng=_rng)

    for fold in folds:

        print(f"Running fold {fold.fold_id}...")
        train_arrays = preprocessed_arrays[fold.train_start:fold.train_end]
        eval_arrays = preprocessed_arrays[fold.inference_start:fold.inference_end]
        episode_start_indices = fold.episode_start_indices
        print(f"  Train length: {fold.train_length} bars")
        print(f"  Evaluation length: {fold.evaluation_length} bars")
        print(f"  Number of training episodes: {len(episode_start_indices)}")
        actor_state, critic_state, metrics = run_fold(
            config=config,
            actor_state=actor_state,
            critic_state=critic_state,
            train_arrays=train_arrays,
            eval_arrays=eval_arrays,
            episode_start_indices=episode_start_indices,
            rng=rng,
        )
from functools import partial

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from .config import PPOVolScalpingConfig, make_default_config
from .contracts import Fold, PreprocessedArrays, Transition
from .data import (
    build_preprocessed_arrays,
    build_walk_forward_folds,
    load_raw_bars_dataframe,
)
from .env import EnvParam, env_reset, env_step, build_env_param
from .model import (
    create_train_states,
    deterministic_action,
    get_entropy,
    get_log_prob,
    sample_and_log_prob,
)
from .visual import prepare_inference_payload, push_inference_metrics, start_dashboard_server


def _summarize_inference(
    config: PPOVolScalpingConfig,
    step_pnl: jax.Array,
    step_returns: jax.Array,
    is_bankrupt: jax.Array,
    bid_fill: jax.Array,
    ask_fill: jax.Array,
) -> dict[str, jax.Array]:
    epsilon = jnp.asarray(config.reward.reward_epsilon, dtype=jnp.float32)
    annualization = jnp.sqrt(
        jnp.asarray(config.logging.evaluation_annualization_factor, dtype=jnp.float32)
    )
    cumulative_pnl = jnp.cumsum(step_pnl)
    bankruptcy_mask = is_bankrupt > 0.0
    path_returns = jnp.where(bankruptcy_mask, -1.0, step_returns)
    wealth_curve = jnp.concatenate(
        [jnp.ones((1,), dtype=step_returns.dtype), jnp.cumprod(1.0 + path_returns)]
    )
    cumulative_return = wealth_curve[1:] - 1.0
    running_peak = jax.lax.associative_scan(jnp.maximum, wealth_curve)
    drawdown = 1.0 - wealth_curve / (running_peak + epsilon)
    mean_return = jnp.mean(step_returns)
    return_std = jnp.std(step_returns)
    downside_deviation = jnp.sqrt(jnp.mean(jnp.square(jnp.minimum(step_returns, 0.0))))
    bankrupt_any = jnp.any(bankruptcy_mask)
    sharpe_ratio = jnp.where(
        bankrupt_any,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        annualization * mean_return / (return_std + epsilon),
    )
    sortino_ratio = jnp.where(
        bankrupt_any,
        jnp.asarray(jnp.nan, dtype=jnp.float32),
        annualization * mean_return / (downside_deviation + epsilon),
    )
    max_drawdown = jnp.max(drawdown)
    final_cumulative_return = cumulative_return[-1]
    bid_counts = jnp.asarray(bid_fill, dtype=jnp.int32)
    ask_counts = jnp.asarray(ask_fill, dtype=jnp.int32)
    transaction_count = jnp.sum(bid_counts + ask_counts, dtype=jnp.int32)
    return {
        "cumulative_pnl": cumulative_pnl,
        "total_pnl": cumulative_pnl[-1],
        "cumulative_return": cumulative_return,
        "final_cumulative_return": final_cumulative_return,
        "max_drawdown": max_drawdown,
        "bankruptcy": bankrupt_any,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "transaction_count": transaction_count,
    }


def _compute_episode_max_drawdown(
    portfolio_value_before: jax.Array,
    portfolio_value_after: jax.Array,
    epsilon: float,
) -> jax.Array:
    starting_portfolio_value = portfolio_value_before[:1]
    portfolio_path = jnp.concatenate([starting_portfolio_value, portfolio_value_after], axis=0)
    running_peak = jax.lax.associative_scan(jnp.maximum, portfolio_path)
    drawdown = 1.0 - portfolio_path / (running_peak + epsilon)
    return jnp.max(drawdown, axis=0)

@partial(jax.jit, static_argnames=("config",))
def run_fold_inference(
    config: PPOVolScalpingConfig,
    actor_state: TrainState,  
    env_param: EnvParam,
):

    num_steps = env_param.static_features.shape[0] - 1

    init_obs, init_state = env_reset(env_param, global_index=0)

    def _env_step(step_carry, unused):
        obs, env_state = step_carry

        dist = actor_state.apply_fn({"params": actor_state.params}, obs)
        action = deterministic_action(dist)

        next_obs, next_state, _, _, info = env_step(
            env_state,
            action,
            env_param,
            config.reward,
        )
        step_info = {
            "actions": info["action"],
            "ask_fill": info["ask_fill"],
            "ask_price": info["ask_price"],
            "ask_size": info["ask_size"],
            "bid_fill": info["bid_fill"],
            "bid_price": info["bid_price"],
            "bid_size": info["bid_size"],
            "cash": info["cash_after"],
            "is_bankrupt": info["is_bankrupt"],
            "inventory": info["inventory_after"],
            "pnl": info["pnl"],
            "portfolio_value": info["portfolio_value_after"],
            "return": info["return"],
        }
        return (next_obs, next_state), step_info

    _, step_info = jax.lax.scan(
        _env_step,
        (init_obs, init_state),
        None,
        length=num_steps,
    )

    summary_metrics = _summarize_inference(
        config=config,
        step_pnl=step_info["pnl"],
        step_returns=step_info["return"],
        is_bankrupt=step_info["is_bankrupt"],
        bid_fill=step_info["bid_fill"],
        ask_fill=step_info["ask_fill"],
    )
    metrics = {
        **step_info,
        **summary_metrics,
    }
    return metrics

@partial(jax.jit, static_argnames=("config",))
def run_fold_update(
    config: PPOVolScalpingConfig,
    actor_state: TrainState,
    critic_state: TrainState,
    env_param: EnvParam,
    episode_start_indices: jnp.ndarray,
    rng: jax.Array,
):
    num_envs = config.ppo.num_env
    episode_length = config.environment.episode_length
    num_episodes = episode_start_indices.shape[0]
    num_runs = num_episodes // num_envs
    discount_factors = jnp.power(
        jnp.asarray(config.ppo.discount, dtype=jnp.float32),
        jnp.arange(episode_length, dtype=jnp.float32),
    )

    # Shuffle episode start indices and drop the tail so shape is (num_runs, num_envs).
    rng, shuffle_rng = jax.random.split(rng)
    shuffled = jax.random.permutation(shuffle_rng, episode_start_indices)
    run_indices = shuffled[: num_runs * num_envs].reshape((num_runs, num_envs))

    def _run_step(carry, start_indices):
        actor_state, critic_state, rng = carry

        init_obs, init_states = jax.vmap(
            lambda idx: env_reset(env_param, global_index=idx)
        )(start_indices)

        def _env_step(step_carry, unused):
            obs, env_states, rng = step_carry

            rng, sample_rng = jax.random.split(rng)

            dist = actor_state.apply_fn({"params": actor_state.params}, obs)
            actions, log_probs = sample_and_log_prob(sample_rng, dist)

            values = critic_state.apply_fn({"params": critic_state.params}, obs)

            next_obs, next_states, rewards, dones, infos = jax.vmap(
                lambda state, action: env_step(state, action, env_param, config.reward)
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

        (last_obs, _, rng), traj_batch = jax.lax.scan(
            _env_step,
            (init_obs, init_states, rng),
            None,
            length=episode_length,
        )

        episode_max_drawdown = _compute_episode_max_drawdown(
            portfolio_value_before=traj_batch.info["portfolio_value_before"],
            portfolio_value_after=traj_batch.info["portfolio_value_after"],
            epsilon=config.reward.reward_epsilon,
        )

        last_val = critic_state.apply_fn({"params": critic_state.params}, last_obs)

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
        discounted_episode_reward = jnp.sum(
            traj_batch.reward * discount_factors[:, None],
            axis=0,
        ).mean()

        def _update_epoch(update_state, unused):
            actor_state, critic_state, traj_batch, advantages, targets, rng = update_state

            rng, permutation_rng = jax.random.split(rng)
            batch_size = episode_length * num_envs
            num_minibatches = batch_size // config.ppo.minibatch_size

            permutation = jax.random.permutation(permutation_rng, batch_size)

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

                def _actor_loss(actor_params):
                    dist = actor_state.apply_fn({"params": actor_params}, traj.obs)
                    log_prob = get_log_prob(dist, traj.action)
                    ratio = jnp.exp(log_prob - traj.log_prob)
                    norm_adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    norm_adv = jnp.clip(norm_adv, -3.0, 3.0)
                    loss1 = ratio * norm_adv
                    loss2 = jnp.clip(
                        ratio,
                        1.0 - config.ppo.clip_epsilon,
                        1.0 + config.ppo.clip_epsilon,
                    ) * norm_adv
                    actor_loss = -jnp.minimum(loss1, loss2).mean()
                    entropy = get_entropy(dist).mean()
                    total = actor_loss - config.ppo.entropy_coefficient * entropy
                    return total, (actor_loss, entropy)

                def _critic_loss(critic_params):
                    value = critic_state.apply_fn({"params": critic_params}, traj.obs)
                    vf_loss = 0.5 * jnp.square(value - tgt).mean()
                    return vf_loss, vf_loss

                (actor_total, (actor_loss, entropy)), actor_grads = jax.value_and_grad(
                    _actor_loss, has_aux=True
                )(actor_state.params)
                (critic_total, _), critic_grads = jax.value_and_grad(
                    _critic_loss, has_aux=True
                )(critic_state.params)

                if config.ppo.actor_l1 > 0.0:
                    actor_grads = jax.tree_util.tree_map(
                        lambda grad, param: grad + config.ppo.actor_l1 * jnp.sign(param),
                        actor_grads,
                        actor_state.params,
                    )
                if config.ppo.critic_l1 > 0.0:
                    critic_grads = jax.tree_util.tree_map(
                        lambda grad, param: grad + config.ppo.critic_l1 * jnp.sign(param),
                        critic_grads,
                        critic_state.params,
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

        run_metrics = (
            discounted_episode_reward,
            jnp.mean(episode_max_drawdown),
        )
        return (actor_state, critic_state, rng), (loss_info, run_metrics)

    (actor_state, critic_state, rng), (run_loss_info, run_metrics) = jax.lax.scan(
        _run_step,
        (actor_state, critic_state, rng),
        run_indices,
    )

    _, critic_losses, actor_losses, entropies = run_loss_info
    run_discounted_episode_rewards, run_avg_max_drawdown = run_metrics
    avg_actor_loss_per_epoch = jnp.mean(actor_losses, axis=-1)
    avg_critic_loss_per_epoch = jnp.mean(critic_losses, axis=-1)
    avg_entropy_per_epoch = jnp.mean(entropies, axis=-1)
    training_metrics = {
        "avg_actor_loss": jnp.mean(avg_actor_loss_per_epoch[:, -1]),
        "avg_critic_loss": jnp.mean(avg_critic_loss_per_epoch[:, -1]),
        "avg_discounted_episode_reward": jnp.mean(run_discounted_episode_rewards),
        "avg_entropy": jnp.mean(avg_entropy_per_epoch[:, -1]),
        "avg_episode_max_drawdown": jnp.mean(run_avg_max_drawdown),
    }

    return actor_state, critic_state, rng, training_metrics

def run(
    folds: list[Fold],
    preprocessed_arrays: PreprocessedArrays,
    config: PPOVolScalpingConfig | None = None,
) -> None:
    rng = jax.random.PRNGKey(config.seed)
    rng, _rng = jax.random.split(rng)
    actor_state, critic_state = create_train_states(config=config, rng=_rng)

    for fold in folds:
        print(f"Running fold {fold.fold_id}...")
        train_arrays = preprocessed_arrays[fold.train_start:fold.train_end]
        infer_arrays = preprocessed_arrays[fold.inference_start:fold.inference_end]
        episode_start_indices = jnp.asarray(fold.episode_start_indices)
        print(f"  Train data length: {fold.train_length} bars")
        print(f"  Evaluation data length: {fold.inference_length} bars")
        print(f"  Episode length: {config.environment.episode_length} bars")
        print(f"  Minibatch size: {config.ppo.minibatch_size}")
        print(f"  Number of updates per fold: {config.ppo.num_update}")
        print(f"  Number of epochs per minibatch: {config.ppo.epochs}")
        print(f"  Number of environments: {config.ppo.num_env}")
        print(f"  Number of training episodes: {len(episode_start_indices)}")
        print(f"  Number of runs per update: {len(episode_start_indices) // config.ppo.num_env}")
        print(f"  Number of minibatches per run: {(config.environment.episode_length * config.ppo.num_env) // config.ppo.minibatch_size}    ")
        print(f"  Number of timesteps per update: {(len(episode_start_indices) // config.ppo.num_env) * config.ppo.num_env}")
        
        env_param = build_env_param(config, train_arrays)
        infer_env_param = build_env_param(config, infer_arrays)

        for update_step in range(config.ppo.num_update):
            actor_state, critic_state, rng, training_metrics = run_fold_update(
                config=config,
                actor_state=actor_state,
                critic_state=critic_state,
                env_param=env_param,
                episode_start_indices=episode_start_indices,
                rng=rng,
            )
            metrics = run_fold_inference(
                config=config,
                actor_state=actor_state,
                env_param=infer_env_param,
            )
            training_metrics_host = jax.device_get(training_metrics)
            metrics_host = jax.device_get(metrics)
            if update_step == config.ppo.num_update - 1:
                start_dashboard_server()
                push_inference_metrics(
                    prepare_inference_payload(
                        metrics=metrics_host,
                        ohlc=infer_arrays.ohlc,
                        fold_id=fold.fold_id,
                        update_step=update_step + 1,
                        num_updates=config.ppo.num_update,
                    )
                )
            print(
                f"  Update {update_step + 1}/{config.ppo.num_update}: "
                f"train_actor_loss={float(training_metrics_host['avg_actor_loss']):.6f}, "
                f"train_critic_loss={float(training_metrics_host['avg_critic_loss']):.6f}, "
                f"train_entropy={float(training_metrics_host['avg_entropy']):.6f}, "
                f"train_discounted_episode_reward={float(training_metrics_host['avg_discounted_episode_reward']):.6f}, "
                f"train_avg_episode_max_drawdown={float(training_metrics_host['avg_episode_max_drawdown']):.6f}, "
                f"infer_total_pnl={float(metrics_host['total_pnl']):.6f}, "
                f"infer_cumulative_return={float(metrics_host['final_cumulative_return']):.6f}, "
                f"infer_sharpe={float(metrics_host['sharpe_ratio']):.6f}, "
                f"infer_sortino={float(metrics_host['sortino_ratio']):.6f}, "
                f"infer_max_drawdown={float(metrics_host['max_drawdown']):.6f}, "
                f"infer_transactions={int(metrics_host['transaction_count'])}, "
                f"infer_bankruptcy={bool(metrics_host['bankruptcy'])}"
            )
        # pause between folds to allow dashboard inspection
        input("Press Enter to continue to the next fold...")


def main() -> None:
    config = make_default_config()
    raw_bars_df = load_raw_bars_dataframe(config.data)
    preprocessed_arrays = build_preprocessed_arrays(raw_bars_df, config.features)
    folds = build_walk_forward_folds(
        preprocessed_arrays=preprocessed_arrays,
        train_window_bars=config.data.train_window_bars,
        inference_window_bars=config.data.inference_window_bars,
        fold_stride_bars=config.data.fold_stride_bars,
        episode_length=config.environment.episode_length,
        episode_stride=config.environment.episode_stride,
    )
    run(folds=folds, preprocessed_arrays=preprocessed_arrays, config=config)


if __name__ == "__main__":
    main()
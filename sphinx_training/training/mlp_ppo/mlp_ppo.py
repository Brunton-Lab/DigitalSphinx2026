# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from .. import masked_running_statistics as running_statistics
from ..masked_running_statistics import RunningStatisticsState
from brax.training.acme import specs
from mujoco_playground import wrapper
from .. import ckpt_utils
from . import losses as ppo_losses
from . import ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from etils import epath
from pathlib import Path
from omegaconf import DictConfig, OmegaConf


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: ppo_losses.PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: types.UInt64


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.astype(leaf, leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _validate_madrona_args(
    madrona_backend: bool,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    eval_env: Optional[envs.Env] = None,
):
    """Validates arguments for Madrona-MJX."""
    if madrona_backend:
        if eval_env:
            raise ValueError("Madrona-MJX doesn't support multiple env instances")
        if num_eval_envs != num_envs:
            raise ValueError("Madrona-MJX requires a fixed batch size")
        if action_repeat != 1:
            raise ValueError(
                "Implement action_repeat using PipelineEnv's _n_frames to avoid"
                " unnecessary rendering!"
            )


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError("episode_length must be specified in ppo.train")
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)
    if wrap_env_fn is not None:
        wrap_for_training = wrap_env_fn
    else:
        wrap_for_training = wrapper.wrap_for_brax_training
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args
    return env


def _random_translate_pixels(
    obs: Mapping[str, jax.Array], key: PRNGKey
) -> Mapping[str, jax.Array]:
    """Apply random translations to B x T x ... pixel observations.

    The same shift is applied across the unroll_length (T) dimension.

    Args:
      obs: a dictionary of observations
      key: a PRNGKey

    Returns:
      A dictionary of observations with translated pixels
    """

    @jax.vmap
    def rt_all_views(
        ub_obs: Mapping[str, jax.Array], key: PRNGKey
    ) -> Mapping[str, jax.Array]:
        # Expects dictionary of unbatched observations.
        def rt_view(img: jax.Array, padding: int, key: PRNGKey) -> jax.Array:  # TxHxWxC
            # Randomly translates a set of pixel inputs.
            # Adapted from
            # https://github.com/ikostrikov/jaxrl/blob/main/jaxrl/agents/drq/augmentations.py
            crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
            zero = jnp.zeros((1,), dtype=jnp.int32)
            crop_from = jnp.concatenate([zero, crop_from, zero])
            padded_img = jnp.pad(
                img,
                ((0, 0), (padding, padding), (padding, padding), (0, 0)),
                mode="edge",
            )
            return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

        out = {}
        for k_view, v_view in ub_obs.items():
            if k_view.startswith("pixels/"):
                key, key_shift = jax.random.split(key)
                out[k_view] = rt_view(v_view, 4, key_shift)
        return {**ub_obs, **out}

    bdim = next(iter(obs.items()), None)[1].shape[0]
    keys = jax.random.split(key, bdim)
    obs = rt_all_views(obs, keys)
    return obs


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
    """Removes pixel observations from the observation dict."""
    if not isinstance(obs, Mapping):
        return obs
    return {k: v for k, v in obs.items() if not k.startswith("pixels/")}


def train(
    environment: envs.Env,
    num_timesteps: int,
    max_devices_per_host: Optional[int] = None,
    # high-level control flow
    wrap_env: bool = True,
    madrona_backend: bool = False,
    augment_pixels: bool = False,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_intention_ppo_networks,
    seed: int = 0,
    use_pmap_on_reset: bool = True,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = True,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
    restore_value_fn: bool = True,
    run_evals: bool = True,
    # custom args
    use_kl_schedule: bool = False,
    use_mmd_schedule: bool = False,
    kl_ramp: float = 1.25,
    kl_ramp_cycles: int = 4,
    kl_loss: bool = False,
    mmd_loss: bool = False,
    kl_weight: float = 1e-3,
    mmd_weight: float = 1e-3,
    warmup_steps: int = 0,
    freeze_mask_fn: Optional[Callable] = None,
    continue_training: bool = False,
    fine_tuning: bool = False,
):
    """PPO training.

    Args:
      environment: the environment to train
      num_timesteps: the total number of environment steps to use during training
      max_devices_per_host: maximum number of chips to use per host process
      wrap_env: If True, wrap the environment for training. Otherwise use the
        environment as is.
      madrona_backend: whether to use Madrona backend for training
      augment_pixels: whether to add image augmentation to pixel inputs
      num_envs: the number of parallel environments to use for rollouts
        NOTE: `num_envs` must be divisible by the total number of chips since each
          chip gets `num_envs // total_number_of_chips` environments to roll out
        NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
          data generated by `num_envs` parallel envs gets used for gradient
          updates over `num_minibatches` of data, where each minibatch has a
          leading dimension of `batch_size`
      episode_length: the length of an environment episode
      action_repeat: the number of timesteps to repeat an action
      wrap_env_fn: a custom function that wraps the environment for training. If
        not specified, the environment is wrapped with the default training
        wrapper.
      randomization_fn: a user-defined callback function that generates randomized
        environments
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss, higher values increase entropy of
        the policy
      discounting: discounting rate
      unroll_length: the number of timesteps to unroll in each environment. The
        PPO loss is computed over `unroll_length` timesteps
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step, each with a
        different minibatch with leading dimension of `batch_size`
      num_updates_per_batch: the number of times to run the gradient update over
        all minibatches before doing a new environment rollout
      num_resets_per_eval: the number of environment resets to run between each
        eval. The environment resets occur on the host
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      max_grad_norm: gradient clipping norm value. If None, no clipping is done
      normalize_advantage: whether to normalize advantage estimate
      network_factory: function that generates networks for policy and value
        functions
      seed: random seed
      num_evals: the number of evals to run during the entire training run.
        Increasing the number of evals increases total training time
      eval_env: an optional environment for eval only, defaults to `environment`
      num_eval_envs: the number of envs to use for evluation. Each env will run 1
        episode, and all envs run in parallel during eval.
      deterministic_eval: whether to run the eval with a deterministic policy
      log_training_metrics: whether to log training metrics and callback to
        progress_fn
      training_metrics_steps: the number of environment steps between logging
        training metrics
      progress_fn: a user-defined callback function for reporting/plotting metrics
      policy_params_fn: a user-defined callback function that can be used for
        saving custom policy checkpoints or creating policy rollouts and videos
      save_checkpoint_path: the path used to save checkpoints. If None, no
        checkpoints are saved.
      restore_checkpoint_path: the path used to restore previous model params
      restore_params: raw network parameters to restore the TrainingState from.
        These override `restore_checkpoint_path`. These paramaters can be obtained
        from the return values of ppo.train().
      restore_value_fn: whether to restore the value function from the checkpoint
        or use a random initialization
      run_evals: if True, use the evaluator num_eval times to collect distinct
        eval rollouts. If False, num_eval_envs and eval_env are ignored.
        progress_fn is then expected to use training_metrics.
      use_pmap_on_reset: default to True. if True, use pmap instead of vmap for
        env.reset across devices.

    Returns:
      Tuple of (make_policy function, network params, metrics)
    """
    assert (batch_size * num_minibatches) % num_envs == 0, f'batch_size: {batch_size}, num_minibatches: {num_minibatches}, num_envs: {num_envs}'
    _validate_madrona_args(
        madrona_backend, num_envs, num_eval_envs, action_repeat, eval_env
    )

    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)
    
    logging.info(f'num_evals_after_init: {num_evals_after_init}, num_training_steps_per_epoch: {num_training_steps_per_epoch}, env_step_per_training_step: {env_step_per_training_step}')

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value, policy_params_fn_key = jax.random.split(global_key, 3)
    del global_key

    assert num_envs % device_count == 0

    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
    )

    if local_devices_to_use > 1 or use_pmap_on_reset:
        reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    else:
        reset_fn = jax.jit(jax.vmap(env.reset))

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)
    init_eval_step = jnp.zeros_like(env_state.info['start_frame'])
    env_state.info['eval_step'] = init_eval_step
    # Discard the batch axes over devices and envs.
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)[0]
    
    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    task_obs_size = int(_unpmap(env_state.info['obs_sizes']['task_obs_size'])[0])
    network_factory = functools.partial(network_factory, task_obs_size=task_obs_size)
    ppo_network = network_factory(
        obs_shape, env.action_size, preprocess_observations_fn=normalize
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)

    optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        # TODO(btaba): Move gradient clipping to `training/gradients.py`.
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    # Setup KL/MMD scheduling if enabled
    if use_kl_schedule and kl_loss:
        kl_schedule = ppo_losses.cosine_annealing_schedule(
            final_value=kl_weight, 
            max_steps=num_evals//kl_ramp, 
            restart_steps=(num_evals//kl_ramp)//kl_ramp_cycles, 
            warmup_steps=warmup_steps
        )
    else:
        kl_schedule = kl_weight
        
    if use_mmd_schedule and mmd_loss:
        mmd_schedule = ppo_losses.cosine_annealing_schedule(
            final_value=mmd_weight, 
            max_steps=num_evals//kl_ramp, 
            restart_steps=(num_evals//kl_ramp)//kl_ramp_cycles, 
            warmup_steps=warmup_steps
        )
    else:
        mmd_schedule = mmd_weight

    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        kl_loss=kl_loss,
        mmd_loss=mmd_loss,
        kl_weight=kl_schedule if use_kl_schedule else kl_weight,
        mmd_weight=mmd_schedule if use_mmd_schedule else mmd_weight,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        use_kl_schedule=use_kl_schedule,
        use_mmd_schedule=use_mmd_schedule,
    )


    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )
    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        steps_between_logging=training_metrics_steps or env_step_per_training_step,
        progress_fn=progress_fn,
    )

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key, it = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params, normalizer_params, data, key_loss, it, optimizer_state=optimizer_state
        )

        return (optimizer_state, params, key, it), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key, it = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        if augment_pixels:
            key, key_rt = jax.random.split(key)
            r_translate = functools.partial(_random_translate_pixels, key=key_rt)
            data = types.Transition(
                observation=r_translate(data.observation),
                action=data.action,
                reward=data.reward,
                discount=data.discount,
                next_observation=r_translate(data.next_observation),
                extras=data.extras,
            )

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad, it),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key, it), metrics

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey, int], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey, int], Metrics]:
        training_state, state, key, it = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

        policy = make_policy(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation", "episode_metrics", "episode_done"),
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)

        if log_training_metrics:  # log unroll metrics
            jax.debug.callback(
                metrics_aggregator.update_episode_metrics,
                data.extras["state_extras"]["episode_metrics"],
                data.extras["state_extras"]["episode_done"],
            )

        # Update normalization params and normalize observations.
        # Use mask if policy freezing is enabled
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            _remove_pixels(data.observation),
            mask=running_statistics_mask,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd, it),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key, it), metrics

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey, it: int
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key, it),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, key: PRNGKey, it: int
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        step = jnp.ones((env_state.info['steps'].shape[0],)) * it
        result = training_epoch(training_state, env_state, key, step)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (
            training_state,
            env_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    # Initialize model params and training state.
    init_params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )

    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    normalizer_params = running_statistics.init_state(_remove_pixels(obs_shape))
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(
            init_params
        ),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=normalizer_params,
        env_steps=types.UInt64(hi=0, lo=0),
    )

    # Advanced checkpoint loading with policy freezing support
    decoder_params = None
    running_statistics_mask = None
    start_eval_it = 0  # Starting evaluation iteration
    current_step = types.UInt64(hi=0, lo=0)  # Current training step
    if checkpoint_path is not None and epath.Path(checkpoint_path).exists():
        logging.info("restoring from checkpoint %s", checkpoint_path)
        
        # Load checkpoint config and params
        config = ckpt_utils.load_config(checkpoint_path)
        params = ckpt_utils.load(checkpoint_path)
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)
        # Handle policy freezing if freeze_mask_fn is provided
        if freeze_mask_fn is not None:
            task_obs_size = getattr(config, 'task_obs_size', task_obs_size)
            decoder_params = params[1]["params"]["decoder"] if "decoder" in params[1]["params"] else None
            
            if continue_training:
                # Use the full loaded policy when continuing training
                init_params = init_params.replace(policy=params[1])
            else:
                # Only use decoder from checkpoint, keep new encoder
                if decoder_params is not None:
                    full_policy = init_params.policy
                    full_policy["params"]["decoder"] = decoder_params
                    init_params = init_params.replace(policy=full_policy)
            
            # Setup multi-transform optimizer for freezing
            optimizer = optax.multi_transform(
                {
                    "learned": optax.adam(learning_rate=learning_rate),
                    "frozen": optax.set_to_zero(),
                },
                freeze_mask_fn(init_params),
            )
            if max_grad_norm is not None:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(max_grad_norm),
                    optimizer,
                )
            
            print("Freezing layers with custom optimizer")
            
            # Handle observation normalization for different obs sizes using config metadata
            checkpoint_obs_size = getattr(config, 'total_obs_size', config.observation_size)
            checkpoint_task_obs_size = getattr(config, 'task_obs_size', 0)
            current_obs_size = jax.tree_util.tree_map(lambda x: x.shape[-1], obs_shape)
            if isinstance(current_obs_size, dict):
                current_obs_size = sum(current_obs_size.values())
                
            if task_obs_size > 0 and checkpoint_obs_size != current_obs_size:
                # Create mask for selective normalization updates
                running_statistics_mask = jnp.arange(current_obs_size) < task_obs_size
                
                # Initialize normalizer with current obs size
                normalizer_params = running_statistics.init_state(_remove_pixels(obs_shape))
                
                # Transfer relevant parts from checkpoint normalizer
                if task_obs_size < current_obs_size:
                    checkpoint_task_obs_size = checkpoint_obs_size - (current_obs_size - task_obs_size)
                    if checkpoint_task_obs_size > 0:
                        # Transfer the non-task observations (proprioceptive)
                        mean = normalizer_params.mean.at[task_obs_size:].set(
                            params[0].mean[checkpoint_task_obs_size:]
                        )
                        std = normalizer_params.std.at[task_obs_size:].set(
                            params[0].std[checkpoint_task_obs_size:]
                        )
                        summed_variance = normalizer_params.summed_variance.at[task_obs_size:].set(
                            params[0].summed_variance[checkpoint_task_obs_size:]
                        )
                        normalizer_params = running_statistics.RunningStatisticsState(
                            count=jnp.zeros(()), mean=mean, summed_variance=summed_variance, std=std
                        )
            else:
                normalizer_params = params[0]
        else:
            # Standard checkpoint loading without freezing
            value_params = params[2] if restore_value_fn else init_params.value
            policy_params = params[1]
            init_params = init_params.replace(policy=policy_params, value=value_params)
            normalizer_params = running_statistics.init_state(_remove_pixels(obs_shape))
            normalizer_params = normalizer_params.replace(
                count=params[0].count,
                mean=params[0].mean,
                summed_variance=params[0].summed_variance,
                std=params[0].std,
            )

        if continue_training:
            current_step = types._sanitize_uint64_input(np.array(int(checkpoint_path.stem)))
            steps_per_eval = env_step_per_training_step * max(num_resets_per_eval, 1) * num_training_steps_per_epoch
            start_eval_it = int(current_step.to_numpy() // steps_per_eval)
            training_state = training_state.replace(env_steps=current_step)
            logging.info(f"Resuming standard training from evaluation iteration {start_eval_it} (step {current_step.to_numpy()})")

            training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
                optimizer_state=optimizer.init(
                    init_params
                ),  # pytype: disable=wrong-arg-types  # numpy-scalars
                params=init_params,
                normalizer_params=normalizer_params,
                env_steps=current_step,
            )


    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ),
            {},
        )

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    eval_env = _maybe_wrap_env(
        eval_env or environment,
        wrap_env,
        num_eval_envs,
        episode_length,
        action_repeat,
        device_count=1,  # eval on the host only
        key_env=eval_key,
        wrap_env_fn=wrap_env_fn,
        randomization_fn=randomization_fn,
    )
    evaluator = acting.Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
    )

    training_metrics = {}
    training_walltime = 0

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1 and run_evals:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (
                    training_state.normalizer_params,
                    training_state.params.policy,
                    training_state.params.value,
                )
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(current_step, metrics)

    # Run initial policy_params_fn.
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    # policy_params_fn(start_eval_it, current_step, make_policy, params, policy_params_fn_key, decoder_params, eval_env)

    # Check if training is already complete
    if start_eval_it >= num_evals_after_init:
        logging.info(f"Training already complete. Checkpoint at iteration {start_eval_it}, target is {num_evals_after_init}")
        total_steps = int(_unpmap(training_state.env_steps))
        params = _unpmap(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )
        return (make_policy, params, {})
    
    for it in range(start_eval_it, num_evals_after_init):
        logging.info("starting iteration %s/%s %s", it, num_evals_after_init, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys, it
            )
            
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            # TODO(brax-team): move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state
            init_eval_step = it*jnp.ones_like(env_state.info['start_frame'])
            env_state.info['eval_step'] = init_eval_step
            
        if process_id != 0:
            continue

        # Process id == 0.
        params = _unpmap(
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            )
        )
        policy_params_fn_key, subkey = jax.random.split(policy_params_fn_key)
        policy_params_fn(it, current_step, make_policy, params, subkey, decoder_params, eval_env)
        # print('params:',params[1])
        
        if save_checkpoint_path is not None:
            # Save extended metadata for policy freezing compatibility
            current_obs_size = jax.tree_util.tree_map(lambda x: x.shape[-1], obs_shape)
            if isinstance(current_obs_size, dict):
                total_obs_size = sum(current_obs_size.values())
            else:
                total_obs_size = current_obs_size
                
            ckpt_config = ckpt_utils.network_config(
                observation_size=total_obs_size,
                action_size=env.action_size,
                normalize_observations=normalize_observations,
                network_factory=network_factory,
            )
            
            # Add metadata to config for easier loading
            # network_kwargs = getattr(network_factory, 'keywords', {})
            # ckpt_config.task_obs_size = int(task_obs_size or 0)
            # ckpt_config.total_obs_size = int(total_obs_size)
            # ckpt_config.policy_hidden_layer_sizes = network_kwargs.get('policy_hidden_layer_sizes', [])
            # ckpt_config.encoder_hidden_layer_sizes = network_kwargs.get('encoder_hidden_layer_sizes', [])
            # ckpt_config.decoder_hidden_layer_sizes = network_kwargs.get('decoder_hidden_layer_sizes', [])
            # ckpt_config.value_hidden_layer_sizes = network_kwargs.get('value_hidden_layer_sizes', [])
            # ckpt_config.intention_latent_size = network_kwargs.get('intention_latent_size', 8)
            # print(ckpt_config)
            ckpt_utils.save(save_checkpoint_path, current_step, params, ckpt_config)

        if num_evals > 0:
            metrics = training_metrics
            if run_evals:
                metrics = evaluator.run_evaluation(
                    params,
                    training_metrics,
                )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f"Total steps {total_steps} is less than `num_timesteps`="
            f" {num_timesteps}."
        )

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)

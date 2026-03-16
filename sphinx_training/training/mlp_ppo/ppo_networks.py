"""
Custom network definitions.
This is needed because we need to route the observations 
to proper places in the network in the case of the VAE (CoMic, Hasenclever 2020)
"""

import dataclasses
from typing import Any, Callable, Sequence, Tuple, Optional
import warnings

from brax.training import networks
from brax.training import types
from brax.training import distribution

from brax.training.types import PRNGKey

import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn

from . import custom_networks  # Relative import

@flax.struct.dataclass
class PPONetworks:
    policy_network: custom_networks.IntentionNetwork
    value_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(ppo_networks: PPONetworks):
    """Creates params and inference function for the PPO agent."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False
    ) -> types.Policy:
        policy_network = ppo_networks.policy_network
        parametric_action_distribution = ppo_networks.parametric_action_distribution

        def policy(
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            param_subset = (params[0], params[1])  # normalizer and policy params
            logits, extras = policy_network.apply(*param_subset, observations, key_network)

            if deterministic:
                return ppo_networks.parametric_action_distribution.mode(logits), extras

            # Sample action based on logits (mean and logvar)
            raw_actions = parametric_action_distribution.sample_no_postprocessing(
                logits, key_sample
            )

            log_prob = parametric_action_distribution.log_prob(logits, raw_actions)

            postprocessed_actions = parametric_action_distribution.postprocess(
                raw_actions
            )
            return postprocessed_actions, {
                "log_prob": log_prob,
                "raw_action": raw_actions,
                "logits": logits,
                "policy_rng": key_network,
            }

        return policy

    return make_policy


# intention policy
def make_intention_ppo_networks(
    observation_size: int,
    action_size: int,
    task_obs_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    continue_training: bool = False,
) -> PPONetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = custom_networks.make_intention_policy(
        param_size=parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        total_obs_size=observation_size,
        task_obs_size=task_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )


def make_encoderdecoder_ppo_networks(
    observation_size: int,
    task_obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    continue_training: bool = False,
) -> PPONetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = custom_networks.make_encoderdecoder_policy(
        parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        total_obs_size=observation_size,
        task_obs_size=task_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
        continue_training=continue_training,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )




# random_decoder policy
def make_randomdecoder_ppo_networks(
    observation_size: int,
    task_obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPONetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = custom_networks.make_random_intention_policy(
        parametric_action_distribution.param_size,
        latent_size=intention_latent_size,
        total_obs_size=observation_size,
        task_obs_size=task_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )

    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

def make_encoder_ppo_networks(
    observation_size: int,
    task_obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPONetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=intention_latent_size
    )
    policy_network = custom_networks.make_encoder_policy(
        parametric_action_distribution.param_size,
        task_obs_size=task_obs_size,
        preprocess_observations_fn=preprocess_observations_fn,
        encoder_hidden_layer_sizes=encoder_hidden_layer_sizes,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )
    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
    
def make_decoder_ppo_networks(
    observation_size: int,
    task_obs_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    intention_latent_size: int = 60,
    encoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    decoder_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
    value_hidden_layer_sizes: Sequence[int] = (1024,) * 2,
) -> PPONetworks:
    """Make Imitation PPO networks with preprocessor."""
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    policy_network = custom_networks.make_decoder_policy(
        parametric_action_distribution.param_size,
        decoder_obs_size=(observation_size-task_obs_size)+intention_latent_size,
        preprocess_observations_fn=preprocess_observations_fn,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )
    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
    
    
    
def make_decoder_policy_fn(ckpt_path, policy_params, action_size=None):
    from omegaconf import DictConfig, OmegaConf
    import masked_running_statistics as running_statistics
    from masked_running_statistics import RunningStatisticsState
    
    def make_decoder_policy(params, policy_network, parametric_action_distribution) -> types.Policy:
        def policy(
            observations: types.Observation,
            key_sample: PRNGKey,
        ) -> Tuple[types.Action, types.Extra]:
            key_sample, key_network = jax.random.split(key_sample)
            logits, extras = policy_network.apply(*params, observations, key_network)

            return parametric_action_distribution.mode(logits), extras

        return policy

    meta_data = OmegaConf.load((ckpt_path / "networks_metadata.yaml").as_posix())
    # meta_data = OmegaConf.load('/data/users/eabe/biomech_model/Flybody/RL_Flybody/debug/run_id=Testing/ckpt/networks_metadata.yaml')
    observation_size = meta_data['total_obs_size']
    task_obs_size = meta_data['task_obs_size']
    intention_latent_size = meta_data['intention_latent_size']
    decoder_hidden_layer_sizes = meta_data['decoder_hidden_layer_sizes']
    if action_size is None:
        action_size = meta_data['action_size']

    parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
    )
    policy_network = custom_networks.make_decoder_policy(
        parametric_action_distribution.param_size,
        decoder_obs_size=(observation_size-task_obs_size)+intention_latent_size,
        preprocess_observations_fn=running_statistics.normalize,
        decoder_hidden_layer_sizes=decoder_hidden_layer_sizes,
    )
    decoder_normalizer_params = RunningStatisticsState(
        count=jnp.zeros(()),
        mean=policy_params[0].mean[task_obs_size:],
        summed_variance=policy_params[0].summed_variance[task_obs_size:],
        std=policy_params[0].std[task_obs_size:],
    )
    decoder_params = (decoder_normalizer_params,{'params':{'decoder': policy_params[1]}})
    decoder_policy = make_decoder_policy(decoder_params, policy_network, parametric_action_distribution)
    return decoder_policy

# ==================== Simple MLP PPO Networks ====================

def make_simple_mlp_ppo_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (512, 512),
    value_hidden_layer_sizes: Sequence[int] = (512, 512),
    task_obs_size: Optional[int] = None,
) -> PPONetworks:
    """Make PPO networks with simple MLP policy (no VAE).
    
    Suitable for closed-loop neuromechanical control where the policy
    directly maps observations to control parameters.
    
    Args:
        observation_size: Size of observation input
        action_size: Size of action output
        preprocess_observations_fn: Function to preprocess observations
        policy_hidden_layer_sizes: Hidden layers for policy MLP
        value_hidden_layer_sizes: Hidden layers for value network
    
    Returns:
        PPONetworks with simple MLP policy and value networks
    """
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size
    )
    
    policy_network = custom_networks.make_simple_mlp_policy(
        param_size=parametric_action_distribution.param_size,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
    )
    
    value_network = networks.make_value_network(
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=value_hidden_layer_sizes,
    )
    
    return PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )

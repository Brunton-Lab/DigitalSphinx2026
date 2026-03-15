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

"""Checkpointing functions with support for custom networks.

This module extends Brax checkpointing to support custom network architectures
(IntentionNetwork, EncoderDecoder, SimpleMLP) while maintaining compatibility
with the standard Brax checkpoint format.
"""

import inspect
import json
import logging
from typing import Any, Dict, Tuple, Union

from brax.training import networks
from brax.training import types

from brax.training.acme import running_statistics
from brax.training.agents.bc import networks as bc_networks
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.sac import networks as sac_networks
from etils import epath
from flax.training import orbax_utils
import jax
from jax import numpy as jp
from ml_collections import config_dict
import numpy as np
from orbax import checkpoint as ocp

# Import custom network factories
try:
    from .mlp_ppo import ppo_networks as custom_ppo_networks
    _CUSTOM_NETWORKS_AVAILABLE = True
except ImportError:
    _CUSTOM_NETWORKS_AVAILABLE = False
    logging.warning('Custom networks not available, using standard Brax networks only')

# Keywords for kernel init functions that need registration
_KERNEL_INIT_FN_KEYWORDS = (
    'policy_network_kernel_init_fn',
    'value_network_kernel_init_fn',
    'q_network_kernel_init_fn',
)

# Registry for custom network factories (similar to networks.ACTIVATION)
NETWORK_FACTORY_REGISTRY = {
    # Standard Brax networks
    'make_ppo_networks': ppo_networks.make_ppo_networks,
    'make_bc_networks': bc_networks.make_bc_networks,
    'make_sac_networks': sac_networks.make_sac_networks,
}

if _CUSTOM_NETWORKS_AVAILABLE:
    NETWORK_FACTORY_REGISTRY.update({
        'make_intention_ppo_networks': custom_ppo_networks.make_intention_ppo_networks,
        'make_encoderdecoder_ppo_networks': custom_ppo_networks.make_encoderdecoder_ppo_networks,
        'make_randomdecoder_ppo_networks': custom_ppo_networks.make_randomdecoder_ppo_networks,
        'make_encoder_ppo_networks': custom_ppo_networks.make_encoder_ppo_networks,
        'make_decoder_ppo_networks': custom_ppo_networks.make_decoder_ppo_networks,
        'make_simple_mlp_ppo_networks': custom_ppo_networks.make_simple_mlp_ppo_networks,
    })


def _get_function_kwargs(func: Any) -> Dict[str, Any]:
  """Gets kwargs of a function."""
  return {
      p.name: p.default
      for p in inspect.signature(func).parameters.values()
      if p.default is not inspect.Parameter.empty
  }


def _get_function_defaults(func: Any) -> Dict[str, Any]:
  """Gets default kwargs of a function potentially wrapped in partials."""
  kwargs = _get_function_kwargs(func)
  if hasattr(func, 'func'):
    kwargs.update(_get_function_defaults(func.func))
  return kwargs


def _np_jp_to_python_types(data: Any) -> Any:
  if isinstance(data, (np.ndarray, jp.ndarray)):
    return data.item() if data.ndim == 0 else data.tolist()
  if isinstance(data, dict):
    return {key: _np_jp_to_python_types(value) for key, value in data.items()}
  if isinstance(data, (list, tuple)):
    return type(data)(_np_jp_to_python_types(item) for item in data)
  return data


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[
        Union[
            bc_networks.BCNetworks,
            ppo_networks.PPONetworks,
            sac_networks.SACNetworks,
        ]
    ],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint.
  
  Args:
    observation_size: Size of observations
    action_size: Size of actions
    normalize_observations: Whether observations should be normalized
    network_factory: Factory function for creating networks
    
  Returns:
    ConfigDict containing network configuration
    
  Raises:
    ValueError: If network_factory not registered or uses non-standard preprocessor
  """
  config = config_dict.ConfigDict()
  kwargs = _get_function_kwargs(network_factory)
  defaults = _get_function_defaults(network_factory)

  if 'preprocess_observations_fn' in kwargs:
    if (
        kwargs['preprocess_observations_fn']
        != defaults['preprocess_observations_fn']
    ):
      raise ValueError(
          'checkpointing only supports identity_observation_preprocessor as the'
          ' preprocess_observations_fn'
      )
    del kwargs['preprocess_observations_fn']

  # Store network factory name for later reconstruction
  # Handle functools.partial objects (e.g., from network_factory = functools.partial(...))
  import functools
  if isinstance(network_factory, functools.partial):
    factory_name = network_factory.func.__name__
  else:
    factory_name = network_factory.__name__
    
  if factory_name not in NETWORK_FACTORY_REGISTRY:
    logging.warning(
        f'Network factory {factory_name} not in registry. '
        'Checkpointing may fail on reload. Register it in NETWORK_FACTORY_REGISTRY.'
    )
  
  config.network_factory_name = factory_name
  config.network_factory_kwargs = kwargs
  config.normalize_observations = normalize_observations
  config.observation_size = observation_size
  config.action_size = action_size
  return config


def get_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[
        Union[
            bc_networks.BCNetworks,
            ppo_networks.PPONetworks,
            sac_networks.SACNetworks,
        ]
    ],
) -> Union[
    bc_networks.BCNetworks, ppo_networks.PPONetworks, sac_networks.SACNetworks
]:
  """Generates a network given config.
  
  Args:
    config: Network configuration
    network_factory: Factory function for creating networks
    
  Returns:
    Network instance (BCNetworks, PPONetworks, or SACNetworks)
  """
  normalize = lambda x, y: x
  if config.normalize_observations:
    normalize = running_statistics.normalize
  network = network_factory(
      config.to_dict()['observation_size'],
      config.action_size,
      preprocess_observations_fn=normalize,
      **config.network_factory_kwargs,
  )
  return network


def get_network_factory(
    config: config_dict.ConfigDict,
) -> types.NetworkFactory:
  """Retrieves network factory from registry based on config.
  
  Args:
    config: Network configuration containing 'network_factory_name'
    
  Returns:
    Network factory function
    
  Raises:
    KeyError: If network factory name not found in registry
    ValueError: If config missing network_factory_name
  """
  if 'network_factory_name' not in config:
    raise ValueError(
        'Config missing network_factory_name. '
        'This may be an old checkpoint format.'
    )
  
  factory_name = config.network_factory_name
  if factory_name not in NETWORK_FACTORY_REGISTRY:
    raise KeyError(
        f'Network factory {factory_name} not found in registry. '
        f'Available factories: {list(NETWORK_FACTORY_REGISTRY.keys())}. '
        'Make sure to import and register custom network factories.'
    )
  
  return NETWORK_FACTORY_REGISTRY[factory_name]


def load_network(
    checkpoint_path: Union[str, epath.Path],
    config_fname: str = 'ppo_network_config.json',
) -> Tuple[Any, Union[bc_networks.BCNetworks, ppo_networks.PPONetworks, sac_networks.SACNetworks]]:
  """Loads both checkpoint params and reconstructed network.
  
  Convenience function that combines load(), load_config(), and get_network().
  
  Args:
    checkpoint_path: Path to checkpoint directory
    config_fname: Name of config file (default: config.json)
    
  Returns:
    Tuple of (params, network)
    
  Example:
    ```python
    params, network = load_network('/path/to/checkpoint/000010000000')
    policy = make_inference_fn(network)(params)
    ```
  """
  checkpoint_path = epath.Path(checkpoint_path)
  
  # Load parameters
  params = load(checkpoint_path)
  
  # Load and parse config
  config_path = checkpoint_path / config_fname
  config = load_config(config_path)
  
  # Get network factory and create network
  network_factory = get_network_factory(config)
  network = get_network(config, network_factory)
  
  logging.info(
      f'Loaded network {config.network_factory_name} with '
      f'observation_size={config.observation_size}, '
      f'action_size={config.action_size}'
  )
  
  return params, network


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Tuple[Any, ...],
    config: config_dict.ConfigDict,
    config_fname: str = 'config.json',
):
  """Saves a checkpoint.
  
  Args:
    path: Directory path to save checkpoint
    step: Training step number (used in directory name)
    params: Network parameters to save
    config: Network configuration
    config_fname: Name of config file (default: config.json)
  """
  ckpt_path = epath.Path(path) / f'{step:012d}'
  logging.info('saving checkpoint to %s', ckpt_path.as_posix())

  if not ckpt_path.exists():
    ckpt_path.mkdir(parents=True)

  # Save the network params using Orbax
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  save_args = orbax_utils.save_args_from_target(params)
  orbax_checkpointer.save(ckpt_path, params, force=True, save_args=save_args)

  config_cp_dict = config.to_dict()
  
  # Convert activation functions to registered names
  if 'activation' in config_cp_dict['network_factory_kwargs'] and callable(
      config_cp_dict['network_factory_kwargs']['activation']
  ):
    name_ = config_cp_dict['network_factory_kwargs']['activation'].__name__
    if name_ not in networks.ACTIVATION:
      raise ValueError(
          f'Activation function {name_} not registered for checkpointing. '
          f'Available activations: {list(networks.ACTIVATION.keys())}'
      )
    config_cp_dict['network_factory_kwargs']['activation'] = name_
    
  # Convert kernel init functions to registered names
  for init_fn_name in _KERNEL_INIT_FN_KEYWORDS:
    if init_fn_name not in config_cp_dict['network_factory_kwargs']:
      continue
    name_ = config_cp_dict['network_factory_kwargs'][init_fn_name].__name__
    if name_ not in networks.KERNEL_INITIALIZER:
      raise ValueError(
          f'Kernel init function {name_} not registered for checkpointing. '
          f'Available initializers: {list(networks.KERNEL_INITIALIZER.keys())}'
      )
    config_cp_dict['network_factory_kwargs'][init_fn_name] = name_
    
  # Convert numpy/jax arrays to python types for JSON serialization
  config_cp_dict = _np_jp_to_python_types(config_cp_dict)
  config = config_dict.ConfigDict(config_cp_dict)

  # Save the config as JSON
  config_path = ckpt_path / config_fname
  config_path.write_text(config.to_json_best_effort())
  logging.info('saved config to %s', config_path.as_posix())


def load(
    path: Union[str, epath.Path],
):
  """Loads checkpoint.
  
  Args:
    path: Path to checkpoint directory
    
  Returns:
    Tuple of (normalizer_params, policy_params, value_params)
    
  Raises:
    ValueError: If checkpoint path doesn't exist
  """
  path = epath.Path(path)
  if not path.exists():
    raise ValueError(f'checkpoint path does not exist: {path.as_posix()}')

  logging.info('restoring from checkpoint %s', path.as_posix())

  # Get checkpoint metadata
  orbax_checkpointer = ocp.PyTreeCheckpointer()
  try:
    metadata = orbax_checkpointer.metadata(path)
  except Exception as e:
    logging.error(f'Failed to read checkpoint metadata: {e}')
    raise

  # Prepare restore arguments
  restore_args = jax.tree.map(
      lambda _: ocp.RestoreArgs(restore_type=np.ndarray), 
      metadata.item_metadata
  )
  
  # Restore checkpoint
  target = orbax_checkpointer.restore(
      path, ocp.args.PyTreeRestore(restore_args=restore_args), item=None
  )

  # Reconstruct RunningStatisticsState from first element
  state_dict = target[0]
  if isinstance(state_dict, dict):
    # Reconstruct UInt64 count if it was saved as dict
    if isinstance(state_dict.get('count'), dict) and 'hi' in state_dict['count']:
      state_dict['count'] = types.UInt64(**state_dict['count'])
    target[0] = running_statistics.RunningStatisticsState(**state_dict)
  
  # Log some statistics about loaded data to help debug NaN issues
  logging.info('Checkpoint loaded successfully')
  nan_count = 0
  total_count = 0
  
  def count_nans(x):
    nonlocal nan_count, total_count
    if isinstance(x, (np.ndarray, jp.ndarray)):
      total_count += x.size
      nan_count += np.isnan(x).sum()
  
  jax.tree.map(count_nans, target)
  if nan_count > 0:
    logging.warning(
        f'Found {nan_count}/{total_count} NaN values in loaded checkpoint!'
    )
  else:
    logging.info(f'No NaN values detected in checkpoint ({total_count} parameters)')

  return target


def load_config(
    config_path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
  """Loads config from config path.
  
  Args:
    config_path: Path to config JSON file
    
  Returns:
    ConfigDict with deserialized network configuration
    
  Raises:
    ValueError: If config file not found
    KeyError: If activation/initializer name not registered
  """
  config_path = epath.Path(config_path)
  if not config_path.exists():
    raise ValueError(f'Config file not found at {config_path.as_posix()}')

  logging.info('loading config from %s', config_path.as_posix())
  loaded_dict = json.loads(config_path.read_text())

  # Convert activation name back to function
  if 'activation' in loaded_dict['network_factory_kwargs']:
    activation_name = loaded_dict['network_factory_kwargs']['activation']
    if activation_name not in networks.ACTIVATION:
      raise KeyError(
          f'Activation function {activation_name} not found in registry. '
          f'Available: {list(networks.ACTIVATION.keys())}'
      )
    loaded_dict['network_factory_kwargs']['activation'] = networks.ACTIVATION[
        activation_name
    ]
    
  # Convert kernel init function names back to functions
  for init_fn_name in _KERNEL_INIT_FN_KEYWORDS:
    if init_fn_name not in loaded_dict['network_factory_kwargs']:
      continue
    init_fn_name_ = loaded_dict['network_factory_kwargs'][init_fn_name]
    if init_fn_name_ not in networks.KERNEL_INITIALIZER:
      raise KeyError(
          f'Kernel initializer {init_fn_name_} not found in registry. '
          f'Available: {list(networks.KERNEL_INITIALIZER.keys())}'
      )
    loaded_dict['network_factory_kwargs'][init_fn_name] = (
        networks.KERNEL_INITIALIZER[init_fn_name_]
    )

  return config_dict.create(**loaded_dict)
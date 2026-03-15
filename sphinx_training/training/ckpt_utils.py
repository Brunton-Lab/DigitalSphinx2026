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

"""Checkpointing for PPO."""

from typing import Any, Union

from . import checkpointing  # Relative import to avoid ambiguity with brax.training
from brax.training import types
from .mlp_ppo import ppo_networks  # Relative import for local package
from etils import epath
from ml_collections import config_dict

_CONFIG_FNAME = 'ppo_network_config.json'


def save(
    path: Union[str, epath.Path],
    step: int,
    params: Any,
    config: config_dict.ConfigDict,
):
  """Saves a checkpoint."""
  return checkpointing.save(path, step, params, config, _CONFIG_FNAME)


def load(
    path: Union[str, epath.Path],
):
  """Loads checkpoint."""
  return checkpointing.load(path)


def network_config(
    observation_size: types.ObservationSize,
    action_size: int,
    normalize_observations: bool,
    network_factory: types.NetworkFactory[ppo_networks.PPONetworks],
) -> config_dict.ConfigDict:
  """Returns a config dict for re-creating a network from a checkpoint."""
  return checkpointing.network_config(
      observation_size, action_size, normalize_observations, network_factory
  )


def _get_ppo_network(
    config: config_dict.ConfigDict,
    network_factory: types.NetworkFactory[ppo_networks.PPONetworks],
) -> ppo_networks.PPONetworks:
  """Generates a PPO network given config."""
  return checkpointing.get_network(config, network_factory)  # pytype: disable=bad-return-type


def load_config(
    path: Union[str, epath.Path],
) -> config_dict.ConfigDict:
  """Loads PPO config from checkpoint."""
  path = epath.Path(path)
  config_path = path / _CONFIG_FNAME
  return checkpointing.load_config(config_path)


def load_policy(
    path: Union[str, epath.Path],
    deterministic: bool = True,
):
  """Loads policy inference function from PPO checkpoint."""
  path = epath.Path(path)
  config = load_config(path)
  network_factory = checkpointing.get_network_factory(config)
  params = load(path)
  ppo_network = _get_ppo_network(config, network_factory)
  make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

  return make_inference_fn(params, deterministic=deterministic), params

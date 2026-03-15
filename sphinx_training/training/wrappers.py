# Copyright 2025 DeepMind Technologies Limited
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
# ==============================================================================
"""Wrappers for MuJoCo Playground environments."""

import functools
from typing import Any, Callable, List, Optional, Sequence, Tuple

from brax.envs.wrappers import training as brax_training
import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.wrapper import Wrapper

class FlattenObsWrapper(Wrapper):

    def __init__(self, env: mjx_env.MjxEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array, clip_idx: Optional[int] = None) -> mjx_env.State:
        state = self.env.reset(rng, clip_idx)
        return self._flatten(state)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state = self.env.step(state, action)
        return self._flatten(state)

    def _flatten(self, state: mjx_env.State) -> mjx_env.State:
        state = state.replace(
            obs = jnp.nan_to_num(jax.flatten_util.ravel_pytree(state.obs)[0]),
            metrics = self._flatten_metrics(state.metrics),
        )
        return state
    
    def _flatten_metrics(self, metrics: dict) -> dict:
        new_metrics = {}
        def rec(d: dict, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    rec(v, prefix + k + '/')
                else:
                    new_metrics[prefix + k] = v
        rec(metrics)
        return new_metrics
    
    @property
    def observation_size(self) -> int:
        rng_shape = jax.eval_shape(jax.random.key, 0)
        #flat_obs = lambda rng: self._flatten(self.env.reset(rng).obs)
        obs_size = len(jax.eval_shape(self.reset, rng_shape))
        return obs_size


class HighLevelWrapper(Wrapper):
    """Loads a 'decoder' and calls it to get the ctrl used in the actual step"""

    def __init__(self, env, decoder_inference_fn, policy_key, fast_eval_mode=False):
        # Pre-JIT the decoder inference function to avoid nested compilation
        self._decoder_inference_fn = jax.jit(decoder_inference_fn)
        super().__init__(env)

    def step(self, state: mjx_env.State, latents: jax.Array) -> mjx_env.State:
        rng, action_rng, cmd_rng = jax.random.split(state.info["rng"], 3)
        
        obs = state.obs
        # Full mode with complete observation
        decoder_input = jnp.concatenate(
            [latents[..., : latents.shape[-1]], obs[..., self.task_obs_size :]],
            axis=-1,
        )
        
        # Use pre-compiled decoder inference
        action, _ = self._decoder_inference_fn(decoder_input, action_rng)
        return self.env.step(state, action)


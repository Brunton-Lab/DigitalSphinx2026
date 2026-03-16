# Goal: Intake observations and generate hidden states for the actor. The weights are static. 
import os
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from mujoco_playground._src import mjx_env
from ml_collections import config_dict
from typing import Optional
# from training import masked_running_statistics


class RMSNorm:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, x):
        # x: [..., dim]
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms


class Obs_Adapter:
    """
    Adapter for extracting target and proprioceptive observations from the environment state.
    Assumes the observation is a concatenation of target and proprioception, strictly in this order.
    """
    def __init__(self, normalize_sensory: bool = True):
        self.normalize_sensory = normalize_sensory
        self.task_obs_size = None
        self.sensory_size = None
        self.sensory_norm = None

    def set_sizes(self, info) -> None:
        '''
        Set up sizes of different observation components.
        '''
        # info is from env.reset().info
        # info['obs_sizes'] = {'task_obs_size': imitation_target.shape[-1],
        #                       'proprioception': proprioception.shape[-1]}
        self.task_obs_size = info['obs_sizes']['task_obs_size']
        self.sensory_size = info['obs_sizes']['proprioception']
        if self.normalize_sensory:
            self.sensory_norm = RMSNorm(self.sensory_size)
        
    def target(self, obs):
        # obs = jnp.concatenate([imitation_target, proprioception])
        return obs[..., :self.task_obs_size]

    def sensory(self, obs):
        # keep explicit proprio_size to avoid silently depending on trailing dims
        x = obs[..., self.task_obs_size : self.task_obs_size+self.sensory_size]
        if self.normalize_sensory:
            x = self.sensory_norm(x)
        return x

class LatentStateWrapper:
    def __init__(self, cfg, rng, env, 
                 adapter: Obs_Adapter):
        '''
        Initialize the Latent State Wrapper.
        Wrapper handles:
        - RNN initialization (random or from file)
            - If from file, normalize recurrent weights
            - Set up motor and sensory neuron indices s.t. only those neurons are used for input/output
        - RNN hidden state maintenance and updates
        - Take sensory inputs from the environment observations and feed into RNN
        - Construct actor observations by concatenating target + RNN hidden state
        # Does render_ghost use sensory obs? If so, that info needs to be passed along separately. Tried render_ghost and it was fine
        '''
        # 1. store env and adapter for observation processing
        self.env = env
        self.adapter = adapter
        # probe once to set adapter sizes + infer proprio dim 
        rng, subkey = jax.random.split(rng)
        probe_state = self.env.reset(subkey)

        self.adapter.set_sizes(probe_state.info)
        sensory_dim = self.adapter.sensory(probe_state.obs).shape[-1]

        # 2. VRNN init 
        config = config_dict.ConfigDict(cfg.connectome)
        if config.rnn_weights == 'random':
            self.hidden_size = config.hidden_size 
            self.cell = nn.recurrent.SimpleCell(features=self.hidden_size) # VRNN only for now
            rng, init_key = jax.random.split(rng)
            h0 = jnp.zeros((self.hidden_size,))
            x0 = jnp.zeros((sensory_dim,))
            self.rnn_params = self.cell.init(init_key, h0, x0)["params"] # static weights
         # check if the string is a file path and the file exists
        elif isinstance(config.rnn_weights, str) and config.rnn_weights.endswith('.npy') and os.path.isfile(config.rnn_weights):
            '''
            1. load RNN weights from npy file
            2. normalize recurrent weights
            3. init RNN cell
            '''
            # 1. load params
            w_hh = np.load(config.rnn_weights, allow_pickle=True)
            assert w_hh.shape[0] == w_hh.shape[1], f"RNN weight matrix must be square. check {config.rnn_weights}"
            assert config.rec_nonlinearity in ['tanh','sigmoid','relu'], "Currently only tanh and relu are supported non-linearities."
            assert config.sensory_nonlinearity in ['tanh','sigmoid','relu'], "Currently only tanh and relu are supported non-linearities."
            
            # 2. weights normalization
            rho = np.max(np.abs(np.linalg.eigvals(w_hh)))
            target_rho = config.norm_eigen_target
            w_hh = (target_rho / (rho + 1e-8)) * w_hh
            self.hidden_size = w_hh.shape[0]

            # 3. create rnn as two parts: cell + params
            activation_fn = None
            if config.rec_nonlinearity == 'tanh':
                activation_fn = nn.tanh
            elif config.rec_nonlinearity == 'relu':
                activation_fn = nn.relu
            elif config.rec_nonlinearity == 'sigmoid':
                activation_fn = nn.sigmoid
            self.cell = nn.recurrent.SimpleCell(
                features=self.hidden_size,
                activation_fn=activation_fn,        # or nn.relu, nn.softplus, etc.
            )
            
            # # 4. set up input output neuron indices
            # assert hasattr(config, 'motor_neuron_indices') and hasattr(config, 'sensory_neuron_indices'), \
            #     "When loading RNN weights from file, motor and sensory neuron index files must be provided in config."
            motor_idx = np.loadtxt(config.motor_neuron_indices, dtype=int)
            sensory_idx = np.loadtxt(config.sensory_neuron_indices, dtype=int)  
            self.motor_neuron_idx = jnp.asarray(motor_idx)
            self.sensory_neuron_idx = jnp.asarray(sensory_idx)

            # Sensory input layer: only sensory neurons have non-zero input weights
            W_in_sensory = jax.random.normal(
                rng, (sensory_dim, self.sensory_neuron_idx.shape[0])
            )
            W_in_full = jnp.zeros((sensory_dim, self.hidden_size))
            b_in_full = jnp.zeros((self.hidden_size,))
            W_in_full = W_in_full.at[:, self.sensory_neuron_idx].set(W_in_sensory) # non-zero weights only for sensory neurons

            self.rnn_params = {
                "i": {  # input layer randomly initialized
                    "kernel": W_in_full,  # shape (input, hidden)   # only sensory neurons have non-zero weights
                    "bias": b_in_full,    # shape (hidden,)         # zero bias
                },
                "h": {  # recurrence
                    "kernel": jnp.asarray(w_hh),   # shape (hidden, hidden)
                    "bias": jnp.zeros((self.hidden_size,)),
                },
            }
            
            # Vestigial code for testing RNN forward pass and all units being used for input/output
            
            # self.rnn_params = {
            #     "i": {  # input layer randomly initialized
            #         "kernel": jax.random.normal(rng, (sensory_dim, self.hidden_size)),
            #         "bias": jnp.zeros((self.hidden_size,)),
            #     },
            #     "h": {  # recurrence
            #         "kernel": jnp.asarray(w_hh),   # shape (hidden, hidden)
            #         "bias": jnp.zeros((self.hidden_size,)),
            #     },
            # }
            
            # # random initial hidden state
            # key, subkey = jax.random.split(rng)
            # h = jax.random.normal(subkey, (self.hidden_size,))

            # # random inputs
            # key, subkey = jax.random.split(key)
            # xs = jax.random.normal(subkey, (5, sensory_dim))  # 5 timesteps

            # # run a few steps
            # for t in range(xs.shape[0]):
            #     h, y = self.cell.apply({"params": self.rnn_params}, h, xs[t])
            #     print(f"Step {t}, h mean: {jnp.mean(h):.4f}, y mean: {jnp.mean(y):.4f}")

        else:
            raise NotImplementedError("RNN weights must be 'random' or a valid path to .npy file. Now config.rnn_weights: {}".format(config.rnn_weights))
        
        # Initialize sensory nonlinearity to prevent large inputs
        if config.sensory_nonlinearity == 'tanh':
            sensory_nl = jnp.tanh
        elif config.sensory_nonlinearity == 'relu':
            sensory_nl = jax.nn.relu
        elif config.sensory_nonlinearity == 'sigmoid':
            sensory_nl = jax.nn.sigmoid

        self.sensory_nl = sensory_nl
        
    def _init_h(self, batch_shape=()):
        # Initialize an array of zeros for the RNN hidden state.
        return jnp.zeros(batch_shape + (self.hidden_size,))

    def reset(self, rng: jax.Array, clip_idx: Optional[int] = None) -> mjx_env.State:
        """
        Arguments:
            rng: JAX random number generator key.
            clip_idx: Optional index for selecting specific reference clip.
        1. reset physics env
        2. reset RNN hidden state by taking one step with initial proprioception
        3. return env state with updated obs and h_state
        """
        # 1. reset physics env
        state = self.env.reset(rng, clip_idx=clip_idx) # reset physics env 
        self.adapter.set_sizes(state.info) # set sizes for extracting target and proprio from obs
        # 2. reset RNN hidden state
        target = self.adapter.target(state.obs)
        h = self._init_h(batch_shape=target.shape[:-1])
        # 2.1 process initial proprioception through RNN to get initial h
        sensory_x_raw = self.adapter.sensory(state.obs)
        sensory_x = self.sensory_nl(sensory_x_raw)
        h, _ = self.cell.apply({"params": self.rnn_params}, h, sensory_x)
        # 2.2 build actor obs: concat(target, h)
        motor_h = h[self.motor_neuron_idx]
        actor_obs = jnp.concatenate([target, motor_h], axis=-1)
        # 3. return env state with updated obs and h_state
        info = dict(state.info)
        info["h"] = h
        # Update obs_sizes to reflect the wrapper's observation shape
        info["obs_sizes"] = {
            "task_obs_size": self.adapter.task_obs_size,
            "proprioception": motor_h.shape[-1],  # motor neuron output replaces raw proprio
        }
        return state.replace(obs=actor_obs, info=info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        """
        # 1. resolve action to create next env state (env.step(action) -> obs)
        # 2. update RNN hidden state with new sensory observations (h_t = RNN(h_{t-1}, proprio_t))
        # 3. return next state for the agent
        """
        # 1. Step mujoco env for new sensory inputs
        next_state = self.env.step(state, action) 
        
        # 2. Update RNN hidden state
        h_prev = state.info["h"]
        # 2.a Extract proprioception
        sensory_x_raw = self.adapter.sensory(next_state.obs)
        # 2.a.1 sensory nonlinearity
        sensory_x = self.sensory_nl(sensory_x_raw)
        # 2.b RNN forward
        h_new, _ = self.cell.apply(
            {"params": self.rnn_params},
            h_prev,
            sensory_x,
        )
        
        # 3. return next state for the agent
        # 3.1 build obs for actor: concat(target, h)
        target = self.adapter.target(next_state.obs)
        motor_h = h_new[self.motor_neuron_idx]
        actor_obs = jnp.concatenate([target, motor_h], axis=-1)
        # 3.2 replace obs and h_state in next_state
        info = dict(next_state.info)
        info["h"] = h_new
        
        return next_state.replace(obs=actor_obs, info=info)
    
    def __getattr__(self, name: str):
        # Attribute forwarding - find attribute in wrapped env if not found in wrapper itself.
        env = object.__getattribute__(self, "env")
        return getattr(env, name)


# obs_shape = jax.tree_util.tree_map(
#     lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
# )
# normalizer_params = running_statistics.init_state(_remove_pixels(obs_shape))

# normalizer_params = running_statistics.update(
#     training_state.normalizer_params,
#     _remove_pixels(data.observation),
#     mask=running_statistics_mask,
#     pmap_axis_name=_PMAP_AXIS_NAME,
# )


# class VanillaRNNEnv:
#     """
#     Minimal JAX env:
#       - input  : x_t
#       - output : hidden state h_t
#       - state  : mjx_env.State
#       - RNN    : vanilla (tanh) via flax.linen.SimpleCell
#     """

#     def __init__(
#         self,
#         cfg,
#         rng: jax.Array = jax.random.PRNGKey(0),
#     ):
#         config = config_dict.ConfigDict(cfg.connectome)
#         self.input_size = config.input_size # all observation dimensions
#         self.hidden_size = config.hidden_size # size of RNN hidden state
#         self._cell = nn.recurrent.SimpleCell(features=self.hidden_size)
#         # SimpleCell implements:
#         #   \begin{array}{ll}
#         #   h' = \tanh(W_i x + b_i + W_h h)
#         #   \end{array}

#         if config.rnn_type == 'random':
#             h0 = jnp.zeros((self.hidden_size,))
#             x0 = jnp.zeros((self.input_size,))
#             variables = self._cell.init(rng, h0, x0)
#             self.params = variables["params"]
#         else:
#             self.params = config.rnn_type  # placeholder for loaded params

#     def reset(self) -> mjx_env.State:
#         h0 = jnp.zeros((self.hidden_size,))
#         obs = jnp.zeros((self.input_size,))
#         h_state = {
#             "h": h0,
#             "obs": obs,
#         }

#         return mjx_env.State(h_state)

#     def step(self, state: mjx_env.State) -> mjx_env.State:

#         h_prev = state.h_state["h"]
#         h_new, _ = self._cell.apply({"params": self.params}, h_prev, state.h_state["obs"])

#         h_state = {
#             "h": h_new,
#         }
        
#         state = state.replace(
#             h_state=h_state,
#         )

#         return state

"""
HDF5-based rollout saving and loading utilities.

This module provides efficient storage and retrieval of rollout trajectories
from MJX environments, with support for lazy loading and multiple rollouts.

Features:
- Direct saving from jax.lax.scan outputs (automatic transposition)
- Efficient HDF5 chunking and compression
- JIT-compatible loading via jax.pure_callback
- Support for standard fields (qpos, qvel) and custom nested data (info, metrics, extras)
"""

from dataclasses import dataclass
from typing import Optional, Union, List, Any, Dict
from pathlib import Path
import h5py
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Rollout:
    """
    Container for rollout data from MJX environment states.

    Attributes:
        qpos: Joint positions, shape (n_timesteps, n_qpos)
        qvel: Joint velocities, shape (n_timesteps, n_qvel)
        ctrl: Control inputs, shape (n_timesteps, n_ctrl)
        time: Simulation time for each step, shape (n_timesteps,)
        obs: Observations, shape (n_timesteps, n_obs)
        qfrc_fluid: Fluid forces, shape (n_timesteps, n_qfrc)
        info: Optional dictionary of info data
        metrics: Optional dictionary of metrics data
        extras: Optional dictionary for additional variables (e.g., activations, policy outputs)
    """
    qpos: jnp.ndarray
    qvel: jnp.ndarray
    ctrl: Optional[jnp.ndarray] = None
    time: Optional[jnp.ndarray] = None
    obs: Optional[jnp.ndarray] = None
    qfrc_fluid: Optional[jnp.ndarray] = None
    sensordata: Optional[jnp.ndarray] = None
    info: Optional[Dict[str, jnp.ndarray]] = struct.field(pytree_node=False, default=None)
    metrics: Optional[Dict[str, jnp.ndarray]] = struct.field(pytree_node=False, default=None)
    extras: Optional[Dict[str, jnp.ndarray]] = struct.field(pytree_node=False, default=None)

    @property
    def n_timesteps(self) -> int:
        """Number of timesteps in the rollout."""
        return self.qpos.shape[0]

    @classmethod
    def from_states(
        cls,
        states: List[Any],
        include_info_keys: Optional[List[str]] = None,
        include_metrics_keys: Optional[List[str]] = None,
        include_obs: bool = True,
        include_qfrc_fluid: bool = True,
        include_sensordata: bool = True
    ) -> Union['Rollout', List['Rollout']]:
        """
        Create Rollout(s) from a list of mjx_env.State objects.

        Args:
            states: List of mjx_env.State objects where each state has shape:
                   - (n_envs, ...) if batched (parallel environments)
                   - (...) if single environment
            include_info_keys: Optional list of keys from state.info to include.
                              If None, attempts to save all info keys.
            include_metrics_keys: Optional list of keys from state.metrics to include.
                                 If None, attempts to save all metrics keys.
            include_obs: Whether to include observations
            include_qfrc_fluid: Whether to include fluid forces
            include_sensordata: Whether to include sensor data
        Returns:
            - If states are batched: List of Rollout objects, one per environment
            - If states are single: Single Rollout object
        """
        # Get the CPU device
        cpu_device = jax.devices('cpu')[0]

        # Use the context manager to create arrays on CPU
        with jax.default_device(cpu_device):
            # Check if states are batched by looking at first state's qpos shape
            first_qpos = states[0].data.qpos
            is_batched = len(first_qpos.shape) > 1  # Has batch dimension

            if is_batched:
                n_envs = first_qpos.shape[0]
                # Stack along time axis: (n_timesteps, n_envs, ...)
                qpos = np.stack([state.data.qpos for state in states], axis=0)
                qvel = np.stack([state.data.qvel for state in states], axis=0)
                ctrl = np.stack([state.data.ctrl for state in states], axis=0)
                time = np.stack([state.data.time for state in states], axis=0)

                # Transpose to (n_envs, n_timesteps, ...)
                qpos = np.transpose(qpos, (1, 0) + tuple(range(2, len(qpos.shape))))
                qvel = np.transpose(qvel, (1, 0) + tuple(range(2, len(qvel.shape))))
                ctrl = np.transpose(ctrl, (1, 0) + tuple(range(2, len(ctrl.shape))))
                time = np.transpose(time, (1, 0))
            else:
                n_envs = 1
                # Single environment: stack along time axis
                qpos = np.stack([state.data.qpos for state in states], axis=0)[np.newaxis, ...]
                qvel = np.stack([state.data.qvel for state in states], axis=0)[np.newaxis, ...]
                ctrl = np.stack([state.data.ctrl for state in states], axis=0)[np.newaxis, ...]
                time = np.array([state.data.time for state in states])[np.newaxis, ...]

            # Extract observations
            obs = None
            if include_obs:
                try:
                    obs_stacked = np.stack([state.obs for state in states], axis=0)
                    if is_batched:
                        obs = np.transpose(obs_stacked, (1, 0) + tuple(range(2, len(obs_stacked.shape))))
                    else:
                        obs = obs_stacked[np.newaxis, ...]
                except (AttributeError, ValueError) as e:
                    print(f"Warning: Could not extract observations: {e}")

            # Extract fluid forces
            qfrc_fluid = None
            if include_qfrc_fluid:
                try:
                    qfrc_stacked = np.stack([state.data.qfrc_fluid for state in states], axis=0)
                    if is_batched:
                        qfrc_fluid = np.transpose(qfrc_stacked, (1, 0) + tuple(range(2, len(qfrc_stacked.shape))))
                    else:
                        qfrc_fluid = qfrc_stacked[np.newaxis, ...]
                except (AttributeError, ValueError) as e:
                    print(f"Warning: Could not extract qfrc_fluid: {e}")

            # Extract sensor data
            sensordata = None
            if include_sensordata:
                try:
                    sensor_stacked = np.stack([state.data.sensordata for state in states], axis=0)
                    if is_batched:
                        sensordata = np.transpose(sensor_stacked, (1, 0) + tuple(range(2, len(sensor_stacked.shape))))
                    else:
                        sensordata = sensor_stacked[np.newaxis, ...]
                except (AttributeError, ValueError) as e:
                    print(f"Warning: Could not extract sensordata: {e}")

            # Extract info
            info = {}
            if include_info_keys is None:
                # Default to only essential info keys
                include_info_keys = ['aero_forces', 'action', 'reference_clip', 'start_frame', 'truncated']

            for key in include_info_keys:
                try:
                    # Special handling for aero_forces which is a list of 4 arrays
                    if key == 'aero_forces':
                        # Extract the list from first state to check structure
                        first_aero = states[0].info.get('aero_forces', None)
                        if first_aero is not None and isinstance(first_aero, (list, tuple)):
                            # Stack the list of forces across time
                            # Input: list of (n_timesteps,) where each element is list of 4 arrays with shape (n_envs, ...)
                            # Stack all timesteps: shape (n_timesteps, 4, n_envs, ...)
                            aero_stacked = np.stack([state.info['aero_forces'] for state in states], axis=0)

                            if is_batched:
                                # Transpose to (n_envs, n_timesteps, n_forces=4, ...)
                                info[key] = np.transpose(aero_stacked, (2, 0, 1) + tuple(range(3, len(aero_stacked.shape))))
                            else:
                                # Add env dimension: (1, n_timesteps, n_forces=4, ...)
                                info[key] = aero_stacked[np.newaxis, ...]
                        continue

                    info_stacked = np.stack([state.info[key] for state in states], axis=0)
                    if is_batched:
                        # Check if this info field has batch dimension
                        if len(info_stacked.shape) > 1 and info_stacked.shape[1] == n_envs:
                            info[key] = np.transpose(info_stacked, (1, 0) + tuple(range(2, len(info_stacked.shape))))
                        else:
                            # Scalar or non-batched field - replicate for each env
                            info[key] = np.tile(info_stacked[np.newaxis, :], (n_envs, 1) + (1,) * (len(info_stacked.shape) - 1))
                    else:
                        info[key] = info_stacked[np.newaxis, ...]
                except (KeyError, ValueError, AttributeError) as e:
                    print(f"Warning: Could not extract info key '{key}': {e}")

            # Extract metrics
            metrics = {}
            if include_metrics_keys is None:
                # Try to get all metrics keys from first state
                try:
                    include_metrics_keys = list(states[0].metrics.keys())
                except (AttributeError, KeyError):
                    include_metrics_keys = []

            for key in include_metrics_keys:
                try:
                    metrics_stacked = np.stack([state.metrics[key] for state in states], axis=0)
                    if is_batched:
                        # Check if this metrics field has batch dimension
                        if len(metrics_stacked.shape) > 1 and metrics_stacked.shape[1] == n_envs:
                            metrics[key] = np.transpose(metrics_stacked, (1, 0) + tuple(range(2, len(metrics_stacked.shape))))
                        else:
                            # Scalar or non-batched field - replicate for each env
                            metrics[key] = np.tile(metrics_stacked[np.newaxis, :], (n_envs, 1) + (1,) * (len(metrics_stacked.shape) - 1))
                    else:
                        metrics[key] = metrics_stacked[np.newaxis, ...]
                except (KeyError, ValueError, AttributeError) as e:
                    print(f"Warning: Could not extract metrics key '{key}': {e}")

            # Create rollout objects per environment
            rollouts = []
            for env_idx in range(n_envs):
                rollout = cls(
                    qpos=np.array(qpos[env_idx]),
                    qvel=np.array(qvel[env_idx]),
                    ctrl=np.array(ctrl[env_idx]),
                    time=np.array(time[env_idx]),
                    obs=np.array(obs[env_idx]) if obs is not None else None,
                    qfrc_fluid=np.array(qfrc_fluid[env_idx]) if qfrc_fluid is not None else None,
                    sensordata=np.array(sensordata[env_idx]) if sensordata is not None else None,
                    info={k: np.array(v[env_idx]) for k, v in info.items()} if info else None,
                    metrics={k: np.array(v[env_idx]) for k, v in metrics.items()} if metrics else None,
                    extras=None  # Not extracted from states, must be added separately
                )
                rollouts.append(rollout)

        # Return single rollout if not batched, otherwise return list
        return rollouts if is_batched else rollouts[0]


class HDF5RolloutSaver:
    """
    Efficient HDF5-based storage for multiple rollouts with lazy loading support.

    This class allows saving multiple rollout trajectories to a single HDF5 file
    and provides efficient loading mechanisms similar to HDF5ReferenceClips.
    """

    def __init__(self, file_path: Union[str, Path], enable_jax: bool = True):
        """
        Initialize HDF5RolloutSaver.

        Args:
            file_path: Path to the HDF5 file
            enable_jax: Whether to convert arrays to JAX arrays when loading
        """
        self.file_path = Path(file_path) if isinstance(file_path, str) else file_path
        self.enable_jax = enable_jax

        if self.file_path.exists():
            # Load metadata
            with h5py.File(self.file_path, 'r') as f:
                self.n_rollouts = f.attrs.get('n_rollouts', 0)
                self.rollout_lengths = jnp.array(f['rollout_lengths'][()]) if 'rollout_lengths' in f else None

                # Check if qpos/qvel are datasets (not groups)
                self._qpos_shape = f['qpos'].shape if 'qpos' in f and isinstance(f['qpos'], h5py.Dataset) else None
                self._qvel_shape = f['qvel'].shape if 'qvel' in f and isinstance(f['qvel'], h5py.Dataset) else None
                self._has_ctrl = 'ctrl' in f and isinstance(f['ctrl'], h5py.Dataset)
                self._has_time = 'time' in f and isinstance(f['time'], h5py.Dataset)
                self._has_obs = 'obs' in f and isinstance(f['obs'], h5py.Dataset)
                self._obs_shape = f['obs'].shape if 'obs' in f and isinstance(f['obs'], h5py.Dataset) else None
                self._has_qfrc_fluid = 'qfrc_fluid' in f and isinstance(f['qfrc_fluid'], h5py.Dataset)
                self._qfrc_fluid_shape = f['qfrc_fluid'].shape if 'qfrc_fluid' in f and isinstance(f['qfrc_fluid'], h5py.Dataset) else None
                self._has_sensordata = 'sensordata' in f and isinstance(f['sensordata'], h5py.Dataset)
                self._sensordata_shape = f['sensordata'].shape if 'sensordata' in f and isinstance(f['sensordata'], h5py.Dataset) else None
                self._info_keys = list(f['info'].keys()) if 'info' in f and isinstance(f['info'], h5py.Group) else []
                self._metrics_keys = list(f['metrics'].keys()) if 'metrics' in f and isinstance(f['metrics'], h5py.Group) else []
                self._extras_keys = list(f['extras'].keys()) if 'extras' in f and isinstance(f['extras'], h5py.Group) else []
        else:
            self.n_rollouts = 0
            self.rollout_lengths = None
            self._qpos_shape = None
            self._qvel_shape = None
            self._has_ctrl = False
            self._has_time = False
            self._has_obs = False
            self._obs_shape = None
            self._has_qfrc_fluid = False
            self._qfrc_fluid_shape = None
            self._has_sensordata = False
            self._sensordata_shape = None
            self._info_keys = []
            self._metrics_keys = []
            self._extras_keys = []

    @classmethod
    def create_from_rollouts(
        cls,
        rollouts: Union[List[List[Any]], List[Rollout]],
        save_path: Union[str, Path],
        include_info_keys: Optional[List[str]] = None,
        extras: Optional[Dict[str, np.ndarray]] = None,
        chunk_size: int = 1,
        compression: str = 'gzip',
        compression_opts: int = 4
    ) -> 'HDF5RolloutSaver':
        """
        Create HDF5 file from multiple rollouts.

        Args:
            rollouts: List of rollouts, where each rollout is either:
                     - List of mjx_env.State objects
                     - Rollout object
            save_path: Path where to save the HDF5 file
            include_info_keys: Keys from state.info to save (only for State lists)
            extras: Optional dict of additional arrays to save, where each value has
                   shape (n_rollouts, n_timesteps, ...). E.g., {'activations': array}.
                   Each extra will be chunked by rollout for parallel loading.
            chunk_size: Chunk size for HDF5 datasets (1 = one rollout per chunk)
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_opts: Compression level (0-9 for gzip)

        Returns:
            New HDF5RolloutSaver instance
        """
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert state lists to Rollout objects if needed
        processed_rollouts = []
        for rollout in rollouts:
            if isinstance(rollout, Rollout):
                processed_rollouts.append(rollout)
            else:  # Assume it's a list of States
                processed_rollouts.append(Rollout.from_states(rollout, include_info_keys))

        # Get dimensions
        n_rollouts = len(processed_rollouts)
        rollout_lengths = np.array([r.n_timesteps for r in processed_rollouts])
        max_timesteps = int(np.max(rollout_lengths))

        # Get shapes from first rollout
        first_rollout = processed_rollouts[0]
        n_qpos = first_rollout.qpos.shape[1]
        n_qvel = first_rollout.qvel.shape[1]
        n_ctrl = first_rollout.ctrl.shape[1] if first_rollout.ctrl is not None else 0
        n_obs = first_rollout.obs.shape[1] if first_rollout.obs is not None else 0
        n_qfrc = first_rollout.qfrc_fluid.shape[1] if first_rollout.qfrc_fluid is not None else 0
        n_sensor = first_rollout.sensordata.shape[1] if first_rollout.sensordata is not None else 0

        with h5py.File(save_path, 'w') as f:
            # Save metadata
            f.attrs['n_rollouts'] = n_rollouts
            f.create_dataset('rollout_lengths', data=rollout_lengths)

            # Create datasets with padding
            chunk_qpos = (chunk_size, max_timesteps, n_qpos)
            chunk_qvel = (chunk_size, max_timesteps, n_qvel)

            qpos_dset = f.create_dataset(
                'qpos',
                shape=(n_rollouts, max_timesteps, n_qpos),
                dtype=np.float32,
                chunks=chunk_qpos,
                compression=compression,
                compression_opts=compression_opts if compression == 'gzip' else None
            )

            qvel_dset = f.create_dataset(
                'qvel',
                shape=(n_rollouts, max_timesteps, n_qvel),
                dtype=np.float32,
                chunks=chunk_qvel,
                compression=compression,
                compression_opts=compression_opts if compression == 'gzip' else None
            )

            # Optional ctrl dataset
            if n_ctrl > 0:
                chunk_ctrl = (chunk_size, max_timesteps, n_ctrl)
                ctrl_dset = f.create_dataset(
                    'ctrl',
                    shape=(n_rollouts, max_timesteps, n_ctrl),
                    dtype=np.float32,
                    chunks=chunk_ctrl,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

            # Optional time dataset
            if first_rollout.time is not None:
                chunk_time = (chunk_size, max_timesteps)
                time_dset = f.create_dataset(
                    'time',
                    shape=(n_rollouts, max_timesteps),
                    dtype=np.float32,
                    chunks=chunk_time,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

            # Optional obs dataset
            if n_obs > 0:
                chunk_obs = (chunk_size, max_timesteps, n_obs)
                obs_dset = f.create_dataset(
                    'obs',
                    shape=(n_rollouts, max_timesteps, n_obs),
                    dtype=np.float32,
                    chunks=chunk_obs,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

            # Optional qfrc_fluid dataset
            if n_qfrc > 0:
                chunk_qfrc = (chunk_size, max_timesteps, n_qfrc)
                qfrc_dset = f.create_dataset(
                    'qfrc_fluid',
                    shape=(n_rollouts, max_timesteps, n_qfrc),
                    dtype=np.float32,
                    chunks=chunk_qfrc,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

            # Optional sensordata dataset
            if n_sensor > 0:
                chunk_sensor = (chunk_size, max_timesteps, n_sensor)
                sensor_dset = f.create_dataset(
                    'sensordata',
                    shape=(n_rollouts, max_timesteps, n_sensor),
                    dtype=np.float32,
                    chunks=chunk_sensor,
                    compression=compression,
                    compression_opts=compression_opts if compression == 'gzip' else None
                )

            # Fill datasets
            for i, rollout in enumerate(processed_rollouts):
                length = rollout.n_timesteps
                qpos_dset[i, :length] = np.array(rollout.qpos)
                qvel_dset[i, :length] = np.array(rollout.qvel)

                if n_ctrl > 0 and rollout.ctrl is not None:
                    ctrl_dset[i, :length] = np.array(rollout.ctrl)

                if rollout.time is not None:
                    time_dset[i, :length] = np.array(rollout.time)

                if n_obs > 0 and rollout.obs is not None:
                    obs_dset[i, :length] = np.array(rollout.obs)

                if n_qfrc > 0 and rollout.qfrc_fluid is not None:
                    qfrc_dset[i, :length] = np.array(rollout.qfrc_fluid)

                if n_sensor > 0 and rollout.sensordata is not None:
                    sensor_dset[i, :length] = np.array(rollout.sensordata)

            # Save info if available
            if processed_rollouts[0].info:
                info_group = f.create_group('info')
                for key in processed_rollouts[0].info.keys():
                    # Get shape from first rollout
                    sample_data = processed_rollouts[0].info[key]
                    if len(sample_data.shape) == 1:
                        info_shape = (n_rollouts, max_timesteps)
                    else:
                        info_shape = (n_rollouts, max_timesteps) + sample_data.shape[1:]

                    info_dset = info_group.create_dataset(
                        key,
                        shape=info_shape,
                        dtype=np.float32,
                        chunks=(chunk_size,) + info_shape[1:],
                        compression=compression,
                        compression_opts=compression_opts if compression == 'gzip' else None
                    )

                    for i, rollout in enumerate(processed_rollouts):
                        if key in rollout.info:
                            length = rollout.n_timesteps
                            info_dset[i, :length] = np.array(rollout.info[key])

            # Save metrics if available
            if processed_rollouts[0].metrics:
                metrics_group = f.create_group('metrics')
                for key in processed_rollouts[0].metrics.keys():
                    # Get shape from first rollout
                    sample_data = processed_rollouts[0].metrics[key]
                    if len(sample_data.shape) == 1:
                        metrics_shape = (n_rollouts, max_timesteps)
                    else:
                        metrics_shape = (n_rollouts, max_timesteps) + sample_data.shape[1:]

                    metrics_dset = metrics_group.create_dataset(
                        key,
                        shape=metrics_shape,
                        dtype=np.float32,
                        chunks=(chunk_size,) + metrics_shape[1:],
                        compression=compression,
                        compression_opts=compression_opts if compression == 'gzip' else None
                    )

                    for i, rollout in enumerate(processed_rollouts):
                        if key in rollout.metrics:
                            length = rollout.n_timesteps
                            metrics_dset[i, :length] = np.array(rollout.metrics[key])

            # Save extras if provided
            if extras:
                extras_group = f.create_group('extras')
                for key, data in extras.items():
                    # Validate shape: should be (n_rollouts, n_timesteps, ...)
                    if data.shape[0] != n_rollouts:
                        raise ValueError(
                            f"Extra '{key}' has shape {data.shape} but expected "
                            f"first dimension to be {n_rollouts} (number of rollouts)"
                        )

                    # Get the extra's shape and create chunked dataset
                    extra_shape = data.shape
                    # Chunk by rollout: (chunk_size, timesteps, features...)
                    chunk_shape = (chunk_size,) + extra_shape[1:]

                    extras_dset = extras_group.create_dataset(
                        key,
                        shape=extra_shape,
                        dtype=np.float32,
                        chunks=chunk_shape,
                        compression=compression,
                        compression_opts=compression_opts if compression == 'gzip' else None
                    )

                    # Write data
                    extras_dset[:] = np.array(data)

        return cls(save_path, enable_jax=True)

    @classmethod
    def create_from_scan_output(
        cls,
        scan_out: Dict[str, Any],
        save_path: Union[str, Path],
        chunk_size: int = 1,
        compression: str = 'gzip',
        compression_opts: int = 4
    ) -> 'HDF5RolloutSaver':
        """
        Create HDF5 file directly from scan output dictionary.

        This method accepts the output from jax.lax.scan and automatically:
        - Transposes arrays from (timesteps, num_envs, ...) to (num_envs, timesteps, ...)
        - Detects standard fields (qpos, qvel, etc.) vs nested fields (info, metrics, extras)
        - Creates appropriate HDF5 groups and datasets

        Args:
            scan_out: Dictionary output from jax.lax.scan with structure:
                     - Standard keys: 'qpos', 'qvel', 'ctrl', 'time', 'obs', 
                       'qfrc_fluid', 'sensordata' (shape: (timesteps, num_envs, ...))
                     - Nested dicts: 'info', 'metrics', 'extras' containing sub-dictionaries
                       with arrays of shape (timesteps, num_envs, ...)
            save_path: Path where to save the HDF5 file
            chunk_size: Chunk size for HDF5 datasets (1 = one rollout per chunk)
            compression: Compression algorithm ('gzip', 'lzf', None)
            compression_opts: Compression level (0-9 for gzip)

        Returns:
            New HDF5RolloutSaver instance

        Example:
            >>> def scan_step(carry, _):
            ...     state = env.step(carry, action)
            ...     return state, {'qpos': state.data.qpos, 'qvel': state.data.qvel,
            ...                    'info': {'vnc_rates': state.info['vnc_rates']}}
            >>> final_state, scan_out = jax.lax.scan(scan_step, init_state, xs)
            >>> saver = HDF5RolloutSaver.create_from_scan_output(scan_out, 'rollouts.h5')
        """
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Define standard top-level fields
        STANDARD_FIELDS = {'qpos', 'qvel', 'ctrl', 'time', 'obs', 'qfrc_fluid', 'sensordata'}
        
        # Define nested field groups (map to HDF5 groups)
        NESTED_GROUPS = {'info', 'metrics', 'extras'}

        # Validate scan_out structure
        if not isinstance(scan_out, dict):
            raise TypeError(f"scan_out must be a dictionary, got {type(scan_out)}")
        
        if 'qpos' not in scan_out or 'qvel' not in scan_out:
            raise ValueError("scan_out must contain at least 'qpos' and 'qvel' keys")

        # Determine dimensions from qpos
        qpos_array = np.array(scan_out['qpos'])
        if qpos_array.ndim < 2:
            raise ValueError(f"qpos must have at least 2 dimensions (timesteps, features), got shape {qpos_array.shape}")
        
        # Detect if batched: (timesteps, num_envs, features) or (timesteps, features)
        if qpos_array.ndim == 2:
            # Single environment case: (timesteps, features) -> add env dimension
            n_timesteps, n_qpos = qpos_array.shape
            num_envs = 1
            is_batched = False
        else:
            # Batched case: (timesteps, num_envs, features)
            n_timesteps, num_envs, n_qpos = qpos_array.shape
            is_batched = True

        # Helper function to transpose arrays from (timesteps, envs, ...) to (envs, timesteps, ...)
        def transpose_array(arr):
            """Transpose from scan output format to rollout format."""
            arr = np.array(arr)
            if arr.ndim == 1:
                # Single dimension: (timesteps,) -> add env dim -> (1, timesteps)
                return arr[np.newaxis, :]
            elif arr.ndim == 2:
                if is_batched:
                    # Batched: (timesteps, envs) -> (envs, timesteps)
                    return np.transpose(arr, (1, 0))
                else:
                    # Single env: (timesteps, features) -> (1, timesteps, features)
                    return arr[np.newaxis, :]
            else:
                # Multi-dimensional: transpose first two dims
                if is_batched:
                    # (timesteps, envs, ...) -> (envs, timesteps, ...)
                    perm = (1, 0) + tuple(range(2, arr.ndim))
                    return np.transpose(arr, perm)
                else:
                    # (timesteps, ...) -> (1, timesteps, ...)
                    return arr[np.newaxis, :]

        # Process standard fields
        standard_data = {}
        for key in STANDARD_FIELDS:
            if key in scan_out:
                standard_data[key] = transpose_array(scan_out[key])

        # Process nested groups
        nested_data = {group: {} for group in NESTED_GROUPS}
        for group in NESTED_GROUPS:
            if group in scan_out and isinstance(scan_out[group], dict):
                for key, value in scan_out[group].items():
                    nested_data[group][key] = transpose_array(value)

        # Get actual rollout lengths for each environment
        # Assume all environments have same length (from scan)
        rollout_lengths = np.full(num_envs, n_timesteps, dtype=np.int32)

        # Create HDF5 file
        with h5py.File(save_path, 'w') as f:
            # Save metadata
            f.attrs['n_rollouts'] = num_envs
            f.create_dataset('rollout_lengths', data=rollout_lengths)

            # Write standard fields as top-level datasets
            for key, data in standard_data.items():
                if data is not None:
                    data_shape = data.shape  # (num_envs, timesteps, ...)
                    chunk_shape = (chunk_size,) + data_shape[1:]
                    
                    dset = f.create_dataset(
                        key,
                        shape=data_shape,
                        dtype=np.float32,
                        chunks=chunk_shape,
                        compression=compression,
                        compression_opts=compression_opts if compression == 'gzip' else None
                    )
                    dset[:] = data

            # Write nested groups
            for group_name, group_data in nested_data.items():
                if group_data:  # Only create group if it has data
                    group = f.create_group(group_name)
                    for key, data in group_data.items():
                        if data is not None:
                            data_shape = data.shape
                            chunk_shape = (chunk_size,) + data_shape[1:]
                            
                            dset = group.create_dataset(
                                key,
                                shape=data_shape,
                                dtype=np.float32,
                                chunks=chunk_shape,
                                compression=compression,
                                compression_opts=compression_opts if compression == 'gzip' else None
                            )
                            dset[:] = data

        return cls(save_path, enable_jax=True)

    def load_rollout(self, rollout_idx: int, clip_length: Optional[int] = None) -> Rollout:
        """
        JIT-compatible method to load a single rollout using jax.pure_callback.
        Returns the full padded rollout (not trimmed to actual length).

        Args:
            rollout_idx: Index of the rollout to load (can be JAX array or Python int)
            clip_length: Optional length to clip rollout to (takes timesteps 0:clip_length)

        Returns:
            Rollout object containing the full rollout's data (padded)
        """
        def host_load_function(rollout_idx_val):
            """Pure callback that performs I/O operations on CPU."""
            # Convert to Python int
            idx = int(rollout_idx_val.item() if hasattr(rollout_idx_val, 'item') else rollout_idx_val)

            # Load the entire padded rollout data
            with h5py.File(self.file_path, 'r') as f:
                # Determine length to load
                if clip_length is not None and clip_length > 0:
                    length = min(clip_length, f['qpos'].shape[1])
                else:
                    length = f['qpos'].shape[1]

                qpos = np.array(f['qpos'][idx, :length])
                qvel = np.array(f['qvel'][idx, :length])

                ctrl = np.array(f['ctrl'][idx, :length]) if self._has_ctrl else np.zeros((length, 1), dtype=np.float32)
                time = np.array(f['time'][idx, :length]) if self._has_time else np.zeros(length, dtype=np.float32)
                obs = np.array(f['obs'][idx, :length]) if self._has_obs else np.zeros((length, 1), dtype=np.float32)
                qfrc_fluid = np.array(f['qfrc_fluid'][idx, :length]) if self._has_qfrc_fluid else np.zeros((length, 1), dtype=np.float32)
                sensordata = np.array(f['sensordata'][idx, :length]) if self._has_sensordata else np.zeros((length, 1), dtype=np.float32)

                # Load info arrays if they exist (only datasets)
                info_arrays = []
                for key in self._info_keys:
                    if isinstance(f['info'][key], h5py.Dataset):
                        info_arrays.append(np.array(f['info'][key][idx, :length]))

                # Load metrics arrays if they exist (only datasets)
                metrics_arrays = []
                for key in self._metrics_keys:
                    if isinstance(f['metrics'][key], h5py.Dataset):
                        metrics_arrays.append(np.array(f['metrics'][key][idx, :length]))

                # Load extras arrays if they exist (only datasets)
                extras_arrays = []
                for key in self._extras_keys:
                    if isinstance(f['extras'][key], h5py.Dataset):
                        extras_arrays.append(np.array(f['extras'][key][idx, :length]))

            return qpos, qvel, ctrl, time, obs, qfrc_fluid, sensordata, *info_arrays, *metrics_arrays, *extras_arrays

        # Determine output timestep length
        with h5py.File(self.file_path, 'r') as f:
            max_timesteps = f['qpos'].shape[1]
            if clip_length is not None and clip_length > 0:
                timestep_length = min(clip_length, max_timesteps)
            else:
                timestep_length = max_timesteps

        # Build output structure for pure_callback
        n_qpos = self._qpos_shape[2]
        n_qvel = self._qvel_shape[2]

        output_shapes = [
            jax.ShapeDtypeStruct((timestep_length, n_qpos), jnp.float32),
            jax.ShapeDtypeStruct((timestep_length, n_qvel), jnp.float32),
        ]

        if self._has_ctrl:
            with h5py.File(self.file_path, 'r') as f:
                n_ctrl = f['ctrl'].shape[2]
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length, n_ctrl), jnp.float32))
        else:
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length, 1), jnp.float32))

        if self._has_time:
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length,), jnp.float32))
        else:
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length,), jnp.float32))

        if self._has_obs:
            n_obs = self._obs_shape[2]
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length, n_obs), jnp.float32))
        else:
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length, 1), jnp.float32))

        if self._has_qfrc_fluid:
            n_qfrc = self._qfrc_fluid_shape[2]
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length, n_qfrc), jnp.float32))
        else:
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length, 1), jnp.float32))

        if self._has_sensordata:
            n_sensor = self._sensordata_shape[2]
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length, n_sensor), jnp.float32))
        else:
            output_shapes.append(jax.ShapeDtypeStruct((timestep_length, 1), jnp.float32))

        # Add info shapes
        with h5py.File(self.file_path, 'r') as f:
            for key in self._info_keys:
                if isinstance(f['info'][key], h5py.Dataset):
                    info_shape = f['info'][key].shape[1:]
                    # Replace first dimension with timestep_length
                    clipped_shape = (timestep_length,) + info_shape[1:]
                    output_shapes.append(jax.ShapeDtypeStruct(clipped_shape, jnp.float32))

            # Add metrics shapes
            for key in self._metrics_keys:
                if isinstance(f['metrics'][key], h5py.Dataset):
                    metrics_shape = f['metrics'][key].shape[1:]
                    # Replace first dimension with timestep_length
                    clipped_shape = (timestep_length,) + metrics_shape[1:]
                    output_shapes.append(jax.ShapeDtypeStruct(clipped_shape, jnp.float32))

            # Add extras shapes
            for key in self._extras_keys:
                if isinstance(f['extras'][key], h5py.Dataset):
                    extras_shape = f['extras'][key].shape[1:]
                    # Replace first dimension with timestep_length
                    clipped_shape = (timestep_length,) + extras_shape[1:]
                    output_shapes.append(jax.ShapeDtypeStruct(clipped_shape, jnp.float32))

        # Use pure_callback to perform I/O from within JIT
        results = jax.pure_callback(
            host_load_function,
            tuple(output_shapes),
            rollout_idx,
            vmap_method='sequential'
        )

        qpos, qvel, ctrl, time, obs, qfrc_fluid, sensordata = results[:7]
        remaining_results = results[7:]

        # Count actual dataset keys (not groups)
        with h5py.File(self.file_path, 'r') as f:
            n_info_datasets = sum(1 for k in self._info_keys if isinstance(f['info'][k], h5py.Dataset))
            n_metrics_datasets = sum(1 for k in self._metrics_keys if isinstance(f['metrics'][k], h5py.Dataset))
            n_extras_datasets = sum(1 for k in self._extras_keys if isinstance(f['extras'][k], h5py.Dataset))

        info_arrays = remaining_results[:n_info_datasets]
        metrics_arrays = remaining_results[n_info_datasets:n_info_datasets + n_metrics_datasets]
        extras_arrays = remaining_results[n_info_datasets + n_metrics_datasets:]

        # Convert to JAX arrays if needed
        if self.enable_jax:
            qpos = jnp.array(qpos)
            qvel = jnp.array(qvel)
            ctrl = jnp.array(ctrl) if self._has_ctrl else None
            time = jnp.array(time) if self._has_time else None
            obs = jnp.array(obs) if self._has_obs else None
            qfrc_fluid = jnp.array(qfrc_fluid) if self._has_qfrc_fluid else None
            sensordata = jnp.array(sensordata) if self._has_sensordata else None

        # Build info dict (only for dataset keys)
        info = {}
        with h5py.File(self.file_path, 'r') as f:
            info_dataset_keys = [k for k in self._info_keys if isinstance(f['info'][k], h5py.Dataset)]
            for i, key in enumerate(info_dataset_keys):
                info[key] = jnp.array(info_arrays[i]) if self.enable_jax else info_arrays[i]

            # Build metrics dict (only for dataset keys)
            metrics = {}
            metrics_dataset_keys = [k for k in self._metrics_keys if isinstance(f['metrics'][k], h5py.Dataset)]
            for i, key in enumerate(metrics_dataset_keys):
                metrics[key] = jnp.array(metrics_arrays[i]) if self.enable_jax else metrics_arrays[i]

            # Build extras dict (only for dataset keys)
            extras = {}
            extras_dataset_keys = [k for k in self._extras_keys if isinstance(f['extras'][k], h5py.Dataset)]
            for i, key in enumerate(extras_dataset_keys):
                extras[key] = jnp.array(extras_arrays[i]) if self.enable_jax else extras_arrays[i]

        return Rollout(
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            time=time,
            obs=obs,
            qfrc_fluid=qfrc_fluid,
            sensordata=sensordata,
            info=info if info else None,
            metrics=metrics if metrics else None,
            extras=extras if extras else None
        )

    def get_qpos_qvel(self, rollout_idx: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Load only qpos and qvel for a rollout (efficient for rendering).

        Args:
            rollout_idx: Index of the rollout to load

        Returns:
            Tuple of (qpos, qvel) arrays
        """
        with h5py.File(self.file_path, 'r') as f:
            length = int(self.rollout_lengths[rollout_idx])
            qpos = f['qpos'][rollout_idx, :length]
            qvel = f['qvel'][rollout_idx, :length]

            if self.enable_jax:
                return jnp.array(qpos), jnp.array(qvel)
            return qpos, qvel

    def print_info(self):
        """Print information about the HDF5 file structure and contents."""
        print(f"HDF5 Rollout File: {self.file_path}")
        print(f"Number of rollouts: {self.n_rollouts}")
        print(f"Rollout lengths: min={np.min(self.rollout_lengths)}, max={np.max(self.rollout_lengths)}, mean={np.mean(self.rollout_lengths):.1f}")
        print()

        with h5py.File(self.file_path, 'r') as f:
            print("Datasets:")
            if self._qpos_shape:
                print(f"  qpos: {self._qpos_shape}")
            if self._qvel_shape:
                print(f"  qvel: {self._qvel_shape}")
            if self._has_ctrl:
                print(f"  ctrl: {f['ctrl'].shape}")
            if self._has_time:
                print(f"  time: {f['time'].shape}")
            if self._has_obs:
                print(f"  obs: {self._obs_shape}")
            if self._has_qfrc_fluid:
                print(f"  qfrc_fluid: {self._qfrc_fluid_shape}")
            if self._has_sensordata:
                print(f"  sensordata: {self._sensordata_shape}")
            print()

            if self._info_keys:
                print("Info keys:")
                for key in self._info_keys:
                    item = f['info'][key]
                    if isinstance(item, h5py.Dataset):
                        print(f"  {key}: {item.shape} (Dataset)")
                    else:
                        print(f"  {key}: Group with {len(item.keys())} items")
                print()

            if self._metrics_keys:
                print("Metrics keys:")
                for key in self._metrics_keys:
                    item = f['metrics'][key]
                    if isinstance(item, h5py.Dataset):
                        print(f"  {key}: {item.shape} (Dataset)")
                    else:
                        print(f"  {key}: Group with {len(item.keys())} items")
                print()

            if self._extras_keys:
                print("Extras keys:")
                for key in self._extras_keys:
                    item = f['extras'][key]
                    if isinstance(item, h5py.Dataset):
                        print(f"  {key}: {item.shape} (Dataset)")
                    else:
                        print(f"  {key}: Group with {len(item.keys())} items")
                print()

    def get_extra(self, extra_key: str, rollout_idx: int) -> jnp.ndarray:
        """
        Load a specific extra variable for a rollout.

        Args:
            extra_key: Name of the extra variable (e.g., 'activations')
            rollout_idx: Index of the rollout to load

        Returns:
            Array containing the extra variable data
        """
        if extra_key not in self._extras_keys:
            raise KeyError(f"Extra '{extra_key}' not found. Available extras: {self._extras_keys}")

        with h5py.File(self.file_path, 'r') as f:
            length = int(self.rollout_lengths[rollout_idx])
            data = f['extras'][extra_key][rollout_idx, :length]

            if self.enable_jax:
                return jnp.array(data)
            return data

    def append_rollout(
        self,
        rollout: Union[List[Any], Rollout],
        include_info_keys: Optional[List[str]] = None
    ):
        """
        Append a new rollout to an existing HDF5 file.

        Note: This is less efficient than batch creation. For many rollouts,
        prefer create_from_rollouts.

        Args:
            rollout: Either a list of mjx_env.State objects or a Rollout object
            include_info_keys: Keys from state.info to save (only for State lists)
        """
        if isinstance(rollout, Rollout):
            new_rollout = rollout
        else:
            new_rollout = Rollout.from_states(rollout, include_info_keys)

        with h5py.File(self.file_path, 'a') as f:
            # Update metadata
            old_n = f.attrs['n_rollouts']
            f.attrs['n_rollouts'] = old_n + 1

            # Resize datasets
            max_timesteps = f['qpos'].shape[1]
            new_length = max(max_timesteps, new_rollout.n_timesteps)

            if new_rollout.n_timesteps > max_timesteps:
                # Need to resize all datasets
                f['qpos'].resize((old_n + 1, new_length, f['qpos'].shape[2]))
                f['qvel'].resize((old_n + 1, new_length, f['qvel'].shape[2]))
                if 'ctrl' in f:
                    f['ctrl'].resize((old_n + 1, new_length, f['ctrl'].shape[2]))
                if 'time' in f:
                    f['time'].resize((old_n + 1, new_length))
            else:
                f['qpos'].resize((old_n + 1, max_timesteps, f['qpos'].shape[2]))
                f['qvel'].resize((old_n + 1, max_timesteps, f['qvel'].shape[2]))
                if 'ctrl' in f:
                    f['ctrl'].resize((old_n + 1, max_timesteps, f['ctrl'].shape[2]))
                if 'time' in f:
                    f['time'].resize((old_n + 1, max_timesteps))

            # Add new data
            length = new_rollout.n_timesteps
            f['qpos'][old_n, :length] = np.array(new_rollout.qpos)
            f['qvel'][old_n, :length] = np.array(new_rollout.qvel)

            if new_rollout.ctrl is not None and 'ctrl' in f:
                f['ctrl'][old_n, :length] = np.array(new_rollout.ctrl)

            if new_rollout.time is not None and 'time' in f:
                f['time'][old_n, :length] = np.array(new_rollout.time)

            # Update rollout lengths
            old_lengths = f['rollout_lengths'][()]
            new_lengths = np.append(old_lengths, new_rollout.n_timesteps)
            del f['rollout_lengths']
            f.create_dataset('rollout_lengths', data=new_lengths)

            # Update instance metadata
            self.n_rollouts = old_n + 1
            self.rollout_lengths = jnp.array(new_lengths)

import collections
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Union, Sequence, List, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.rollout_saver import Rollout

import brax.math
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from sphinx_training.utils.data_utils import ReferenceClips, HDF5ReferenceClips
from sphinx_training.utils.utils import add_cross_hair_sites, add_trajectory_sites_spheres
from . import base as fruitfly_base
from .constants import *

try:
    from sphinx_training.envs.quasi_aero import compute_apply_all_fluid_forces_to_state
    from sphinx_training.envs.ellipsoid_fluid import compute_apply_ellipsoid_fluid_forces
    from sphinx_training.envs.pattern_generators_new import JITWingBeatPatternGenerator
except ImportError:
    compute_apply_all_fluid_forces_to_state = None
    compute_apply_ellipsoid_fluid_forces = None
    JITWingBeatPatternGenerator = None

_REWARD_FCN_REGISTRY: dict[str, Callable] = {}
_TERMINATION_FCN_REGISTRY: dict[str, Callable] = {}


def _bounded_quat_dist(source: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Computes a quaternion distance limiting the difference to a max of pi/2.

    This function supports an arbitrary number of batch dimensions, B.

    Args:
        source: a quaternion, shape (B, 4).
        target: another quaternion, shape (B, 4).

    Returns:
        Quaternion distance, shape (B, 1).
    """
    source /= jnp.linalg.norm(source, axis=-1, keepdims=True)
    target /= jnp.linalg.norm(target, axis=-1, keepdims=True)
    # "Distance" in interval [-1, 1].
    dist = 2 * jnp.einsum("...i,...i", source, target) ** 2 - 1
    # Clip at 1 to avoid occasional machine epsilon leak beyond 1.
    dist = jnp.minimum(1.0, dist)
    # Divide by 2 and add an axis to ensure consistency with expected return
    # shape and magnitude.
    return 0.5 * jnp.arccos(dist)[..., np.newaxis]

class Imitation(fruitfly_base.FruitflyEnv):
    """Multi-clip imitation environment supporting both walking and flying behaviors."""

    def __init__(
        self,
        cfg,
        reference_clips: Optional[ReferenceClips] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any], dict]]] = None,
    ) -> None:
        """
        Initialize the fruitfly imitation environment supporting both walking and flying.
        Args:
            cfg: Configuration object containing environment parameters
            reference_clips: Optional reference clips for imitation
            config_overrides: Optional configuration overrides
            enable_flight: Whether to enable flight physics (aerodynamics)
        """
        env_args = cfg.dataset.env_args
        config = config_dict.ConfigDict(env_args)
        config.ctrl_dt = 1/env_args.mocap_hz
        config.sim_dt = env_args.physics_timestep
        config._walker_xml_path = env_args.mjcf_path
        config._arena_xml_path = env_args.arena_path
        config._floor_height = env_args.floor_height
        config.rescale_factor = 1.0
        config.torque_actuators = True
        config.mujoco_impl = env_args.mujoco_impl
        super().__init__(config, config_overrides)
        # Flight-specific parameters
        if self._enable_flight:
            if self._enable_wbpg:
                self._wbpg = JITWingBeatPatternGenerator(base_pattern_path=env_args.wbpg_path)
            self._quasi_aero = getattr(env_args, 'quasi_aero', True)
        self.add_fly(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
            only_T1=self._config.only_T1
        )
        self.compile(mjx_model=True)

        if reference_clips is not None:
            self.reference_clips = reference_clips
        else:
            # 030226 changed this since mine is named datasets not clips
            (ref_train, ref_test) = HDF5ReferenceClips(cfg.paths.data_dir / f"datasets/{cfg.dataset.clip_idx}").split(test_ratio=env_args.test_ratio)
            # (ref_train, ref_test) = ReferenceClips.from_path(cfg.paths.data_dir / f"clips/{cfg.dataset.clip_idx}").split(test_ratio=env_args.test_ratio)
            self.reference_clips = ref_train.load_all_clips() if cfg.dataset.train else ref_test.load_all_clips()
        print(f"Loaded {self.reference_clips.num_clips} reference clips for {'training' if cfg.dataset.train else 'testing'}.")


    def reset(self, rng: jax.Array, clip_idx: Optional[int] = None) -> mjx_env.State:
        """
        Resets the environment state: draws a new reference clip and initializes the fly's pose to match.
        Args:
            rng (jax.Array): JAX random number generator stare.
        Returns:
            mjx_env.State: The initial state of the environment after reset.
        """

        start_rng, clip_rng, reset_rng = jax.random.split(rng, 3)
        if clip_idx is None:
            clip_idx = jax.random.choice(clip_rng, self.reference_clips.num_clips)
            start_frame = jax.random.randint(start_rng, (), *self._config.start_frame_range)
        else:
            start_frame = 0
        info: dict[str, Any] = {
            "start_frame": start_frame,
            "reference_clip": clip_idx,
        }
        data, info = self._reset_data(clip_idx, start_frame, reset_rng, info)
        last_valid_frame = self._clip_length(info['reference_clip']) - self._config.ref_len - 1
        truncated = self._get_cur_frame(data, info) > last_valid_frame
        info["truncated"] = jnp.astype(truncated, float)
        info["prev_action"] = self.null_action()
        info["action"] = self.null_action()
        metrics = {
            "current_frame": jnp.astype(self._get_cur_frame(data, info), float),
        }
        imitation_target, proprioception = self._get_obs(data, info, flatten=True)
        reward = self._get_reward(data, info, metrics, None)
        done = self._is_done(data, info, metrics, None)
        info['obs_sizes'] = {'task_obs_size': imitation_target.shape[-1],
                            'proprioception': proprioception.shape[-1]}
        obs = jnp.concatenate([imitation_target, proprioception])

        return mjx_env.State(data, obs, reward, jnp.astype(done, float), metrics, info)

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
    ) -> mjx_env.State:
        """Step the environment forward.
        Args:
            state (mjx_env.State): Current environment state.
            action (jax.Array): Action to apply.
        Returns:
            mjx_env.State: The new state of the environment.
        """
        # floor_contact = data.sensordata[self._contact_idxs]
        # if jnp.any(floor_contact>0):
        #     self._enable_flight = False

        n_steps = int(self._config.ctrl_dt / self._config.sim_dt)

        if self._enable_flight:
            if self._enable_wbpg:
                # Apply aerodynamic forces during physics step
                action_mod, info = self.before_step(state.data, action, state.info)
            else:
                action_mod = action
                info = state.info
            data, force_components = self._step_with_aero(state.data, action_mod, n_steps)
            info['aero_forces'] = [force[self._wing_fluid_idxs] for force in force_components]
        else:
            # Standard physics step for walking
            data = mjx_env.step(self.mjx_model, state.data, action, n_steps)

        info = state.info
        last_valid_frame = self._clip_length(info['reference_clip']) - self._config.ref_len - 1
        truncated = self._get_cur_frame(data, info) > last_valid_frame
        info["truncated"] = jnp.astype(truncated, float)
        info["prev_action"] = state.info["action"]
        info["action"] = action

        # Extract current target once for efficiency - this is the key optimization
        current_target = self._get_current_target(data, info)

        imitation_target, proprioception = self._get_obs(data, info, flatten=True)
        obs = jnp.concatenate([imitation_target, proprioception])
        terminated = self._is_done(data, info, state.metrics, current_target)
        done = jnp.logical_or(terminated, info["truncated"])
        reward = self._get_reward(data, info, state.metrics, current_target)
        obs = jnp.nan_to_num(obs)
        state = state.replace(
            data=data,
            obs=obs,
            info=info,
            reward=reward,
            done=done.astype(float),
        )
        current_frame = self._get_cur_frame(data, info)
        state.metrics["current_frame"] = jnp.astype(current_frame, float)
        return state

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any], flatten: bool) -> Mapping[str, Any]:
        if flatten:
            imitation_target = self._get_imitation_target(data, info, flatten=flatten)
            proprioception = self._get_proprioception(data, flatten=flatten)
            return imitation_target, proprioception
        else:
            obs = collections.OrderedDict(
                imitation_target=self._get_imitation_target(data, info, flatten=flatten),
                proprioception=self._get_proprioception(data, flatten=flatten),
            )
            return obs

    def _get_reward(
        self, data: mjx.Data, info: Mapping[str, Any], metrics: Dict, current_target: ReferenceClips = None
    ) -> float:
        net_reward = 0.0
        for name, kwargs in self._config.reward_terms.items():
            net_reward += jnp.nan_to_num(_REWARD_FCN_REGISTRY[name](self, data, info, metrics, current_target, **kwargs))
        return net_reward

    def _is_done(self, data: mjx.Data, info: Mapping[str, Any], metrics, current_target: ReferenceClips = None) -> bool:
        any_terminated = False
        for name, kwargs in self._config.termination_criteria.items():
            termination_fcn = _TERMINATION_FCN_REGISTRY[name]
            terminated = termination_fcn(self, data, info, current_target, **kwargs)
            any_terminated = jnp.logical_or(any_terminated, terminated)
            #Also log terminations as floats so averaging -> hazard rate
            metrics["terminations/" + name] = jnp.astype(terminated, float)
        metrics["terminations/any"] = jnp.astype(any_terminated, float)
        return any_terminated

    def _reset_data(self, clip_idx: int, start_frame: int, rng: jax.Array, info: dict[str, Any]) -> mjx.Data:
        data = mjx.make_data(
            self.mj_model,
            impl=self._config.mujoco_impl,
            naconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        reference = self.reference_clips.extract_frame(frame_index=start_frame, clip_index=clip_idx)
        wbpg_rng, rng = jax.random.split(rng)
        ##### Check if initial position is below the floor #####
        leg_below = jnp.where((reference.xpos[self._end_eff_idxs, -1] - self._floor_height) < 0,1,0,)
        result = jax.lax.select(
            leg_below,
            reference.xpos[self._end_eff_idxs, -1]
            - self._floor_height,
            jnp.zeros_like(reference.xpos[self._end_eff_idxs, -1]),
        )

        # set the z position of the root joint such that the lowest end effector is above the floor
        new_qpos = reference.qpos.at[2].set(reference.qpos[2] - jnp.min(result))# + 7.5e-3)
        # Retract wings to default position
        new_qpos = new_qpos.at[self._wing_joint_idxs].set(self._default_wing_pos)   
        data = data.replace(qpos=new_qpos)
        if self._config.qvel_init == "default":
            pass
        elif self._config.qvel_init == "zeros":
            data = data.replace(qvel=jnp.zeros(self.mjx_model.nv))
        elif self._config.qvel_init == "noise":
            low, hi = -self._config.reset_noise_scale, self._config.reset_noise_scale
            data = data.replace(qvel=jax.random.uniform(rng, (self.mjx_model.nv,), minval=low, maxval=hi))
        elif self._config.qvel_init == "reference":
            data = data.replace(qvel=reference.qvel)
        if self._enable_flight:
            # Initialize wingbeat pattern generator state
            if self._enable_wbpg:
                init_phase = jax.random.uniform(wbpg_rng, 1, minval=0, maxval=1)
                wbpg_state, wing_qpos, wing_qvel = self._wbpg.reset(ctrl_freq=self._wbpg.data.base_beat_freq, initial_phase=init_phase, return_qvel=True)
                info["wbpg_state"] = wbpg_state
                data = data.replace(qpos=data.qpos.at[self._wing_joint_idxs].set(wing_qpos), 
                                    qvel=data.qvel.at[self._wing_joint_idxs-1].set(wing_qvel),)
            else:
                # If not using WBPG, set wing joints to default positions and zero velocity
                data = data.replace(qpos=data.qpos.at[self._wing_joint_idxs].set(reference.qpos[self._wing_joint_idxs]),
                                    qvel=data.qvel.at[self._wing_joint_idxs-1].set(reference.qvel[self._wing_joint_idxs-1]),)
            data = mjx.forward(self.mjx_model, data)
            fluid_forces, force_components = self._compute_aero_forces(data)
            info['aero_forces'] = [force[self._wing_fluid_idxs] for force in force_components] # Initialize aero forces 
            data = data.replace(qfrc_fluid=fluid_forces)
            
        data = mjx.forward(self.mjx_model, data)
        return data, info

    def null_action(self) -> jnp.ndarray:
        return jnp.zeros(self.action_size)

    def _clip_length(self, clip_idx: int) -> int:
        return self.reference_clips.clip_lengths[clip_idx]
        # return self.reference_clips.qpos.shape[1]

    def _get_cur_frame(self, data: mjx.Data, info: Mapping[str, Any]) -> int:
        time_in_frames = data.time * self._config.mocap_hz
        return jnp.round(time_in_frames + info["start_frame"]).astype(int)

    def _get_current_target(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get the reference data at the current frame."""
        return self.reference_clips.extract_frame(
            clip_index=info["reference_clip"], frame_index=self._get_cur_frame(data, info)
        )

    def _get_imitation_reference(
        self, data: mjx.Data, info: Mapping[str, Any]
    ) -> ReferenceClips:
        """Get the reference slice that is to be part of the observation."""
        return self.reference_clips.extract_clip_sequence(
            clip_index=info["reference_clip"],
            start_frame=self._get_cur_frame(data, info) + 1,
            length=self._config.ref_len,
        )

    def _get_imitation_target(
        self, data: mjx.Data, info: Mapping[str, Any], flatten: bool = True,
    ) -> Mapping[str, jnp.ndarray]:
        """Get the imitation target, i.e. the imitation reference transformed to
        egocentric coordinates."""
        reference = self._get_imitation_reference(data, info)
        # TODO look at this to get the pecis of the data ?
        root_pos = data.qpos[:3]
        root_quat = data.qpos[3:7]
        ref_root_pos = reference.qpos[..., :3]
        ref_root_quat = reference.qpos[..., 3:7]
        root_targets = jax.vmap(lambda a, b: brax.math.rotate(a, b), in_axes=(0, None))(ref_root_pos - root_pos, root_quat)
        quat_targets = jax.vmap(lambda a, b: brax.math.relative_quat(a, b), in_axes=(0, None))(ref_root_quat, root_quat)
        joint_targets = reference.qpos[..., self._joint_idxs] - self._get_joint_angles(data)
        to_egocentric = jax.vmap(lambda diff_vec: brax.math.rotate(diff_vec, root_quat))
        body_targets = jax.vmap(to_egocentric)(reference.xpos[:,self._body_idxs] - data.xpos[self._body_idxs])
        
        if flatten:
            return jnp.concatenate([root_targets.flatten(),
                                    quat_targets.flatten(),
                                    joint_targets.flatten(),
                                    body_targets.flatten()])
        else:
            return collections.OrderedDict(
                    root=root_targets,
                    quat=quat_targets,
                    joint=joint_targets,
                    body=body_targets
                )

    # Rewards
    def _named_reward(name: str):
        def decorator(reward_fcn: Callable):
            _REWARD_FCN_REGISTRY[name] = reward_fcn
            return reward_fcn

        return decorator

    @_named_reward("root_pos")
    def _root_pos_reward(self, data, info, metrics, current_target, weight, exp_scale) -> float:
        target = current_target if current_target is not None else self._get_current_target(data, info)
        root_pos = data.qpos[:3]
        pos_distance = root_pos - target.qpos[...,:3]
        distance_sq = jnp.sum(pos_distance**2)
        metrics["root_pos_distance"] = distance_sq
        reward = weight * jnp.exp(-exp_scale * distance_sq)
        metrics["rewards/root_pos"] = reward
        return reward

    @_named_reward("root_quat")
    def _root_quat_reward(self, data, info, metrics, current_target, weight, exp_scale) -> float:
        target = current_target if current_target is not None else self._get_current_target(data, info)
        root_quat = data.qpos[3:7]
        quat_distance = jnp.sum(
            _bounded_quat_dist(root_quat, target.qpos[..., 3:7]) ** 2
        )
        metrics["root_angular_error"] = quat_distance
        reward = weight * jnp.exp(-exp_scale * quat_distance)
        metrics["rewards/root_quat"] = reward
        return reward

    @_named_reward("root_ang_vel")
    def _root_ang_vel_reward(self, data, info, metrics, current_target, weight, exp_scale) -> float:
        target = current_target if current_target is not None else self._get_current_target(data, info)
        root_ang_vel = data.qvel[..., 3:6]
        ang_vel_diff = target.qvel[..., 3:6] - root_ang_vel
        distance_sq = jnp.sum(ang_vel_diff**2)
        metrics["root_ang_vel_distance"] = distance_sq
        reward = weight * jnp.exp(-exp_scale * distance_sq)
        metrics["rewards/root_ang_vel"] = reward
        return reward

    @_named_reward("joints")
    def _joints_reward(self, data, info, metrics, current_target, weight, exp_scale) -> float:
        target = current_target if current_target is not None else self._get_current_target(data, info)
        # Use _joint_reward_idxs (wing-free) — wings are clamped to default pos
        # in walking mode creating a constant irreducible error that dilutes the
        # gradient signal for leg joints. See base.py for index computation.
        joints = data.qpos[..., self._joint_reward_idxs]
        joint_diff = joints - target.qpos[..., self._joint_reward_idxs]
        distance_sq = jnp.sum(joint_diff**2)
        metrics["joint_error"] = distance_sq
        reward = weight * jnp.exp(-exp_scale * distance_sq)
        metrics["rewards/joints"] = reward
        return reward

    @_named_reward("joints_vel")
    def _joint_vels_reward(self, data, info, metrics, current_target, weight, exp_scale) -> float:
        target = current_target if current_target is not None else self._get_current_target(data, info)
        joint_vels = data.qvel[..., self._joint_vel_reward_idxs]
        joint_vel_diff = target.qvel[..., self._joint_vel_reward_idxs] - joint_vels
        distance_sq = jnp.sum(joint_vel_diff**2)
        metrics["joint_vel_error"] = distance_sq
        reward = weight * jnp.exp(-exp_scale * distance_sq)
        metrics["rewards/joints_vel"] = reward
        return reward

    def _get_bodies_dist(self, data, info, metrics, current_target, bodies, body_idxs) -> float:
        target = current_target if current_target is not None else self._get_current_target(data, info)
        body_pos = self._get_bodies_pos(data, flatten=False)
        total_dist_sqr = 0.0
        for idx, body_name in enumerate(bodies):
            dist_sqr = jnp.sum((body_pos[body_name] - target.xpos[body_idxs[idx]])**2)
            # metrics["body_errors/" + body_name] = dist_sqr
            total_dist_sqr += dist_sqr
        return total_dist_sqr

    @_named_reward("bodies_pos")
    def _body_pos_reward(self, data, info, metrics, current_target, weight, exp_scale) -> float:
        # total_dist = self._get_bodies_dist(data, info, metrics, current_target, self._config.body_names, self._body_idxs)
        target = current_target if current_target is not None else self._get_current_target(data, info)
        target_body_pos = target.xpos[self._body_idxs]
        body_pos = data.xpos[self._body_idxs]
        total_dist = jnp.sum((body_pos - target_body_pos).flatten()**2)
        metrics["body_errors/total"] = total_dist
        reward = weight * jnp.exp(-exp_scale * total_dist)
        metrics["rewards/bodies_pos"] = reward
        return reward

    @_named_reward("end_eff")
    def _end_eff_reward(self, data, info, metrics, current_target, weight, exp_scale) -> float:
        # total_dist = self._get_bodies_dist(data, info, metrics, current_target, self._config.end_eff_names, self._end_eff_idxs)
        target = current_target if current_target is not None else self._get_current_target(data, info)
        target_body_pos = target.xpos[self._end_eff_idxs]
        body_pos = data.xpos[self._end_eff_idxs]
        total_dist = jnp.sum((body_pos - target_body_pos).flatten()**2)
        metrics["body_errors/end_eff_total"] = total_dist
        reward = weight * jnp.exp(-exp_scale * total_dist)
        metrics["rewards/end_eff"] = reward
        return reward

    @_named_reward("thorax_z_range")
    def _thorax_z_range_reward(self, data, info, metrics, current_target, weight,
                              healthy_z_range) -> float:
        metrics["thorax_z"] = thorax_z = self._get_body_height(data)
        min_z, max_z = healthy_z_range
        in_range = jnp.logical_and(thorax_z >= min_z, thorax_z <= max_z)
        metrics["in_range"] = in_range.astype(float)
        reward = weight * in_range
        metrics["rewards/thorax_z_range"] = reward
        return reward
    
    @_named_reward("last_frame_reward")
    def _last_frame_reward(self, data, info, metrics, current_target, weight) -> float:
        lastframe_reward = weight * info["truncated"]
        metrics["rewards/last_frame"] = lastframe_reward
        return lastframe_reward

    @_named_reward("control_cost")
    def _control_cost(self, data, info, metrics, current_target, weight) -> float:
        action = jnp.nan_to_num(info["action"], nan=0.0)
        metrics["ctrl_sqr"] = ctrl_sqr = jnp.sum(jnp.square(action))
        cost = weight * ctrl_sqr
        metrics["costs/control"] = cost
        return -cost

    @_named_reward("control_diff_cost")
    def _control_diff_cost(self, data, info, metrics, current_target, weight) -> float:
        prev_action = jnp.nan_to_num(info["prev_action"], nan=0.0)
        action = jnp.nan_to_num(info["action"], nan=0.0)
        metrics["ctrl_diff_sqr"] = ctrl_diff_sqr = jnp.sum(jnp.square(prev_action - action))
        cost = weight * ctrl_diff_sqr
        metrics["costs/control_diff"] = cost
        return -cost

    @_named_reward("energy_cost")
    def _energy_cost(self, data, info, metrics, current_target, weight, max_value) -> float:
        energy_use = jnp.sum(jnp.abs(data.qvel[self._joint_vel_idxs]) * jnp.abs(data.qfrc_actuator[self._joint_vel_idxs]))
        metrics["energy_use"] = energy_use
        cost = weight * jnp.minimum(energy_use, max_value)
        metrics["costs/energy"] = cost
        return -cost

    @_named_reward("jerk_cost")
    def _jerk_cost(self, data, info, metrics, current_target, weight, window_len) -> float:
        raise NotImplementedError("jerk_cost is not implemented")

    # Termination
    def _named_termination_criterion(name: str):
        def decorator(termination_fcn: Callable):
            _TERMINATION_FCN_REGISTRY[name] = termination_fcn
            return termination_fcn

        return decorator

    @_named_termination_criterion("root_too_far")
    def _root_too_far(self, data, info, current_target, max_distance) -> bool:
        target = current_target if current_target is not None else self._get_current_target(data, info)
        root_pos = self.root_body(data).xpos
        # pos_distance = target.qpos[..., :3] - root_pos
        # root_pos = data.qpos[:3]
        pos_distance = root_pos - target.qpos[...,:3]
        distance = jnp.sum((pos_distance) ** 2)
        return distance > max_distance

    @_named_termination_criterion("root_too_rotated")
    def _root_too_rotated(self, data, info, current_target, max_degrees) -> bool:
        target = current_target if current_target is not None else self._get_current_target(data, info)
        root_quat = self.root_body(data).xquat
        quat_distance = jnp.sum(
            _bounded_quat_dist(root_quat, target.qpos[..., 3:7]) ** 2
        )
        return quat_distance > max_degrees

    @_named_termination_criterion("pose_error")
    def _bad_pose(self, data, info, current_target, max_l2_error) -> bool:
        target = current_target if current_target is not None else self._get_current_target(data, info)
        # joints = self._get_joint_angles(data)
        joints = data.qpos[..., self._joint_reward_idxs]
        joint_diff = joints - target.qpos[..., self._joint_reward_idxs]
        pose_error = jnp.sum((joint_diff)**2)
        return pose_error > max_l2_error
    
    @_named_termination_criterion("nan_state")
    def _nan_state(self, data, info, current_target, unused) -> bool:
        flattened_vals, _ = jax.flatten_util.ravel_pytree(data)
        num_nans = jnp.sum(jnp.isnan(flattened_vals))
        return num_nans > 0
    
    @_named_termination_criterion("thorax_z_range")
    def _thorax_z_range_termination(self, data, info, current_target, enable_flight, healthy_z_range) -> float:
        if enable_flight:
            return False
        thorax_z = self._get_body_height(data)
        min_z, max_z = healthy_z_range
        in_range = jnp.logical_and(thorax_z >= min_z, thorax_z <= max_z)
        return ~in_range
    
    def verify_reference_data(self, atol: float = 5e-3) -> bool:
        """A set of non-exhaustive sanity checks that the reference data found in
        `config.REFERENCE_DATA_PATH` matches the environment's model. Most
        importantly, it verifies that the global coordinates of the
        body parts (xpos) match those the model produces when initialized to
        the corresponding qpos from the file. This can catch issues with nested
        free joints, mismatched joint orders, and incorrect scaling of the model
        but it's not exhaustive). This current implementation tests all
        frames of all clips in the reference data, and so is rather slow and
        does not have to be run every time.

        Args:
            atol (float): Absolute floating-point tolerance for the checks.
                          Defaults to 5e-3, because this seems to be the precision
                          of the reference data (TODO: why are there several mm
                          of error?).
        Returns:
            bool: True if all checks passed, False if any check failed.
        """

        def test_frame(clip_idx: int, frame: int) -> dict[str, bool]:
            data, _ = self._reset_data(clip_idx, frame, jax.random.PRNGKey(0), {})
            reference = self.reference_clips.extract_frame(clip_index=clip_idx, frame_index=frame)
            checks = collections.OrderedDict()
            checks["root_pos"] = jnp.allclose(
                self.root_body(data).xpos, reference.qpos[..., :3], atol=atol
            )
            checks["root_quat"] = jnp.allclose(
                self.root_body(data).xquat, reference.qpos[..., 3:7], atol=atol
            )
            checks["joints"] = jnp.allclose(
                self._get_joint_angles(data), reference.qpos[..., self._joint_reward_idxs], atol=atol
            )
            body_pos = self._get_bodies_pos(data, flatten=False)
            for idx, (body_name, body_pos) in enumerate(body_pos.items()):
                checks[f"body_xpos/{body_name}"] = jnp.allclose(
                    body_pos, reference.xpos[self._body_idxs[idx]], atol=atol
                )
            if self._config.qvel_init == "reference":
                checks["joints_ang_vel"] = jnp.allclose(
                    self._get_joint_ang_vels(data), reference.qvel[..., 6:], atol=atol
                )
            return checks

        @jax.jit
        def test_clip(clip_idx: int):
            return jax.vmap(test_frame, in_axes=(None, 0))(
                clip_idx, jnp.arange(self._clip_length(clip_idx))
            )

        clip_idxs = jnp.arange(self.reference_clips.num_clips)

        any_failed = False
        for clip in clip_idxs:
            if clip < 0 or clip >= self.reference_clips.qpos.shape[0]:
                raise ValueError(
                    f"Clip index {clip} is out of range. Reference"
                    f"data has {self.reference_clips.qpos.shape[0]} clips."
                )
            test_result = test_clip(clip)

            for name, result in test_result.items():
                n_failed = jnp.sum(np.logical_not(result))
                if n_failed > 0:
                    first_failed_frame = jnp.argmax(np.logical_not(result))
                    warnings.warn(
                        f"Reference data verification failed for {n_failed} frames"
                        f" for check '{name}' for clip {clip}."
                        f" First failure at frame {first_failed_frame}."
                    )
                    any_failed = True
        return not any_failed

    def verify_body_indexing(self, atol: float = 1e-3) -> bool:
        """Verify that body and end-effector indices correctly map between the
        simulation model and the stored reference data.

        Three checks are performed without any JAX compilation:

          1. Name round-trip: each entry in ``_body_idxs`` / ``_end_eff_idxs``
             is looked up in the MuJoCo model and the returned name must match
             ``body_name + self._suffix``.
          2. Index range: the maximum body / end-effector index must be
             strictly less than ``reference.xpos.shape[-2]`` so that the
             direct-indexing path used in reward functions never reads
             out-of-bounds or wraps into a wrong body.
          3. Cross-reference consistency on frame 0 of clip 0.  A clean
             forward pass is used (``qpos = reference.qpos``, zeroed qvel,
             ``mjx.forward``) rather than ``_reset_data``, because
             ``_reset_data`` modifies qpos before kinematics (z-floor clamp
             + wing retraction) introducing ~3 mm legitimate differences
             that are unrelated to index correctness.

             Before the forward pass, joint qpos ordering is verified using
             ``reference.qpos_names``: each ``self._joint_idxs[i]`` is used
             directly to index ``reference.qpos`` in reward and observation
             functions (e.g. ``target.qpos[..., self._joint_idxs]``), so
             ``reference.qpos_names[joint_idx]`` must match the expected
             joint name.  A mismatch means joint rewards and the joint
             component of the imitation observation are comparing wrong
             joints.

               a. ``data.xpos[_body_idxs]`` (reward path) must equal
                  ``_get_bodies_pos(data)`` (spec-binding path used by
                  ``verify_reference_data``).  A disagreement here means the
                  two verification methods are measuring different things.
               b. ``data.xpos[_body_idxs]`` must match
                  ``reference.xpos[_body_idxs]``.  A failure here means the
                  rewards are comparing the sim against the wrong body
                  positions in the reference.
               c. Same as (b) but for end effectors.

        Args:
            atol (float): Absolute tolerance for floating-point comparisons.
                          Defaults to 1e-3.
        Returns:
            bool: True if every check passed, False otherwise.
        """
        all_passed = True

        # ── Check 1: name round-trip ─────────────────────────────────────────
        print("verify_body_indexing | Check 1: body index name round-trip...")
        for body_name, body_idx in zip(self._config.body_names, self._body_idxs):
            expected = body_name + self._suffix
            actual = self._mj_model.body(int(body_idx)).name
            if actual != expected:
                warnings.warn(
                    f"Body index mismatch: idx={int(body_idx)} → '{actual}' "
                    f"but expected '{expected}'"
                )
                all_passed = False
        for end_eff_name, end_eff_idx in zip(self._config.end_eff_names, self._end_eff_idxs):
            expected = end_eff_name + self._suffix
            actual = self._mj_model.body(int(end_eff_idx)).name
            if actual != expected:
                warnings.warn(
                    f"End-effector index mismatch: idx={int(end_eff_idx)} → '{actual}' "
                    f"but expected '{expected}'"
                )
                all_passed = False
        if all_passed:
            print("  ✓ All body and end-effector names match their indices.")

        # Print a full inventory and flag any wing names that leaked in.
        print("\n  Body parts used in reward/observation calculations:")
        print(f"  {'─'*60}")
        print(f"  bodies_pos / imitation target body ({len(self._config.body_names)} bodies):")
        wing_in_bodies, wing_in_end_eff = [], []
        for name, idx in zip(self._config.body_names, self._body_idxs):
            is_wing = 'wing' in name.lower()
            flag = '  ⚠ WING' if is_wing else ''
            print(f"    [{int(idx):3d}] {name}{flag}")
            if is_wing:
                wing_in_bodies.append(name)
        print(f"  end_eff reward ({len(self._config.end_eff_names)} bodies):")
        for name, idx in zip(self._config.end_eff_names, self._end_eff_idxs):
            is_wing = 'wing' in name.lower()
            flag = '  ⚠ WING' if is_wing else ''
            print(f"    [{int(idx):3d}] {name}{flag}")
            if is_wing:
                wing_in_end_eff.append(name)
        print(f"  {'─'*60}")
        if wing_in_bodies:
            warnings.warn(
                f"Wing bodies found in body_names (used in bodies_pos reward and "
                f"imitation target): {wing_in_bodies}. Wings should not be tracked "
                f"as imitation targets because their motion is decoupled from qpos "
                f"(driven by WBPG/reset overrides)."
            )
            all_passed = False
        if wing_in_end_eff:
            warnings.warn(
                f"Wing bodies found in end_eff_names (used in end_eff reward): "
                f"{wing_in_end_eff}."
            )
            all_passed = False
        if not wing_in_bodies and not wing_in_end_eff:
            print("  ✓ No wing bodies in body_names or end_eff_names.")

        # ── Check 2: index range vs reference.xpos shape ─────────────────────
        print("verify_body_indexing | Check 2: index range vs reference xpos shape...")
        ref_n_bodies = self.reference_clips.xpos.shape[-2]  # (..., n_bodies, 3)
        max_body_idx = int(jnp.max(self._body_idxs))
        max_end_eff_idx = int(jnp.max(self._end_eff_idxs))
        check2_passed = True
        if max_body_idx >= ref_n_bodies:
            warnings.warn(
                f"_body_idxs max={max_body_idx} is out of range for reference.xpos "
                f"which only has {ref_n_bodies} bodies along axis -2."
            )
            all_passed = False
            check2_passed = False
        else:
            print(f"  ✓ _body_idxs OK  (max={max_body_idx} < ref n_bodies={ref_n_bodies})")
        if max_end_eff_idx >= ref_n_bodies:
            warnings.warn(
                f"_end_eff_idxs max={max_end_eff_idx} is out of range for reference.xpos "
                f"which only has {ref_n_bodies} bodies along axis -2."
            )
            all_passed = False
            check2_passed = False
        else:
            print(f"  ✓ _end_eff_idxs OK (max={max_end_eff_idx} < ref n_bodies={ref_n_bodies})")

        # ── Check 3: cross-reference consistency (single frame, no JIT) ──────
        # We deliberately do NOT use _reset_data here because that method
        # modifies qpos before running forward kinematics (z-floor clamp and
        # wing retraction to defaults), which introduces legitimate position
        # differences of ~3 mm that have nothing to do with index correctness.
        # Instead we set qpos = reference.qpos exactly and run a clean forward
        # pass, so the only source of disagreement can be a wrong body index.
        print("verify_body_indexing | Check 3: cross-reference consistency on clip 0, frame 0...")
        if not check2_passed:
            warnings.warn(
                "Skipping Check 3 because Check 2 failed (indices are out of bounds)."
            )
            return False

        reference = self.reference_clips.extract_frame(clip_index=0, frame_index=0)

        # 3-pre: joint qpos ordering check.
        # self._joint_idxs are qpos slot numbers derived from the *sim* model.
        # They are used as-is to index reference.qpos in rewards and observations
        # (e.g. target.qpos[..., self._joint_idxs]).  For this to be correct the
        # reference must have been preprocessed from a model with the same qpos
        # ordering.  qpos_names gives us a ground-truth label for each slot in
        # the reference, so we can verify the mapping directly.
        if reference.qpos_names is not None:
            print("  Checking joint qpos ordering against reference.qpos_names...")
            qpos_check_passed = True
            _wing_joints = set(self._config.wing_names)
            joint_reward_names = [j for j in self._config.joint_names if j not in _wing_joints]
            for joint_name, joint_idx in zip(joint_reward_names, self._joint_reward_idxs):
                idx = int(joint_idx)
                if idx >= len(reference.qpos_names):
                    warnings.warn(
                        f"joint_idx={idx} for '{joint_name}' is out of range "
                        f"for reference.qpos_names (len={len(reference.qpos_names)})."
                    )
                    all_passed = False
                    qpos_check_passed = False
                    continue
                ref_name = reference.qpos_names[idx]
                # The reference may have been generated without self._suffix; strip
                # it from the reference name if present before comparing.
                ref_name_stripped = ref_name.removesuffix(self._suffix) if self._suffix else ref_name
                if ref_name_stripped != joint_name:
                    warnings.warn(
                        f"Joint qpos mismatch: sim joint '{joint_name}' uses "
                        f"qpos idx={idx}, but reference.qpos_names[{idx}]='{ref_name}' "
                        f"(stripped: '{ref_name_stripped}'). "
                        f"Joint rewards/observations may compare wrong joints."
                    )
                    all_passed = False
                    qpos_check_passed = False
            if qpos_check_passed:
                print("  ✓ All joint qpos indices match reference.qpos_names.")
        else:
            print("  (skipping joint qpos name check: reference.qpos_names is None)")

        # Clean forward pass: set qpos from reference, zero qvel, run kinematics.
        data = mjx.make_data(self.mj_model, impl=self._config.mujoco_impl,
                             naconmax=self._config.nconmax, njmax=self._config.njmax)
        data = data.replace(qpos=reference.qpos, qvel=jnp.zeros(self.mjx_model.nv))
        data = mjx.forward(self.mjx_model, data)

        # 3a — reward path vs spec-binding path (should always agree if Check 1 passes)
        direct_body_pos = np.array(data.xpos[self._body_idxs])   # reward path
        spec_body_pos = np.array(jnp.stack([                       # spec-binding path
            data.bind(self.mjx_model, self._spec.body(f"{name}{self._suffix}")).xpos
            for name in self._config.body_names
        ]))
        if not np.allclose(direct_body_pos, spec_body_pos, atol=atol):
            max_diff = float(np.max(np.abs(direct_body_pos - spec_body_pos)))
            warnings.warn(
                f"3a FAILED: reward path (data.xpos[_body_idxs]) disagrees with "
                f"spec-binding path (_get_bodies_pos). Max diff: {max_diff:.2e}. "
                f"verify_reference_data() can pass while rewards are still wrong."
            )
            all_passed = False
        else:
            print("  ✓ 3a: reward path and spec-binding path agree.")

        # 3b — reward path vs reference xpos (body positions)
        ref_body_pos = np.array(reference.xpos[self._body_idxs])
        if not np.allclose(direct_body_pos, ref_body_pos, atol=atol):
            all_passed = False
            print("  ✗ 3b: reward body positions do NOT match reference. Per-body breakdown:")
            for i, body_name in enumerate(self._config.body_names):
                diff = float(np.max(np.abs(direct_body_pos[i] - ref_body_pos[i])))
                status = "FAIL" if diff > atol else "ok  "
                print(f"       [{status}] '{body_name}' (idx={int(self._body_idxs[i])}): "
                      f"sim={direct_body_pos[i]}, ref={ref_body_pos[i]}, max_diff={diff:.2e}")
        else:
            print("  ✓ 3b: reward body positions match reference.")

        # 3c — reward path vs reference xpos (end effectors)
        direct_end_eff_pos = np.array(data.xpos[self._end_eff_idxs])
        ref_end_eff_pos = np.array(reference.xpos[self._end_eff_idxs])
        if not np.allclose(direct_end_eff_pos, ref_end_eff_pos, atol=atol):
            all_passed = False
            print("  ✗ 3c: reward end-effector positions do NOT match reference. Per-body breakdown:")
            for i, name in enumerate(self._config.end_eff_names):
                diff = float(np.max(np.abs(direct_end_eff_pos[i] - ref_end_eff_pos[i])))
                status = "FAIL" if diff > atol else "ok  "
                print(f"       [{status}] '{name}' (idx={int(self._end_eff_idxs[i])}): "
                      f"sim={direct_end_eff_pos[i]}, ref={ref_end_eff_pos[i]}, max_diff={diff:.2e}")
        else:
            print("  ✓ 3c: reward end-effector positions match reference.")

        if all_passed:
            print("verify_body_indexing | ✓ All checks passed.")
        else:
            warnings.warn(
                "verify_body_indexing: one or more checks FAILED. "
                "Body rewards may be comparing the wrong body parts. "
                "Review the warnings and per-body output above."
            )

        n_frames = 10
        clip_idx = 0

        frames_qpos = []
        for fi in range(n_frames):
            frame = self.reference_clips.extract_frame(clip_index=clip_idx, frame_index=fi)
            frames_qpos.append(np.array(frame.qpos))

        qpos_stack = np.stack(frames_qpos, axis=0)  # (10, nq)
        print(f"qpos shape per frame: {qpos_stack.shape[1:]}")
        print(f"qpos stack shape: {qpos_stack.shape}")

        always_zero = np.all(qpos_stack == 0, axis=0)
        zero_indices = np.where(always_zero)[0]
        print(f"\nIndices always 0 across {n_frames} frames: {zero_indices.tolist()}")
        print(f"Number always zero: {len(zero_indices)}")
        print(f"\nPer-index values across frames:")
        for i in range(qpos_stack.shape[1]):
            vals = qpos_stack[:, i]
            marker = "  <-- ALWAYS ZERO" if always_zero[i] else ""
            print(f"  qpos[{i:3d}]: {np.array2string(vals, precision=4, suppress_small=True)}{marker}")
        return all_passed

    def _calc_obs_size(self) -> int:
        joint_size = len(self._joint_idxs)
        joint_vel_size = len(self._joint_vel_idxs)
        body_size = len(self._body_idxs)
        root_size = 3 + 4  # pos + quat
        # Proprioception components:
        # - joint_angles: joint_size
        # - joint_ang_vels: joint_vel_size
        # - actuator_ctrl: len(data.qfrc_actuator[self._joint_vel_idxs]) = joint_vel_size (joint DOFs only)
        # - body_height: 1 (scalar)
        # - world_zaxis: 3 (z-axis vector from rotation matrix)
        # - appendages_pos: 3 * num_appendages (3D pos for each end effector)
        num_appendages = len(self._config.end_eff_names)
        proprioception_size = joint_size + joint_vel_size + joint_vel_size + 1 + 3 + (3 * num_appendages)
        task_obs_size = (root_size + joint_size + body_size*3) * self._config.ref_len
        return task_obs_size, proprioception_size
    
    def before_step(self, data, action, info):
        """Combine action with WPG base pattern."""
        # Get target wing joint angles at beat frequency requested by the agent.
        base_freq, rel_range = self._wbpg.data.base_beat_freq, self._wbpg.data.rel_freq_range
        act = jnp.clip(action[-1], -1.0, 1.0) # action in [-1, 1].
        ctrl_freq = base_freq * (1 + rel_range * act)
        wbpg_state, ctrl = self._wbpg.step(info["wbpg_state"], ctrl_freq=ctrl_freq)

        # Get the current wing joint angles.
        length = data.qpos[self._wing_joint_idxs]
        # jax.debug.print("length: {length}", length=length)
        # Convert position control to force control.
        full_action = action.at[self._wing_actuator_idxs].set(action[self._wing_actuator_idxs] + (ctrl - length))
        full_action = full_action[:-1].copy()
        
        # Save updated wbpg_state back to info for next step
        info["wbpg_state"] = wbpg_state
        return full_action, info
    
    def _step_with_aero(self, data: mjx.Data, action: jax.Array, n_steps: int) -> tuple[mjx.Data, jax.Array]:
        """Physics step with aerodynamic forces for flight behavior."""
        # Apply control action

        data = data.replace(ctrl=action)

        # Initialize force_components with xyz zeros for each of the 4 force components
        dummy_force_components = (jnp.zeros_like(data.geom_xpos),)*4

        # Step physics multiple times with aerodynamic forces
        def step_fn(carry, _):
            data_inner, _ = carry

            # Compute aerodynamic forces based on wing motion and body velocity
            fluid_force, force_components = self._compute_aero_forces(data_inner)
            # Apply aerodynamic forces as external forces
            data_inner = data_inner.replace(qfrc_fluid=fluid_force)

            # Forward dynamics step
            data_inner = mjx.step(self.mjx_model, data_inner)
            return (data_inner, force_components), force_components

        (data, final_force_components), _ = jax.lax.scan(
            step_fn, (data, dummy_force_components), None, length=n_steps
        )

        return data, final_force_components

    def _compute_aero_forces(self, data: mjx.Data) -> jax.Array:
        """Compute aerodynamic forces on wing segments."""
        if self._quasi_aero:
            fluid_force, force_components = compute_apply_all_fluid_forces_to_state(self._mjx_model, data)
            # print('Using quasi-steady aerodynamic forces')
        else:
            fluid_force, force_components = compute_apply_ellipsoid_fluid_forces(self._mjx_model, data)
            # print('Using ellipsoid fluid forces')

        return fluid_force, force_components

    def render_ghost(
        self,
        trajectory: Union[List[mjx_env.State], 'Rollout', tuple],
        height: int = 240,
        width: int = 320,
        clip_idx: Optional[int] = None,
        camera: Optional[str] = None,
        scene_option: Optional[mujoco.MjvOption] = None,
        modify_scene_fns: Optional[Sequence[Callable[[mujoco.MjvScene], None]]] = None,
        add_ghost: bool = True,
        add_labels=False,
        termination_extra_frames=0,
        video_path: str = "imitation_render.mp4",
        vid_fps: int = 50,
        verbose: bool = True,
        colorize_dict: Optional[dict] = None,
        max_frames: Optional[int] = None,
        interp_factor: int = 1,
    ) -> Sequence[np.ndarray]:
        """
        Renders a sequence of states (trajectory). The video includes the imitation
        target as a white transparent "ghost".

        Args:
            trajectory: Can be one of:
                - List[mjx_env.State]: Sequence of environment states to render
                - Rollout: Rollout object from HDF5RolloutSaver
                - tuple: (qpos, qvel) arrays with shape (n_timesteps, n_qpos/n_qvel)
            height (int, optional): Height of the rendered frames in pixels. Defaults to 240.
            width (int, optional): Width of the rendered frames in pixels. Defaults to 320.
            clip_idx (int, optional): Clip index to use. Required for Rollout or tuple input.
            camera (str, optional): Camera name or index to use for rendering.
            scene_option (mujoco.MjvOption, optional): Additional scene rendering options.
            modify_scene_fns (Sequence[Callable[[mujoco.MjvScene], None]], optional):
                Sequence of functions to modify the scene before rendering each frame.
                Defaults to None.
            add_labels (bool, optional): Whether to overlay clip and termination cause
                labels on frames. Defaults to False. Only works with State trajectory.
            termination_extra_frames (int, optional): If larger than 0, then repeat the
                frame triggering the termination this number of times. This gives
                a freeze-on-done effect that may help debug termination criteria.
                Additionally, a simple fade-out effect is applied during those frames
                to smooth the tranisition between clips. If this is larger than 0, the
                number of returned frames might be larger than `len(trajectory)`.
                Only works with State trajectory.
            interp_factor (int, optional): Number of sub-frames to render between each
                pair of original frames. When > 1, qpos/qvel are linearly interpolated
                (with SLERP for the root quaternion) and the video fps is scaled by
                this factor so playback speed stays the same. Defaults to 1 (no
                interpolation).
        Returns:
            Sequence[np.ndarray]: List of rendered frames as numpy arrays.
        """
        import imageio
        # Determine input type and extract data
        # Use duck typing instead of isinstance to avoid import path issues
        # Check if trajectory has Rollout-like attributes (qpos, qvel)
        is_rollout = (
            hasattr(trajectory, 'qpos') and 
            hasattr(trajectory, 'qvel') and
            not isinstance(trajectory, tuple) and
            not isinstance(trajectory, list)
        )
        is_tuple = isinstance(trajectory, tuple) and len(trajectory) == 2
        is_states = not is_rollout and not is_tuple

        if is_rollout:
            # Rollout object
            qpos_array = np.array(trajectory.qpos)
            qvel_array = np.array(trajectory.qvel)
            n_timesteps = qpos_array.shape[0]
            # Get clip_idx from rollout info or use provided
            if clip_idx is None:
                if trajectory.info and 'reference_clip' in trajectory.info:
                    clip_idx = int(trajectory.info['reference_clip'][0])
                else:
                    raise ValueError("clip_idx must be provided for Rollout without reference_clip info")
            # Disable state-dependent features
            termination_extra_frames = 0
        elif is_tuple:
            # Tuple of (qpos, qvel) arrays
            qpos_array, qvel_array = trajectory
            n_timesteps = qpos_array.shape[0]
            if clip_idx is None:
                raise ValueError("clip_idx must be provided when trajectory is a tuple of arrays")
            # Disable state-dependent features
            termination_extra_frames = 0
        else:
            # List of States
            qpos_array = None
            qvel_array = None
            n_timesteps = len(trajectory)
            if clip_idx is None and len(trajectory) > 0:
                clip_idx = trajectory[0].info["reference_clip"]

        # Create a new spec with a ghost, without modifying the existing one
        if add_ghost:
            spec_pair = self.add_ghost_fly(only_T1=self._config.only_T1)
        else:
            spec_pair = self._spec.copy()

        # Extract reference trajectory for visualization
        traj = self.reference_clips.extract_clip(clip_idx)
        spec_pair = add_cross_hair_sites(spec_pair, init_pos=traj.qpos[0,:3])
        spec_pair = add_trajectory_sites_spheres(spec=spec_pair,
                                        n_traj_sites=traj.clip_lengths,
                                        traj=traj.qpos[:,:3])
        spec_pair.add_exclude(name='wings',  bodyname1="wing_left"+self._suffix, bodyname2="wing_right"+self._suffix)
        # Add a tracking camera via spec
        cam = spec_pair.worldbody.add_camera(
            name="follow",
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
            pos=[-0.5, 0, 0.6],
            # pos=[-0.3, 0, 0.2],
            xyaxes=[0, -1, 0, 
                    0.45, 0, 0.3],
        )

        cam = spec_pair.worldbody.add_camera(
            name="static_lookat",
            mode=mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY,
            pos=[0.5, 0.4, 0.15],
            targetbody="thorax-fly",
        )

        cam = spec_pair.worldbody.add_camera(
            name="static_side",
            mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
            pos=[0.5, 0.3, 0.1],   # world position
            xyaxes=[0, -1, 0,
                    0, 0, 1],
        )

        mj_model = spec_pair.compile()
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height

        # --- PATCH FLOOR + SKYBOX DIRECTLY ON THE COMPILED MODEL ---
        # Floor texture colors (checker pattern RGB values in tex_data)
        floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        mat_id = mj_model.geom_matid[floor_id]
        tex_id = int(mj_model.mat_texid[mat_id, 1])  # <-- channel 1, not 0!

        if tex_id >= 0:
            w = int(mj_model.tex_width[tex_id])
            h = int(mj_model.tex_height[tex_id])
            adr = int(mj_model.tex_adr[tex_id])
            f = 1.5
            c1 = np.array([15/f, 10/f, 25/f], dtype=np.uint8)   # dark purple-black
            c2 = np.array([38, 30, 55], dtype=np.uint8)   # slightly lighter purple
            # c2 = np.array([0, 0, 0], dtype=np.uint8)        # black zenith
            for row in range(h):
                for col in range(w):
                    is_c1 = ((row * 2 // h) + (col * 2 // w)) % 2 == 0
                    px = adr + (row * w + col) * 3
                    mj_model.tex_data[px:px+3] = c1 if is_c1 else c2
            mj_model.mat_texrepeat[mat_id] = [90, 90]
            mj_model.mat_reflectance[mat_id] = 0

        # Patch skybox to dark
        sky_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_TEXTURE, 'skybox')
        if sky_id >= 0:
            sw = int(mj_model.tex_width[sky_id])
            sh = int(mj_model.tex_height[sky_id])
            sadr = int(mj_model.tex_adr[sky_id])
            f = 1.03
            # sky_top = np.array([0, 0, 0], dtype=np.uint8)        # black zenith
            sky_top =  np.array([13*f, 13*f, 26*f], dtype=np.uint8)  # dark blue horizon
            sky_horizon = np.array([13*f, 13*f, 26*f], dtype=np.uint8)  # dark blue horizon
            for row in range(sh):
                t = row / max(sh - 1, 1)
                color = (sky_top * (1 - t) + sky_horizon * t).astype(np.uint8)
                for col in range(sw):
                    px = sadr + (row * sw + col) * 3
                    mj_model.tex_data[px:px+3] = color
        # --- END PATCH ---

        # --- DEBUG: verify XML asset changes ---
        floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        mat_id = mj_model.geom_matid[floor_id]
        tex_id = mj_model.mat_texid[mat_id, 1] if mat_id >= 0 else -1  # was [mat_id, 0]
        print(f"floor geom_id={floor_id}, mat_id={mat_id}, tex_id={tex_id}")
        print(f"floor rgba={mj_model.geom_rgba[floor_id]}")
        if mat_id >= 0:
            print(f"mat texrepeat={mj_model.mat_texrepeat[mat_id]}")
            print(f"mat reflectance={mj_model.mat_reflectance[mat_id]}")
        if tex_id >= 0:
            tex_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_TEXTURE, tex_id)
            print(f"tex name={tex_name}, width={mj_model.tex_width[tex_id]}, height={mj_model.tex_height[tex_id]}")
            print(f"tex type={mj_model.tex_type[tex_id]}")  # 0=2d, 1=cube, 2=skybox

        # check skybox
        sky_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_TEXTURE, 'skybox')
        print(f"skybox tex_id={sky_id}")
        if sky_id >= 0:
            print(f"skybox width={mj_model.tex_width[sky_id]}, height={mj_model.tex_height[sky_id]}")
        # --- END DEBUG ---

        # colorize the mjco model for better visualization if colorize_dict is provided
        if colorize_dict is not None:
            def hex_to_rgba(hex_color, alpha=1.0):
                h = hex_color.lstrip('#')
                r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
                return [r, g, b, alpha]

            # Get the exact lookup if passed alongside colorize_dict
            abdomen_exact = colorize_dict.pop('__abdomen_exact__', {})

            for geom_id in range(mj_model.ngeom):
                geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                if geom_name is None:
                    continue

                # Check exact match first (for abdomen gradient)
                if geom_name in abdomen_exact:
                    hex_color, alpha = abdomen_exact[geom_name]
                    mj_model.geom_rgba[geom_id] = hex_to_rgba(hex_color, alpha)
                    continue

                # Fall back to substring matching
                for group_name, (hex_color, alpha, keywords) in colorize_dict.items():
                    if any(kw in geom_name for kw in keywords):
                        mj_model.geom_rgba[geom_id] = hex_to_rgba(hex_color, alpha)
                        break
        mj_data = mujoco.MjData(mj_model)

        renderer = mujoco.Renderer(mj_model, height=height, width=width)
        if camera is None:
            camera = -1

        rendered_frames = []
        if max_frames is not None:
            n_timesteps = min(n_timesteps, max_frames)
        # Create video writer only if video_path is provided
        # Helpers for sub-frame interpolation
        def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
            """Spherical linear interpolation between two unit quaternions."""
            dot = np.dot(q1, q2)
            if dot < 0.0:          # shortest-path convention
                q2, dot = -q2, -dot
            dot = float(np.clip(dot, -1.0, 1.0))
            if dot > 0.9995:       # nearly identical – fall back to normalised lerp
                result = q1 + t * (q2 - q1)
                return result / np.linalg.norm(result)
            theta_0 = np.arccos(dot)
            theta = theta_0 * t
            sin_theta = np.sin(theta)
            sin_theta_0 = np.sin(theta_0)
            s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s2 = sin_theta / sin_theta_0
            return s1 * q1 + s2 * q2

        def _lerp_qpos(qp1: np.ndarray, qp2: np.ndarray, t: float) -> np.ndarray:
            """Linear interpolation of qpos, with SLERP for root quaternion (indices 3:7)."""
            result = qp1 + t * (qp2 - qp1)
            result[3:7] = _slerp(qp1[3:7], qp2[3:7], t)
            return result

        video_writer = imageio.get_writer(video_path, fps=vid_fps * interp_factor) if video_path is not None else None
        try:
            if verbose:
                from tqdm.auto import tqdm
                iterator = tqdm(range(n_timesteps))
            else:
                iterator = range(n_timesteps)

            for i in iterator:
                if is_rollout or is_tuple:
                    # Use arrays directly
                    qpos = qpos_array[i]
                    qvel = qvel_array[i]
                    # For rollout/tuple, use frame index = i
                    frame = i
                    clip = clip_idx
                    ref = self.reference_clips.extract_frame(clip_index=clip, frame_index=frame)
                    state = None  # No state object available
                else:
                    # Use State objects
                    state = trajectory[i]
                    qpos = state.data.qpos
                    qvel = state.data.qvel
                    time_in_frames = state.data.time * self._config.mocap_hz
                    frame = jnp.round(time_in_frames + state.info["start_frame"]).astype(int)
                    clip = state.info["reference_clip"] if clip_idx is None else clip_idx
                    ref = self.reference_clips.extract_frame(clip_index=clip, frame_index=frame)

                # Fetch the next frame's data for sub-frame interpolation
                if interp_factor > 1 and i < n_timesteps - 1:
                    if is_rollout or is_tuple:
                        next_qpos = np.array(qpos_array[i + 1])
                        next_qvel = np.array(qvel_array[i + 1])
                        next_ref = self.reference_clips.extract_frame(clip_index=clip, frame_index=i + 1)
                    else:
                        _ns = trajectory[i + 1]
                        next_qpos = np.array(_ns.data.qpos)
                        next_qvel = np.array(_ns.data.qvel)
                        _nf = int(jnp.round(_ns.data.time * self._config.mocap_hz + _ns.info["start_frame"]))
                        next_ref = self.reference_clips.extract_frame(clip_index=clip, frame_index=_nf)
                else:
                    next_qpos = np.array(qpos)
                    next_qvel = np.array(qvel)
                    next_ref = ref

                qpos_np = np.array(qpos)
                qvel_np = np.array(qvel)
                ref_qpos_np = np.array(ref.qpos)
                ref_qvel_np = np.array(ref.qvel)

                for sub_i in range(interp_factor):
                    alpha = sub_i / interp_factor if interp_factor > 1 else 0.0
                    if alpha > 0.0:
                        i_qpos = _lerp_qpos(qpos_np.copy(), next_qpos, alpha)
                        i_qvel = qvel_np + alpha * (next_qvel - qvel_np)
                        i_ref_qpos = _lerp_qpos(ref_qpos_np.copy(), np.array(next_ref.qpos), alpha)
                        i_ref_qvel = ref_qvel_np + alpha * (np.array(next_ref.qvel) - ref_qvel_np)
                    else:
                        i_qpos = qpos_np
                        i_qvel = qvel_np
                        i_ref_qpos = ref_qpos_np
                        i_ref_qvel = ref_qvel_np

                    if add_ghost:
                        mj_data.qpos = np.concatenate((i_qpos, i_ref_qpos))
                        mj_data.qvel = np.concatenate((i_qvel, i_ref_qvel))
                    else:
                        mj_data.qpos = i_qpos
                        mj_data.qvel = i_qvel

                    mujoco.mj_forward(mj_model, mj_data)
                    renderer.update_scene(
                        mj_data, camera=camera, scene_option=scene_option
                    )

                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                    if modify_scene_fns is not None:
                        for k in range(len(modify_scene_fns)):
                            # Pass additional info for Rollout rendering
                            if is_rollout or is_tuple:
                                # For Rollout/tuple, pass rollout info, timestep, and geom positions
                                rollout_info = trajectory.info if is_rollout else None
                                modify_scene_fns[k](
                                    scene=renderer.scene,
                                    state=state,
                                    rollout_info=rollout_info,
                                    timestep_idx=i,
                                    geom_xpos=mj_data.geom_xpos
                                )
                            else:
                                # For States, use original signature
                                modify_scene_fns[k](scene=renderer.scene, state=state)

                    rendered_frame = renderer.render()

                    # Add labels for both State and Rollout
                    if add_labels:
                        import cv2

                        # Add clip label
                        label = f"Clip {clip} "
                        cv2.putText(
                            rendered_frame,
                            label,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    rendered_frames.append(rendered_frame)

                    # Write to video only if video_writer exists
                    if video_writer is not None:
                        video_writer.append_data(rendered_frame)

                # Check for termination (works for both State and Rollout)
                # Evaluated once per original frame, not per sub-frame
                is_done = False
                truncated = False
                termination_reason = "<Unknown>"

                if state is not None and hasattr(state, 'done'):
                    # Using State object
                    is_done = state.done
                    if is_done:
                        truncated = state.info.get("truncated", False)
                        # Find termination reason from metrics
                        for name in self._config.termination_criteria.keys():
                            if state.metrics.get("terminations/" + name, 0) > 0:
                                termination_reason = name
                                break
                        if truncated:
                            termination_reason = "truncated"
                elif is_rollout:
                    # Using Rollout object - check termination at this timestep
                    # Check if any termination metric is > 0 at this timestep
                    if trajectory.metrics is not None:
                        for name in self._config.termination_criteria.keys():
                            metric_key = "terminations/" + name
                            if metric_key in trajectory.metrics:
                                metric_val = trajectory.metrics[metric_key]
                                val = metric_val[i] if hasattr(metric_val, '__len__') and len(metric_val) > i else metric_val
                                if val > 0:
                                    is_done = True
                                    termination_reason = name
                                    break

                    # Check truncation
                    if trajectory.info is not None and 'truncated' in trajectory.info:
                        truncated_data = trajectory.info['truncated']
                        truncated = bool(truncated_data[i] if hasattr(truncated_data, '__len__') and len(truncated_data) > i else truncated_data)
                        if truncated:
                            is_done = True
                            termination_reason = "truncated"

                # Add termination labels and fade-out
                if is_done:
                    if add_labels:
                        import cv2
                        cv2.putText(
                            rendered_frame,
                            termination_reason,
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    for t in range(termination_extra_frames):
                        rel_t = t / termination_extra_frames
                        fade_factor = 1 / (
                            1 + np.exp(10 * (rel_t - 0.5))
                        )  # Logistic fade-out
                        faded_frame = (rendered_frame * fade_factor).astype(np.uint8)
                        rendered_frames.append(faded_frame)
        finally:
            # Ensure video writer is properly closed
            if video_writer is not None:
                video_writer.close()
        return rendered_frames

def _assert_all_are_prefix(a, b, a_name="a", b_name="b"):
    if isinstance(a, map):
        a = list(a)
    if isinstance(b, map):
        b = list(b)
    if len(a) != len(b):
        raise AssertionError(
            f"{a_name} has length {len(a)} but {b_name} has length {len(b)}."
        )
    for a_el, b_el in zip(a, b):
        if not b_el.startswith(a_el):
            raise AssertionError(
                f"Comparing {a_name} and {b_name}. Expected {a_el} to match {b_el}."
            )
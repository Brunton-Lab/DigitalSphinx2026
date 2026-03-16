"""Base classes for fruitfly"""

from typing import Any, Dict, Optional, Union, Mapping
import collections

import logging
import mujoco
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco.mjx._src import math as mj_math
from sphinx_training.utils.utils import change_body_frame, neg_quat


from .env_utils import _scale_body_tree, _recolour_tree, dm_scale_spec
from .constants import *


def get_assets(xml_paths) -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, xml_paths, "*.xml")
    mjx_env.update_assets(assets, xml_paths / "assets")
    return assets


def default_config(cfg) -> config_dict.ConfigDict:
    return config_dict.create(
        walker_xml_path=cfg.mjcf_path,
        arena_xml_path=cfg.arena_xml_path,
        sim_dt=cfg.physics_timestep,
        ctrl_dt=cfg.physics_steps_per_control_step * cfg.physics_timestep,
        solver="cg",
        iterations=4,
        ls_iterations=4,
        noslip_iterations=0,
        mujoco_impl="jax"
    )


class FruitflyEnv(mjx_env.MjxEnv):
    """Base class for fruitfly environments."""

    def __init__(
        self,
        config,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        """
        Initialize the FruitflyEnv class with only arena

        Args:
            config (config_dict.ConfigDict): Configuration dictionary for the environment.
            config_overrides (Optional[Dict[str, Union[str, int, list[Any]]]], optional): Optional overrides for the configuration. Defaults to None.
            compile_spec (bool, optional): Whether to compile the model. Defaults to False.
        """
        super().__init__(config, config_overrides)
        self._walker_xml_path = str(config._walker_xml_path)
        self._arena_xml_path = str(config._arena_xml_path)
        self._spec = mujoco.MjSpec.from_file(str(config._arena_xml_path))
        floor_height = getattr(config,'floor_height', -0.125)
        self._spec.geom('floor').pos = [0,0,floor_height]
        self._compiled = False
        self._enable_flight = config.enable_flight
        self._enable_wbpg = getattr(config, 'enable_wbpg', True)


    def filter_model_to_config_joints(self, spec: Optional[mujoco.MjSpec] = None, xml_path: Optional[str] = None):
        """Remove joints, actuators, tendons, and sensors not matching
        ``config.joint_names`` from *spec*.

        Only joints whose name appears in ``self._config.joint_names`` are kept.
        Actuators are kept when their name matches a kept joint.  Tendons are
        kept when any kept joint name is a substring of the tendon name or
        vice-versa (handles coupled tendons like ``abduct_abdomen``).  Sensors
        are kept when any kept joint name appears in the sensor name, or the
        sensor has no joint reference (global sensors like accelerometer/gyro).

        Args:
            spec: Optional MjSpec to filter. If None, will load from xml_path
                  or ``self._walker_xml_path``.
            xml_path: Optional path to MuJoCo XML file. Used if spec is None.

        Returns:
            Filtered MjSpec.
        """
        if spec is None:
            path = xml_path if xml_path is not None else self._walker_xml_path
            spec = mujoco.MjSpec.from_file(path)

        allowed_joints = set(self._config.joint_names)

        # --- Joints ---
        joints_to_delete = []
        for body in spec.bodies:
            for joint in body.joints:
                if joint.name not in allowed_joints:
                    joints_to_delete.append(joint)

        # --- Actuators (names match joint names in this model) ---
        actuators_to_delete = []
        for actuator in spec.actuators:
            if actuator.name not in allowed_joints:
                actuators_to_delete.append(actuator)

        # --- Tendons (coupled joints — keep if any allowed joint name is a
        #     substring of the tendon name or vice-versa) ---
        tendons_to_delete = []
        for tendon in spec.tendons:
            keep = any(
                jname in tendon.name or tendon.name in jname
                for jname in allowed_joints
            )
            if not keep:
                tendons_to_delete.append(tendon)

        # --- Sensors (keep global sensors + those referencing allowed joints) ---
        # Global sensors (accelerometer, gyro, velocimeter) have no body/joint
        # reference in their name so we keep them.  Body-specific sensors
        # contain a body segment identifier (e.g. ``force_tarsus_T1_left``).
        sensors_to_delete = []
        for sensor in spec.sensors:
            # Keep if any allowed joint name appears as substring in sensor name
            keep = any(jname in sensor.name for jname in allowed_joints)
            # Also keep global sensors (no underscore-separated body reference)
            if not keep and '_' not in sensor.name:
                keep = True
            if not keep:
                sensors_to_delete.append(sensor)

        print(f"filter_model_to_config_joints: keeping {len(allowed_joints)} joint names")
        print(f"  Joints:    removing {len(joints_to_delete)} of {sum(len(list(b.joints)) for b in spec.bodies)}")
        print(f"  Actuators: removing {len(actuators_to_delete)} of {len(list(spec.actuators))}")
        print(f"  Tendons:   removing {len(tendons_to_delete)} of {len(list(spec.tendons))}")
        print(f"  Sensors:   removing {len(sensors_to_delete)} of {len(list(spec.sensors))}")

        for joint in joints_to_delete:
            spec.delete(joint)
        for actuator in actuators_to_delete:
            spec.delete(actuator)
        for tendon in tendons_to_delete:
            spec.delete(tendon)
        for sensor in sensors_to_delete:
            spec.delete(sensor)

        return spec

    def add_fly(
        self,
        torque_actuators: bool,
        rescale_factor: float = 1.0,
        pos: tuple[float, float, float] = (0, 0, -0.05),
        quat: tuple[float, float, float, float] = (1, 0, 0, 0),
        rgba: Optional[tuple[float, float, float, float]] = None,
        suffix: str = "-fly",
        new_spec: Optional[mujoco.MjSpec] = None,
        filter_joints: bool = False,
    ) -> None:
        """Adds the fly model to the environment.

        Args:
            torque_actuators: Whether to convert motors to torque-mode actuators.
            rescale_factor: Factor to rescale the fly body. Defaults to 1.0.
            pos: Position (x, y, z) to spawn the fly. Defaults to (0, 0, 0.05).
            quat: Quaternion (w, x, y, z) for fly orientation. Defaults to (1, 0, 0, 0).
            rgba: RGBA color values (red, green, blue, alpha) for recoloring the body.
                If None, no recoloring is applied. Defaults to None.
            suffix: Suffix to append to body names. Defaults to "-fly".
            filter_joints: Whether to filter the model to only keep joints
                listed in ``config.joint_names``. Defaults to False.
        """
        # Load base fly spec
        fly = mujoco.MjSpec.from_file(self._walker_xml_path)

        # Filter to config joints if requested
        if filter_joints:
            fly = self.filter_model_to_config_joints(spec=fly)

        if rescale_factor != 1.0:
            logging.info(f"Rescaling body tree with scale factor {rescale_factor}")
            fly = dm_scale_spec(fly, rescale_factor)

        # Recolor the body if rgba is specified
        if rgba is not None:
            for body in fly.worldbody.bodies:
                _recolour_tree(body, rgba=rgba)
                
        self._suffix = suffix
        if new_spec is None:
            spawn_frame = self._spec.worldbody.add_frame(
                pos=pos,
                quat=quat,
            )
            spawn_body = spawn_frame.attach_body(fly.body("thorax"), "", suffix=suffix)
        else:
            spawn_frame = new_spec.worldbody.add_frame(
                pos=pos,
                quat=quat,
            )
            spawn_body = spawn_frame.attach_body(fly.body("thorax"), "", suffix=suffix)
            return new_spec
        
    def add_ghost_fly(
        self,
        rescale_factor: float = 1.0,
        pos=(0, 0, 0.05),
        ghost_rgba=(0.8, 0.8, 0.8, 0.3),
        suffix="-ghost",
        filter_joints: bool = False,
    ):
        """Adds a ghost fly model to the environment.

        Args:
            rescale_factor: Factor to rescale the ghost body. Defaults to 1.0.
            pos: Position to spawn the ghost. Defaults to (0, 0, 0.05).
            ghost_rgba: RGBA color for the ghost. Defaults to (0.8, 0.8, 0.8, 0.3).
            suffix: Suffix for the ghost body names. Defaults to "-ghost".
            filter_joints: Whether to filter model to only keep joints
                listed in ``config.joint_names``. Defaults to False.
        """
        new_spec = mujoco.MjSpec.from_file(str(self._config._arena_xml_path))
        new_spec = self.add_fly(
            rescale_factor=self._config.rescale_factor,
            torque_actuators=self._config.torque_actuators,
            new_spec=new_spec,
            filter_joints=filter_joints)

        # Load ghost fly spec
        fly_spec = mujoco.MjSpec.from_file(self._walker_xml_path)

        # Filter to config joints if requested
        if filter_joints:
            fly_spec = self.filter_model_to_config_joints(spec=fly_spec)
        # Scale and recolor the ghost body
        for body in fly_spec.worldbody.bodies:
            _scale_body_tree(body, rescale_factor)
            _recolour_tree(body, rgba=ghost_rgba)
        # Attach as ghost at the offset frame
        spawn_frame = new_spec.worldbody.add_frame(pos=pos, quat=[1, 0, 0, 0])
        spawn_body = spawn_frame.attach_body(fly_spec.body("thorax"), "", suffix=suffix)
        for geom in new_spec.geoms:
            if suffix in geom.name:
                geom.contype = 0
                geom.conaffinity = 0
                if 'wing' in geom.name:
                    geom.group = 5  # put ghost in group 5
        for site in new_spec.sites:
            if suffix in site.name:
                if 'tracking' in site.name:
                    site.rgba = [1, 0, 0, 1]
        if self._enable_flight:
            new_spec = self._set_up_flight(new_spec)
            # Only reset ghost wings since main fly wings are handled in _set_up_flight
            # if suffix != self._suffix:
                # new_spec = self._reset_wing_orientation(spec=new_spec, suffix=[suffix])
        # else:
            # new_spec = self._reset_wing_orientation(spec=new_spec, suffix=[self._suffix, suffix])
        return new_spec
    
    def compile(self, forced=False, mjx_model=False) -> None:
        """Compiles the model from the mj_spec and put models to mjx"""
        if not self._compiled or forced:
            self._spec.option.noslip_iterations = self._config.noslip_iterations
            if self._enable_flight:
                print("Setting up flight")
                self._spec = self._set_up_flight()
            else:
                # self._spec = self._reset_wing_orientation(self._spec.copy(), suffix=self._suffix)
                self._default_wing_pos = jnp.asarray(_WING_PARAMS['default_qpos'])
                
            self._mj_model = self._spec.compile()
            self._mj_model.opt.timestep = self._config.sim_dt
            # Increase offscreen framebuffer size to render at higher resolutions.
            self._mj_model.vis.global_.offwidth = 3840
            self._mj_model.vis.global_.offheight = 2160
            self._mj_model.opt.iterations = self._config.iterations
            self._mj_model.opt.ls_iterations = self._config.ls_iterations
            if mjx_model:
                self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.mujoco_impl)
            if self._enable_flight:
                if self._enable_wbpg:
                    self._joint_idxs = jnp.array(mjx_env.get_qpos_ids(self._mj_model, [joint + self._suffix for joint in self._config.joint_names if 'wing' not in joint]))
                else:
                    self._joint_idxs = jnp.array(mjx_env.get_qpos_ids(self._mj_model, [joint + self._suffix for joint in self._config.joint_names]))
            else:
                self._joint_idxs = jnp.array(mjx_env.get_qpos_ids(self._mj_model, [joint + self._suffix for joint in self._config.joint_names]))
            self._joint_vel_idxs = jnp.array(mjx_env.get_qvel_ids(self._mj_model, [joint + self._suffix for joint in self._config.joint_names]))
            # Wing-free indices used only for rewards/termination (not proprioception).
            # Wings are clamped to default pos in walking mode, creating a constant
            # irreducible error vs reference clips that dilutes gradient for leg joints.
            # Uses config.wing_names (anatomy YAML) rather than string-matching.
            _wing_joints = set(self._config.wing_names)
            self._joint_reward_idxs = jnp.array(mjx_env.get_qpos_ids(
                self._mj_model,
                [j + self._suffix for j in self._config.joint_names if j not in _wing_joints]
            ))
            self._joint_vel_reward_idxs = jnp.array(mjx_env.get_qvel_ids(
                self._mj_model,
                [j + self._suffix for j in self._config.joint_names if j not in _wing_joints]
            ))
            self._body_idxs = jnp.array([mujoco.mj_name2id(self._mj_model, mujoco.mju_str2Type("body"), body+self._suffix) for body in self._config.body_names])
            self._end_eff_idxs = jnp.array([mujoco.mj_name2id(self._mj_model, mujoco.mju_str2Type("body"), body+self._suffix) for body in self._config.end_eff_names])
            self._wing_joint_idxs = jnp.array(mjx_env.get_qpos_ids(self._mj_model, [joint + self._suffix for joint in self._config.wing_names]))
            self._wing_actuator_idxs = jnp.array([act.id for act in self._spec.actuators if 'wing' in act.name])
            # Mask to zero out wing actuator ctrl (1.0=keep, 0.0=zero)
            zero_mask = np.ones(self._mj_model.nu, dtype=np.float32)
            zero_mask[self._wing_actuator_idxs] = 0.0
            self._zero_ctrl_mask = jnp.array(zero_mask)
            self._wing_fluid_idxs = jnp.array([geom.id for geom in self._spec.geoms if 'fluid' in geom.name])
            self._sensor_idxs = jnp.array([sensor.id for sensor in self._spec.sensors if 'contact' in sensor.name])
            self._contact_idxs = jnp.array([self.mj_model.sensor(sensor_idx).adr[0] for sensor_idx in self._sensor_idxs])
            self._floor_height = self._spec.geom("floor").pos[2]
            self._compiled = True

    def _get_joint_angles(self, data: mjx.Data) -> jnp.ndarray:
        return data.qpos[self._joint_idxs] 
    
    def _get_joint_ang_vels(self, data: mjx.Data) -> jnp.ndarray:
        return data.qvel[self._joint_vel_idxs] 
    
    def _get_actuator_ctrl(self, data: mjx.Data) -> jnp.ndarray:
        return data.qfrc_actuator[self._joint_vel_idxs]

    def _get_body_height(self, data: mjx.Data) -> jnp.ndarray:
        thorax_pos = data.bind(self.mjx_model, self._spec.body(f"thorax{self._suffix}")).xpos
        thorax_z = thorax_pos[2]
        return thorax_z
    
    def _get_world_zaxis(self, data: mjx.Data) -> jnp.ndarray:
        return self.root_body(data).xmat.flatten()[6:]
    
    def _get_appendages_pos(self, data: mjx.Data, flatten: bool = True) -> Union[dict[str, jnp.ndarray], jnp.ndarray]:
        """Get _egocentric_ position of the appendages."""
        thorax = data.bind(self.mjx_model, self._spec.body(f"thorax{self._suffix}"))
        appendages_pos = collections.OrderedDict()
        for appendage_name in self._config.end_eff_names:
            global_xpos = data.bind(self.mjx_model, self._spec.body(f"{appendage_name}{self._suffix}")).xpos
            egocentric_xpos = jnp.dot(global_xpos - thorax.xpos, thorax.xmat)
            appendages_pos[appendage_name] = egocentric_xpos
        if flatten:
            appendages_pos, _ = jax.flatten_util.ravel_pytree(appendages_pos)
        return appendages_pos

    def _get_bodies_pos(self, data: mjx.Data, flatten: bool = True) -> Union[dict[str, jnp.ndarray], jnp.ndarray]:
        """Get _global_ positions of the body parts."""
        bodies_pos = collections.OrderedDict()
        for body_name, body_idx in zip(self._config.body_names, self._body_idxs):
            global_xpos = data.bind(self.mjx_model, self._spec.body(f"{body_name}{self._suffix}")).xpos
            bodies_pos[body_name] = global_xpos
        if flatten:
            bodies_pos, _ = jax.flatten_util.ravel_pytree(bodies_pos)
        return bodies_pos

    def _get_proprioception(self, data: mjx.Data, flatten: bool = True) -> Union[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Get proprioception data from the environment."""
        if flatten:
            proprioception = jnp.concatenate([self._get_joint_angles(data).flatten(), 
                                              self._get_joint_ang_vels(data).flatten(),
                                              self._get_actuator_ctrl(data).flatten(),
                                              self._get_body_height(data).flatten(),
                                              self._get_world_zaxis(data).flatten(),
                                              self._get_appendages_pos(data, flatten=flatten).flatten(),
                                              ])
        else:
            proprioception = collections.OrderedDict(
                joint_angles = self._get_joint_angles(data),
                joint_ang_vels = self._get_joint_ang_vels(data),
                actuator_ctrl = self._get_actuator_ctrl(data),
                body_height = self._get_body_height(data),
                world_zaxis = self._get_world_zaxis(data),
                appendages_pos = self._get_appendages_pos(data, flatten=flatten)
            )
        return proprioception

    def _get_kinematic_sensors(self, data: mjx.Data, flatten: bool = True) -> Union[Mapping[str, jnp.ndarray], jnp.ndarray]:
        """Get kinematic sensors data from the environment."""
        accelerometer = data.bind(self.mjx_model, self._spec.sensor("accelerometer" + self._suffix)).sensordata
        velocimeter = data.bind(self.mjx_model, self._spec.sensor("velocimeter" + self._suffix)).sensordata
        gyro = data.bind(self.mjx_model, self._spec.sensor("gyro" + self._suffix)).sensordata
        sensors = collections.OrderedDict(
            accelerometer = accelerometer,
            velocimeter = velocimeter,
            gyro = gyro,
        )
        if flatten:
            sensors, _ = jax.flatten_util.ravel_pytree(sensors)
        return sensors

    def _get_touch_sensors(self, data: mjx.Data) -> jnp.ndarray:
        """Get touch sensors data from the environment."""
        touches = [data.bind(self.mjx_model, self._spec.sensor(f"{name}{self._suffix}")).sensordata for name in self._config.touch_sensor_names]
        return jnp.array(touches)

    def _get_origin(self, data: mjx.Data) -> jnp.ndarray:
        """Get origin position in the thorax frame."""
        thorax = data.bind(self.mjx_model, self._spec.body(f"thorax{self._suffix}"))
        thorax_frame = thorax.xmat
        thorax_pos = thorax.xpos
        return jnp.dot(-thorax_pos, thorax_frame)

    def _get_egocentric_camera(self, data: mjx.Data):
        """Get egocentric camera data from the environment."""
        raise NotImplementedError(
            "Egocentric camera is not implemented for this environment."
        )

    def get_joint_names(self):
        return map(lambda j: j.name, self._spec.joints[1:])

    def root_body(self, data):
        #TODO: Double-check which body should be considered the root (walker or thorax)
        return data.bind(self.mjx_model, self._spec.body(f"thorax{self._suffix}"))
    
    def _set_up_flight(self, spec: Optional[mujoco.MjSpec] = None) -> mujoco.MjSpec:
        """Set up the model for flight by configuring wing joints and actuators."""
        if spec is None:
            spec = self._spec.copy()
        if self._quasi_aero:
            print("Using quasi-steady aerodynamic model.")
            for geom in spec.geoms:
                if 'fluid' in geom.name:
                    geom.fluid_coefs = _WING_PARAMS['quasi_fluidcoef']
        else:
            print("Using ellipsoid aerodynamic model.")
            for geom in spec.geoms:
                if 'fluid' in geom.name:
                    geom.fluid_coefs = _WING_PARAMS['ellipsoid_fluidcoef']
                    
        spec.option.density = _AIR_DENSITY
        spec.option.viscosity = _AIR_VISCOSITY
        spec.option.timestep = _FLY_PHYSICS_TIMESTEP
        self._default_wing_pos = jnp.asarray(_WING_PARAMS['default_qpos'])
        for joint in self._config.wing_names:
            spec.joint(joint + self._suffix).stiffness = _WING_PARAMS['stiffness']
            spec.joint(joint + self._suffix).damping = _WING_PARAMS['damping']
            spec.actuator(joint + self._suffix).gainprm[0] = _WING_PARAMS['gainprm']
        # spec = self._reset_wing_orientation(spec, suffix=self._suffix)
        return spec

    def _reset_wing_orientation(self, spec: Optional[mujoco.MjSpec]=None, suffix: Optional[list[str]] = None, body_pitch_angle: float = _BODY_PITCH_ANGLE) -> mujoco.MjSpec:
        """Reset wing orientation to default stroke plane angle."""
        if spec is None:
            spec = self._spec.copy()
        if isinstance(suffix, str):
            suffixes = [suffix]
        elif isinstance(suffix, list):
            suffixes = suffix
            suffix = suffix[0]
        elif suffix is None:
            suffixes = [self._suffix]
            suffix = self._suffix
        site_name = 'hover_up_dir' + suffix
        up_dir = jnp.array(spec.site(site_name).quat)
        up_dir_angle = 2 * jnp.arccos(up_dir[0])
        delta = (jnp.deg2rad(body_pitch_angle) - up_dir_angle)
        dquat = jnp.array([jnp.cos(delta / 2), 0, jnp.sin(delta / 2), 0])
        # Rotate up_dir to new angle.
        up_dir = up_dir.at[:].set(mj_math.quat_mul(dquat, up_dir))
        # == Set stroke plane angle.
        stroke_plane_angle = jnp.deg2rad(body_pitch_angle)
        stroke_plane_quat = jnp.array([
            jnp.cos(stroke_plane_angle / 2), 0,
            jnp.sin(stroke_plane_angle / 2), 0
        ])
        for sx in suffixes:
            wing_body_names = ['wing_left'+sx, 'wing_right'+sx]
            for quat, wing in [(jnp.array([0., 0, 0, 1]), wing_body_names[0]),
                                (jnp.array([0., -1, 0, 0]), wing_body_names[1])]:
                # Rotate wing-joint frame.
                dquat = mj_math.quat_mul(neg_quat(stroke_plane_quat), quat)
                new_wing_quat = mj_math.quat_mul(dquat, neg_quat(up_dir))
                body = spec.body(wing)
                change_body_frame(body, body.pos, new_wing_quat)
        return spec

    @property
    def action_size(self) -> int:
        if self._enable_flight & self._enable_wbpg:
            return self._mj_model.nu + 1
        return self._mj_model.nu

    @property
    def xml_path(self) -> str:
        return self._walker_xml_path

    @property
    def walker_xml_path(self) -> str:
        return self._walker_xml_path

    @property
    def arena_xml_path(self) -> str:
        return self._arena_xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
    
    @property
    def enable_flight(self) -> bool:
        return self._enable_flight
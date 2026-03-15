import wandb
import jax
import mujoco
import os
import numpy as np
import imageio
from fly_mimic.utils.utils import add_trajectory_sites_spheres, add_cross_hair_sites

def log_eval_rollout(cfg, rollout, state, env, reference_clip, model_path, num_steps, EVAL_STEPS=0):
    '''Log the rollout to wandb'''

    # Log the metrics for the rollout
    # for metric in cfg.dataset.train.state_metric_list:
    for metric in state.metrics:
        metric_values = [state.metrics[metric] for state in rollout]
        table = wandb.Table(
            data=[[x, y] for (x, y) in zip(range(len(metric_values)), metric_values)],
            columns=["frame", metric],
        )
        wandb.log(
            {
                f"eval/rollout_{metric}": wandb.plot.line(
                    table,
                    "frame",
                    metric,
                    title=f"{metric} for each rollout frame",
                )
            },
            commit=False,
        )
        
    
    # Render the walker with the reference expert demonstration trajectory
    # os.environ["MUJOCO_GL"] = "osmesa"
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    
    scene_option = mujoco.MjvOption()
    scene_option.sitegroup[:] = [1, 1, 1, 1, 1, 0]
    scene_option.geomgroup[:] = [1, 1, 1, 1, 0, 0]
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
    camera = "track1-fly"
    print(f"clip_id:{rollout[0].info}")
    if cfg.dataset.env_args.enable_flight:
        from functools import partial
        video_path = cfg.paths.fig_dir/f"{num_steps}.mp4"
        add_arrow_part = partial(add_arrows, radius=0.003, wing_fluid_idxs=env._wing_fluid_idxs, rgba=[[0, 0, 1, 1],[0, 1, 0, 1],[1, 0, 0, 1]])
        rendered_frames = env.render_ghost(rollout, camera=camera, scene_option=scene_option, video_path=video_path, 
                                           add_labels=True, add_ghost=True, verbose=False,
                                           modify_scene_fns=[add_arrow_part])
        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})
    else:
        video_path = cfg.paths.fig_dir/f"{num_steps}.mp4"
        rendered_frames = env.render_ghost(rollout, camera=camera, scene_option=scene_option, video_path=video_path, add_labels=True, add_ghost=True, verbose=False)
        wandb.log({"eval/rollout": wandb.Video(video_path, format="mp4")})


def add_arrows(scene, state, wing_fluid_idxs, radius=0.03, rgba=[0.2, 0.2, 0.6, 1], scale_vectors=0.1,
               rollout_info=None, timestep_idx=None, geom_xpos=None):
    """
    Add force arrows to the scene.

    Args:
        scene: Mujoco scene
        state: mjx_env.State object (or None if using Rollout)
        wing_fluid_idxs: Indices of wing geometry
        radius: Arrow radius
        rgba: Color for each force component
        scale_vectors: Scale factor for force vectors
        rollout_info: Dict containing 'aero_forces' from Rollout (shape: timesteps, n_forces, n_wings, xyz)
        timestep_idx: Current timestep index when using Rollout
        geom_xpos: Geometry positions from mj_data.geom_xpos when using Rollout
    """
    # Determine source of aero_forces
    if state is not None and hasattr(state, 'info') and state.info is not None and 'aero_forces' in state.info:
        # Using mjx_env.State
        aero_forces = state.info['aero_forces']
        geom_positions = state.data.geom_xpos
    elif rollout_info is not None and 'aero_forces' in rollout_info and timestep_idx is not None:
        # Using Rollout - extract forces for this timestep
        # rollout_info['aero_forces'] shape: (timesteps, n_forces, n_wings, xyz)
        aero_forces = rollout_info['aero_forces'][timestep_idx]  # (n_forces, n_wings, xyz)
        geom_positions = geom_xpos
    else:
        # No aero_forces available
        return

    for wing in range(len(wing_fluid_idxs)):
        for m in range(len(aero_forces)-1):
            wing_xpos = geom_positions[wing_fluid_idxs[wing]]
            forces = aero_forces[m][wing]  # (xyz,)
            from_ = wing_xpos
            to = wing_xpos + scale_vectors * forces
            """Add an arrow to the scene."""
            scene.geoms[scene.ngeom].category = mujoco.mjtCatBit.mjCAT_STATIC
            mujoco.mjv_initGeom(
                geom=scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.zeros(9),
                rgba=np.asarray(rgba[m]).astype(np.float32),
            )
            mujoco.mjv_connector(
                geom=scene.geoms[scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                width=radius,
                from_=from_,
                to=to,
            )
            scene.ngeom += 1

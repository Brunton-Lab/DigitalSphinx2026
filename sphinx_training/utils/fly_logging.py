import wandb
import jax
import mujoco
import os
import numpy as np
import imageio
import time
import glob
from pathlib import Path
from fly_mimic.utils.utils import add_trajectory_sites_spheres, add_cross_hair_sites


def cleanup_old_eval_artifacts(keep_n_recent=1):
    """
    Delete old local evaluation artifacts (videos, tables) while keeping only
    the N most recent evaluations. This reduces local disk usage while
    preserving full history in WandB cloud.
    
    Args:
        keep_n_recent: Number of recent evaluation artifacts to keep locally (default: 1)
    """
    if wandb.run is None:
        return
    
    try:
        import re
        
        # Get the wandb run directory
        run_dir = Path(wandb.run.dir).parent
        media_dir = run_dir / "files" / "media"
        
        if not media_dir.exists():
            return
        
        # Find all artifact files (videos and tables)
        artifact_patterns = [
            "videos/**/*.mp4",
            "videos/**/*.webm",
            "table/**/*.json",
            "table/**/*.table.json"
        ]
        
        # Regex patterns to extract evaluation number from filenames
        # Videos: rollout_2_abc123.mp4 -> eval 2
        # Tables: rollout_rewards_table_2_def456.table.json -> eval 2
        video_pattern = re.compile(r'rollout_(\d+)_[a-f0-9]+\.(mp4|webm)$')
        table_pattern = re.compile(r'_table_(\d+)_[a-f0-9]+\.(table\.)?json$')
        
        artifacts_by_eval = {}  # {eval_num: [file_paths]}
        
        for pattern in artifact_patterns:
            files = list(media_dir.glob(pattern))
            for f in files:
                # Try to extract evaluation number from filename
                eval_num = None
                
                video_match = video_pattern.search(f.name)
                if video_match:
                    eval_num = int(video_match.group(1))
                else:
                    table_match = table_pattern.search(f.name)
                    if table_match:
                        eval_num = int(table_match.group(1))
                
                # Group files by evaluation number
                if eval_num is not None:
                    if eval_num not in artifacts_by_eval:
                        artifacts_by_eval[eval_num] = []
                    artifacts_by_eval[eval_num].append(f)
        
        if not artifacts_by_eval:
            return  # No evaluation artifacts found
        
        # Determine which evaluations to keep based on most recent file timestamp
        eval_timestamps = {}
        for eval_num, files in artifacts_by_eval.items():
            # Use the most recent file's timestamp to represent this evaluation
            max_mtime = max(f.stat().st_mtime for f in files)
            eval_timestamps[eval_num] = max_mtime
        
        # Sort evaluation numbers by timestamp (newest first)
        sorted_evals = sorted(eval_timestamps.keys(), 
                            key=lambda e: eval_timestamps[e], 
                            reverse=True)
        
        # Keep only the N most recent evaluations
        keep_evals = set(sorted_evals[:keep_n_recent])
        
        print(f"[Cleanup] Found {len(sorted_evals)} evaluations, keeping {keep_n_recent} most recent: {sorted(keep_evals)}")
        
        # Delete all files from old evaluations
        files_deleted = 0
        for eval_num, files in artifacts_by_eval.items():
            if eval_num not in keep_evals:
                for old_file in files:
                    try:
                        if old_file.exists():  # Check existence (avoid race conditions)
                            old_file.unlink()
                            files_deleted += 1
                            print(f"[Cleanup] Deleted artifact from eval {eval_num}: {old_file.name}")
                    except Exception as e:
                        # Don't fail if we can't delete (file might be in use)
                        pass
        
        if files_deleted > 0:
            print(f"[Cleanup] Deleted {files_deleted} artifact files from {len(sorted_evals) - len(keep_evals)} old evaluations")
        
        # Clean up empty directories
        for subdir in ["videos", "table"]:
            subdir_path = media_dir / subdir
            if subdir_path.exists():
                # Remove empty subdirectories
                for item in subdir_path.rglob("*"):
                    if item.is_dir() and not any(item.iterdir()):
                        try:
                            item.rmdir()
                        except:
                            pass
    
    except Exception as e:
        # Don't fail the training run if cleanup fails
        print(f"[Cleanup] Warning: Could not clean old artifacts: {e}")


def log_eval_rollout(cfg, rollout, state, env, reference_clip, model_path, num_steps, EVAL_STEPS=0, cleanup_old_artifacts=True):
    '''Log the rollout to wandb'''

    # Log the metrics for the rollout
    n_frames = len(rollout)
    xs = list(range(n_frames))
    grouped_prefixes = {"terminations/": "Terminations", "rewards/": "Rewards"}
    grouped = {prefix: {} for prefix in grouped_prefixes}
    individual = {}

    for metric in state.metrics:
        metric_values = [float(rollout[i].metrics[metric]) for i in range(n_frames)]
        matched = False
        for prefix in grouped_prefixes:
            if metric.startswith(prefix):
                grouped[prefix][metric] = metric_values
                matched = True
                break
        if not matched:
            individual[metric] = metric_values

    # Log grouped metrics as line_series (1 table file per group)
    for prefix, title in grouped_prefixes.items():
        if grouped[prefix]:
            keys = list(grouped[prefix].keys())
            ys = [grouped[prefix][k] for k in keys]
            wandb.log({
                f"eval/rollout_{title.lower()}": wandb.plot.line_series(
                    xs=xs, ys=ys, keys=keys, title=f"Rollout {title}", xname="frame",
                )
            }, commit=False)

    # Log remaining metrics as individual line plots
    for metric, values in individual.items():
        table = wandb.Table(
            data=[[x, y] for x, y in zip(xs, values)],
            columns=["frame", metric],
        )
        wandb.log({
            f"eval/rollout_{metric}": wandb.plot.line(
                table, "frame", metric, title=f"{metric} for each rollout frame",
            )
        }, commit=False)
        
    
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
    # print(f"clip_id:{rollout[0].info}")
    if cfg.dataset.env_args.enable_flight:
        from functools import partial
        add_arrow_part = partial(add_arrows, radius=0.003, wing_fluid_idxs=env._wing_fluid_idxs, rgba=[[0, 0, 1, 1],[0, 1, 0, 1],[1, 0, 0, 1]])
        rendered_frames = env.render_ghost(rollout, camera=camera, scene_option=scene_option, video_path=None,
                                           add_labels=True, add_ghost=True, verbose=False,
                                           modify_scene_fns=[add_arrow_part])
    else:
        rendered_frames = env.render_ghost(rollout, camera=camera, scene_option=scene_option, video_path=None, add_labels=True, add_ghost=True, verbose=False)
    frames_array = np.stack(rendered_frames).transpose(0, 3, 1, 2)  # (T,H,W,C) -> (T,C,H,W)
    wandb.log({"eval/rollout": wandb.Video(frames_array, fps=50, format="mp4")})
    
    # Clean up old local artifacts to save disk space (cloud history is preserved)
    if cleanup_old_artifacts:
        # Get cleanup settings from config if available
        keep_n = cfg.dataset.get('keep_n_eval_artifacts', 1) if hasattr(cfg, 'dataset') else 1
        # Small delay to allow wandb to queue uploads before local cleanup
        time.sleep(2)
        cleanup_old_eval_artifacts(keep_n_recent=keep_n)


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
    if rollout_info is not None and 'aero_forces' in rollout_info and timestep_idx is not None:
        # Using Rollout - extract forces for this timestep
        # rollout_info['aero_forces'] shape: (timesteps, n_forces, n_wings, xyz)
        aero_forces = rollout_info['aero_forces'][timestep_idx]  # (n_forces, n_wings, xyz)
        geom_positions = geom_xpos
    elif state is not None and hasattr(state, 'info') and state.info is not None and 'aero_forces' in state.info:
        # Using mjx_env.State
        aero_forces = state.info['aero_forces']
        geom_positions = state.data.geom_xpos
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

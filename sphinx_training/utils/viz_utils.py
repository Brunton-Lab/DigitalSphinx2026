
import jax
import mujoco
import jax.numpy as jnp
import numpy as np
from typing import List, Optional, Sequence, Callable, Union
from mujoco import mjx
from mujoco_playground._src import mjx_env



def compile(env, spec, mjx_model=False) -> None:
    """Compiles the model from the mj_spec and put models to mjx"""
    spec.option.noslip_iterations = env._config.noslip_iterations
    mj_model = spec.compile()
    mj_model.opt.timestep = env._config.sim_dt
    # Increase offscreen framebuffer size to render at higher resolutions.
    mj_model.vis.global_.offwidth = 3840
    mj_model.vis.global_.offheight = 2160
    mj_model.opt.iterations = env._config.iterations
    mj_model.opt.ls_iterations = env._config.ls_iterations
    idx_dict = {}
    idx_dict['joint_idxs'] = jnp.array(mjx_env.get_qpos_ids(mj_model, [joint + env._suffix for joint in env._config.joint_names]))
    idx_dict['joint_vel_idxs'] = jnp.array(mjx_env.get_qvel_ids(mj_model, [joint + env._suffix for joint in env._config.joint_names]))
    idx_dict['body_idxs'] = jnp.array([mujoco.mj_name2id(mj_model, mujoco.mju_str2Type("body"), body+env._suffix) for body in env._config.body_names])
    idx_dict['end_eff_idxs'] = jnp.array([mujoco.mj_name2id(env._mj_model, mujoco.mju_str2Type("body"), body+env._suffix) for body in env._config.end_eff_names])
    if mjx_model:
        mj_model = mjx.put_model(mj_model, impl=env._config.mujoco_impl)
    return mj_model, idx_dict

def render_pair_array(
    mj_model: mujoco.MjModel,
    trajectory: Union[List[mujoco.MjData], mujoco.MjData],
    height: int = 480,
    width: int = 640,
    camera: Optional[str] = None,
    scene_option: Optional[mujoco.MjvOption] = None,
    modify_scene_fns: Optional[
        Sequence[Callable[[mujoco.MjvScene], None]]
    ] = None,
    hfield_data: Optional[jax.Array] = None,
):
  """Renders a trajectory as an array of images."""
  renderer = mujoco.Renderer(mj_model, height=height, width=width)
  camera = camera or -1

  if hfield_data is not None:
    mj_model.hfield_data = hfield_data.reshape(mj_model.hfield_data.shape)
    mujoco.mjr_uploadHField(mj_model, renderer._mjr_context, 0)

  def get_image(data, modify_scn_fn=None) -> np.ndarray:
    d = mujoco.MjData(mj_model)
    d.qpos, d.qvel = data.qpos, data.qvel
    d.mocap_pos, d.mocap_quat = data.mocap_pos, data.mocap_quat
    d.xfrc_applied = data.xfrc_applied
    mujoco.mj_forward(mj_model, d)
    renderer.update_scene(d, camera=camera, scene_option=scene_option)
    if modify_scn_fn is not None:
      modify_scn_fn(renderer.scene)
    return renderer.render()

  if isinstance(trajectory, list):
    from tqdm.auto import tqdm
    out = []
    for i, data in enumerate(tqdm(trajectory)):
      if modify_scene_fns is not None:
        modify_scene_fn = modify_scene_fns[i]
      else:
        modify_scene_fn = None
      out.append(get_image(data, modify_scene_fn))
  else:
    out = get_image(trajectory)

  renderer.close()
  return out

def copy_data_to_pair(mj_data, ref_clip_data, data_pair, frame_idx=0, debug=False):
    """Copy mjx data to first half and reference clip data to second half of data_pair."""
    
    # Get reference clip data for the specified frame
    ref_qpos = ref_clip_data.qpos[frame_idx]
    ref_qvel = ref_clip_data.qvel[frame_idx]
    
    # Get sizes
    n_qpos_mj = len(mj_data.qpos)
    n_qvel_mj = len(mj_data.qvel)
    n_qpos_ref = len(ref_qpos)
    n_qvel_ref = len(ref_qvel)
    
    # Calculate starting indices for second half
    n_qpos_half = len(data_pair.qpos) // 2
    n_qvel_half = len(data_pair.qvel) // 2
    
    if debug:
        print(f"Copying mjx data to first half:")
        print(f"  qpos: {n_qpos_mj} elements")
        print(f"  qvel: {n_qvel_mj} elements")
        print(f"Copying ref clip data to second half:")
        print(f"  qpos: starting at {n_qpos_half}, copying {n_qpos_ref} elements")
        print(f"  qvel: starting at {n_qvel_half}, copying {n_qvel_ref} elements")
    
    # Copy mjx data to first half
    data_pair.qpos[:n_qpos_mj] = mj_data.qpos
    data_pair.qvel[:n_qvel_mj] = mj_data.qvel
    
    # Copy reference clip data to second half
    data_pair.qpos[n_qpos_half:n_qpos_half+n_qpos_ref] = ref_qpos
    data_pair.qvel[n_qvel_half:n_qvel_half+n_qvel_ref] = ref_qvel
    
    # Copy other relevant arrays from mjx data
    if hasattr(mj_data, 'qacc') and mj_data.qacc is not None:
        data_pair.qacc[:len(mj_data.qacc)] = mj_data.qacc
    
    if hasattr(mj_data, 'xpos') and mj_data.xpos is not None:
        n_bodies = len(mj_data.xpos)
        data_pair.xpos[:n_bodies] = mj_data.xpos
    
    if hasattr(mj_data, 'xquat') and mj_data.xquat is not None:
        n_bodies = len(mj_data.xquat)
        data_pair.xquat[:n_bodies] = mj_data.xquat
    
    if hasattr(mj_data, 'xmat') and mj_data.xmat is not None:
        n_bodies = len(mj_data.xmat)
        data_pair.xmat[:n_bodies] = mj_data.xmat
        
    if hasattr(mj_data, 'xipos') and mj_data.xipos is not None:
        n_bodies = len(mj_data.xipos)
        data_pair.xipos[:n_bodies] = mj_data.xipos
        
    if hasattr(mj_data, 'ximat') and mj_data.ximat is not None:
        n_bodies = len(mj_data.ximat)
        data_pair.ximat[:n_bodies] = mj_data.ximat
    
    # Copy mocap data if it exists
    if hasattr(mj_data, 'mocap_pos') and mj_data.mocap_pos is not None:
        n_mocap = len(mj_data.mocap_pos)
        data_pair.mocap_pos[:n_mocap] = mj_data.mocap_pos
        
    if hasattr(mj_data, 'mocap_quat') and mj_data.mocap_quat is not None:
        n_mocap = len(mj_data.mocap_quat)
        data_pair.mocap_quat[:n_mocap] = mj_data.mocap_quat
    
    # Copy applied forces if they exist
    if hasattr(mj_data, 'xfrc_applied') and mj_data.xfrc_applied is not None:
        n_bodies = len(mj_data.xfrc_applied)
        data_pair.xfrc_applied[:n_bodies] = mj_data.xfrc_applied
    
    # Copy reference clip qacc if it exists
    if hasattr(ref_clip_data, 'qacc') and ref_clip_data.qacc is not None:
        ref_qacc = ref_clip_data.qacc[frame_idx]
        n_qacc_half = len(data_pair.qacc) // 2
        data_pair.qacc[n_qacc_half:n_qacc_half+len(ref_qacc)] = ref_qacc
    return data_pair


def create_paired_mjx_data_single(mjx_single, ref_qpos, ref_qvel):
    """Create paired mjx data by concatenating simulation and reference data."""
    # Create paired arrays by concatenation - only handle qpos and qvel
    paired_qpos = jnp.concatenate([mjx_single.qpos, ref_qpos])
    paired_qvel = jnp.concatenate([mjx_single.qvel, ref_qvel])
    
    # Only modify the essential fields and let mjx handle the rest
    replacement_dict = {
        'qpos': paired_qpos,
        'qvel': paired_qvel,
    }
    
    # Create a new mjx data structure with only the essential paired arrays
    paired_mjx_data = mjx_single.replace(**replacement_dict)
    
    return paired_mjx_data

def create_paired_mjx_data_batched_clips(mjx_data_batch, ref_clip_data_batch, frame_indices):
    """Create paired mjx data with different reference clips for each batch element.
    
    Args:
        mjx_data_batch: Batch of mjx data (with leading batch dimension)
        ref_clip_data_batch: Batch of reference clip data (different clips per batch element)
        frame_indices: Array of frame indices for each batch element
    
    Returns:
        Batched mjx data with paired arrays
    """
    # Extract reference data at specified frame indices for each clip
    batch_size = len(frame_indices)
    ref_qpos_batch = ref_clip_data_batch.qpos[jnp.arange(batch_size), frame_indices]
    ref_qvel_batch = ref_clip_data_batch.qvel[jnp.arange(batch_size), frame_indices]
    
    # Use vmap to process the batch
    vmapped_fn = jax.vmap(create_paired_mjx_data_single, in_axes=(0, 0, 0))
    paired_mjx_data_batch = vmapped_fn(mjx_data_batch, ref_qpos_batch, ref_qvel_batch)
    
    return paired_mjx_data_batch

# Combined fast extraction and conversion function
def extract_and_convert_trajectories_fast(rollout_pair, model_pair, num_envs):
    """Extract all environment trajectories and convert to MjData format in one go."""
    # Stack all timesteps into a single array: (timesteps, envs, ...)
    stacked_rollout = jax.tree.map(lambda *arrays: jnp.stack(arrays, axis=0), *rollout_pair)
    
    # Transpose to get (envs, timesteps, ...)
    all_trajectories_stacked = jax.tree.map(lambda x: jnp.transpose(x, (1, 0, *range(2, x.ndim))), stacked_rollout)
    
    # Convert all trajectories at once
    all_mjdata_trajectories = []
    
    for env_idx in range(num_envs):
        # Extract single environment using tree.map
        env_trajectory_arrays = jax.tree.map(lambda x: x[env_idx], all_trajectories_stacked)
        
        # Get trajectory length
        trajectory_length = env_trajectory_arrays.qpos.shape[0]
        
        # Convert entire trajectory at once using vectorized operations
        mjdata_trajectory = []
        
        # Pre-create all MjData objects
        mjdata_frames = [mujoco.MjData(model_pair) for _ in range(trajectory_length)]
        
        # Vectorized assignment of qpos and qvel
        for t in range(trajectory_length):
            mjdata_frames[t].qpos[:] = env_trajectory_arrays.qpos[t]
            mjdata_frames[t].qvel[:] = env_trajectory_arrays.qvel[t]
            # Run forward dynamics
            mujoco.mj_forward(model_pair, mjdata_frames[t])
        
        all_mjdata_trajectories.append(mjdata_frames)
    
    return all_mjdata_trajectories

# Even more optimized version using batched operations where possible
def extract_and_convert_single_trajectory(rollout_pair, model_pair, env_idx=0):
    """Extract and convert a single environment's trajectory efficiently."""
    # Stack timesteps: (timesteps, envs, ...)
    stacked_rollout = jax.tree.map(lambda *arrays: jnp.stack(arrays, axis=0), *rollout_pair)
    
    # Extract single environment: (timesteps, ...)
    single_env_stacked = jax.tree.map(lambda x: x[:, env_idx], stacked_rollout)
    
    # Get trajectory length
    trajectory_length = single_env_stacked.qpos.shape[0]
    
    # Pre-allocate MjData objects
    mjdata_trajectory = [mujoco.MjData(model_pair) for _ in range(trajectory_length)]
    
    # Batch copy data and forward dynamics
    qpos_batch = single_env_stacked.qpos
    qvel_batch = single_env_stacked.qvel
    
    for t in range(trajectory_length):
        mjdata_trajectory[t].qpos[:] = qpos_batch[t]
        mjdata_trajectory[t].qvel[:] = qvel_batch[t]
        mujoco.mj_forward(model_pair, mjdata_trajectory[t])
    
    return mjdata_trajectory
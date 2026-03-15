import os
import subprocess as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax
import jax.numpy as jnp
import mujoco 
import brax.math as brax_math

from omegaconf import OmegaConf
from pathlib import Path

import fly_mimic.utils.io_dict_to_hdf5 as ioh5

from typing import Sequence

def any_substr_in_str(substrings: Sequence[str], string: str) -> bool:
    """Checks if any of substrings is in string."""
    return any(s in string for s in substrings)

def body_quat_from_springrefs(body: 'mjcf.element') -> np.ndarray:
    """Computes new body quat from all joint springrefs and current quat."""
    joints = body.joints
    if not joints:
        return None
    # Construct quaternions for all joint axes.
    quats = []
    for joint in joints:
        theta = joint.springref
        axis = joint.axis
        if axis is None:
            axis = joint.dclass.parent.joint.axis
        quats.append(np.hstack((np.cos(theta / 2), np.sin(theta / 2) * axis)))
    # Compute the new orientation quaternion.
    quat = np.array([1., 0, 0, 0])
    for i in range(len(quats)):
        quat = brax_math.quat_mul(quats[-1 - i], quat)
    if body.quat is not None:
        quat = brax_math.quat_mul(body.quat, quat)
    return quat

def add_colorbar(mappable,linewidth=2,location='right',**kwargs):
    ''' modified from https://supy.readthedocs.io/en/2021.3.30/_modules/supy/util/_plot.html'''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="5%", pad=0.1)
    cbar = fig.colorbar(mappable, cax=cax, drawedges=False,**kwargs)
    cbar.outline.set_linewidth(linewidth)
    plt.sca(last_axes)
    return cbar

def map_discrete_cbar(cmap,N):
    cmap = plt.get_cmap(cmap,N+1)
    bounds = np.arange(-.5,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

def load_cfg(cfg_path, default_model_config=None):
    """ Load configuration file and merge with default model configuration

    Args:
        cfg_path (string): path to configuration file
        default_model_config (string): path to default model configuration file to merge with

    Returns:
        cfg: returns the merged configuration data
    """
    cfg = OmegaConf.load(cfg_path)
    for k in cfg.paths.keys():
        if k != 'user':
            cfg.paths[k] = Path(cfg.paths[k])
            cfg.paths[k].mkdir(parents=True, exist_ok=True)
    params_curr = cfg.dataset
    if default_model_config is not None:
        params = OmegaConf.load(default_model_config)
        cfg.dataset = OmegaConf.merge(params, params_curr)
    return cfg



def ema(data, length, axis=-1, initial_value=None):
    """
    Calculates the exponential moving average of an array along a specified axis.

    Args:
        data: The input array.
        length: The smoothing window 
        axis: The axis along which to calculate the EMA (default: -1).
        initial_value: Optional initial value for the EMA. If None, the first 
                       element along the axis is used.

    Returns:
        An array with the same shape as data, containing the EMA values.
    """
    alpha = 2 / (length + 1)
    if initial_value is None:
        initial_value = jnp.take(data, indices=0, axis=axis)
        initial_value = jnp.zeros_like(initial_value)

    def ema_step(carry, x):
        ema_prev = carry
        ema_current = alpha * x + (1 - alpha) * ema_prev
        return ema_current, ema_current

    _, result = jax.lax.scan(
        f=ema_step,
        init=initial_value,
        xs=jnp.swapaxes(data, axis, 0),
        length=data.shape[axis]
    )
    return jnp.swapaxes(result, 0, axis)

def neg_quat(quat_a):
    """Returns neg(quat_a)."""
    quat_b = quat_a.copy()
    quat_b = quat_b.at[0].set(quat_b[0]*-1)
    return quat_b

def change_body_frame(body, frame_pos, frame_quat):
    """Change the frame of a body while maintaining child locations."""
    frame_pos = jnp.zeros(3) if frame_pos is None else frame_pos
    frame_quat = jnp.array((1., 0, 0, 0)) if frame_quat is None else frame_quat
    # Get frame transformation.
    body_pos = jnp.zeros(3) if body.pos is None else body.pos
    dpos = body_pos - frame_pos
    body_quat = jnp.array((1., 0, 0, 0)) if body.quat is None else body.quat
    dquat = brax_math.quat_mul(neg_quat(frame_quat), body_quat)
    # Translate and rotate the body to the new frame.
    body.pos = frame_pos
    body.quat = frame_quat
    # Move all its children to their previous location.
    for child in body.find_all('body'):
        if not hasattr(child, 'pos'):
            continue
        # Rotate:
        if hasattr(child, 'quat'):
            child_quat = jnp.array(
                (1., 0, 0, 0)) if child.quat is None else child.quat
            child.quat = brax_math.quat_mul(dquat, child_quat)
        # Translate, accounting for rotations.
        child_pos = jnp.zeros(3) if child.pos is None else child.pos
        pos_in_parent = brax_math.rotate(child_pos, body_quat) + dpos
        child.pos = brax_math.rotate(pos_in_parent, neg_quat(frame_quat))


def load_mujoco_model(mjcf_path, cfg):
    paired = False
    if 'pair' in mjcf_path.stem:
        paired = True
    ##### Create model from xml file #####
    spec = mujoco.MjSpec()
    spec = spec.from_file(mjcf_path.as_posix())

    if 'flight' in cfg.dataset.env_name:
        body_pitch_angle = 47.5
        stroke_plane_angle = 0.
        worldbody = spec.worldbody
        # Change quaternions of all leg bodies to be the same as the springrefs.
        leg_bodies = [b for b in worldbody.find_all('body') if any_substr_in_str(['T1', 'T2', 'T3'], b.name)]
        for body in leg_bodies:
            body.quat = body_quat_from_springrefs(body)

        # Remove all leg joints, actuators and tendons.
        leg_joints = [b for b in worldbody.find_all('joint') if any_substr_in_str(['T1', 'T2', 'T3'], b.name)]
        for joint in leg_joints:
            spec.delete(joint)#.delete()
            
        leg_acts = [b for b in spec.actuators if any_substr_in_str(['T1', 'T2', 'T3'], b.name)]
        for actuator in leg_acts:
            spec.delete(actuator)#.delete()

        leg_tendons = [b for b in spec.tendons if any_substr_in_str(['T1', 'T2', 'T3'], b.name)]
        for tendon in leg_tendons:
            spec.delete(tendon)#.delete()
        leg_sensors = [b for b in spec.sensors if any_substr_in_str(['T1', 'T2', 'T3'], b.name)]
        for sensor in leg_sensors:
            spec.delete(sensor)#.delete()
        # == Set body pitch angle.
        site_name = 'hover_up_dir-1' if 'pair' in mjcf_path.stem else 'hover_up_dir'
        up_dir = jnp.array(spec.site(site_name).quat)
        up_dir_angle = 2 * jnp.arccos(up_dir[0])
        delta = (jnp.deg2rad(body_pitch_angle) - up_dir_angle)
        dquat = jnp.array([jnp.cos(delta / 2), 0, jnp.sin(delta / 2), 0])
        # Rotate up_dir to new angle.
        up_dir = up_dir.at[:].set(brax_math.quat_mul(dquat, up_dir))
        # == Set stroke plane angle.
        stroke_plane_angle = jnp.deg2rad(stroke_plane_angle)
        stroke_plane_quat = jnp.array([
            jnp.cos(stroke_plane_angle / 2), 0,
            jnp.sin(stroke_plane_angle / 2), 0
        ])
        wing_body_names = ['wing_left-1', 'wing_right-1'] if 'pair' in mjcf_path.stem else ['wing_left', 'wing_right']
        for quat, wing in [(jnp.array([0., 0, 0, 1]), wing_body_names[0]),
                            (jnp.array([0., -1, 0, 0]), wing_body_names[1])]:
            # Rotate wing-joint frame.
            dquat = brax_math.quat_mul(neg_quat(stroke_plane_quat), quat)
            new_wing_quat = brax_math.quat_mul(dquat, neg_quat(up_dir))
            body = spec.body(wing)
            change_body_frame(body, body.pos, new_wing_quat)
        
    mj_model = spec.compile()
    ##### Set solver options ##### 
    mj_model.opt.solver = {
        "cg": mujoco.mjtSolver.mjSOL_CG,
        "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    }["cg"]
    mj_model.opt.iterations = cfg.dataset.env_args.iterations
    mj_model.opt.ls_iterations = cfg.dataset.env_args.ls_iterations
    mj_model.opt.timestep = cfg.dataset.env_args.physics_timestep
    ##### Create mj data #####
    mj_data = mujoco.MjData(mj_model)
    if paired:
        ###### change site colors to show policy and target #####
        site_names = [
            mj_model.site(i).name
            for i in range(mj_model.nsite)
            if "-0" in mj_model.site(i).name
        ]
        site_id = [
            mj_model.site(i).id
            for i in range(mj_model.nsite)
            if "-0" in mj_model.site(i).name
        ]
        for id in site_id:
            mj_model.site(id).rgba = [1, 0, 0, 1]
    return mj_model, mj_data, spec

def local_vel_step(carry):
    vel_t, rot_t = carry
    print(vel_t.shape, rot_t.shape)
    return brax_math.rotate(vel_t, brax_math.quat_inv(rot_t))

def add_trajectory_sites_spheres(spec, n_traj_sites, traj, skip=10, group=4):
    """Add or update trajectory sites as spheres (single size value)."""
    for i in range(0, n_traj_sites, skip):
        x, y, z = traj[i]
        site_name = f'traj_{i}'

        # Try to find existing site
        existing_site = spec.site(site_name)
        if existing_site is not None:
                # Update position of existing site
                existing_site.pos = [x, y, z]
        else:
            # Site doesn't exist, create new one
            spec.worldbody.add_site(name=site_name,
                            pos=[x, y, z],
                            size=(0.005, 0.005, 0.005),
                            rgba=(0, 1, 1, 0.5),
                            group=group,
                            type=mujoco.mjtGeom.mjGEOM_SPHERE)
    return spec

def add_cross_hair_sites(spec, init_pos, group=4):
    """Add or update crosshair sites."""
    x, y, z = init_pos
    fromto_list = [[x, y, z-0.5, x, y, z + 0.5],     # Vertical line (z-axis)
                    [x - 0.5, y, z, x + 0.5, y, z],  # Horizontal line (x-axis)
                    [x, y - 0.5, z, x, y + 0.5, z]]  # Horizontal line (y-axis)

    for i, fromto in enumerate(fromto_list):
        site_name = f'crosshair_{i}'
        # Try to find existing site
        existing_site = spec.site(site_name)
        if existing_site is not None:
            # Update fromto of existing site
            existing_site.fromto = fromto
        else:
            # Site doesn't exist, create new one
            spec.worldbody.add_site(name=site_name,
                            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                            size=(0.002, 0.002, 0.002),
                            fromto=fromto,
                            rgba=(0, 0.8, 0.5, 0.5),
                            group=group)
    return spec



def add_arrow(scene, from_, to, radius=0.03, rgba=[0.2, 0.2, 0.6, 1]):
    """Add an arrow to the scene."""
    scene.geoms[scene.ngeom].category = mujoco.mjtCatBit.mjCAT_STATIC
    mujoco.mjv_initGeom(
        geom=scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=np.asarray(rgba).astype(np.float32),
    )
    mujoco.mjv_connector(
        geom=scene.geoms[scene.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        width=radius,
        from_=from_,
        to=to,
    )
    scene.ngeom += 1

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        print('Warning: scene.ngeom >= scene.maxgeom, cannot add more geoms')
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_connector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.asarray(rgba).astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                        point1, point2)

def add_text_labels(frame, labels, label_clrs, init_xy=(10, 5), font_size=25):
    from PIL import Image, ImageDraw, ImageFont
    # Create a blank image or load an existing one
    width, height = frame.shape[1]//2, frame.shape[0]//2
    img = Image.fromarray(np.asarray(frame).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Define font and text properties
    font_path = "arial.ttf" # Replace with a valid font file path on your system
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except IOError:
        font = ImageFont.load_default() # Fallback if font not found

    for i, label in enumerate(labels):
        # Calculate text position (e.g., center)
        bbox = draw.textbbox((0, 0), label, font=font)
        x = init_xy[0]
        y = init_xy[1] + i * (font_size)  # Adjust vertical spacing
        # Draw the text on the image
        draw.text((x, y), label, fill=label_clrs[i], font=font)
    return np.asarray(img)



def get_gpu_memory_info():
    """Get comprehensive GPU memory information using multiple methods.
    
    Returns:
        dict: Contains total_memory, free_memory, and used_memory lists for each GPU
    """
    n_gpus = jax.device_count(backend="gpu")
    
    # Method 1: Try JAX device memory info (most reliable for JAX workloads)
    try:
        import jax.extend.backend as xb
        devices = jax.devices('gpu')
        total_memory = []
        free_memory = []
        used_memory = []
        
        for device in devices:
            # Get memory info from JAX device
            try:
                # This gives us the memory info from XLA
                backend = xb.get_backend('gpu')
                memory_info = backend.get_memory_info(device.id)
                total_mb = memory_info.bytes_limit // (1024 * 1024)
                used_mb = memory_info.bytes_in_use // (1024 * 1024)
                free_mb = total_mb - used_mb
                
                total_memory.append(total_mb)
                free_memory.append(free_mb)
                used_memory.append(used_mb)
            except (AttributeError, RuntimeError):
                # Fallback for older JAX versions or if memory info unavailable
                raise ValueError("JAX memory info not available")
        
        print("Using JAX device memory information")
        return {
            'total_memory': total_memory,
            'free_memory': free_memory, 
            'used_memory': used_memory,
            'method': 'jax'
        }
    except (ImportError, ValueError, AttributeError):
        pass
    
    # Method 2: Try nvidia-ml-py (more reliable than nvidia-smi)
    try:
        try:
            import pynvml
        except ImportError:
            print("pynvml not found. Attempting to install...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "nvidia-ml-py"])
            import pynvml
        
        pynvml.nvmlInit()
        
        total_memory = []
        free_memory = []
        used_memory = []
        
        for i in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            total_mb = mem_info.total // (1024 * 1024)
            used_mb = mem_info.used // (1024 * 1024)
            free_mb = mem_info.free // (1024 * 1024)
            
            total_memory.append(total_mb)
            free_memory.append(free_mb)
            used_memory.append(used_mb)
        
        print("Using nvidia-ml-py for GPU memory information")
        return {
            'total_memory': total_memory,
            'free_memory': free_memory,
            'used_memory': used_memory,
            'method': 'pynvml'
        }
    except (ImportError, Exception) as e:
        print(f"pynvml method failed: {e}")
        pass
    
    # Method 3: nvidia-smi command line (fallback)
    try:
        # Get total, used, and free memory in one command
        command = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits"
        output = sp.check_output(command.split()).decode("ascii").strip().split("\n")
        
        total_memory = []
        free_memory = []
        used_memory = []
        
        for line in output:
            if line.strip():
                total, used, free = map(int, line.split(', '))
                total_memory.append(total)
                free_memory.append(free)
                used_memory.append(used)
        
        print("Using nvidia-smi for GPU memory information")
        return {
            'total_memory': total_memory,
            'free_memory': free_memory,
            'used_memory': used_memory,
            'method': 'nvidia-smi'
        }
    except (sp.CalledProcessError, FileNotFoundError, ValueError) as e:
        print(f"nvidia-smi method failed: {e}")
        pass
    
    # Method 4: Final fallback with conservative estimates
    print("Warning: No GPU memory detection method available. Using conservative estimates.")
    # Conservative estimates based on common GPU types
    default_total = 11000  # MB - reasonable for mid-range GPUs
    default_used = 500     # MB - assume some baseline usage
    default_free = default_total - default_used
    
    return {
        'total_memory': [default_total] * n_gpus,
        'free_memory': [default_free] * n_gpus,
        'used_memory': [default_used] * n_gpus,
        'method': 'fallback'
    }


def get_jax_memory_usage():
    """Get current JAX memory usage if available.
    
    Returns:
        dict: Memory usage info per device or None if unavailable
    """
    try:
        devices = jax.devices('gpu')
        memory_usage = {}
        
        for device in devices:
            try:
                # Try to get JAX's view of memory usage
                import jax.lib.xla_bridge as xb
                backend = xb.get_backend('gpu')
                memory_info = backend.get_memory_info(device.id)
                
                memory_usage[device.id] = {
                    'bytes_in_use': memory_info.bytes_in_use,
                    'bytes_limit': memory_info.bytes_limit,
                    'mb_in_use': memory_info.bytes_in_use // (1024 * 1024),
                    'mb_limit': memory_info.bytes_limit // (1024 * 1024)
                }
            except (AttributeError, RuntimeError):
                memory_usage[device.id] = None
        
        return memory_usage
    except Exception:
        return None


def load_reference_clip_for_estimation(cfg):
    """
    Load reference clip using the same logic as main training script.
    
    Args:
        cfg: Configuration object
    
    Returns:
        reference_clip: Loaded reference clip data
    """
    try:
        # Import required modules
        import utils.io_dict_to_hdf5 as ioh5
        from utils.data_utils import ReferenceClips
        import jax.numpy as jnp
        
        env_cfg = cfg.dataset
        reference_path = cfg.paths.data_dir / f"clips/{env_cfg['clip_idx']}"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        
        if "flight" in cfg.dataset.env_name:
            reference_clip = ioh5.load(cfg.paths.data_dir / f'clips/{cfg.dataset.clip_idx}')
            reference_clip = {key: jnp.array(value) for key, value in reference_clip.items()}
        else:
            (ref_train, ref_test) = ReferenceClips.from_path(cfg.paths.data_dir/ f"clips/{cfg.dataset.clip_idx}").split()
            reference_clip = ref_train if cfg.dataset.train else ref_test

        print(f"  Successfully loaded reference clip from {reference_path}")
        return reference_clip
        
    except Exception as e:
        print(f"  Failed to load reference clip ({str(e)}), memory estimation will use heuristics")
        return None


def estimate_memory_per_env(cfg, reference_clip=None, test_batch_size=8, safety_margin=1.2):
    """
    Estimate memory consumption per environment by creating a test environment
    and measuring memory usage.
    
    Args:
        cfg: Configuration object
        reference_clip: Reference clip data (will be auto-loaded if None)
        test_batch_size: Number of environments to test with (default: 8)
        safety_margin: Safety factor for memory estimation (default: 1.2 = 20% buffer)
    
    Returns:
        float: Estimated memory per environment in MB
    """
    try:
        from brax import envs
        
        print(f"Estimating memory per environment...")
        
        # Auto-load reference clip if not provided
        if reference_clip is None:
            print("  Loading reference clip for memory estimation...")
            reference_clip = load_reference_clip_for_estimation(cfg)
            
            if reference_clip is None:
                print("  Could not load reference clip, using heuristic estimation...")
                return estimate_memory_heuristic(cfg, safety_margin)
        
        # Get initial memory state
        initial_memory = get_jax_memory_usage()
        
        # Prepare environment arguments without modifying original config
        env_kwargs = {}
        for key, value in cfg.dataset.env_args.items():
            env_kwargs[key] = value
        
        # Add reference clip separately to avoid serialization issues
        env_kwargs['reference_clip'] = reference_clip
        
        # Create a test environment with small batch
        try:
            test_env = envs.get_environment(cfg.dataset.env_name, **env_kwargs)
        except TypeError as e:
            if 'reference_clip' in str(e):
                print("  Environment requires reference_clip but loading failed. Using heuristic estimation...")
                return estimate_memory_heuristic(cfg, safety_margin)
            else:
                raise e
        
        # Reset environments to get initial state
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, test_batch_size)
        
        # Initialize environment states (this allocates memory)
        reset_fn = jax.jit(jax.vmap(test_env.reset))
        states = reset_fn(keys)
        
        # Force compilation and memory allocation
        _ = jax.block_until_ready(states)
        
        # Get memory after environment creation
        final_memory = get_jax_memory_usage()
        
        if initial_memory and final_memory:
            # Calculate memory increase
            device_id = list(initial_memory.keys())[0]
            if (initial_memory[device_id] is not None and 
                final_memory[device_id] is not None):
                
                initial_mb = initial_memory[device_id]['mb_in_use']
                final_mb = final_memory[device_id]['mb_in_use']
                memory_increase = final_mb - initial_mb
                
                if memory_increase > 0:
                    mb_per_env = (memory_increase / test_batch_size) * safety_margin
                    print(f"  Test batch size: {test_batch_size}")
                    print(f"  Memory increase: {memory_increase:.1f} MB")
                    print(f"  Estimated memory per env: {mb_per_env:.1f} MB (with {safety_margin:.1f}x safety margin)")
                    return float(mb_per_env)  # Ensure it's a regular Python float
        
        # Fallback: estimate based on environment characteristics
        print("  Could not measure memory directly, using heuristic estimation...")
        return estimate_memory_heuristic(cfg, safety_margin)
        
    except Exception as e:
        print(f"  Memory estimation failed ({str(e)}), using heuristic estimation...")
        return estimate_memory_heuristic(cfg, safety_margin)


def estimate_memory_heuristic(cfg, safety_margin=1.2):
    """
    Estimate memory per environment using heuristics based on environment characteristics.
    
    Args:
        cfg: Configuration object
        safety_margin: Safety factor for memory estimation
    
    Returns:
        float: Estimated memory per environment in MB
    """
    env_args = cfg.dataset.env_args
    
    # Base memory estimation factors
    base_memory = 50  # MB - basic environment overhead
    
    # Factor in clip length (longer clips = more memory for trajectory storage)
    clip_length_factor = getattr(env_args, 'clip_length', 1000) / 1000.0
    clip_memory = clip_length_factor * 30  # MB per 1000 timesteps
    
    # Factor in reference length
    ref_len_factor = getattr(env_args, 'ref_len', 100) / 100.0
    ref_memory = ref_len_factor * 20  # MB per 100 reference steps
    
    # Factor in environment complexity
    env_name = cfg.dataset.env_name.lower()
    if 'flight' in env_name:
        complexity_memory = 80  # Flight environments are more complex
    elif 'multiclip' in env_name:
        complexity_memory = 60  # Multi-clip environments need more memory
    else:
        complexity_memory = 40  # Single clip environments
    
    # Factor in physics timestep (smaller timesteps = more computation/memory)
    physics_timestep = getattr(env_args, 'physics_timestep', 0.001)
    timestep_factor = 0.001 / physics_timestep  # Normalize to 0.001
    timestep_memory = timestep_factor * 10  # MB adjustment
    
    # Factor in solver iterations
    iterations = getattr(env_args, 'iterations', 1)
    iteration_memory = iterations * 5  # MB per iteration
    
    # Calculate total estimated memory
    total_memory = (base_memory + clip_memory + ref_memory + 
                   complexity_memory + timestep_memory + iteration_memory)
    
    # Apply safety margin
    estimated_memory = total_memory * safety_margin
    
    print(f"  Heuristic estimation breakdown:")
    print(f"    Base memory: {base_memory:.1f} MB")
    print(f"    Clip length factor: {clip_memory:.1f} MB")
    print(f"    Reference length factor: {ref_memory:.1f} MB")
    print(f"    Environment complexity: {complexity_memory:.1f} MB")
    print(f"    Physics timestep factor: {timestep_memory:.1f} MB")
    print(f"    Solver iterations: {iteration_memory:.1f} MB")
    print(f"    Total (with {safety_margin:.1f}x safety): {estimated_memory:.1f} MB")
    
    return float(estimated_memory)  # Ensure it's a regular Python float


def auto_configure_memory_settings(cfg, reference_clip=None, force_recalculate=False):
    """
    Automatically configure memory settings for training.
    
    Args:
        cfg: Configuration object
        reference_clip: Reference clip data (will be auto-loaded if None)
        force_recalculate: If True, recalculates even if mb_per_env is already set
    
    Returns:
        cfg: Updated configuration object
    """
    # Check if mb_per_env is already set and we're not forcing recalculation
    if hasattr(cfg.train_setup, 'mb_per_env') and cfg.train_setup.mb_per_env > 0 and not force_recalculate:
        print(f"Using existing mb_per_env setting: {cfg.train_setup.mb_per_env} MB")
        return cfg
    
    print("Auto-configuring memory settings...")
    
    # Auto-load reference clip if not provided
    if reference_clip is None:
        print("  Loading reference clip for memory configuration...")
        reference_clip = load_reference_clip_for_estimation(cfg)
    
    # Estimate memory per environment (don't store reference_clip in config to avoid serialization issues)
    estimated_mb_per_env = estimate_memory_per_env(cfg, reference_clip=reference_clip)
    
    # Update configuration - ensure train_setup exists and set mb_per_env
    if not hasattr(cfg, 'train_setup'):
        from omegaconf import DictConfig
        cfg.train_setup = DictConfig({})
    if not hasattr(cfg.train_setup, 'mb_per_env'):
        cfg.train_setup.mb_per_env = 0
    
    # Set the estimated value (as a float to avoid serialization issues)
    cfg.train_setup.mb_per_env = float(estimated_mb_per_env)
    
    print(f"Set mb_per_env to {estimated_mb_per_env:.1f} MB")
    
    return cfg


def closest_power_of_two(x):
    # Start with the largest power of 2 less than or equal to x
    power = 1
    while power * 2 <= x:
        power *= 2
    return power


def calculate_optimal_envs(cfg, reference_clip=None, use_free_memory=False, memory_safety_factor=0.85, 
                          min_envs_per_gpu=1, max_envs_per_gpu=None, fallback_envs_per_gpu=32,
                          auto_estimate_mb_per_env=True):
    """
    Automatically calculate optimal number of environments based on available devices and GPU memory.
    
    Args:
        cfg: Configuration object containing train_setup.mb_per_env
        reference_clip: Reference clip data (will be auto-loaded if None and needed)
        use_free_memory (bool): If True, use available memory; if False, use total memory
        memory_safety_factor (float): Safety factor to avoid OOM (0.0-1.0)
        min_envs_per_gpu (int): Minimum environments per GPU
        max_envs_per_gpu (int): Maximum environments per GPU (None for no limit)
        fallback_envs_per_gpu (int): Fallback value if memory calculation fails
        auto_estimate_mb_per_env (bool): If True, automatically estimate mb_per_env if not set
    
    Returns:
        dict: Contains calculated values for envs, batch_size, and memory info
    """
    n_gpus = jax.device_count(backend="gpu")
    
    try:
        # Auto-configure memory settings if needed
        if auto_estimate_mb_per_env:
            if (not hasattr(cfg.train_setup, 'mb_per_env') or 
                cfg.train_setup.mb_per_env <= 0):
                print("mb_per_env not set or invalid, estimating automatically...")
                
                # Auto-load reference clip if not provided
                if reference_clip is None:
                    print("  Loading reference clip for memory estimation...")
                    reference_clip = load_reference_clip_for_estimation(cfg)
                
                cfg = auto_configure_memory_settings(cfg, reference_clip=reference_clip)
        
        # Get comprehensive memory information
        memory_info = get_gpu_memory_info()
        
        if use_free_memory:
            gpu_memories = memory_info['free_memory']
            memory_type = f"free ({memory_info['method']})"
        else:
            gpu_memories = memory_info['total_memory']
            memory_type = f"total ({memory_info['method']})"
        
        # Use the minimum memory across all GPUs (bottleneck approach)
        min_memory = min(gpu_memories) if gpu_memories else 0
        
        if min_memory == 0:
            raise ValueError("No GPU memory detected")
        
        # Apply safety factor to avoid OOM
        usable_memory = int(min_memory * memory_safety_factor)
        
        # Calculate environments per GPU based on memory
        mb_per_env = cfg.train_setup.mb_per_env
        envs_per_gpu_memory = max(1, usable_memory // mb_per_env)
        
        # Round down to nearest power of 2 for better performance
        envs_per_gpu = closest_power_of_two(envs_per_gpu_memory)
        
        # Apply min/max constraints
        envs_per_gpu = max(envs_per_gpu, min_envs_per_gpu)
        if max_envs_per_gpu is not None:
            envs_per_gpu = min(envs_per_gpu, max_envs_per_gpu)
        
        # Calculate memory usage
        memory_per_gpu = envs_per_gpu * mb_per_env
        memory_utilization = (memory_per_gpu / min_memory) * 100 if min_memory > 0 else 0
        
        success = True
        detailed_memory_info = memory_info
        
    except Exception as e:
        print(f"Warning: GPU memory calculation failed ({str(e)}). Using fallback values.")
        envs_per_gpu = fallback_envs_per_gpu
        gpu_memories = ["unknown"] * n_gpus
        min_memory = "unknown"
        usable_memory = "unknown"
        memory_per_gpu = "unknown"
        memory_utilization = "unknown"
        memory_type = "fallback"
        success = False
        detailed_memory_info = None
        
        # Set a default mb_per_env for fallback calculation
        if not hasattr(cfg.train_setup, 'mb_per_env') or cfg.train_setup.mb_per_env <= 0:
            cfg.train_setup.mb_per_env = 256  # Conservative default
    
    # Calculate total environments and batch size
    total_envs = n_gpus * envs_per_gpu
    batch_size = max(1, total_envs // 4)  # Keep the existing ratio
    
    result = {
        'total_envs': total_envs,
        'envs_per_gpu': envs_per_gpu,
        'batch_size': batch_size,
        'n_gpus': n_gpus,
        'success': success,
        'memory_info': {
            'min_gpu_memory_mb': min_memory,
            'usable_memory_mb': usable_memory,
            'memory_per_gpu_mb': memory_per_gpu,
            'memory_utilization_pct': memory_utilization,
            'memory_type': memory_type,
            'safety_factor': memory_safety_factor,
            'mb_per_env': cfg.train_setup.mb_per_env,
            'gpu_memories': gpu_memories,
            'detailed_info': detailed_memory_info
        }
    }
    
    print(f"GPU Environment Calculation:")
    print(f"  GPUs: {n_gpus}")
    if success:
        print(f"  {memory_type.title()} memory per GPU: {gpu_memories} MB")
        print(f"  Min {memory_type.split()[0]} memory: {min_memory} MB")
        print(f"  Usable memory (with {memory_safety_factor:.1%} safety): {usable_memory} MB")
        print(f"  MB per env: {cfg.train_setup.mb_per_env}")
        print(f"  Memory utilization: {memory_utilization:.1f}%")
        
        # Show additional memory details if available
        if detailed_memory_info and detailed_memory_info['method'] != 'fallback':
            total_mem = detailed_memory_info['total_memory']
            used_mem = detailed_memory_info['used_memory']
            free_mem = detailed_memory_info['free_memory']
            print(f"  Detailed memory status:")
            for i in range(len(total_mem)):
                print(f"    GPU {i}: {used_mem[i]}/{total_mem[i]} MB used ({used_mem[i]/total_mem[i]*100:.1f}%), {free_mem[i]} MB free")
    else:
        print(f"  Using fallback calculation")
    print(f"  Envs per GPU: {envs_per_gpu}")
    print(f"  Total envs: {total_envs}")
    print(f"  Batch size: {batch_size}")
    
    return result


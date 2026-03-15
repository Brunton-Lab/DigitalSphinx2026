import sys
from pathlib import Path
import os
import functools

# Set JAX configuration
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

# Add fly_neuromechanics to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Add fly_mimic submodule to path
fly_mimic_path = repo_root / 'fly_mimic'
sys.path.insert(0, str(fly_mimic_path))

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import wandb
import warnings

jax.config.update("jax_default_matmul_precision", "high")  
# Enable persistent compilation cache
jax.config.update("jax_compilation_cache_dir", "tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

import jax.numpy as jnp
from mujoco_playground import wrapper

num_gpus = jax.device_count(backend="gpu")

from utils.path_utils import register_custom_resolvers, convert_dict_to_path, convert_dict_to_string

# Import fly_mimic components
from sphinx_training.training.mlp_ppo import mlp_ppo
from sphinx_training.training.mlp_ppo import ppo_networks
from sphinx_training.utils.data_utils import HDF5ReferenceClips
from sphinx_training.utils.fly_logging import log_eval_rollout

# Import basic environments
from sphinx_training.envs.fruitfly import imitation
from sphinx_training.envs.fruitfly import imitation_unified

# Import neuromechanics support for muscle patching
from fly_neuromechanics.core import support

warnings.filterwarnings("ignore", category=DeprecationWarning)
register_custom_resolvers()


def make_basic_environment(cfg: DictConfig):
    """
    Create basic imitation environment with direct actuator control.
    
    Args:
        cfg: Hydra config from fly_neuromechanics
    
    Returns:
        Environment instance and test clips
    """
    # Load reference clips from fly_mimic data structure
    dataset_path = cfg.paths.data_dir / f"datasets/{cfg.dataset.clip_idx}"
    
    ref_clips = HDF5ReferenceClips(dataset_path)
    train_clips, test_clips = ref_clips.split(test_ratio=cfg.dataset.env_args.test_ratio)
    
    # Load clips into memory (required for JAX tracing)
    train_clips_loaded = train_clips.load_all_clips()
    test_clips_loaded = test_clips.load_all_clips()
    
    # Create environment based on dataset name
    if "unified" in cfg.dataset.env_name:
        env = imitation_unified.ImitationUnified(cfg, reference_clips=train_clips_loaded)
        env_type = "ImitationUnified (adaptive walk/flight control)"
    else:
        env = imitation.Imitation(cfg, reference_clips=train_clips_loaded)
        env_type = "Imitation (direct actuator control)"
    
    # Calculate episode length using fly_mimic formula
    episode_length = (
        cfg.dataset.env_args.clip_length 
        - cfg.dataset.env_args.start_frame_range[-1] 
        - cfg.dataset.env_args.ref_len
    )
    
    print(f"\n{'='*70}")
    print(f"Environment Created Successfully")
    print(f"{'='*70}")
    print(f"Environment type: {env_type}")
    print(f"Action size: {env.action_size} (direct actuator commands)")
    print(f"Episode length: {episode_length} steps")
    print(f"Control timestep: {cfg.dataset.env_args.physics_timestep * cfg.dataset.env_args.physics_steps_per_control_step}s")
    print(f"Physics timestep: {cfg.dataset.env_args.physics_timestep}s")
    print(f"Dataset: {cfg.dataset.clip_idx}")
    print(f"Train clips: {len(train_clips_loaded.clip_lengths)}")
    print(f"Test clips: {len(test_clips_loaded.clip_lengths)}")
    print(f"{'='*70}\n")
    
    return env, test_clips_loaded, episode_length


def make_network_factory(cfg: DictConfig):
    """Create PPO network factory with config from fly_neuromechanics."""
    network_type = cfg.training.network.network_type
    
    if network_type == 'intention':
        return ppo_networks.make_intention_ppo_networks
    elif network_type == 'encoderdecoder':
        return ppo_networks.make_encoderdecoder_ppo_networks
    elif network_type == 'simple_mlp':
        return ppo_networks.make_simple_mlp_ppo_networks
    else:
        raise ValueError(f"Unknown network type: {network_type}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training loop."""
    num_envs = num_gpus * cfg.train_args.num_envs
    print(f"Using {num_gpus} GPUs with total {num_envs} environments.")
    
    # Handle load_jobid for HPC job requeuing
    print("run_id:", cfg.run_id)
    if (
        ("load_jobid" in cfg)
        and (cfg["load_jobid"] is not None)
        and (cfg["load_jobid"] != "")
    ):
        run_id = cfg.load_jobid
        load_cfg_path = (
            Path(cfg.paths.base_dir) / f"{run_id}/logs/run_config.yaml"
        )
        cfg = OmegaConf.load(load_cfg_path)
        continue_training = True
        print(f"Loading job config from: {load_cfg_path}")
    else:
        run_id = cfg.run_id
        continue_training = False

    cfg.paths = convert_dict_to_path(cfg.paths)
    
    # Simple checkpoint discovery - let mlp_ppo.train handle the details
    restore_checkpoint = None
    if cfg.paths.ckpt_dir.exists() and any(cfg.paths.ckpt_dir.iterdir()):
        from natsort import natsorted
        ckpt_files = natsorted([Path(f.path) for f in os.scandir(cfg.paths.ckpt_dir) if f.is_dir()])
        if len(ckpt_files) > 0:
            restore_checkpoint = ckpt_files[-1]  # Latest checkpoint
            continue_training = True
            config_path = Path(cfg.paths.base_dir) / f"{run_id}/logs/run_config.yaml"
            if config_path.exists():
                print(f"Loading job config from: {config_path}")
                cfg = OmegaConf.load(config_path)
                cfg.paths = convert_dict_to_path(cfg.paths)
            print(f"Found checkpoint to resume from: {restore_checkpoint}")
    # Fallback to explicit restore checkpoint if specified and no auto-discovery
    if restore_checkpoint is None and cfg.training.network.restore_checkpoint != "":
        restore_checkpoint = Path(cfg.training.network.restore_checkpoint)
        print(f"Using explicit restore checkpoint: {restore_checkpoint}")
    
    print(f"\n{'='*70}")
    print(f"Basic Imitation Training (Direct Actuator Control)")
    print(f"{'='*70}")
    print(f"Run ID: {run_id}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Environment: {cfg.dataset.env_name}")
    print(f"Network: {cfg.training.network.network_type}")
    print(f"Continue training: {continue_training}")
    if restore_checkpoint:
        print(f"Restore checkpoint: {restore_checkpoint}")
    print(f"{'='*70}\n")
    
    # Patch MJX for custom muscles
    support.patch_mjx_muscle_functions(enable=True)
    print("✓ MJX muscle functions patched")
    
    # Create environment
    train_env, test_clips, episode_length = make_basic_environment(cfg)
    
    # Setup wandb logging
    wandb.init(
        dir=cfg.paths.log_dir,
        project=cfg.train_setup.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),
        notes=cfg.get("note", "Basic imitation training with direct actuator control"),
        id=f"{run_id}",
        resume="allow",
    )
    wandb.run.name = f"{cfg.train_setup['task_name']}_{cfg.train_setup['algo_name']}_{run_id}"
    
    def wandb_progress(num_steps, metrics):
        num_steps = int(num_steps)
        metrics["num_steps"] = num_steps
        wandb.log(metrics, commit=False)
    
    # Setup evaluation rollout function
    reference_clips = train_env.reference_clips
    jit_reset = jax.jit(train_env.reset)
    jit_step = jax.jit(train_env.step)
    
    def policy_params_fn(
        eval_steps,
        num_steps,
        make_policy,
        params,
        policy_params_fn_key,
        decoder_params=None,
        rollout_env=train_env,
        jit_reset=jit_reset,
        jit_step=jit_step,
    ):
        # Calculate current eval step from num_steps
        print(f"Eval Step: {eval_steps}, num_steps: {num_steps}")
        policy_params = (params[0], params[1])
        
        jit_inference_fn = jax.jit(make_policy(policy_params, deterministic=True))
        reset_rng, act_rng = jax.random.split(policy_params_fn_key)

        state = jit_reset(reset_rng)
        init_eval_step = eval_steps * jnp.ones_like(rollout_env._get_cur_frame(state.data, state.info))
        state.info["eval_step"] = init_eval_step

        rollout = [state]
        extras_all = []
        rollout_len = reference_clips.clip_lengths[state.info['reference_clip']].astype(int)-1
        for _ in range(rollout_len):
            _, act_rng = jax.random.split(act_rng)
            obs = state.obs
            ctrl, extras = jit_inference_fn(obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)
            extras_all.append(extras)
        
        log_eval_rollout(
            cfg, rollout, state, rollout_env, reference_clips, cfg.paths.ckpt_dir, eval_steps,
        )
    
    # Create network factory with proper parameters
    network_factory_fn = make_network_factory(cfg)
    network_factory = functools.partial(
        network_factory_fn,
        **OmegaConf.to_container(cfg.training.network.network_factory_args, resolve=True)
    )
    
    # Training configuration for mlp_ppo.train()
    train_config = {
        'num_timesteps': cfg.train_setup.num_timesteps,
        'episode_length': episode_length,
        'num_envs': num_envs,
        'learning_rate': cfg.train_args.learning_rate,
        'batch_size': cfg.train_args.batch_size,
        'num_minibatches': cfg.train_args.num_minibatches,
        'unroll_length': cfg.train_args.unroll_length,
        'normalize_observations': cfg.train_args.normalize_observations,
        'action_repeat': cfg.train_args.action_repeat,
        'num_updates_per_batch': cfg.train_args.num_updates_per_batch,
        'entropy_cost': cfg.train_args.entropy_cost,
        'discounting': cfg.train_args.discounting,
        'reward_scaling': cfg.train_args.reward_scaling,
        'gae_lambda': 0.95,
        'clipping_epsilon': cfg.train_args.clipping_epsilon,
        'max_grad_norm': cfg.train_args.max_grad_norm,
        'kl_loss': cfg.train_args.kl_loss,
        'kl_weight': cfg.train_args.kl_weight,
        'use_kl_schedule': cfg.train_args.use_kl_schedule,
        'kl_ramp': cfg.train_args.kl_ramp,
        'kl_ramp_cycles': cfg.train_args.kl_ramp_cycles,
        'warmup_steps': cfg.train_args.warmup_steps,
        'network_factory': network_factory,
        'seed': cfg.train_args.seed,
    }
    
    # Add eval and checkpoint parameters
    num_evals = int(cfg.train_setup.num_timesteps / cfg.train_setup.eval_every)
    num_resets_per_eval = cfg.train_setup.eval_every // cfg.train_setup.reset_every
    
    # Determine number of GPUs for pmap
    use_pmap_on_reset = True if num_gpus > 1 else False
    
    print("\nStarting PPO training...")
    print(f"Total timesteps: {train_config['num_timesteps']:,}")
    print(f"Evaluations: {num_evals}")
    print(f"Num resets per eval: {num_resets_per_eval}")
    print(f"Batch size: {train_config['batch_size']}")
    print(f"Num envs: {train_config['num_envs']}")
    print(f"Learning rate: {train_config['learning_rate']}")
    print(f"Episode length: {train_config['episode_length']}")
    print(f"Using pmap on reset: {use_pmap_on_reset}\n")
    
    # Save config if first run
    if not (cfg.paths.log_dir / "run_config.yaml").exists():
        cfg_temp = cfg.copy()
        cfg_temp.paths = convert_dict_to_string(cfg_temp.paths)
        OmegaConf.save(cfg_temp, cfg.paths.log_dir / "run_config.yaml")
        print(OmegaConf.to_yaml(cfg_temp, resolve=True))
    else:
        print(f"Config already exists: {cfg.paths.log_dir / 'run_config.yaml'}")
        print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # Run training with all parameters from mlp_ppo.train
    make_inference_fn, params, metrics = mlp_ppo.train(
        environment=train_env,
        num_timesteps=train_config['num_timesteps'],
        num_evals=num_evals,
        num_resets_per_eval=num_resets_per_eval,
        episode_length=train_config['episode_length'],
        num_envs=train_config['num_envs'],
        learning_rate=train_config['learning_rate'],
        seed=train_config['seed'],
        batch_size=train_config['batch_size'],
        num_minibatches=train_config['num_minibatches'],
        unroll_length=train_config['unroll_length'],
        normalize_observations=train_config['normalize_observations'],
        action_repeat=train_config['action_repeat'],
        num_updates_per_batch=train_config['num_updates_per_batch'],
        entropy_cost=train_config['entropy_cost'],
        discounting=train_config['discounting'],
        reward_scaling=train_config['reward_scaling'],
        clipping_epsilon=train_config['clipping_epsilon'],
        network_factory=train_config['network_factory'],
        progress_fn=wandb_progress,
        policy_params_fn=policy_params_fn,
        wrap_env=True,  # Enable environment wrapping
        wrap_env_fn=wrapper.wrap_for_brax_training,
        save_checkpoint_path=str(cfg.paths.ckpt_dir),
        checkpoint_path=restore_checkpoint,
        continue_training=continue_training,
        use_pmap_on_reset=use_pmap_on_reset,
        max_grad_norm=train_config['max_grad_norm'],
        gae_lambda=train_config['gae_lambda'],
        kl_loss=train_config.get('kl_loss', False),
        kl_weight=train_config.get('kl_weight', 0.0),
        use_kl_schedule=train_config.get('use_kl_schedule', False),
        kl_ramp=train_config.get('kl_ramp', 1.0),
        kl_ramp_cycles=train_config.get('kl_ramp_cycles', 1),
        warmup_steps=train_config.get('warmup_steps', 0),
    )
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Final metrics: {metrics}")
    print(f"{'='*70}\n")
    
    return params, metrics


if __name__ == '__main__':
    main()

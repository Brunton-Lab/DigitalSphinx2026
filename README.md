# DigitalSphinx

Integrated neuromechanical simulation for *Drosophila* combining C. elegans connectome dynamics with physics-based body models.

This project trains reinforcement learning policies for fruit fly locomotion control using JAX, MuJoCo/MJX, and Brax. It supports imitation learning from real fly kinematics data with connectome-informed neural architectures.

## Requirements

- Python >= 3.12

## Installation

### uv (recommended)

```bash
# CPU only
uv sync

# GPU (CUDA 12)
uv sync --extra gpu
```

### Conda / Mamba

```bash
# GPU (CUDA 12)
conda env create -f environment-gpu.yaml
conda activate sphinx

# CPU only
conda env create -f environment-cpu.yaml
conda activate sphinx-cpu
```

### pip

```bash
# Basic install
pip install -e .

# With GPU support and dev tools
pip install -e ".[gpu,dev]"
```

## Quick Start

```bash
# Run training with default config
python scripts/train_basic_imitation.py

# Override config groups from the command line
python scripts/train_basic_imitation.py paths=hyak dataset=imitation_walk_anipose_data_v1 seed=42
```

Training logs to [Weights & Biases](https://wandb.ai) by default.

## Configuration System

The project uses [Hydra](https://hydra.cc/) with [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management. All configs live in the `configs/` directory.

### Directory Layout

```
configs/
  config.yaml                  # Root config — composes everything below
  training/
    ppo_base.yaml              # Base PPO hyperparameters
    ppo_basic_imitation_low_kl.yaml  # Training variant with lower KL weight
    anatomy/
      v1.yaml                  # Anatomical model (legs, joints, wings)
    network/
      intention.yaml           # Network architecture selection
  connectome/
    C_elegans_VRNN.yaml        # Connectome weights and neuron indices
  dataset/
    imitation_walk_anipose_data_v1.yaml  # Dataset and environment parameters
  paths/
    default.yaml               # Local workstation paths
    hyak.yaml                  # UW Hyak cluster paths
    tillicum.yaml              # Tillicum cluster paths
```

### How It Works

The root `configs/config.yaml` declares a `defaults:` list that tells Hydra which sub-configs to compose together:

```yaml
defaults:
  - _self_
  - training: ppo_basic_imitation_low_kl
  - training/anatomy: v1
  - connectome: C_elegans_VRNN
  - paths: default
  - dataset: imitation_walk_anipose_data_v1
```

Hydra merges all of these into a single resolved config object that gets passed to the training script. Each sub-config can define its own nested keys using the `@package` directive (e.g., `@package _global_` to merge at root level).

### Overriding Configs

Any config group or value can be overridden from the command line:

```bash
# Switch to a different path template for cluster runs
python scripts/train_basic_imitation.py paths=hyak

# Change multiple groups at once
python scripts/train_basic_imitation.py paths=hyak seed=123 run_id=my_experiment

# Override nested values directly
python scripts/train_basic_imitation.py training.train_args.num_envs=2048
```

### Path Templates

The `configs/paths/` directory contains environment-specific path configs so the same codebase works across different machines. Each file defines base directories, data paths, and save locations using OmegaConf interpolation:

- `default.yaml` — local workstation
- `hyak.yaml` — UW Hyak cluster
- `tillicum.yaml` — Tillicum cluster

Switch environments by overriding the `paths` group: `paths=hyak`.

### Custom Resolvers

The project registers custom OmegaConf resolvers (in `sphinx_training/utils/path_utils.py`) that can be used in YAML configs:

| Resolver | Description |
|----------|-------------|
| `${multirun_save_dir:base,run_id}` | Generates save directories aware of Hydra multirun |
| `${eq:a,b}` | Case-insensitive string equality check |
| `${divide:x,y}` | Integer division |
| `${contains:x,y}` | Case-insensitive substring check |
| `${resolve_default:default,arg}` | Returns `default` if `arg` is empty, otherwise `arg` |

## License

MIT

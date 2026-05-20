# DigitalSphinx2026 Repo

[<ins>**The digital sphinx: Can a worm brain control a fly body?**</ins>](https://www.biorxiv.org/cgi/content/short/2026.03.20.713233v1)  
Bingni W. Brunton\*, Elliott T.T. Abe\*, Lawrence Jianqiao Hu, and John C. Tuthill. biorxiv, 2026. doi: 10.64898/2026.03.20.713233  
\* These authors contributed equally to this work

[Video](https://www.youtube.com/watch?v=3KBZ6nrZxDY)  


<video width="630" height="300" src="https://github.com/user-attachments/assets/66840ff4-9779-43ce-99e8-61cdfb8b26c1" autoplay loop muted playsinline></video>



## Abstract
Animal intelligence is not purely a product of abstract computation in the brain, but emerges from dynamic interactions between the nervous system and the body. New connectome datasets and musculoskeletal models now enable integrated, closed-loop simulations of the neural and biomechanical systems of the fruit fly Drosophila, an ideal model organism to investigate embodied intelligence. However, many biological parameters of the nervous system and the body, as well as how they interface, remain unknown. To fill such gaps, researchers are turning to deep reinforcement learning (DRL), a data-driven optimization framework, to create virtual animals that imitate the behavior of real animals. Here, we provide a cautionary tale about the interpretation of such models. We constructed a virtual chimera of two phylogenetically distant species: a connectome of the C. elegans nematode worm and a biomechanical model of the fly body. The worm connectome receives sensory information from the fly body, and an artificial neural network is trained with DRL to map worm motor neuron activations to the fly’s leg actuators. The resulting digital sphinx produces highly realistic fly walking—yet it is biologically meaningless. This exercise teaches us nothing about either animal and exposes a core peril of connectome-body models: behavioral fidelity is achievable without biological fidelity, making such models easy to overinterpret. Done carefully, virtual animals can be powerful partners to biological experiments, but only if their components and interfaces are grounded in biology.

## Hardware requirements

| Task | Hardware |
| --- | --- |
| Training the policy | **NVIDIA GPU with CUDA 12 is required.** Tested on A100 / H100. Single-GPU is sufficient. |
| Visualization, rollout rendering, notebooks | CPU is sufficient (Linux x86_64 or Apple Silicon macOS). |

Training on macOS is not supported — JAX falls back to CPU and the PPO loop is impractically slow. Use a Linux machine or HPC cluster for training; use any machine for visualization.

Python 3.12 or newer is required.

## Installation

The repo uses a minimal conda environment for Python + system deps (mainly `ffmpeg`), then `uv pip` to install Python packages from [pyproject.toml](pyproject.toml).

```bash
# 1. Create the bootstrap environment (Python 3.12 + uv + ffmpeg).
conda env create -f environment.yaml
conda activate sphinx

# 2. Install the project. Pick the right command for your hardware:

# Linux + NVIDIA GPU (training and viz):
uv pip install -e ".[gpu]"

# macOS or any CPU-only machine (viz only):
uv pip install -e .
```

Both commands install the notebook and dev tooling (`jupyter`, `ipython`, `pytest`, `black`, `mypy`, `flake8`) as part of the base dependencies, so you can launch `jupyter lab` and run the visualization notebooks immediately. They also pull in [`mujoco_visualizer`](https://github.com/elliottabe/mujoco_visualizer), which powers the rendering, camera presets, and pan animations used by `notebooks/Viz.ipynb`.

To verify the install picked up the right backend:

```bash
python -c "import jax; print(jax.devices())"
# Expect: [CudaDevice(id=0)] on a GPU machine, [CpuDevice(id=0)] on macOS / CPU.
```

## Quick Start

### Visualize a trained policy (no GPU needed)

Open `notebooks/RL_basic_viz.ipynb` and run all cells. It loads a saved policy
rollout and renders the fly walking. Set `SPHINX_BASE_DIR` to the directory
where your training outputs live before launching Jupyter, e.g.:

```bash
export SPHINX_BASE_DIR=$HOME/TheSphinx/walk
jupyter lab notebooks/RL_basic_viz.ipynb
```

### Train a policy (GPU required)

```bash
# Default training run.
python scripts/train_basic_imitation.py

# Override config groups from the command line, e.g. to switch to your own
# paths config (see `configs/paths/`).
python scripts/train_basic_imitation.py paths=mylaptop dataset=imitation_walk_anipose_data_v1 seed=42
```

Training logs to [Weights & Biases](https://wandb.ai) by default.

## HPC submission

The [scripts/](scripts/) directory ships two reference SLURM submission wrappers from the original lab environment:

- `scripts/klone_run.py` — UW Hyak (klone) HPC
- `scripts/tillicum_run.py` — Tillicum HPC

Both hardcode lab-specific SBATCH directives (account, partition, email address). They are useful as a *starting point* if you want to run training on a SLURM cluster; copy one of them to a new file and adjust the directives for your site.

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
    template.yaml              # Annotated template — copy this for new machines
    default.yaml               # Original-lab local workstation paths
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

The `configs/paths/` directory contains environment-specific path configs so the same codebase works across different machines. Each file defines base directories, data paths, and save locations using OmegaConf interpolation. The `user` field auto-resolves from the `$USER` environment variable, so on a typical machine you only need to update `project_dir`, `base_dir`, and `data_dir`.

**Setting up a new machine:**

```bash
# Copy the template to a name that identifies your machine.
cp configs/paths/template.yaml configs/paths/mylaptop.yaml

# Edit the three CHANGEME entries in mylaptop.yaml (project_dir, base_dir, data_dir).

# Pass `paths=mylaptop` to any script.
python scripts/train_basic_imitation.py paths=mylaptop
```

The shipped configs (`default.yaml`, `hyak.yaml`, `tillicum.yaml`) are reference layouts for the original lab. They are kept in the repo as concrete examples — you do not need to use them.

## License

GPL

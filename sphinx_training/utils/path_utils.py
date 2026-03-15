from pathlib import Path
from omegaconf import OmegaConf
import logging
import os
from hydra import initialize, compose
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)


def register_custom_resolvers():
    """Register custom OmegaConf resolvers for interpolations."""

    
    def multirun_aware_save_dir_resolver(base_dir: str, run_id: str):
        """
        Custom resolver that checks if it's a multirun.
        If multirun: use hydra.runtime.output_dir (which will be the subdir in sweep)
        If not multirun: use ${base_dir}/${run_id}/
        
        Args:
            base_dir: The base directory path
            run_id: The run ID
        """
        try:
            # Check if we're in a multirun context
            from hydra.core.hydra_config import HydraConfig
            
            # Try to get the current Hydra config
            if HydraConfig.initialized():
                hydra_cfg = HydraConfig.get()
                # Check if it's a multirun by looking at the job config
                is_multirun = hydra_cfg.mode.name == "MULTIRUN"
                
                if is_multirun:
                    # For multirun, use the current output directory (which is the subdir)
                    # This should be something like .../Testing/sim.noiseStdvProp=0.0/
                    hydra_output_dir = hydra_cfg.runtime.output_dir
                    if hydra_output_dir:
                        return str(hydra_output_dir)
                    
                    # Fallback: construct the path manually using override_dirname
                    override_dirname = hydra_cfg.job.override_dirname
                    if override_dirname:
                        return f"{base_dir}/{run_id}/{override_dirname}"
                    else:
                        return f"{base_dir}/{run_id}"
                else:
                    # For single run, use the standard path
                    return f"{base_dir}/{run_id}"
            else:
                # If Hydra not initialized, assume single run
                return f"{base_dir}/{run_id}"
                
        except Exception as e:
            logger.debug(f"Could not determine if multirun: {e}")
            # If we can't determine, use the standard path
            return f"{base_dir}/{run_id}"
    
    ##### Custom Resolver #####
    # Register all resolvers (with replace=True to avoid conflicts if already registered)
    OmegaConf.register_new_resolver(
        "multirun_save_dir",
        multirun_aware_save_dir_resolver,
        use_cache=False,
        replace=True
    )
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower(), use_cache=False,replace=True)
    OmegaConf.register_new_resolver('divide', lambda x, y: x//y, use_cache=False,replace=True)
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower(), use_cache=False,replace=True)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg, use_cache=False,replace=True)

# Auto-register when module is imported
register_custom_resolvers()


def convert_to_string(value):
    """Convert a value to a string, handling Path objects."""
    if isinstance(value, Path):
        return str(value)
    return value

def convert_to_path(value):
    """Convert a value to a Path object, handling strings."""
    if isinstance(value, str):
        return Path(value)
    return value

def convert_dict_to_string(d):
    """Convert all values in a dictionary to strings."""
    return {k: convert_to_string(v) for k, v in d.items()}

def convert_dict_to_path(d):
    """Convert all values in a dictionary to Path objects."""
    for k in d.keys():
        if k != 'user':
            d[k] = convert_to_path(d[k])
            d[k].mkdir(parents=True, exist_ok=True)
    return d


def save_config(cfg, path):
    """Save the configuration to a file."""
    # Create a copy of the config to avoid modifying the original
    cfg_copy = OmegaConf.create(cfg)
    
    # Resolve all interpolations in the paths section before converting to strings
    if 'paths' in cfg_copy:
        # Resolve interpolations first
        OmegaConf.resolve(cfg_copy.paths)
        # Then convert to strings
        cfg_copy.paths = convert_dict_to_string(cfg_copy.paths)
    
    OmegaConf.save(cfg_copy, path)
    

def smart_path_replacement(original_paths, target_template_paths, preserve_relative=True, verbose=True):
    """
    Intelligently replace paths by mapping base directories while preserving relative structure.
    
    Args:
        original_paths: Dict of original paths from loaded config
        target_template_paths: Dict of target paths from template
        preserve_relative: Whether to preserve relative path structure
        verbose: Whether to print detailed information about path analysis and mapping
    
    Returns:
        Dict with updated paths
    """
    from pathlib import Path
    
    # Convert everything to Path objects for easier manipulation
    orig_paths = {k: Path(v) if isinstance(v, str) else v for k, v in original_paths.items()}
    target_paths = {k: Path(v) if isinstance(v, str) else v for k, v in target_template_paths.items()}
    
    if verbose:
        print("🔍 Analyzing path differences...")
    
    # Find common base mappings between original and target
    base_mappings = {}
    
    # Look for key paths that define the base structure
    key_paths = ['cwd_dir', 'base_dir', 'data_dir']
    
    for key in key_paths:
        if key in orig_paths and key in target_paths:
            orig_base = orig_paths[key]
            target_base = target_paths[key]
            if verbose:
                print(f"   {key}: {orig_base} -> {target_base}")
            base_mappings[str(orig_base)] = str(target_base)
    
    # If we don't have direct mappings, try to infer them
    if not base_mappings and preserve_relative:
        # Try to find common patterns
        orig_str_paths = [str(p) for p in orig_paths.values() if isinstance(p, Path)]
        target_str_paths = [str(p) for p in target_paths.values() if isinstance(p, Path)]
        
        # Find the longest common prefixes that differ
        orig_prefixes = set()
        target_prefixes = set()
        
        for path_str in orig_str_paths:
            parts = Path(path_str).parts
            for i in range(1, len(parts)):
                orig_prefixes.add(str(Path(*parts[:i])))
        
        for path_str in target_str_paths:
            parts = Path(path_str).parts
            for i in range(1, len(parts)):
                target_prefixes.add(str(Path(*parts[:i])))
        
        # Find potential mappings
        common_suffixes = {}
        for orig_path in orig_str_paths:
            for target_path in target_str_paths:
                orig_parts = Path(orig_path).parts
                target_parts = Path(target_path).parts
                
                # Find common suffix
                min_len = min(len(orig_parts), len(target_parts))
                for i in range(1, min_len + 1):
                    if orig_parts[-i:] == target_parts[-i:]:
                        orig_base = str(Path(*orig_parts[:-i])) if len(orig_parts) > i else str(Path(orig_parts[0]))
                        target_base = str(Path(*target_parts[:-i])) if len(target_parts) > i else str(Path(target_parts[0]))
                        if orig_base != target_base:
                            base_mappings[orig_base] = target_base
                            break
    
    # Apply the base mappings
    updated_paths = {}
    for key, orig_path in orig_paths.items():
        if key == 'user':
            # Keep user as-is or update from target
            updated_paths[key] = target_paths.get(key, orig_path)
            continue
            
        orig_str = str(orig_path)
        updated_str = orig_str
        
        # Apply base mappings (longest match first)
        for orig_base, target_base in sorted(base_mappings.items(), key=len, reverse=True):
            if orig_str.startswith(orig_base):
                updated_str = orig_str.replace(orig_base, target_base, 1)
                if verbose:
                    print(f"   Mapping {key}: {orig_str} -> {updated_str}")
                break
        
        # If no mapping found, use the target template if available
        if updated_str == orig_str and key in target_paths:
            updated_str = str(target_paths[key])
            if verbose:
                print(f"   Direct replacement {key}: {orig_str} -> {updated_str}")
        
        updated_paths[key] = updated_str
    
    return updated_paths

def replace_paths_with_template(cfg, paths_template="workstation", config_dir="../configs", verbose=True):
    """
    Replace all paths in the config with paths from a specified template YAML file.
    
    This function loads a paths template (e.g., glados.yaml) and uses it to replace
    the paths in the current config, while preserving dataset-specific information.
    
    Args:
        cfg: The configuration object to update
        paths_template: Name of the paths template (e.g., "glados", "hyak")
        config_dir: Directory containing the configs
        verbose: Whether to print detailed information about path replacements
    
    Returns:
        Updated configuration with new paths
    """
    from pathlib import Path
    from omegaconf import OmegaConf
    
    # Handle both absolute and relative config_dir paths
    config_dir_path = Path(config_dir)
    if not config_dir_path.is_absolute():
        # Make relative to the current working directory
        config_dir_path = Path.cwd() / config_dir_path
    
    # Resolve to absolute path
    config_dir = config_dir_path.resolve()
    
    paths_file = Path(config_dir) / "paths" / f"{paths_template}.yaml"
    
    if not paths_file.exists():
        if verbose:
            print(f"❌ Paths template file not found: {paths_file}")
            print(f"   Looked in: {paths_file.absolute()}")
        return cfg
    
    if verbose:
        print(f"🔄 Loading paths template from: {paths_file}")
    
    # Load the paths template
    paths_template_cfg = OmegaConf.load(paths_file)
    
    # Store original dataset and version info from current config
    experiment_name = cfg.get('dataset', {}).get('name', 'unknown_experiment')
    version = cfg.get('version', 'debug')
    run_id = cfg.get('run_id', 'Testing')
    
    # Create a temporary config for path resolution
    temp_cfg = OmegaConf.create({
        'paths': paths_template_cfg,
        'dataset': {'name': experiment_name},
        'version': version,
        'run_id': run_id
    })
    
    # Resolve all interpolations in the paths
    try:
        OmegaConf.resolve(temp_cfg.paths)
        resolved_template_paths = temp_cfg.paths
        
        # Get original paths (resolve them too if possible)
        original_paths = cfg.get('paths', {})
        if original_paths:
            try:
                # Try to resolve original paths as well
                temp_orig = OmegaConf.create({
                    'paths': original_paths,
                    'dataset': {'name': experiment_name},
                    'version': cfg.get('version', version),
                    'run_id': cfg.get('run_id', run_id)
                })
                OmegaConf.resolve(temp_orig.paths)
                resolved_original_paths = temp_orig.paths
            except:
                resolved_original_paths = original_paths
                
            # Use smart path replacement
            updated_paths = smart_path_replacement(
                resolved_original_paths, 
                resolved_template_paths,
                verbose=verbose
            )
            cfg.paths = OmegaConf.create(updated_paths)
        else:
            cfg.paths = resolved_template_paths
        
        if verbose:
            print(f"✅ Successfully applied template '{paths_template}':")
            print(f"   Base dir: {cfg.paths.base_dir}")
            print(f"   Save dir: {cfg.paths.save_dir}")
            print(f"   Data dir: {cfg.paths.data_dir}")
        
    except Exception as e:
        print(f"❌ Error resolving paths template: {e}")
        print("Falling back to original paths...")
    
    return cfg


def load_config_with_path_template(config_path, paths_template=None, dataset=None, version=None, run_id=None, config_dir="configs", verbose=False):
    """
    Load a config file and replace paths using a specified template.
    
    Args:
        config_path: Path to the config file to load
        paths_template: Name of the paths template (e.g., "glados", "hyak")
        dataset: Dataset name (if not in config)
        version: Version name (if not in config) 
        run_id: Run ID for path generation
        config_dir: Directory containing the configs
    
    Returns:
        Config with updated paths from template
    """
    
    print(f"📁 Loading config from: {config_path}")
    
    # Load the config
    cfg = OmegaConf.load(config_path)
    
    # If no paths_template specified, keep original paths
    if paths_template is None:
        if verbose:
            print("🔄 No paths template specified, keeping original paths")
        return cfg
    # Override dataset/version if provided
    if dataset:
        if 'dataset' not in cfg:
            cfg.dataset = {}
        cfg.dataset.name = dataset
    
    if version:
        cfg.version = version
        
    if run_id is not None:
        cfg.run_id = run_id
    
    # Resolve config_dir to full path before passing to replace_paths_with_template
    config_dir_path = Path(config_dir)
    if not config_dir_path.is_absolute():
        config_dir_path = Path.cwd() / config_dir_path
    config_dir_resolved = config_dir_path.resolve()
    
    # Replace paths using the specified template
    cfg = replace_paths_with_template(cfg, paths_template, str(config_dir_resolved), verbose=verbose)
    
    return cfg

def create_fresh_config_with_paths(dataset, paths_template="workstation", version="debug", run_id="Testing", config_dir="../configs", verbose=True):
    """
    Create a fresh config using Hydra with specified paths template.
    
    Args:
        dataset: Experiment name
        paths_template: Name of the paths template (e.g., "glados", "hyak")
        version: Version name
        run_id: Run ID for path generation
        config_dir: Directory containing the configs
        verbose: Whether to print detailed information
    
    Returns:
        Fresh config with paths from template
    """
    if verbose:
        print(f"🔄 Creating fresh config for dataset '{dataset}' with paths template '{paths_template}'")
    
    # Handle both absolute and relative config_dir paths
    config_dir_path = Path(config_dir)
    if not config_dir_path.is_absolute():
        config_dir_path = Path.cwd() / config_dir_path
    config_dir = str(config_dir_path.resolve())
    
    try:
        with initialize(version_base=None, config_path=config_dir):
            # Build overrides list
            overrides = [
                f"dataset={dataset}", 
                f"version={version}", 
                f'run_id={run_id}'
            ]
            
            # Add paths override if specified
            if paths_template:
                overrides.append(f"paths={paths_template}")
            
            if verbose:
                print(f"   Hydra overrides: {overrides}")
            
            cfg = compose(
                config_name='config.yaml',
                overrides=overrides,
                return_hydra_config=True
            )
            
            # Set the Hydra config for proper multirun handling
            HydraConfig.instance().set_config(cfg)
    
        if verbose:
            print(f"✅ Created fresh config:")
            print(f"   Experiment: {cfg.dataset.name}")
            print(f"   Version: {cfg.version}")
            print(f"   Paths template: {paths_template}")
            if hasattr(cfg, 'paths') and hasattr(cfg.paths, 'save_dir'):
                print(f"   Save dir: {cfg.paths.save_dir}")
            else:
                print("   ⚠️  Paths not resolved yet")
    
    except Exception as e:
        if verbose:
            print(f"❌ Error creating fresh config: {e}")
        raise
    return cfg


def override_config_paths(cfg, new_paths_template, config_dir="configs", verbose=True):
    """
    Override the paths in an existing config with a new paths template.

    This function takes an already loaded config and replaces its paths section
    with paths from a different template, preserving all other config settings.

    Args:
        cfg: The existing configuration object to update
        new_paths_template: Name of the new paths template (e.g., "workstation", "hyak", "desktop")
        config_dir: Directory containing the configs (relative to current working directory)
        verbose: Whether to print detailed information about the override

    Returns:
        The updated configuration object with new paths

    Example:
        # Load a config with one set of paths
        cfg = load_config_with_path_template("config.yaml", "hyak")

        # Later, override with different paths
        cfg = override_config_paths(cfg, "workstation")
    """
    if verbose:
        old_base_dir = cfg.get('paths', {}).get('base_dir', 'unknown')
        print(f"🔄 Overriding config paths with template '{new_paths_template}'")
        print(f"   Current base_dir: {old_base_dir}")

    # Handle both absolute and relative config_dir paths
    config_dir_path = Path(config_dir)
    if not config_dir_path.is_absolute():
        config_dir_path = Path.cwd() / config_dir_path
    config_dir = str(config_dir_path.resolve())

    paths_file = Path(config_dir) / "paths" / f"{new_paths_template}.yaml"

    if not paths_file.exists():
        if verbose:
            print(f"❌ Paths template file not found: {paths_file}")
            available_templates = list(Path(config_dir).glob("paths/*.yaml"))
            if available_templates:
                print(f"   Available templates: {[f.stem for f in available_templates]}")
        return cfg

    if verbose:
        print(f"📂 Loading new paths template from: {paths_file}")

    # Load the new paths template
    new_paths_template_cfg = OmegaConf.load(paths_file)

    # Extract experiment info from current config
    experiment_name = cfg.get('dataset', {}).get('name', 'unknown_experiment')
    version = cfg.get('version', 'debug')
    run_id = cfg.get('run_id', 'Testing')

    # Create a temporary config for path resolution with the new template
    temp_cfg = OmegaConf.create({
        'paths': new_paths_template_cfg,
        'dataset': {'name': experiment_name},
        'version': version,
        'run_id': run_id
    })

    # Resolve all interpolations in the new paths
    try:
        OmegaConf.resolve(temp_cfg.paths)
        resolved_new_paths = temp_cfg.paths

        # Replace the paths in the original config
        cfg.paths = resolved_new_paths

        if verbose:
            print(f"✅ Successfully overridden paths with template '{new_paths_template}':")
            print(f"   New base_dir: {cfg.paths.base_dir}")
            print(f"   New save_dir: {cfg.paths.save_dir}")
            print(f"   New data_dir: {cfg.paths.data_dir}")

    except Exception as e:
        if verbose:
            print(f"❌ Error resolving new paths template: {e}")
            print("Keeping original paths...")

    return cfg


def load_config_and_override_paths(config_path, new_paths_template, config_dir="configs", verbose=False):
    """
    Load a config file and override its paths with a new paths template.

    This is a convenience function that combines loading a config file with
    overriding its paths section using a different template.

    Args:
        config_path: Path to the config file to load (can be absolute or relative)
        new_paths_template: Name of the new paths template (e.g., "workstation", "hyak", "desktop")
        config_dir: Directory containing the configs (relative to current working directory)
        verbose: Whether to print detailed information about loading and override

    Returns:
        Configuration object with paths overridden from the new template

    Example:
        # Load any config file and immediately override paths
        cfg = load_config_and_override_paths("configs/config.yaml", "workstation")

        # Load a saved config and switch to cluster paths
        cfg = load_config_and_override_paths("outputs/run_123/config.yaml", "hyak")

        # Load from absolute path
        cfg = load_config_and_override_paths("/path/to/saved/config.yaml", "desktop")
    """
    from pathlib import Path

    if verbose:
        print(f"📁 Loading config from: {config_path}")

    # Handle both absolute and relative config paths
    config_file_path = Path(config_path)
    if not config_file_path.is_absolute():
        config_file_path = Path.cwd() / config_file_path

    if not config_file_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    # Load the config file
    cfg = OmegaConf.load(config_file_path)

    if verbose:
        original_template = "unknown"
        if hasattr(cfg, 'paths') and hasattr(cfg.paths, 'base_dir'):
            base_dir = str(cfg.paths.base_dir)
            # Try to guess original template from path patterns
            if '/gscratch/portia/' in base_dir:
                original_template = "hyak"
            elif '/home/' in base_dir and '/Research/data/' in base_dir:
                original_template = "workstation/desktop"
            elif '/Users/' in base_dir:
                original_template = "mbook"

        print(f"   Detected original paths template: {original_template}")

    # Override the paths using the new template
    cfg = override_config_paths(cfg, new_paths_template, config_dir, verbose)

    return cfg
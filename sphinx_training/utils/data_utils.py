
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path
import jax
import h5py
import numpy as np
import jax.numpy as jnp
from flax import struct
from jax import random
from . import io_dict_to_hdf5 as ioh5

@struct.dataclass
class ReferenceClips:
    qpos: jnp.ndarray 
    qvel: jnp.ndarray 
    xpos: jnp.ndarray
    xquat: jnp.ndarray
    clip_lengths: Optional[jnp.ndarray] = None
    qpos_names: Optional[list] = struct.field(pytree_node=False, default=None)  # Mark as non-pytree
    extras: Optional[dict] = struct.field(pytree_node=True, default=None)  # Additional optional fields (e.g., forces, contacts)

    # ===== CREATION METHODS =====
    @classmethod
    def from_path(cls, data_path: Union[str, Path], enable_jax: bool = True):
        """Load ReferenceClip from file path."""
        if isinstance(data_path, str):
            data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data_dict = ioh5.load(data_path, enable_jax=enable_jax)
        return cls.from_dict(data_dict)

    @classmethod
    def from_dict(cls, data_dict):
        """Create ReferenceClip from dictionary"""
        # Separate core fields from extras
        core_fields = {'qpos', 'qvel', 'xpos', 'xquat', 'clip_lengths', 'qpos_names', 'extras'}
        
        # If 'extras' key exists, use it directly; otherwise collect non-core fields
        if 'extras' in data_dict:
            extras = data_dict['extras']
        else:
            extras = {k: v for k, v in data_dict.items() if k not in core_fields}
        
        return cls(
            qpos=data_dict.get('qpos'),
            qvel=data_dict.get('qvel'), 
            xpos=data_dict.get('xpos'),
            xquat=data_dict.get('xquat'),
            clip_lengths=data_dict.get('clip_lengths'),
            qpos_names=data_dict.get('qpos_names'),
            extras=extras if extras else None
        )

    # ===== CONVERSION METHODS =====
    def to_dict(self):
        """Convert to dictionary format"""
        data = {
            'qpos': self.qpos,
            'qvel': self.qvel,
            'xpos': self.xpos,
            'xquat': self.xquat,
            'clip_lengths': self.clip_lengths,
            'qpos_names': self.qpos_names
        }
        # Include extras if present
        if self.extras is not None:
            data.update(self.extras)
        return data
    
    def save(self, data_path: Union[str, Path], compression_opts: int = 3):
        """Save to file path."""
        if isinstance(data_path, str):
            data_path = Path(data_path)
        
        data_path.parent.mkdir(parents=True, exist_ok=True)
        ioh5.save(data_path, self.to_dict(), compression_opts=compression_opts)

    def save_chunked(self, data_path: Union[str, Path], compression_opts: int = 3, clips_per_chunk: int = 1):
        """
        Save to file path with optimal chunking for HDF5ReferenceClips.

        Args:
            data_path: Path to save the file
            compression_opts: Compression level (0-9)
            clips_per_chunk: Number of clips per chunk (1 = optimal for single-clip loading)
        """
        if isinstance(data_path, str):
            data_path = Path(data_path)

        data_path.parent.mkdir(parents=True, exist_ok=True)
        ioh5.save_reference_clips_chunked(data_path, self.to_dict(),
                                        compression_opts=compression_opts,
                                        clips_per_chunk=clips_per_chunk)

    # ===== SLICING METHODS =====
    def slice_to_joints(self, joint_names: list[str], qpos_names: list[str]) -> 'ReferenceClips':
        """Slice qpos/qvel to only include joints listed in joint_names.

        Builds a mapping from qpos_names (joint-only, 0-indexed from the HDF5)
        to find the column index in the full reference for each requested joint.
        Root DOFs are always preserved (first 7 qpos cols, first 6 qvel cols).
        xpos/xquat are unchanged since bodies are not removed by joint filtering.

        Args:
            joint_names: List of joint names to keep (from config).
            qpos_names: List of joint names from the reference HDF5
                        (indices correspond to qpos columns offset by 7).
        Returns:
            New ReferenceClips with sliced qpos/qvel and updated qpos_names.
        """
        name_to_ref_idx = {name: i for i, name in enumerate(qpos_names)}

        # Root DOFs are always kept
        qpos_cols = list(range(7))   # root pos (3) + root quat (4)
        qvel_cols = list(range(6))   # root linear vel (3) + root angular vel (3)
        kept_names = []

        for jname in joint_names:
            if jname not in name_to_ref_idx:
                raise ValueError(
                    f"Joint '{jname}' from config not found in reference "
                    f"qpos_names. Available: {qpos_names}"
                )
            ref_joint_idx = name_to_ref_idx[jname]
            qpos_cols.append(ref_joint_idx + 7)  # +7 for root offset in qpos
            qvel_cols.append(ref_joint_idx + 6)  # +6 for root offset in qvel
            kept_names.append(jname)

        qpos_cols = jnp.array(qpos_cols)
        qvel_cols = jnp.array(qvel_cols)

        new_qpos = self.qpos[..., qpos_cols]
        new_qvel = self.qvel[..., qvel_cols]

        print(f"slice_to_joints: {self.qpos.shape[-1]} -> {new_qpos.shape[-1]} qpos cols, "
              f"{self.qvel.shape[-1]} -> {new_qvel.shape[-1]} qvel cols "
              f"({len(kept_names)} joints kept)")

        return self.replace(qpos=new_qpos, qvel=new_qvel, qpos_names=kept_names)

    # ===== INSTANCE METHODS (operate on self) =====
    def extract_clip(self, clip_index, return_type='class'):
        """Extract a single clip from this multi-clip instance."""
        return self.extract_single_clip(clip_index, return_type)
    
    def extract_clip_sequence(self, clip_index, start_frame=0, length=None, return_type='class'):
        """Extract a sequence from a specific clip in this multi-clip instance."""
        # First extract the single clip
        single_clip = self.extract_single_clip(clip_index)
        
        # Then extract the sequence from that clip
        return single_clip.extract_sequence(start_frame, length)
    
    def extract_sequence(self, start_frame=0, length=None):
        """Extract a sequence from this single clip instance."""
        # Get clip length
        if self.clip_lengths is not None:
            clip_length = self.clip_lengths.astype(int)
        else:
            clip_length = self.qpos.shape[0]
        
        if length is None:
            sequence_length = clip_length - start_frame
        else:
            sequence_length = length
            
        def extract_sequence_slice(x):
            # Only extract from JAX arrays with correct length, leave everything else unchanged
            try:
                return jax.lax.dynamic_slice_in_dim(x, start_frame, sequence_length, axis=0)
            except:
                return x
        
        # Handle core fields and extras separately
        core_dict = {
            'qpos': self.qpos,
            'qvel': self.qvel,
            'xpos': self.xpos,
            'xquat': self.xquat,
            'clip_lengths': self.clip_lengths,
            'qpos_names': self.qpos_names
        }
        trimmed_core = jax.tree_util.tree_map(extract_sequence_slice, core_dict)
        
        # Extract extras if present
        if self.extras is not None:
            trimmed_extras = jax.tree_util.tree_map(extract_sequence_slice, self.extras)
            trimmed_core['extras'] = trimmed_extras
        
        trimmed_dict = trimmed_core
        
        if 'clip_lengths' in trimmed_dict:
            trimmed_dict['clip_lengths'] = sequence_length
        
        return self.from_dict(trimmed_dict)
    
    def extract_frame(self, frame_index, clip_index=None, return_type='class'):
        """Extract a single frame from a clip. If clip_index is provided, extract from that clip first."""
        # If clip_index is provided, first extract the specific clip
        if clip_index is not None:
            single_clip = self.extract_single_clip(clip_index)
            core_dict = {
                'qpos': single_clip.qpos,
                'qvel': single_clip.qvel,
                'xpos': single_clip.xpos,
                'xquat': single_clip.xquat,
                'clip_lengths': single_clip.clip_lengths,
                'qpos_names': single_clip.qpos_names
            }
            extras_dict = single_clip.extras
        else:
            core_dict = {
                'qpos': self.qpos,
                'qvel': self.qvel,
                'xpos': self.xpos,
                'xquat': self.xquat,
                'clip_lengths': self.clip_lengths,
                'qpos_names': self.qpos_names
            }
            extras_dict = self.extras
        
        def extract_frame_slice(x):
            # Only extract from JAX arrays, leave everything else unchanged
            if not hasattr(x, 'shape') or len(x.shape) == 0:
                # Scalar or non-array data, return as-is
                return x
            try:
                # Ensure frame_index is a scalar for dynamic_slice_in_dim
                idx = jnp.asarray(frame_index).squeeze()
                return jax.lax.dynamic_slice_in_dim(x, idx, 1, axis=0).squeeze(0)
            except (TypeError, AttributeError):
                # Not a JAX array, return as-is
                return x
            except Exception as e:
                # Log the actual error for debugging
                print(f"Warning: Failed to extract frame from array with shape {x.shape}: {e}")
                return x
        
        extracted_core = jax.tree_util.tree_map(extract_frame_slice, core_dict)
        extracted_core['clip_lengths'] = None  # Remove clip_lengths for single frame
        
        # Extract extras if present
        if extras_dict is not None:
            extracted_core['extras'] = jax.tree_util.tree_map(extract_frame_slice, extras_dict)
        
        if return_type == 'class':
            return type(self).from_dict(extracted_core)
        else:
            return extracted_core
    
    def extract_single_clip(self, clip_index, return_type='class'):
        """Extract a single clip from this multi-clip instance."""
        core_dict = {
            'qpos': self.qpos,
            'qvel': self.qvel,
            'xpos': self.xpos,
            'xquat': self.xquat,
            'clip_lengths': self.clip_lengths,
            'qpos_names': self.qpos_names
        }
        
        def extract_clip_slice(x):
            # Only extract from JAX arrays, leave everything else unchanged
            if not hasattr(x, 'shape') or len(x.shape) == 0:
                # Scalar or non-array data, return as-is
                return x
            try:
                # Ensure clip_index is a scalar for dynamic_slice_in_dim
                idx = jnp.asarray(clip_index).squeeze()
                return jax.lax.dynamic_slice_in_dim(x, idx, 1, axis=0).squeeze(0)
            except (TypeError, AttributeError):
                # Not a JAX array, return as-is
                return x
            except Exception as e:
                # Log the actual error for debugging
                print(f"Warning: Failed to extract clip from array with shape {x.shape}: {e}")
                return x
        
        extracted_core = jax.tree_util.tree_map(extract_clip_slice, core_dict)
        
        # Extract extras if present
        if self.extras is not None:
            extracted_core['extras'] = jax.tree_util.tree_map(extract_clip_slice, self.extras)
        
        if return_type == 'class':
            return type(self).from_dict(extracted_core)
        else:
            return extracted_core

    def split(self, test_ratio=0.2, random_seed=42, return_type='class'):
        """Split this multi-clip instance into train and test sets."""
        from jax import random
        
        data_dict = self.to_dict()
        n_clips = self.num_clips
        
        if n_clips <= 1:
            raise ValueError("Cannot split data with only one clip")
        
        # Create random indices for splitting using JAX
        key = random.PRNGKey(random_seed)
        indices = jnp.arange(n_clips)
        shuffled_indices = random.permutation(key, indices)
        
        # Calculate split point
        n_test = jnp.floor(n_clips * test_ratio).astype(int)
        n_train = n_clips - n_test
        
        # Use JAX dynamic_slice for fast indexing
        train_indices = jax.lax.dynamic_slice(shuffled_indices, (0,), (n_train,))
        test_indices = jax.lax.dynamic_slice(shuffled_indices, (n_train,), (n_test,))
        
        def extract_train(x):
            if not hasattr(x, 'shape') or len(x.shape) == 0:
                return x
            try:
                return x[train_indices]
            except:
                return x
        
        def extract_test(x):
            if not hasattr(x, 'shape') or len(x.shape) == 0:
                return x
            try:
                return x[test_indices]
            except:
                return x
        
        # Split core fields
        core_dict = {
            'qpos': self.qpos,
            'qvel': self.qvel,
            'xpos': self.xpos,
            'xquat': self.xquat,
            'clip_lengths': self.clip_lengths,
            'qpos_names': self.qpos_names
        }
        train_dict = jax.tree_util.tree_map(extract_train, core_dict)
        test_dict = jax.tree_util.tree_map(extract_test, core_dict)
        
        # Split extras if present
        if self.extras is not None:
            train_dict['extras'] = jax.tree_util.tree_map(extract_train, self.extras)
            test_dict['extras'] = jax.tree_util.tree_map(extract_test, self.extras)
        
        # Update clip_lengths if present
        if train_dict['clip_lengths'] is not None:
            train_dict['clip_lengths'] = train_dict['clip_lengths'][train_indices]
            test_dict['clip_lengths'] = test_dict['clip_lengths'][test_indices]
        
        if return_type == 'class':
            return type(self).from_dict(train_dict), type(self).from_dict(test_dict)
        else:
            return train_dict, test_dict


    @property
    def num_clips(self):
        """Return the number of clips."""
        if self.clip_lengths is not None:
            return len(self.clip_lengths)
        else:
            return 1 if self.qpos is not None and len(self.qpos.shape) > 0 else 0

    def print_info(self, clip_idx=None):
        """
        Print detailed information about reference clips.

        Args:
            clip_idx: Optional specific clip index to analyze
        """
        import numpy as np

        print("=" * 60)
        print("REFERENCE CLIPS INFORMATION")
        print("=" * 60)

        # Overall statistics
        if self.qpos is not None:
            print(f"Total number of clips: {self.qpos.shape[0]}")
            print(f"Frames per clip: {self.qpos.shape[1]}")
            print(f"Total qpos dimensions: {self.qpos.shape[2]}")

        if self.xpos is not None:
            print(f"Total bodies tracked: {self.xpos.shape[2]}")

        # Data arrays available
        arrays = []
        for attr in ['qpos', 'qvel', 'xpos', 'xquat']:
            if getattr(self, attr) is not None:
                arrays.append(attr)
        print(f"\nData arrays available: {arrays}")

        # Joint information
        if self.qpos_names is not None:
            print(f"\nJoint names ({len(self.qpos_names)}):")
            for i, joint_name in enumerate(self.qpos_names):
                print(f"  {i:2d}: {joint_name}")

        # Data shapes
        print(f"\nData array shapes:")
        for array_name in arrays:
            array = getattr(self, array_name)
            if array is not None:
                print(f"  {array_name:8}: {array.shape}")

        # Clip lengths info
        if self.clip_lengths is not None:
            print(f"\nClip lengths: min={self.clip_lengths.min()}, max={self.clip_lengths.max()}, mean={self.clip_lengths.mean():.1f}")

        # Extras information
        if self.extras is not None and len(self.extras) > 0:
            print(f"\nExtra fields ({len(self.extras)}):")
            for key, value in self.extras.items():
                if hasattr(value, 'shape'):
                    print(f"  {key:20}: shape={value.shape}, dtype={value.dtype}")
                    # Show min/max for numeric arrays
                    try:
                        if value.size > 0 and np.issubdtype(value.dtype, np.number):
                            print(f"  {' '*20}  range=[{value.min():.3e}, {value.max():.3e}]")
                    except:
                        pass
                else:
                    print(f"  {key:20}: {type(value).__name__}")

        # Specific clip analysis
        if clip_idx is not None and self.qpos is not None:
            print(f"\n" + "="*40)
            print(f"CLIP {clip_idx} ANALYSIS")
            print("="*40)

            clip_data = self.qpos[clip_idx]
            print(f"Clip frames: {clip_data.shape[0]}")
            print(f"Clip duration: {clip_data.shape[0] * 2e-4:.3f}s (assuming 2e-4s timestep)")

            # Root position statistics (first 3 elements)
            root_pos = self.qpos[clip_idx, :, :3]
            print(f"\nRoot position range:")
            print(f"  X: [{root_pos[:, 0].min():.3f}, {root_pos[:, 0].max():.3f}]")
            print(f"  Y: [{root_pos[:, 1].min():.3f}, {root_pos[:, 1].max():.3f}]")
            print(f"  Z: [{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}]")

            # Joint angle statistics (elements 7 onwards, skipping root pos and quat)
            if self.qpos.shape[2] > 7:
                joints_data = self.qpos[clip_idx, :, 7:]
                print(f"\nJoint angles range (first 10):")
                num_joints_to_show = min(10, joints_data.shape[1])
                for i in range(num_joints_to_show):
                    joint_vals = joints_data[:, i]
                    joint_name = self.qpos_names[7 + i] if self.qpos_names and len(self.qpos_names) > 7 + i else f"joint_{i}"
                    print(f"  {joint_name:20}: [{joint_vals.min():.3f}, {joint_vals.max():.3f}]")
                if joints_data.shape[1] > 10:
                    print(f"  ... and {joints_data.shape[1] - 10} more joints")
class HDF5ReferenceClips:
    """
    Lazy-loading reference clips class that reads data directly from HDF5 files.

    This class provides the same interface as ReferenceClips but with lazy loading,
    allowing efficient access to large datasets without loading everything into memory.
    Designed for use with vmap/pmap where individual clips are loaded into state.info.
    """

    def __init__(self, file_path: Union[str, Path], enable_jax: bool = True):
        """
        Initialize HDF5ReferenceClips with a file path.

        Args:
            file_path: Path to the HDF5 file containing reference clips
            enable_jax: Whether to convert arrays to JAX arrays when loading
        """
        self.file_path = Path(file_path) if isinstance(file_path, str) else file_path
        self.enable_jax = enable_jax

        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")

        # Load metadata without loading the actual data arrays
        with h5py.File(self.file_path, 'r') as f:
            # Debug: print what's in the file
            print(f"Keys in HDF5 file: {list(f.keys())}")

            self.clip_lengths = jnp.array(f['clip_lengths'][()]) if self.enable_jax else np.array(f['clip_lengths'][()])
            
            # Store extra field names (any keys not in core set)
            core_keys = {'qpos', 'qvel', 'xpos', 'xquat', 'clip_lengths', 'qpos_names'}
            self.extras_keys = [key for key in f.keys() if key not in core_keys]

            # Handle qpos_names - could be Dataset or Group
            if 'qpos_names' in f:
                qpos_names_obj = f['qpos_names']
                if isinstance(qpos_names_obj, h5py.Group):
                    # Reconstruct the list from the group
                    qpos_names_dict = {}
                    for key in qpos_names_obj.keys():
                        value = qpos_names_obj[key][()]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        qpos_names_dict[int(key)] = value

                    # Sort by index and create list
                    max_index = max(qpos_names_dict.keys())
                    self.qpos_names = [qpos_names_dict[i] for i in range(max_index + 1)]

                elif isinstance(qpos_names_obj, h5py.Dataset):
                    # qpos_names is stored as a dataset
                    print(f"qpos_names shape: {qpos_names_obj.shape}")
                    qpos_names_raw = qpos_names_obj[()]

                    # Handle different types of string storage
                    if isinstance(qpos_names_raw, np.ndarray):
                        if qpos_names_raw.dtype.kind in ['S', 'U']:  # byte or unicode strings
                            if qpos_names_raw.dtype.kind == 'S':  # byte strings
                                self.qpos_names = [name.decode('utf-8') if isinstance(name, bytes) else str(name)
                                                 for name in qpos_names_raw]
                            else:  # unicode strings
                                self.qpos_names = [str(name) for name in qpos_names_raw]
                        else:
                            self.qpos_names = list(qpos_names_raw)
                    elif isinstance(qpos_names_raw, bytes):
                        self.qpos_names = [qpos_names_raw.decode('utf-8')]
                    else:
                        self.qpos_names = list(qpos_names_raw) if hasattr(qpos_names_raw, '__iter__') else [qpos_names_raw]
                else:
                    print(f"Unexpected qpos_names type: {type(qpos_names_obj)}")
                    self.qpos_names = None
            else:
                self.qpos_names = None

            # Store array shapes for reference
            self._qpos_shape = f['qpos'].shape
            self._qvel_shape = f['qvel'].shape
            self._xpos_shape = f['xpos'].shape
            self._xquat_shape = f['xquat'].shape

    @property
    def num_clips(self) -> int:
        """Number of clips in the dataset."""
        return len(self.clip_lengths)

    def _load_array_slice(self, dataset_name: str, clip_idx: int, start_frame: int = 0, length: int = None) -> jnp.ndarray:
        """
        Load a slice of data from the HDF5 file.

        Args:
            dataset_name: Name of the dataset ('qpos', 'qvel', 'xpos', 'xquat')
            clip_idx: Index of the clip to load
            start_frame: Starting frame within the clip (default: 0)
            length: Number of frames to load (default: all remaining frames)

        Returns:
            JAX array containing the requested data slice
        """
        with h5py.File(self.file_path, 'r') as f:
            dataset = f[dataset_name]

            # Handle end frame calculation
            clip_length = self.clip_lengths[clip_idx]
            if length is None:
                end_frame = clip_length
            else:
                end_frame = min(start_frame + length, clip_length)

            # Load the slice for this specific clip and frame range
            data_slice = dataset[clip_idx, start_frame:end_frame]

            # Convert to JAX array if requested
            if self.enable_jax:
                return jnp.array(data_slice)
            else:
                return data_slice

    def load_single_clip(self, clip_idx) -> 'ReferenceClips':
        """
        JIT-compatible method to load a single clip using jax.pure_callback.
        Returns the full padded clip (not trimmed to actual length).

        Args:
            clip_idx: Index of the clip to load (can be JAX array or Python int)

        Returns:
            ReferenceClips object containing the full clip's data (padded)
        """
        def host_load_function(clip_idx_val):
            """Pure callback that performs I/O operations on CPU."""
            # Convert to Python int
            idx = int(clip_idx_val.item() if hasattr(clip_idx_val, 'item') else clip_idx_val)

            # Load the entire padded clip data
            with h5py.File(self.file_path, 'r') as f:
                qpos = np.array(f['qpos'][idx])  # Full padded length
                qvel = np.array(f['qvel'][idx])  # Full padded length
                xpos = np.array(f['xpos'][idx])  # Full padded length
                xquat = np.array(f['xquat'][idx])  # Full padded length
                
                # Load extras if present
                extras_data = {}
                for key in self.extras_keys:
                    extras_data[key] = np.array(f[key][idx])

            return qpos, qvel, xpos, xquat, extras_data

        # Build result type with extras
        result_type = [
            jax.ShapeDtypeStruct(self._qpos_shape[1:], jnp.float32),  # (max_frames, features)
            jax.ShapeDtypeStruct(self._qvel_shape[1:], jnp.float32),  # (max_frames, features)
            jax.ShapeDtypeStruct(self._xpos_shape[1:], jnp.float32),  # (max_frames, bodies, 3)
            jax.ShapeDtypeStruct(self._xquat_shape[1:], jnp.float32), # (max_frames, bodies, 4)
        ]
        
        # Add shape specs for extras - store as dict result
        # Note: pure_callback doesn't support dict outputs directly, so we use a workaround
        # by returning a tuple and reconstructing the dict after
        result_type.append(dict)  # Placeholder for extras dict
        
        # Use pure_callback to perform I/O from within JIT
        results = jax.pure_callback(
            host_load_function,
            tuple(result_type),
            clip_idx,
            vmap_method='sequential'
        )
        
        qpos, qvel, xpos, xquat, extras_data = results

        # Convert to JAX arrays if needed
        if self.enable_jax:
            qpos = jnp.array(qpos)
            qvel = jnp.array(qvel)
            xpos = jnp.array(xpos)
            xquat = jnp.array(xquat)
            if extras_data:
                extras_data = jax.tree_util.tree_map(jnp.array, extras_data)

        # Return full padded clip - use the clip_lengths from metadata
        return ReferenceClips(
            qpos=qpos,
            qvel=qvel,
            xpos=xpos,
            xquat=xquat,
            clip_lengths=self.clip_lengths,  # Keep original clip lengths array
            qpos_names=self.qpos_names,
            extras=extras_data if extras_data else None
        )

    def extract_frame(self, clip_index: int, frame_index: int) -> 'ReferenceClips':
        """
        Extract a single frame from a clip.

        Args:
            clip_index: Index of the clip
            frame_index: Index of the frame within the clip

        Returns:
            ReferenceClips object containing the single frame's data
        """
        # Load single frame data
        qpos = self._load_array_slice('qpos', clip_index, frame_index, 1)[0]  # Remove time dimension
        qvel = self._load_array_slice('qvel', clip_index, frame_index, 1)[0]
        xpos = self._load_array_slice('xpos', clip_index, frame_index, 1)[0]
        xquat = self._load_array_slice('xquat', clip_index, frame_index, 1)[0]
        # Load extras if present
        extras_data = None
        if self.extras_keys:
            extras_data = {}
            for key in self.extras_keys:
                extras_data[key] = self._load_array_slice(key, clip_index, frame_index, 1)[0]

        return ReferenceClips(
            qpos=qpos,
            qvel=qvel,
            xpos=xpos,
            xquat=xquat,
            clip_lengths=jnp.array([1]) if self.enable_jax else np.array([1]),
            qpos_names=self.qpos_names,
            extras=extras_data
        )

    def extract_clip_sequence(self, clip_index: int, start_frame: int, length: int) -> 'ReferenceClips':
        """
        Extract a sequence of frames from a clip.

        Args:
            clip_index: Index of the clip
            start_frame: Starting frame index within the clip
            length: Number of frames to extract

        Returns:
            ReferenceClips object containing the requested sequence
        """
        if clip_index < 0 or clip_index >= self.num_clips:
            raise IndexError(f"Clip index {clip_index} out of range [0, {self.num_clips})")

        clip_length = int(self.clip_lengths[clip_index])
        end_frame = min(start_frame + length, clip_length)
        actual_length = end_frame - start_frame

        if start_frame >= clip_length:
            raise IndexError(f"Start frame {start_frame} beyond clip length {clip_length}")

        # Load sequence data
        qpos = self._load_array_slice('qpos', clip_index, start_frame, actual_length)
        qvel = self._load_array_slice('qvel', clip_index, start_frame, actual_length)
        xpos = self._load_array_slice('xpos', clip_index, start_frame, actual_length)
        xquat = self._load_array_slice('xquat', clip_index, start_frame, actual_length)
        
        # Load extras if present
        extras_data = None
        if self.extras_keys:
            extras_data = {}
            for key in self.extras_keys:
                extras_data[key] = self._load_array_slice(key, clip_index, start_frame, actual_length)

        return ReferenceClips(
            qpos=qpos,
            qvel=qvel,
            xpos=xpos,
            xquat=xquat,
            clip_lengths=jnp.array([actual_length]) if self.enable_jax else np.array([actual_length]),
            qpos_names=self.qpos_names,
            extras=extras_data
        )

    def load_clips(self, clip_indices: Optional[Union[list, np.ndarray, jnp.ndarray]] = None) -> 'ReferenceClips':
        """
        Load clips from HDF5 file into memory.

        Args:
            clip_indices: List/array of clip indices to load. If None, loads all clips.
                         Can be a Python list, numpy array, or JAX array.

        Returns:
            ReferenceClips object containing the loaded clips in memory
        """
        if clip_indices is None:
            # Load all clips
            clip_indices = np.arange(self.num_clips)
        else:
            # Convert to numpy array for indexing
            if isinstance(clip_indices, jnp.ndarray):
                clip_indices = np.array(clip_indices)
            elif isinstance(clip_indices, list):
                clip_indices = np.array(clip_indices)
            clip_indices = np.asarray(clip_indices)

        if len(clip_indices) == 0:
            raise ValueError("No clips specified to load")

        # Validate indices
        if np.any(clip_indices < 0) or np.any(clip_indices >= self.num_clips):
            raise IndexError(f"Clip indices must be in range [0, {self.num_clips}), got {clip_indices}")

        # Open file once and batch load all required clips
        with h5py.File(self.file_path, 'r') as f:
            # Get datasets
            qpos_dataset = f['qpos']
            qvel_dataset = f['qvel']
            xpos_dataset = f['xpos']
            xquat_dataset = f['xquat']

            # Batch load using fancy indexing - much faster than individual loads
            qpos_batch = qpos_dataset[clip_indices]
            qvel_batch = qvel_dataset[clip_indices]
            xpos_batch = xpos_dataset[clip_indices]
            xquat_batch = xquat_dataset[clip_indices]

            # Get clip lengths for selected clips
            selected_clip_lengths = self.clip_lengths[clip_indices]
            
            # Load extras if present
            extras_data = None
            if self.extras_keys:
                extras_data = {}
                for key in self.extras_keys:
                    extras_data[key] = f[key][clip_indices]

        # Convert to JAX arrays if enabled
        if self.enable_jax:
            qpos_batch = jnp.array(qpos_batch)
            qvel_batch = jnp.array(qvel_batch)
            xpos_batch = jnp.array(xpos_batch)
            xquat_batch = jnp.array(xquat_batch)
            selected_clip_lengths = jnp.array(selected_clip_lengths)
            if extras_data:
                extras_data = jax.tree.map(jnp.array, extras_data)

        return ReferenceClips(
            qpos=qpos_batch,
            qvel=qvel_batch,
            xpos=xpos_batch,
            xquat=xquat_batch,
            clip_lengths=selected_clip_lengths,
            qpos_names=self.qpos_names,
            extras=extras_data
        )

    def load_all_clips(self) -> 'ReferenceClips':
        """
        Load all clips from HDF5 file into memory.
        Convenience method that calls load_clips() with no arguments.

        Returns:
            ReferenceClips object containing all clips in memory
        """
        return self.load_clips(clip_indices=None)

    def split(self, random_seed: int = 1, test_ratio: float = 0.2) -> tuple['HDF5ReferenceClips', 'HDF5ReferenceClips']:
        """
        Split the dataset into train and test sets.

        Note: This creates two separate HDF5ReferenceClips instances that reference
        different subsets of clips from the same file. For true efficiency, you might
        want to create separate HDF5 files for train/test splits.

        Args:
            test_ratio: Fraction of clips to use for testing

        Returns:
            Tuple of (train_clips, test_clips)
        """
        n_clips = self.num_clips

        # Create random indices for splitting using JAX
        key = random.PRNGKey(random_seed)
        indices = jnp.arange(n_clips)
        shuffled_indices = random.permutation(key, indices)

        # Calculate split point
        n_test = jnp.floor(n_clips * test_ratio).astype(int)
        n_train = n_clips - n_test

        # Use JAX dynamic_slice for fast indexing
        train_indices = jax.lax.dynamic_slice(shuffled_indices, (0,), (n_train,)).sort()
        test_indices = jax.lax.dynamic_slice(shuffled_indices, (n_train,), (n_test,)).sort()

        train_clips = HDF5ReferenceClipsSubset(self, train_indices)
        test_clips = HDF5ReferenceClipsSubset(self, test_indices)

        return train_clips, test_clips

    @classmethod
    def from_reference_clips(cls, reference_clips: ReferenceClips, save_path: Union[str, Path],
                           chunk_size: int = 1, compression_opts: int = 3) -> 'HDF5ReferenceClips':
        """
        Convert existing ReferenceClips to HDF5 format with optimized chunking.

        Args:
            reference_clips: ReferenceClips object to convert
            save_path: Path where to save the HDF5 file
            chunk_size: Chunk size for HDF5 datasets (1 = one clip per chunk)
            compression_opts: Compression level (0-9)

        Returns:
            New HDF5ReferenceClips instance
        """
        save_path = Path(save_path) if isinstance(save_path, str) else save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(save_path, 'w') as f:
            # Determine optimal chunk shape (chunk by clip)
            chunk_qpos = (chunk_size, reference_clips.qpos.shape[1], reference_clips.qpos.shape[2])
            chunk_qvel = (chunk_size, reference_clips.qvel.shape[1], reference_clips.qvel.shape[2])
            chunk_xpos = (chunk_size, reference_clips.xpos.shape[1], reference_clips.xpos.shape[2], reference_clips.xpos.shape[3])
            chunk_xquat = (chunk_size, reference_clips.xquat.shape[1], reference_clips.xquat.shape[2], reference_clips.xquat.shape[3])

            # Create datasets with chunking optimized for single-clip access
            f.create_dataset('qpos', data=np.array(reference_clips.qpos),
                           chunks=chunk_qpos, compression='gzip', compression_opts=compression_opts)
            f.create_dataset('qvel', data=np.array(reference_clips.qvel),
                           chunks=chunk_qvel, compression='gzip', compression_opts=compression_opts)
            f.create_dataset('xpos', data=np.array(reference_clips.xpos),
                           chunks=chunk_xpos, compression='gzip', compression_opts=compression_opts)
            f.create_dataset('xquat', data=np.array(reference_clips.xquat),
                           chunks=chunk_xquat, compression='gzip', compression_opts=compression_opts)

            # Save metadata
            f.create_dataset('clip_lengths', data=np.array(reference_clips.clip_lengths))
            if reference_clips.qpos_names:
                f.create_dataset('qpos_names', data=[name.encode('utf-8') for name in reference_clips.qpos_names])
            
            # Save extras if present
            if reference_clips.extras:
                for key, value in reference_clips.extras.items():
                    value_np = np.array(value)
                    # Determine chunk shape based on value dimensions
                    if len(value_np.shape) >= 2:
                        chunk_shape = (chunk_size,) + value_np.shape[1:]
                    else:
                        chunk_shape = (chunk_size,)
                    f.create_dataset(key, data=value_np,
                                   chunks=chunk_shape, compression='gzip', compression_opts=compression_opts)

        return cls(save_path, enable_jax=True)


class HDF5ReferenceClipsSubset:
    """
    A subset view of HDF5ReferenceClips for train/test splits.

    This class provides the same interface as HDF5ReferenceClips but only
    accesses a subset of clips based on provided indices.
    """

    def __init__(self, parent: HDF5ReferenceClips, clip_indices: list):
        """
        Initialize subset with parent HDF5ReferenceClips and clip indices.

        Args:
            parent: Parent HDF5ReferenceClips instance
            clip_indices: List of clip indices to include in this subset
        """
        self.parent = parent
        self.clip_indices = jnp.array(clip_indices)
        # Convert list to jnp.array for JAX indexing
        indices_array = jnp.array(clip_indices)
        self.clip_lengths = parent.clip_lengths[indices_array]
        self.qpos_names = parent.qpos_names

    @property
    def num_clips(self) -> int:
        """Number of clips in this subset."""
        return len(self.clip_indices)

    def load_single_clip(self, subset_clip_idx: int) -> 'ReferenceClips':
        """Load a single clip from the subset."""
        actual_clip_idx = self.clip_indices[subset_clip_idx]
        return self.parent.load_single_clip(actual_clip_idx)

    def extract_frame(self, clip_index: int, frame_index: int) -> 'ReferenceClips':
        """Extract frame from subset."""
        actual_clip_idx = self.clip_indices[clip_index]
        return self.parent.extract_frame(actual_clip_idx, frame_index)

    def extract_clip_sequence(self, clip_index: int, start_frame: int, length: int) -> 'ReferenceClips':
        """Extract clip sequence from subset."""
        actual_clip_idx = self.clip_indices[clip_index]
        return self.parent.extract_clip_sequence(actual_clip_idx, start_frame, length)

    def load_all_clips(self) -> 'ReferenceClips':
        """Efficient batch loading of all clips in this subset."""
        if self.num_clips == 0:
            raise ValueError("No clips to load")

        # Open file once and batch load all required clips
        with h5py.File(self.parent.file_path, 'r') as f:
            # Get datasets
            qpos_dataset = f['qpos']
            qvel_dataset = f['qvel']
            xpos_dataset = f['xpos']
            xquat_dataset = f['xquat']

            # Pre-allocate arrays for all clips
            max_frames = qpos_dataset.shape[1]
            qpos_shape = (self.num_clips, max_frames, qpos_dataset.shape[2])
            qvel_shape = (self.num_clips, max_frames, qvel_dataset.shape[2])
            xpos_shape = (self.num_clips, max_frames, xpos_dataset.shape[2], xpos_dataset.shape[3])
            xquat_shape = (self.num_clips, max_frames, xquat_dataset.shape[2], xquat_dataset.shape[3])

            # Use numpy arrays for initial loading, convert to JAX later
            qpos_batch = np.zeros(qpos_shape, dtype=np.float32)
            qvel_batch = np.zeros(qvel_shape, dtype=np.float32)
            xpos_batch = np.zeros(xpos_shape, dtype=np.float32)
            xquat_batch = np.zeros(xquat_shape, dtype=np.float32)

            # Batch load using fancy indexing - much faster than individual loads
            clip_indices_np = np.array(self.clip_indices)
            qpos_batch[:] = qpos_dataset[clip_indices_np]
            qvel_batch[:] = qvel_dataset[clip_indices_np]
            xpos_batch[:] = xpos_dataset[clip_indices_np]
            xquat_batch[:] = xquat_dataset[clip_indices_np]
            
            # Load extras if present
            extras_data = None
            if self.parent.extras_keys:
                extras_data = {}
                for key in self.parent.extras_keys:
                    extras_dataset = f[key]
                    # Pre-allocate based on dataset shape
                    extras_shape = (self.num_clips,) + extras_dataset.shape[1:]
                    extras_batch = np.zeros(extras_shape, dtype=extras_dataset.dtype)
                    extras_batch[:] = extras_dataset[clip_indices_np]
                    extras_data[key] = extras_batch

        # Convert to JAX arrays if parent has JAX enabled
        if self.parent.enable_jax:
            import jax.numpy as jnp
            qpos_batch = jnp.array(qpos_batch)
            qvel_batch = jnp.array(qvel_batch)
            xpos_batch = jnp.array(xpos_batch)
            xquat_batch = jnp.array(xquat_batch)
            if extras_data:
                extras_data = jax.tree_util.tree_map(jnp.array, extras_data)

        return ReferenceClips(
            qpos=qpos_batch,
            qvel=qvel_batch,
            xpos=xpos_batch,
            xquat=xquat_batch,
            clip_lengths=self.clip_lengths,
            qpos_names=self.qpos_names,
            extras=extras_data
        )


# Example integration functions for the Imitation class
def create_optimized_imitation_workflow(hdf5_clips: HDF5ReferenceClips):
    """
    Example function showing how to integrate HDF5ReferenceClips with the Imitation environment.

    Usage in Imitation class __init__:
    ```python
    # Instead of loading everything into memory:
    # self.reference_clips = ReferenceClips.from_path(data_path)

    # Use lazy loading:
    from utils.data_utils import HDF5ReferenceClips
    self.hdf5_reference_clips = HDF5ReferenceClips(data_path)
    ```

    Usage in reset method:
    ```python
    def reset(self, rng: jax.Array, clip_idx: Optional[int] = None) -> mjx_env.State:
        # Select clip index
        if clip_idx is None:
            clip_idx = jax.random.choice(clip_rng, self.hdf5_reference_clips.num_clips)

        # Load ONLY the selected clip into memory and store in state.info
        selected_clip = self.hdf5_reference_clips.load_single_clip(clip_idx)

        # Store the loaded clip in state.info for efficient access during episode
        info = {
            "reference_clip_idx": clip_idx,
            "loaded_reference_clip": selected_clip,  # This is now a small ReferenceClips object
            "start_frame": start_frame,
        }

        # Rest of reset logic remains the same, but uses info["loaded_reference_clip"]
        # instead of self.reference_clips.extract_*
    ```

    Usage in step method and reward functions:
    ```python
    def _get_current_target(self, data: mjx.Data, info: Mapping[str, Any]) -> ReferenceClips:
        # Instead of: self.reference_clips.extract_frame(...)
        # Use the pre-loaded clip from state.info:
        loaded_clip = info["loaded_reference_clip"]
        current_frame = self._get_cur_frame(data, info)
        return loaded_clip.extract_frame(clip_index=0, frame_index=current_frame)
    ```

    Benefits:
    - Memory usage scales with number of environments, not dataset size
    - Each environment only holds one clip worth of data
    - Perfect for vmap/pmap parallelization
    - Fast episode resets since only metadata changes
    - Can handle arbitrarily large datasets
    """
    pass


def convert_existing_dataset_to_hdf5(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    chunk_size: int = 1,
    compression_opts: int = 3
) -> HDF5ReferenceClips:
    """
    Convert an existing ReferenceClips dataset to HDF5 format for lazy loading.

    Args:
        input_path: Path to existing ReferenceClips file
        output_path: Path where to save the HDF5 version
        chunk_size: HDF5 chunk size (1 = one clip per chunk, optimal for single-clip access)
        compression_opts: Compression level (0-9)

    Returns:
        HDF5ReferenceClips instance pointing to the new file

    Example usage:
    ```python
    # Convert existing dataset
    hdf5_clips = convert_existing_dataset_to_hdf5(
        "data/clips/walking_clips.h5",
        "data/clips/walking_clips_lazy.h5"
    )

    # Use in Imitation environment
    env = Imitation(cfg, reference_clips=None)  # Don't load clips in __init__
    env.hdf5_reference_clips = hdf5_clips       # Set lazy loader instead
    ```
    """
    print(f"Converting {input_path} to HDF5 format...")

    # Load the original dataset
    original_clips = ReferenceClips.from_path(input_path)
    print(f"Loaded {original_clips.num_clips} clips from original dataset")

    # Save with chunking using the new method
    original_clips.save_chunked(output_path, compression_opts=compression_opts, clips_per_chunk=chunk_size)

    # Create and return HDF5ReferenceClips instance
    hdf5_clips = HDF5ReferenceClips(output_path)
    print(f"Dataset contains {hdf5_clips.num_clips} clips")

    return hdf5_clips


# ===== USAGE EXAMPLES =====

if __name__ == "__main__":
    """
    Example usage of the new HDF5ReferenceClips system:

    1. Convert existing dataset to chunked HDF5 format
    2. Load clips lazily for parallel environments
    3. Demonstrate memory efficiency

    Run with: python -m utils.data_utils
    """

    # Example 1: Convert existing dataset to chunked format
    print("=== Example 1: Converting dataset to chunked HDF5 ===")

    # Simulated existing data path - replace with your actual path
    input_path = "data/clips/example_clips.h5"
    output_path = "data/clips/example_clips_chunked.h5"

    try:
        # This would convert your existing dataset
        # hdf5_clips = convert_existing_dataset_to_hdf5(input_path, output_path)

        print("To convert your dataset:")
        print(f"  hdf5_clips = convert_existing_dataset_to_hdf5('{input_path}', '{output_path}')")
        print()

        # Example 2: Direct chunked saving
        print("=== Example 2: Direct chunked saving ===")
        print("If you have a ReferenceClips object:")
        print("  clips = ReferenceClips.from_path('data.h5')")
        print("  clips.save_chunked('data_chunked.h5')  # Optimal for lazy loading")
        print()

        # Example 3: Using HDF5ReferenceClips
        print("=== Example 3: Using HDF5ReferenceClips ===")
        print("# Load lazily - only metadata loaded")
        print("hdf5_clips = HDF5ReferenceClips('data_chunked.h5')")
        print()
        print("# Load single clip efficiently (only reads 1 chunk)")
        print("single_clip = hdf5_clips.load_single_clip(0)")
        print()
        print("# Use in parallel environments (vmap/pmap compatible)")
        print("# Each env gets its own clip in state.info, minimal memory per env")
        print()

        # Example 4: Performance comparison
        print("=== Example 4: Performance Benefits ===")
        print("Traditional ReferenceClips:")
        print("  - Loads ALL clips into memory at once")
        print("  - Memory usage: O(total_dataset_size)")
        print("  - Each env holds reference to full dataset")
        print()
        print("HDF5ReferenceClips:")
        print("  - Loads only metadata initially")
        print("  - Memory usage: O(num_environments × single_clip_size)")
        print("  - Each env loads only its current clip")
        print("  - Perfect for large datasets + parallel training")
        print()

    except Exception as e:
        print(f"Demo mode - files don't exist yet: {e}")
        print("Create actual data files to test the conversion")

    print("=== Chunking Strategy ===")
    print("For optimal parallel performance:")
    print("  clips_per_chunk=1  -> Each clip is its own chunk (best for single-clip access)")
    print("  clips_per_chunk=5  -> 5 clips per chunk (good if you often load multiple clips)")
    print("  clips_per_chunk=N  -> Trade-off between chunk overhead and access patterns")

"""
Fly Tracking Research Project

A comprehensive framework for fly tracking simulations using JAX, Brax, and MuJoCo.
"""

__version__ = "0.1.0"

# Core modules
try:
    from . import envs
    from . import training
    from . import utils
    from . import preprocessing
except ImportError:
    pass

# Optional modules
try:
    from . import scripts
except ImportError:
    pass

# Note: brax_to_onnx is NOT imported by default to avoid TensorFlow overhead
# Import explicitly when needed: from sphinx_training import brax_to_onnx
# try:
#     from . import brax_to_onnx
# except (ImportError, AttributeError):
#     pass  # brax_to_onnx has tf2onnx numpy compatibility issues
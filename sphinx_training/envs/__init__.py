"""
Custom environments for fly tracking research.
"""

# Import main environments for easy access
try:
    from .quasi_aero import *
except ImportError:
    pass

try:
    from .ellipsoid_fluid import *
except ImportError:
    pass

try:
    from .pattern_generators_new import *
except ImportError:
    pass

try:
    from .custom_mjx_env import *
except ImportError:
    pass

# Fruitfly submodule
try:
    from . import fruitfly
except ImportError:
    pass

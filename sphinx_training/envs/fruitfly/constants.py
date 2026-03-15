"""Module defining constants for fruit fly RL tasks."""

__all__ = [
    '_AIR_DENSITY', '_AIR_VISCOSITY',
    '_WALK_CONTROL_TIMESTEP', '_WALK_PHYSICS_TIMESTEP', '_TERMINAL_LINVEL',
    '_TERMINAL_ANGVEL', '_FLY_CONTROL_TIMESTEP', '_FLY_PHYSICS_TIMESTEP',
    '_TERMINAL_HEIGHT', '_TERMINAL_QACC', '_BODY_PITCH_ANGLE', '_WING_PARAMS'
]

_AIR_DENSITY = 0.00128 # mg/cm^3
_AIR_VISCOSITY = 0.000185  # mg/(cm*s)

# Walking constants.
_WALK_CONTROL_TIMESTEP = 2e-3  # s
_WALK_PHYSICS_TIMESTEP = 2e-4
_TERMINAL_LINVEL = 50  # cm/s
_TERMINAL_ANGVEL = 200  # rad/s

# Flight constants.
_FLY_CONTROL_TIMESTEP = 2e-4
_FLY_PHYSICS_TIMESTEP = 5e-5
_BODY_PITCH_ANGLE = 55# 47.5  # deg
_TERMINAL_HEIGHT = 0.2  # cm

_TERMINAL_QACC = 1e14  # mixed units

_WING_PARAMS = {
    'base_freq': 218.,
    'gainprm': 18,
    'damping': 0.007769230,
    'stiffness': 0.01,
    'ellipsoid_fluidcoef': [1.0, 0.5, 1.5, 1.7, 1.0],
    'quasi_fluidcoef': [0.00128, 3.3207, 0.4123, 3.1670, 1],
    # 'default_qpos': [1.5707964, 0., -0.7853982, 1.5707964, 0., -0.7853982],
    'default_qpos': [1.5, 0.7, -0.85, 1.5, 0.7, -0.85],
    'rel_freq_range': 0.05,
    'num_freqs': 201,
}
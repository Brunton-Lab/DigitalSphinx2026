import jax.numpy as jnp
from jax import jit

@jit
def vec_mag(v):
    """Compute magnitude of vector"""
    return jnp.sqrt(jnp.dot(v, v))

@jit
def vec_dot(v, w):
    """Compute dot product of two vectors"""
    return jnp.dot(v, w)

@jit
def vec_norm_squared(v):
    """Compute squared norm of vector"""
    return jnp.dot(v, v)

@jit
def vec_comp(v, w):
    """
    Compute the component of vector v in the direction of vector w
    Returns zero vector if w is zero vector (avoids division by zero)
    """
    w_norm_sq = vec_norm_squared(w)
    # Use jnp.where to avoid division by zero
    # If w_norm_sq is 0, return zero vector, otherwise compute projection
    return jnp.where(w_norm_sq == 0, 
                     jnp.zeros_like(v), 
                     (vec_dot(v, w) / w_norm_sq) * w)

@jit
def vec_sub(v, w):
    """Subtract vector w from vector v"""
    return v - w

@jit
def vec_remove_comp(v, w):
    """
    Remove the component of the vector v in the vector w direction
    If w is zero vector, returns v unchanged (since there's no direction to remove)
    """
    w_comp = vec_comp(v, w)
    return vec_sub(v, w_comp)

def unit_vector_safe(v, eps=1e-12, axis=-1):
    """
    Create a unit vector with numerical stability.
    
    Args:
        v: Input vector (jax array)
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Unit vector with same shape as input
    """
    norm = jnp.linalg.norm(v, axis=axis, keepdims=True)
    return v / jnp.maximum(norm, eps)

@jit
def vec_angle(v, w):
    """
    Get the angle between two vectors
    Returns 0.0 if either vector is zero vector (avoids division by zero)
    """
    v_mag = vec_mag(v)
    w_mag = vec_mag(w)
    
    # Check if either vector is zero
    either_zero = (v_mag == 0) | (w_mag == 0)
    
    # Compute angle components
    denom = vec_dot(v, w)
    num = vec_mag(jnp.cross(v, w))
    
    # Use jnp.where to avoid division by zero
    # If either vector is zero, return 0.0, otherwise compute angle
    return jnp.where(either_zero, 
                     0.0, 
                     jnp.arctan2(num, denom))
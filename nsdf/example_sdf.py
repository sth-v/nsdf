# example_sdf.py --------------------------------------------------------------
import numpy as np

def sdf(points: np.ndarray) -> np.ndarray:
    """
    Analytic signed‑distance field of a unit sphere centred at the origin.
    points : (..., 3) NumPy array
    returns: (...,)   NumPy array of SDF values
    """
    return np.linalg.norm(points, axis=-1) - 1.0

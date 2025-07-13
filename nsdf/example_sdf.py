# example_sdf.py --------------------------------------------------------------
import numpy as np

def sdf_sphere(points: np.ndarray) -> np.ndarray:
    """
    Analytic signed‑distance field of a unit sphere centred at the origin.
    points : (..., 3) NumPy array
    returns: (...,)   NumPy array of SDF values
    """
    return np.linalg.norm(points, axis=-1) - 1.0


import numpy as np

import numpy as np


def sd_cut_hollow_sphere(p: np.ndarray, r: float=1., h: float=0.9, t: float=0.1) -> np.ndarray:
    """
    Signed distance to a hollow sphere truncated by a plane.

    Parameters
    ----------
    p : array_like, shape (..., 3)
        Point(s) at which to evaluate the SDF.
    r : float
        Radius of the sphere.
    h : float
        Distance from center to the cutting plane along the Y-axis.
    t : float
        Shell (wall) thickness.

    Returns
    -------
    d : ndarray, shape (...)
        Signed distance value(s). Negative inside the hollow shell, positive outside.
    """
    p = np.asarray(p)
    # Precompute
    w = np.sqrt(r * r - h * h)  # float
    
    # Convert to "q" coordinates: q.x = sqrt(x^2 + z^2), q.y = y
    q_x = np.hypot(p[..., 0], p[..., 2])  # shape (...)
    q_y = p[..., 1]  # shape (...)
    
    # Branch condition: inside the cut cone vs on the spherical shell
    cond = (h * q_x) < (w * q_y)  # shape (...)
    
    # Distance to the cut circle center (w, h)
    d1 = np.hypot(q_x - w, q_y - h)  # shape (...)
    
    # Distance to the sphere surface
    d2 = np.abs(np.hypot(q_x, q_y) - r)  # shape (...)
    
    # Select per-point and subtract thickness
    d = np.where(cond, d1, d2) - t  # shape (...)
    return d
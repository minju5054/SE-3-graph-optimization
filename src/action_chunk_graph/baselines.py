import numpy as np

from .geometry import wrap_angle


def linear_crossfade(old, new):
    """
    Point-wise Euclidean crossfade.
    alpha goes from 0 (OLD) to 1 (NEW).
    """
    if old.shape != new.shape:
        raise ValueError("old and new must have the same shape")

    n = len(old)
    alpha = np.linspace(0.0, 1.0, n)

    out = np.zeros_like(old)
    out[:, :2] = (
        (1.0 - alpha[:, None]) * old[:, :2]
        + alpha[:, None] * new[:, :2]
    )

    dtheta = wrap_angle(new[:, 2] - old[:, 2])
    out[:, 2] = wrap_angle(old[:, 2] + alpha * dtheta)
    return out


def cubic_hermite_crossfade(old, new):
    """
    Crossfade two time-aligned trajectories with a cubic Hermite blend weight.

    The smoothstep polynomial h(a) = 3a^2 - 2a^3 has zero derivative at both
    ends. Consequently, the blended trajectory inherits the OLD tangent at the
    start and the NEW tangent at the end in the continuous-time formulation.
    """
    if old.shape != new.shape:
        raise ValueError("old and new must have the same shape")

    n = len(old)
    alpha = np.linspace(0.0, 1.0, n)
    weight = 3.0 * alpha**2 - 2.0 * alpha**3

    out = np.zeros_like(old)
    out[:, :2] = (
        (1.0 - weight[:, None]) * old[:, :2]
        + weight[:, None] * new[:, :2]
    )

    dtheta = wrap_angle(new[:, 2] - old[:, 2])
    out[:, 2] = wrap_angle(old[:, 2] + weight * dtheta)
    return out

import numpy as np


def wrap_angle(angle):
    """Map angle(s) to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def se2_relative_log(pose_a, pose_b):
    """
    Compute Log(T_a^{-1} T_b) for SE(2).

    pose = [x, y, theta]
    return = [rho_x, rho_y, theta_error]
    """
    xa, ya, tha = pose_a
    xb, yb, thb = pose_b

    c = np.cos(tha)
    s = np.sin(tha)

    dx = xb - xa
    dy = yb - ya

    tx = c * dx + s * dy
    ty = -s * dx + c * dy

    dtheta = float(wrap_angle(thb - tha))

    if abs(dtheta) < 1e-8:
        return np.array([tx, ty, dtheta], dtype=float)

    A = np.sin(dtheta) / dtheta
    B = (1.0 - np.cos(dtheta)) / dtheta

    det = A * A + B * B
    rho_x = (A * tx + B * ty) / det
    rho_y = (-B * tx + A * ty) / det

    return np.array([rho_x, rho_y, dtheta], dtype=float)


def heading_from_xy(x, y):
    """Compute tangent heading for a 2D trajectory."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.unwrap(np.arctan2(dy, dx))

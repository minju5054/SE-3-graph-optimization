from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .geometry import se2_relative_log, wrap_angle


@dataclass
class GraphConfig:
    lambda_old: float = 2.0
    lambda_new: float = 2.0
    lambda_smooth: float = 25.0
    lambda_collision: float = 1200.0
    rotation_scale: float = 0.5
    max_nfev: int = 120


def optimize_reconciled_trajectory(old, new, obstacle=None, config=None):
    if config is None:
        config = GraphConfig()

    if old.shape != new.shape or old.shape[1] != 3:
        raise ValueError("Expected old/new shape [N, 3] with [x, y, theta].")

    n = len(old)
    current_pose = old[0].copy()

    initial = old.copy()
    z0 = initial[1:].reshape(-1)

    def unpack(z):
        X = np.vstack([current_pose, z.reshape(n - 1, 3)])
        X[:, 2] = wrap_angle(X[:, 2])
        return X

    def scaled_twist(r):
        r = np.asarray(r, dtype=float).copy()
        r[2] *= config.rotation_scale
        return r

    def residual_vector(z):
        X = unpack(z)
        residuals = []

        for i in range(1, n):
            alpha = i / (n - 1)

            w_old = config.lambda_old * (1.0 - alpha) + 1e-4
            w_new = config.lambda_new * alpha + 1e-4

            r_old = scaled_twist(se2_relative_log(old[i], X[i]))
            r_new = scaled_twist(se2_relative_log(new[i], X[i]))

            residuals.extend(np.sqrt(w_old) * r_old)
            residuals.extend(np.sqrt(w_new) * r_new)

        twists = [
            scaled_twist(se2_relative_log(X[i], X[i + 1]))
            for i in range(n - 1)
        ]
        for i in range(len(twists) - 1):
            residuals.extend(
                np.sqrt(config.lambda_smooth) * (twists[i + 1] - twists[i])
            )

        if obstacle is not None:
            center = np.asarray(obstacle["center"], dtype=float)
            safe_radius = obstacle["radius"] + obstacle["margin"]

            for i in range(1, n):
                distance = np.linalg.norm(X[i, :2] - center)
                violation = max(0.0, safe_radius - distance)
                residuals.append(np.sqrt(config.lambda_collision) * violation)

        return np.asarray(residuals, dtype=float)

    result = least_squares(
        residual_vector,
        z0,
        method="trf",
        max_nfev=config.max_nfev,
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
    )

    return unpack(result.x), result

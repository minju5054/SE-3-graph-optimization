from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import se2_relative_log, se3_relative_log, wrap_angle


@dataclass
class GraphConfig:
    lambda_old: float = 2.0
    lambda_new: float = 2.0
    lambda_smooth: float = 25.0
    lambda_collision: float = 1200.0
    rotation_scale: float = 0.5
    max_nfev: int = 120
    collision_factor: str = "nodes"


@dataclass
class GraphConfigSE3:
    lambda_old: float = 2.0
    lambda_new: float = 2.0
    lambda_smooth: float = 25.0
    lambda_collision: float = 1200.0
    rotation_scale: float = 0.35
    max_nfev: int = 160
    collision_factor: str = "segments"
    tolerance: float = 1e-6


def optimize_reconciled_trajectory(old, new, obstacle=None, config=None):
    if config is None:
        config = GraphConfig()

    if old.shape != new.shape or old.shape[1] != 3:
        raise ValueError("Expected old/new shape [N, 3] with [x, y, theta].")
    if config.collision_factor not in {"nodes", "segments"}:
        raise ValueError("collision_factor must be 'nodes' or 'segments'.")

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

            if config.collision_factor == "nodes":
                distances = [
                    np.linalg.norm(X[i, :2] - center) for i in range(1, n)
                ]
            else:
                distances = []
                for i in range(n - 1):
                    start = X[i, :2]
                    segment = X[i + 1, :2] - start
                    squared_length = np.dot(segment, segment)
                    if squared_length > 1e-12:
                        projection = np.dot(center - start, segment) / squared_length
                        projection = np.clip(projection, 0.0, 1.0)
                        closest = start + projection * segment
                    else:
                        closest = start
                    distances.append(np.linalg.norm(closest - center))

            for distance in distances:
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


def optimize_reconciled_trajectory_se3(old, new, obstacle=None, config=None):
    """Optimize [x, y, z, rotation-vector] poses with SE(3) residuals."""
    if config is None:
        config = GraphConfigSE3()

    if old.shape != new.shape or old.ndim != 2 or old.shape[1] != 6:
        raise ValueError(
            "Expected old/new shape [N, 6] with [x, y, z, rotation-vector]."
        )
    if len(old) < 2:
        raise ValueError("Expected at least two poses.")
    if config.collision_factor not in {"nodes", "segments"}:
        raise ValueError("collision_factor must be 'nodes' or 'segments'.")

    n = len(old)
    current_pose = old[0].copy()
    z0 = old[1:].reshape(-1)

    def unpack(z):
        return np.vstack([current_pose, z.reshape(n - 1, 6)])

    def scaled_twist(residual):
        residual = np.asarray(residual, dtype=float).copy()
        residual[3:] *= config.rotation_scale
        return residual

    def residual_vector(z):
        trajectory = unpack(z)
        residuals = []

        for i in range(1, n):
            alpha = i / (n - 1)
            weight_old = config.lambda_old * (1.0 - alpha) + 1e-4
            weight_new = config.lambda_new * alpha + 1e-4
            residual_old = scaled_twist(
                se3_relative_log(old[i], trajectory[i])
            )
            residual_new = scaled_twist(
                se3_relative_log(new[i], trajectory[i])
            )
            residuals.extend(np.sqrt(weight_old) * residual_old)
            residuals.extend(np.sqrt(weight_new) * residual_new)

        twists = [
            scaled_twist(se3_relative_log(trajectory[i], trajectory[i + 1]))
            for i in range(n - 1)
        ]
        for i in range(len(twists) - 1):
            residuals.extend(
                np.sqrt(config.lambda_smooth) * (twists[i + 1] - twists[i])
            )

        if obstacle is not None:
            center = np.asarray(obstacle["center"], dtype=float)
            safe_radius = obstacle["radius"] + obstacle["margin"]
            if config.collision_factor == "nodes":
                distances = [
                    np.linalg.norm(trajectory[i, :3] - center)
                    for i in range(1, n)
                ]
            else:
                distances = []
                for i in range(n - 1):
                    start = trajectory[i, :3]
                    segment = trajectory[i + 1, :3] - start
                    squared_length = np.dot(segment, segment)
                    if squared_length > 1e-12:
                        projection = (
                            np.dot(center - start, segment) / squared_length
                        )
                        projection = np.clip(projection, 0.0, 1.0)
                        closest = start + projection * segment
                    else:
                        closest = start
                    distances.append(np.linalg.norm(closest - center))

            for distance in distances:
                violation = max(0.0, safe_radius - distance)
                residuals.append(np.sqrt(config.lambda_collision) * violation)

        return np.asarray(residuals, dtype=float)

    result = least_squares(
        residual_vector,
        z0,
        method="trf",
        max_nfev=config.max_nfev,
        xtol=config.tolerance,
        ftol=config.tolerance,
        gtol=config.tolerance,
        x_scale="jac",
    )

    trajectory = unpack(result.x)
    trajectory[:, 3:] = Rotation.from_rotvec(trajectory[:, 3:]).as_rotvec()
    return trajectory, result

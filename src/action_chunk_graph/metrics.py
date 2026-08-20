import numpy as np

from .geometry import se2_relative_log, wrap_angle


def minimum_clearance(trajectory, obstacle):
    center = np.asarray(obstacle["center"], dtype=float)
    radius = float(obstacle["radius"])
    distances = np.linalg.norm(trajectory[:, :2] - center, axis=1)
    return float(np.min(distances - radius))


def collision(trajectory, obstacle):
    return minimum_clearance(trajectory, obstacle) < 0.0


def safety_margin_violation(trajectory, obstacle):
    center = np.asarray(obstacle["center"], dtype=float)
    safe_radius = obstacle["radius"] + obstacle["margin"]
    distances = np.linalg.norm(trajectory[:, :2] - center, axis=1)
    return float(np.max(np.maximum(0.0, safe_radius - distances)))


def polyline_minimum_clearance(trajectory, obstacle):
    """Minimum circle clearance over every line segment of a trajectory."""
    if len(trajectory) < 2:
        return minimum_clearance(trajectory, obstacle)

    center = np.asarray(obstacle["center"], dtype=float)
    radius = float(obstacle["radius"])
    starts = trajectory[:-1, :2]
    segments = trajectory[1:, :2] - starts
    squared_lengths = np.sum(segments * segments, axis=1)

    projections = np.divide(
        np.sum((center - starts) * segments, axis=1),
        squared_lengths,
        out=np.zeros_like(squared_lengths),
        where=squared_lengths > 0.0,
    )
    projections = np.clip(projections, 0.0, 1.0)
    closest_points = starts + projections[:, None] * segments
    distances = np.linalg.norm(closest_points - center, axis=1)
    return float(np.min(distances - radius))


def polyline_collision(trajectory, obstacle):
    return polyline_minimum_clearance(trajectory, obstacle) < 0.0


def polyline_safety_margin_violation(trajectory, obstacle):
    safe_clearance = float(obstacle["margin"])
    clearance = polyline_minimum_clearance(trajectory, obstacle)
    return float(max(0.0, safe_clearance - clearance))


def translational_jerk_rms(trajectory, dt=1.0):
    p = trajectory[:, :2]
    if len(p) < 4:
        return 0.0
    v = np.diff(p, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(j * j, axis=1))))


def rotational_increment_rms(trajectory):
    dtheta = wrap_angle(np.diff(trajectory[:, 2]))
    return float(np.sqrt(np.mean(dtheta * dtheta)))


def body_motion_smoothness(trajectory):
    twists = np.array(
        [
            se2_relative_log(trajectory[i], trajectory[i + 1])
            for i in range(len(trajectory) - 1)
        ]
    )
    if len(twists) < 2:
        return 0.0
    delta = np.diff(twists, axis=0)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def mean_translational_deviation(trajectory, reference):
    if trajectory.shape != reference.shape:
        raise ValueError("trajectory and reference must have the same shape")
    delta = trajectory[:, :2] - reference[:, :2]
    return float(np.mean(np.linalg.norm(delta, axis=1)))


def mean_rotational_deviation(trajectory, reference):
    if trajectory.shape != reference.shape:
        raise ValueError("trajectory and reference must have the same shape")
    delta = wrap_angle(trajectory[:, 2] - reference[:, 2])
    return float(np.mean(np.abs(delta)))

import numpy as np

from .geometry import se2_relative_log, se3_relative_log, wrap_angle


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


def transition_velocity_mismatches(
    trajectory, modification_step, transition_window_poses, dt=1.0
):
    """Velocity discontinuity at the start and end of a local transition."""
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if modification_step < 1:
        raise ValueError("modification_step must have an incoming segment.")
    if transition_window_poses < 1:
        raise ValueError("transition_window_poses must be positive.")

    end_step = modification_step + transition_window_poses - 1
    if modification_step + 1 >= len(trajectory) or end_step + 1 >= len(trajectory):
        raise ValueError("Trajectory does not include both transition boundaries.")

    incoming_start = (
        trajectory[modification_step, :2]
        - trajectory[modification_step - 1, :2]
    ) / dt
    outgoing_start = (
        trajectory[modification_step + 1, :2]
        - trajectory[modification_step, :2]
    ) / dt
    incoming_end = (
        trajectory[end_step, :2] - trajectory[end_step - 1, :2]
    ) / dt
    outgoing_end = (
        trajectory[end_step + 1, :2] - trajectory[end_step, :2]
    ) / dt
    return (
        float(np.linalg.norm(outgoing_start - incoming_start)),
        float(np.linalg.norm(outgoing_end - incoming_end)),
    )


def steps_to_aligned_position_tolerance(
    trajectory,
    new,
    observation_step,
    modification_step,
    tolerance,
):
    """First step after modification whose pose is close to aligned NEW."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    start_new = modification_step - observation_step
    if start_new < 0:
        raise ValueError("modification_step precedes the NEW observation origin.")
    executed = trajectory[modification_step:, :2]
    reference = new[start_new:, :2]
    if executed.shape != reference.shape:
        raise ValueError("Executed and aligned NEW suffixes must have equal shape.")

    errors = np.linalg.norm(executed - reference, axis=1)
    reached = np.flatnonzero(errors <= tolerance)
    return int(reached[0]) if len(reached) else None


def spatial_minimum_clearance(trajectory, obstacle):
    center = np.asarray(obstacle["center"], dtype=float)
    radius = float(obstacle["radius"])
    distances = np.linalg.norm(trajectory[:, :3] - center, axis=1)
    return float(np.min(distances - radius))


def spatial_collision(trajectory, obstacle):
    return spatial_minimum_clearance(trajectory, obstacle) < 0.0


def spatial_safety_margin_violation(trajectory, obstacle):
    safe_radius = float(obstacle["radius"] + obstacle["margin"])
    center = np.asarray(obstacle["center"], dtype=float)
    distances = np.linalg.norm(trajectory[:, :3] - center, axis=1)
    return float(np.max(np.maximum(0.0, safe_radius - distances)))


def spatial_polyline_minimum_clearance(trajectory, obstacle):
    if len(trajectory) < 2:
        return spatial_minimum_clearance(trajectory, obstacle)

    center = np.asarray(obstacle["center"], dtype=float)
    radius = float(obstacle["radius"])
    starts = trajectory[:-1, :3]
    segments = trajectory[1:, :3] - starts
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


def spatial_polyline_collision(trajectory, obstacle):
    return spatial_polyline_minimum_clearance(trajectory, obstacle) < 0.0


def spatial_polyline_safety_margin_violation(trajectory, obstacle):
    clearance = spatial_polyline_minimum_clearance(trajectory, obstacle)
    return float(max(0.0, float(obstacle["margin"]) - clearance))


def spatial_translational_jerk_rms(trajectory, dt=1.0):
    positions = trajectory[:, :3]
    if len(positions) < 4:
        return 0.0
    velocity = np.diff(positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt
    jerk = np.diff(acceleration, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1))))


def spatial_rotational_increment_rms(trajectory):
    increments = np.array(
        [
            se3_relative_log(trajectory[i], trajectory[i + 1])[3:]
            for i in range(len(trajectory) - 1)
        ]
    )
    return float(np.sqrt(np.mean(np.sum(increments * increments, axis=1))))


def spatial_body_motion_smoothness(trajectory, rotation_scale=1.0):
    twists = np.array(
        [
            se3_relative_log(trajectory[i], trajectory[i + 1])
            for i in range(len(trajectory) - 1)
        ]
    )
    if len(twists) < 2:
        return 0.0
    twists[:, 3:] *= rotation_scale
    delta = np.diff(twists, axis=0)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))

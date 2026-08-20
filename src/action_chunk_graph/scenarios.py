import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import heading_from_xy


def make_collision_stress_scenario(num_poses=21):
    """
    Two individually safe trajectories go around opposite sides of an obstacle.

    This is a stress test. Point-wise blending crosses the obstacle.
    """
    t = np.linspace(0.0, 1.0, num_poses)
    x = 4.0 * t

    y_old = 0.85 * np.sin(np.pi * t)
    y_new = -0.85 * np.sin(np.pi * t)

    old = np.column_stack([x, y_old, heading_from_xy(x, y_old)])
    new = np.column_stack([x, y_new, heading_from_xy(x, y_new)])

    obstacle = {
        "center": np.array([2.0, 0.0]),
        "radius": 0.45,
        "margin": 0.12,
    }
    return old, new, obstacle


def make_smoothness_stitch_scenario(num_poses=31):
    """
    Two smooth but disagreeing chunks for isolating stitching quality.

    The chunks start at the same position, then differ in curvature and finish
    at different poses. No obstacle is included so that collision avoidance
    cannot confound the smoothness comparison.
    """
    t = np.linspace(0.0, 1.0, num_poses)

    x_old = 4.0 * t
    y_old = 0.70 * np.sin(np.pi * t) + 0.10 * np.sin(2.0 * np.pi * t)

    x_new = 4.2 * t
    y_new = -0.55 * np.sin(np.pi * t) + 0.30 * t

    old = np.column_stack(
        [x_old, y_old, heading_from_xy(x_old, y_old)]
    )
    new = np.column_stack(
        [x_new, y_new, heading_from_xy(x_new, y_new)]
    )
    return old, new


def make_collision_scenario_suite(num_scenarios=12, num_poses=21, seed=42):
    """Create deterministic stress cases whose OLD/NEW paths pass opposite sides."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, num_poses)
    x = 4.0 * t
    scenarios = []

    for index in range(num_scenarios):
        center_x = rng.uniform(1.65, 2.35)
        center_y = rng.uniform(-0.08, 0.08)
        radius = rng.uniform(0.32, 0.48)
        margin = rng.uniform(0.08, 0.14)
        extra_clearance = rng.uniform(0.16, 0.30)
        shape_power = rng.uniform(0.85, 1.20)

        shape = np.sin(np.pi * t) ** shape_power
        center_progress = center_x / 4.0
        shape_at_center = np.sin(np.pi * center_progress) ** shape_power
        safe_radius = radius + margin

        upper_amplitude = (
            center_y + safe_radius + extra_clearance
        ) / shape_at_center
        lower_amplitude = (
            -center_y + safe_radius + extra_clearance
        ) / shape_at_center

        y_old = upper_amplitude * shape
        y_new = -lower_amplitude * shape
        old = np.column_stack([x, y_old, heading_from_xy(x, y_old)])
        new = np.column_stack([x, y_new, heading_from_xy(x, y_new)])
        obstacle = {
            "center": np.array([center_x, center_y]),
            "radius": radius,
            "margin": margin,
        }
        scenarios.append(
            {
                "scenario": f"scenario_{index:02d}",
                "old": old,
                "new": new,
                "obstacle": obstacle,
            }
        )

    return scenarios


def _rotation_vectors_from_tangents(tangents, roll):
    forwards = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)
    world_up = np.tile(np.array([0.0, 0.0, 1.0]), (len(tangents), 1))
    sides = np.cross(world_up, forwards)
    sides /= np.linalg.norm(sides, axis=1, keepdims=True)
    corrected_up = np.cross(forwards, sides)

    frames = np.stack([forwards, sides, corrected_up], axis=2)
    base_rotation = Rotation.from_matrix(frames)
    local_roll = Rotation.from_rotvec(
        np.column_stack([roll, np.zeros_like(roll), np.zeros_like(roll)])
    )
    return (base_rotation * local_roll).as_rotvec()


def make_se3_collision_scenario(num_poses=17):
    """Two safe 6-DoF trajectories whose point-wise blend crosses a sphere."""
    t = np.linspace(0.0, 1.0, num_poses)
    x = 4.0 * t
    shape = np.sin(np.pi * t) ** 2
    shape_derivative = np.pi * np.sin(2.0 * np.pi * t)

    old_position = np.column_stack([x, 0.78 * shape, 0.32 * shape])
    new_position = np.column_stack([x, -0.78 * shape, -0.32 * shape])

    old_tangent = np.column_stack(
        [np.full_like(t, 4.0), 0.78 * shape_derivative, 0.32 * shape_derivative]
    )
    new_tangent = np.column_stack(
        [np.full_like(t, 4.0), -0.78 * shape_derivative, -0.32 * shape_derivative]
    )

    roll_amplitude = np.deg2rad(170.0)
    old_rotation = _rotation_vectors_from_tangents(
        old_tangent, roll_amplitude * shape
    )
    new_rotation = _rotation_vectors_from_tangents(
        new_tangent, -roll_amplitude * shape
    )

    old = np.column_stack([old_position, old_rotation])
    new = np.column_stack([new_position, new_rotation])
    obstacle = {
        "center": np.array([2.0, 0.0, 0.0]),
        "radius": 0.45,
        "margin": 0.15,
    }
    return old, new, obstacle

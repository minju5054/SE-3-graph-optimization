import numpy as np

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

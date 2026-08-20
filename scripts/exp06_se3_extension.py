from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from action_chunk_graph.baselines import (
    se3_euclidean_crossfade,
    se3_geodesic_crossfade,
)
from action_chunk_graph.geometry import (
    pose6_to_matrix,
    se3_exp,
    se3_log,
    se3_relative_log,
)
from action_chunk_graph.metrics import (
    spatial_body_motion_smoothness,
    spatial_collision,
    spatial_polyline_collision,
    spatial_polyline_minimum_clearance,
    spatial_polyline_safety_margin_violation,
    spatial_rotational_increment_rms,
    spatial_translational_jerk_rms,
)
from action_chunk_graph.optimizer import (
    GraphConfigSE3,
    optimize_reconciled_trajectory_se3,
)
from action_chunk_graph.scenarios import make_se3_collision_scenario


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "exp06_se3_extension"

SEED = 42
NUM_POSES = 9
DT = 0.1
ROTATION_SCALE = 0.35

METHOD_ORDER = [
    "OLD proposal",
    "NEW proposal",
    "Euclidean rotvec crossfade",
    "SO(3) geodesic crossfade",
    "SE(3) graph without collision",
    "SE(3) graph with segment collision",
]
COLORS = {
    "OLD proposal": "tab:blue",
    "NEW proposal": "tab:orange",
    "Euclidean rotvec crossfade": "tab:green",
    "SO(3) geodesic crossfade": "tab:red",
    "SE(3) graph without collision": "tab:purple",
    "SE(3) graph with segment collision": "tab:brown",
}


def validate_geometry(num_samples=1000):
    rng = np.random.default_rng(SEED)
    max_exp_log_error = 0.0
    max_relative_error = 0.0

    for _ in range(num_samples):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.0, np.pi - 1e-4)
        twist = np.concatenate([rng.uniform(-2.0, 2.0, size=3), axis * angle])
        transform = se3_exp(twist)
        reconstructed = se3_exp(se3_log(transform))
        max_exp_log_error = max(
            max_exp_log_error,
            float(np.linalg.norm(reconstructed - transform)),
        )

        pose_a = np.concatenate(
            [rng.uniform(-2.0, 2.0, size=3), rng.uniform(-1.0, 1.0, size=3)]
        )
        pose_b = np.concatenate(
            [rng.uniform(-2.0, 2.0, size=3), rng.uniform(-1.0, 1.0, size=3)]
        )
        relative = se3_relative_log(pose_a, pose_b)
        target = pose6_to_matrix(pose_b)
        relative_reconstruction = pose6_to_matrix(pose_a) @ se3_exp(relative)
        max_relative_error = max(
            max_relative_error,
            float(np.linalg.norm(relative_reconstruction - target)),
        )

    pose_positive = np.array(
        [0.0, 0.0, 0.0, np.deg2rad(179.0), 0.0, 0.0]
    )
    pose_negative = np.array(
        [0.0, 0.0, 0.0, np.deg2rad(-179.0), 0.0, 0.0]
    )
    wrapped_rotation = np.rad2deg(
        np.linalg.norm(se3_relative_log(pose_positive, pose_negative)[3:])
    )
    return pd.DataFrame(
        [
            {
                "num_random_samples": num_samples,
                "max_exp_log_matrix_error": max_exp_log_error,
                "max_relative_reconstruction_error": max_relative_error,
                "rotation_179_to_minus179_deg": wrapped_rotation,
            }
        ]
    )


def timed_call(function, *args, **kwargs):
    start = perf_counter()
    value = function(*args, **kwargs)
    runtime_ms = 1000.0 * (perf_counter() - start)
    return value, runtime_ms


def rotation_distance_degrees(pose_a, pose_b):
    relative_rotation = se3_relative_log(pose_a, pose_b)[3:]
    return float(np.rad2deg(np.linalg.norm(relative_rotation)))


def evaluate(
    name,
    trajectory,
    old,
    new,
    obstacle,
    runtime_ms=0.0,
    optimizer_result=None,
):
    midpoint = len(trajectory) // 2
    row = {
        "method": name,
        "sampled_collision": spatial_collision(trajectory, obstacle),
        "polyline_collision": spatial_polyline_collision(trajectory, obstacle),
        "polyline_min_clearance": spatial_polyline_minimum_clearance(
            trajectory, obstacle
        ),
        "polyline_safety_margin_violation": (
            spatial_polyline_safety_margin_violation(trajectory, obstacle)
        ),
        "translational_jerk_rms": spatial_translational_jerk_rms(
            trajectory, dt=DT
        ),
        "rotation_step_rms": spatial_rotational_increment_rms(trajectory),
        "body_motion_smoothness": spatial_body_motion_smoothness(
            trajectory, rotation_scale=ROTATION_SCALE
        ),
        "midpoint_rotation_from_old_deg": rotation_distance_degrees(
            old[midpoint], trajectory[midpoint]
        ),
        "midpoint_rotation_from_new_deg": rotation_distance_degrees(
            new[midpoint], trajectory[midpoint]
        ),
        "generation_runtime_ms": runtime_ms,
        "optimizer_success": None,
        "optimizer_cost": np.nan,
        "optimizer_nfev": np.nan,
        "optimizer_optimality": np.nan,
    }
    if optimizer_result is not None:
        row.update(
            {
                "optimizer_success": bool(optimizer_result.success),
                "optimizer_cost": float(optimizer_result.cost),
                "optimizer_nfev": int(optimizer_result.nfev),
                "optimizer_optimality": float(optimizer_result.optimality),
            }
        )
    return row


def add_sphere(axis, center, radius, color, alpha, wireframe=False):
    longitude = np.linspace(0.0, 2.0 * np.pi, 36)
    latitude = np.linspace(0.0, np.pi, 18)
    x = center[0] + radius * np.outer(np.cos(longitude), np.sin(latitude))
    y = center[1] + radius * np.outer(np.sin(longitude), np.sin(latitude))
    z = center[2] + radius * np.outer(np.ones_like(longitude), np.cos(latitude))
    if wireframe:
        axis.plot_wireframe(x, y, z, color=color, alpha=alpha, linewidth=0.5)
    else:
        axis.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0.0)


def make_figure(metrics, trajectories, obstacle):
    fig = plt.figure(figsize=(14, 10))
    ax_path = fig.add_subplot(2, 2, 1, projection="3d")
    ax_clearance = fig.add_subplot(2, 2, 2)
    ax_smoothness = fig.add_subplot(2, 2, 3)
    ax_orientation = fig.add_subplot(2, 2, 4)

    for method in METHOD_ORDER:
        trajectory = trajectories[method]
        linestyle = "--" if "proposal" in method else "-"
        linewidth = 2.7 if "segment collision" in method else 1.7
        ax_path.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            linestyle=linestyle,
            linewidth=linewidth,
            color=COLORS[method],
            label=method,
        )

    center = obstacle["center"]
    add_sphere(
        ax_path,
        center,
        obstacle["radius"],
        color="tab:blue",
        alpha=0.22,
    )
    add_sphere(
        ax_path,
        center,
        obstacle["radius"] + obstacle["margin"],
        color="black",
        alpha=0.18,
        wireframe=True,
    )
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    ax_path.set_zlabel("z")
    ax_path.set_title("3D reconciled trajectories")
    ax_path.legend(fontsize=7, loc="upper left")
    ax_path.view_init(elev=24, azim=-62)

    colors = [COLORS[method] for method in METHOD_ORDER]
    bars = ax_clearance.bar(
        METHOD_ORDER,
        metrics.set_index("method").loc[METHOD_ORDER, "polyline_min_clearance"],
        color=colors,
    )
    ax_clearance.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax_clearance.axhline(0.0, color="black", linestyle=":")
    ax_clearance.set_ylabel("Polyline minimum clearance")
    ax_clearance.set_title("Sphere collision clearance")
    ax_clearance.tick_params(axis="x", rotation=25, labelsize=8)
    ax_clearance.grid(True, axis="y", alpha=0.3)

    smoothness_columns = [
        "translational_jerk_rms",
        "rotation_step_rms",
        "body_motion_smoothness",
    ]
    graph_rows = metrics[metrics["method"].str.startswith("SE(3) graph")]
    normalized = graph_rows[smoothness_columns].to_numpy(copy=True)
    normalized /= normalized[0]
    x = np.arange(len(smoothness_columns))
    width = 0.35
    for index, (_, row) in enumerate(graph_rows.iterrows()):
        bars = ax_smoothness.bar(
            x + (index - 0.5) * width,
            normalized[index],
            width,
            color=COLORS[row["method"]],
            label=row["method"],
        )
        ax_smoothness.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax_smoothness.axhline(1.0, color="black", linestyle=":")
    ax_smoothness.set_xticks(x, ["Trans. jerk", "Rotation step", "Body motion"])
    ax_smoothness.set_ylabel("Ratio to graph without collision")
    ax_smoothness.set_title("Graph collision-factor trade-off")
    ax_smoothness.legend(fontsize=8)
    ax_smoothness.grid(True, axis="y", alpha=0.3)

    orientation_methods = [
        "Euclidean rotvec crossfade",
        "SO(3) geodesic crossfade",
    ]
    orientation_rows = metrics.set_index("method").loc[orientation_methods]
    x = np.arange(len(orientation_methods))
    width = 0.36
    ax_orientation.bar(
        x - width / 2,
        orientation_rows["midpoint_rotation_from_old_deg"],
        width,
        label="Distance from OLD",
    )
    ax_orientation.bar(
        x + width / 2,
        orientation_rows["midpoint_rotation_from_new_deg"],
        width,
        label="Distance from NEW",
    )
    ax_orientation.set_xticks(x, ["Euclidean rotvec", "SO(3) geodesic"])
    ax_orientation.set_ylabel("Midpoint rotation distance (deg)")
    ax_orientation.set_title("170° / -170° rotation-vector boundary")
    ax_orientation.legend()
    ax_orientation.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Experiment 06: SE(3) Geometry and Graph Optimization")
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    geometry_validation = validate_geometry()
    geometry_path = OUTPUT_DIR / "geometry_validation.csv"
    geometry_validation.to_csv(geometry_path, index=False)

    old, new, obstacle = make_se3_collision_scenario(num_poses=NUM_POSES)
    euclidean, euclidean_runtime = timed_call(se3_euclidean_crossfade, old, new)
    geodesic, geodesic_runtime = timed_call(se3_geodesic_crossfade, old, new)

    config_without_collision = GraphConfigSE3(
        lambda_collision=0.0,
        collision_factor="segments",
    )
    config_with_collision = GraphConfigSE3(
        lambda_collision=1200.0,
        collision_factor="segments",
    )
    graph_without_output, graph_without_runtime = timed_call(
        optimize_reconciled_trajectory_se3,
        old,
        new,
        None,
        config_without_collision,
    )
    graph_without, result_without = graph_without_output
    graph_with_output, graph_with_runtime = timed_call(
        optimize_reconciled_trajectory_se3,
        old,
        new,
        obstacle,
        config_with_collision,
    )
    graph_with, result_with = graph_with_output

    if not result_without.success or not result_with.success:
        raise RuntimeError(
            "SE(3) optimizer failed: "
            f"without={result_without.message}, with={result_with.message}"
        )

    trajectories = {
        "OLD proposal": old,
        "NEW proposal": new,
        "Euclidean rotvec crossfade": euclidean,
        "SO(3) geodesic crossfade": geodesic,
        "SE(3) graph without collision": graph_without,
        "SE(3) graph with segment collision": graph_with,
    }
    for method, trajectory in trajectories.items():
        if not np.all(np.isfinite(trajectory)):
            raise RuntimeError(f"{method} produced non-finite values.")

    rows = [
        evaluate("OLD proposal", old, old, new, obstacle),
        evaluate("NEW proposal", new, old, new, obstacle),
        evaluate(
            "Euclidean rotvec crossfade",
            euclidean,
            old,
            new,
            obstacle,
            euclidean_runtime,
        ),
        evaluate(
            "SO(3) geodesic crossfade",
            geodesic,
            old,
            new,
            obstacle,
            geodesic_runtime,
        ),
        evaluate(
            "SE(3) graph without collision",
            graph_without,
            old,
            new,
            obstacle,
            graph_without_runtime,
            result_without,
        ),
        evaluate(
            "SE(3) graph with segment collision",
            graph_with,
            old,
            new,
            obstacle,
            graph_with_runtime,
            result_with,
        ),
    ]
    metrics = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    figure = make_figure(metrics, trajectories, obstacle)
    figure_path = OUTPUT_DIR / "figure.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print("\n=== SE(3) Geometry Validation ===")
    print(geometry_validation.to_string(index=False))
    print("\n=== Experiment 06: SE(3) Extension ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved: {geometry_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

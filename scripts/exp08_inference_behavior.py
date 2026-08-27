from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from action_chunk_graph.execution import (
    ExecutionConfig,
    assemble_local_transition,
    build_committed_prefix,
    local_transition_windows,
)
from action_chunk_graph.geometry import wrap_angle
from action_chunk_graph.metrics import (
    body_motion_smoothness,
    polyline_collision,
    polyline_minimum_clearance,
    polyline_safety_margin_violation,
    rotational_increment_rms,
    translational_jerk_rms,
)
from action_chunk_graph.optimizer import GraphConfig, optimize_reconciled_trajectory
from action_chunk_graph.scenarios import make_async_obstacle_update_scenario


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "exp08_inference_behavior"

DT = 0.1
NUM_POSES = 31
OBSERVATION_STEP = 8
COMMIT_HORIZON_STEPS = 0
OPTIMIZATION_WINDOW_POSES = 7
LATENCY_STEPS = [0, 2, 4, 6, 8]
INFERENCE_BEHAVIORS = ["continue_old", "hold_pose"]
SEED = 42
REPRESENTATIVE_LATENCY = 6
GRAPH_CONFIG = GraphConfig(
    lambda_old=2.0,
    lambda_new=2.0,
    lambda_smooth=25.0,
    lambda_collision=3000.0,
    rotation_scale=0.5,
    max_nfev=600,
    collision_factor="segments",
    lambda_terminal_new=1200.0,
)
COLORS = {"continue_old": "tab:red", "hold_pose": "tab:green"}


def distance_travelled(trajectory):
    if len(trajectory) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)))


def run_case(old, new, obstacle, inference_behavior, latency_steps):
    config = ExecutionConfig(
        dt=DT,
        observation_step=OBSERVATION_STEP,
        inference_latency_steps=latency_steps,
        commit_horizon_steps=COMMIT_HORIZON_STEPS,
        optimization_window_poses=OPTIMIZATION_WINDOW_POSES,
        inference_behavior=inference_behavior,
    )
    prefix = build_committed_prefix(old, config)
    before_ready = prefix[: config.new_ready_step + 1]
    old_window, new_window = local_transition_windows(old, new, config)
    start = perf_counter()
    graph_window, result = optimize_reconciled_trajectory(
        old_window, new_window, obstacle, GRAPH_CONFIG
    )
    optimizer_runtime_ms = 1000.0 * (perf_counter() - start)
    trajectory = assemble_local_transition(old, new, graph_window, config)
    if not result.success:
        raise RuntimeError(
            f"Optimizer failed for {inference_behavior}, latency={latency_steps}: "
            f"{result.message}"
        )

    inference_slice = prefix[
        config.observation_step : config.new_ready_step + 1
    ]
    aligned_start = config.modification_step - config.observation_step
    row = {
        "inference_behavior": inference_behavior,
        "latency_steps": latency_steps,
        "latency_seconds": latency_steps * config.dt,
        "observation_step": config.observation_step,
        "new_ready_step": config.new_ready_step,
        "modification_step": config.modification_step,
        "commit_horizon_steps": config.commit_horizon_steps,
        "optimization_window_poses": config.optimization_window_poses,
        "collision_before_new_ready": polyline_collision(before_ready, obstacle),
        "minimum_clearance_before_new_ready": polyline_minimum_clearance(
            before_ready, obstacle
        ),
        "full_polyline_collision": polyline_collision(trajectory, obstacle),
        "full_polyline_minimum_clearance": polyline_minimum_clearance(
            trajectory, obstacle
        ),
        "full_polyline_safety_margin_violation": (
            polyline_safety_margin_violation(trajectory, obstacle)
        ),
        "distance_travelled_during_inference": distance_travelled(
            inference_slice
        ),
        "translational_jerk_rms": translational_jerk_rms(
            trajectory, dt=config.dt
        ),
        "rotational_increment_rms": rotational_increment_rms(trajectory),
        "body_motion_smoothness": body_motion_smoothness(trajectory),
        "final_position_error_to_new": float(
            np.linalg.norm(trajectory[-1, :2] - new[-1, :2])
        ),
        "final_rotation_error_to_new": float(
            abs(wrap_angle(trajectory[-1, 2] - new[-1, 2]))
        ),
        "mean_new_position_deviation_after_modify": float(
            np.mean(
                np.linalg.norm(
                    trajectory[config.modification_step :, :2]
                    - new[aligned_start:, :2],
                    axis=1,
                )
            )
        ),
        "optimizer_runtime_ms": optimizer_runtime_ms,
        "optimizer_success": bool(result.success),
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
    }
    return row, trajectory, prefix, config


def make_figure(metrics, representative, old, new, obstacle):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_path, ax_clearance, ax_collision, ax_progress = axes.flat
    ax_path.plot(old[:, 0], old[:, 1], "--", color="black", label="OLD")
    ax_path.plot(new[:, 0], new[:, 1], "--", color="tab:orange", label="NEW")
    center = obstacle["center"]
    ax_path.add_patch(
        Circle(center, obstacle["radius"], color="tab:blue", alpha=0.28)
    )
    ax_path.add_patch(
        Circle(
            center,
            obstacle["radius"] + obstacle["margin"],
            fill=False,
            color="black",
            linestyle=":",
        )
    )
    for behavior in INFERENCE_BEHAVIORS:
        trajectory, prefix, config = representative[behavior]
        ax_path.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=COLORS[behavior],
            linewidth=2.2,
            label=behavior,
        )
        ax_path.scatter(
            prefix[config.new_ready_step, 0],
            prefix[config.new_ready_step, 1],
            marker="x",
            s=70,
            color=COLORS[behavior],
            label=f"{behavior} new-ready",
        )
    observation_pose = old[OBSERVATION_STEP]
    ax_path.scatter(
        observation_pose[0],
        observation_pose[1],
        marker="*",
        s=120,
        color="gold",
        edgecolor="black",
        label="observation",
        zorder=6,
    )
    ax_path.set_title(
        f"Representative safety case (latency={REPRESENTATIVE_LATENCY})"
    )
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(fontsize=7)

    for behavior in INFERENCE_BEHAVIORS:
        group = metrics[metrics["inference_behavior"] == behavior]
        ax_clearance.plot(
            group["latency_seconds"],
            group["minimum_clearance_before_new_ready"],
            marker="o",
            color=COLORS[behavior],
            label=behavior,
        )
        collision_values = group["collision_before_new_ready"].astype(int)
        ax_collision.plot(
            group["latency_seconds"],
            collision_values,
            marker="o",
            color=COLORS[behavior],
            label=behavior,
        )
        ax_progress.plot(
            group["latency_seconds"],
            group["distance_travelled_during_inference"],
            marker="o",
            color=COLORS[behavior],
            label=behavior,
        )
    ax_clearance.axhline(0.0, color="black", linestyle=":", label="collision")
    ax_clearance.axhline(
        obstacle["margin"], color="tab:blue", linestyle=":", label="safety margin"
    )
    ax_clearance.set_title("Clearance before NEW is ready")
    ax_clearance.set_xlabel("Inference latency (s)")
    ax_clearance.set_ylabel("Minimum obstacle clearance")
    ax_clearance.grid(True, alpha=0.3)
    ax_clearance.legend(fontsize=8)
    ax_collision.set_title("Collision before optimizer can act")
    ax_collision.set_xlabel("Inference latency (s)")
    ax_collision.set_ylabel("Collision (0/1)")
    ax_collision.set_yticks([0, 1], ["safe", "collision"])
    ax_collision.set_ylim(-0.15, 1.15)
    ax_collision.grid(True, alpha=0.3)
    ax_collision.legend(fontsize=8)
    ax_progress.set_title("Progress during inference")
    ax_progress.set_xlabel("Inference latency (s)")
    ax_progress.set_ylabel("Distance travelled")
    ax_progress.grid(True, alpha=0.3)
    ax_progress.legend(fontsize=8)

    fig.suptitle("Experiment 08: Inference-Time Behavior")
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old, new, obstacle = make_async_obstacle_update_scenario(
        num_poses=NUM_POSES,
        observation_step=OBSERVATION_STEP,
        seed=SEED,
    )
    new_clearance = polyline_minimum_clearance(new, obstacle)
    if new_clearance < obstacle["margin"]:
        raise RuntimeError(
            f"NEW proposal violates safety margin: clearance={new_clearance:.6f}"
        )

    rows = []
    representative = {}
    for behavior in INFERENCE_BEHAVIORS:
        for latency_steps in LATENCY_STEPS:
            row, trajectory, prefix, config = run_case(
                old, new, obstacle, behavior, latency_steps
            )
            rows.append(row)
            if latency_steps == REPRESENTATIVE_LATENCY:
                representative[behavior] = (trajectory, prefix, config)

    metrics = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    figure = make_figure(metrics, representative, old, new, obstacle)
    figure_path = OUTPUT_DIR / "figure.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print("\n=== Experiment 08: Inference-Time Behavior ===")
    print(metrics.to_string(index=False))
    print(f"\nNEW proposal polyline clearance: {new_clearance:.6f}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from action_chunk_graph.baselines import (
    cubic_hermite_crossfade,
    linear_crossfade,
)
from action_chunk_graph.metrics import (
    body_motion_smoothness,
    mean_rotational_deviation,
    mean_translational_deviation,
    rotational_increment_rms,
    translational_jerk_rms,
)
from action_chunk_graph.optimizer import GraphConfig, optimize_reconciled_trajectory
from action_chunk_graph.scenarios import make_smoothness_stitch_scenario


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "exp02_smoothness"

NUM_POSES = 31
DT = 0.1
GRAPH_CONFIG = GraphConfig(
    lambda_old=2.0,
    lambda_new=2.0,
    lambda_smooth=25.0,
    lambda_collision=0.0,
    rotation_scale=0.5,
    max_nfev=120,
)


def timed_call(function, *args, **kwargs):
    start = perf_counter()
    value = function(*args, **kwargs)
    runtime_ms = 1000.0 * (perf_counter() - start)
    return value, runtime_ms


def boundary_velocity_mismatch(trajectory, old, new):
    start_velocity = (trajectory[1, :2] - trajectory[0, :2]) / DT
    old_start_velocity = (old[1, :2] - old[0, :2]) / DT
    end_velocity = (trajectory[-1, :2] - trajectory[-2, :2]) / DT
    new_end_velocity = (new[-1, :2] - new[-2, :2]) / DT
    return (
        float(np.linalg.norm(start_velocity - old_start_velocity)),
        float(np.linalg.norm(end_velocity - new_end_velocity)),
    )


def evaluate(
    name,
    trajectory,
    old,
    new,
    runtime_ms,
    optimizer_result=None,
):
    start_mismatch, end_mismatch = boundary_velocity_mismatch(
        trajectory, old, new
    )
    row = {
        "method": name,
        "translational_jerk_rms": translational_jerk_rms(trajectory, dt=DT),
        "rotation_step_rms": rotational_increment_rms(trajectory),
        "body_motion_smoothness": body_motion_smoothness(trajectory),
        "mean_position_deviation_old": mean_translational_deviation(
            trajectory, old
        ),
        "mean_position_deviation_new": mean_translational_deviation(
            trajectory, new
        ),
        "mean_rotation_deviation_old": mean_rotational_deviation(
            trajectory, old
        ),
        "mean_rotation_deviation_new": mean_rotational_deviation(
            trajectory, new
        ),
        "start_velocity_mismatch": start_mismatch,
        "end_velocity_mismatch": end_mismatch,
        "generation_runtime_ms": runtime_ms,
        "optimizer_success": None,
        "optimizer_cost": np.nan,
        "optimizer_nfev": np.nan,
    }
    if optimizer_result is not None:
        row.update(
            {
                "optimizer_success": bool(optimizer_result.success),
                "optimizer_cost": float(optimizer_result.cost),
                "optimizer_nfev": int(optimizer_result.nfev),
            }
        )
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old, new = make_smoothness_stitch_scenario(num_poses=NUM_POSES)

    linear, linear_runtime = timed_call(linear_crossfade, old, new)
    spline, spline_runtime = timed_call(cubic_hermite_crossfade, old, new)
    graph_output, graph_runtime = timed_call(
        optimize_reconciled_trajectory,
        old,
        new,
        None,
        GRAPH_CONFIG,
    )
    graph, optimizer_result = graph_output

    if not optimizer_result.success:
        raise RuntimeError(f"Graph optimizer failed: {optimizer_result.message}")
    for name, trajectory in (
        ("Linear crossfade", linear),
        ("Cubic Hermite", spline),
        ("SE(2) graph", graph),
    ):
        if not np.all(np.isfinite(trajectory)):
            raise RuntimeError(f"{name} produced non-finite trajectory values.")

    rows = [
        evaluate("Linear crossfade", linear, old, new, linear_runtime),
        evaluate("Cubic Hermite", spline, old, new, spline_runtime),
        evaluate(
            "SE(2) graph",
            graph,
            old,
            new,
            graph_runtime,
            optimizer_result,
        ),
    ]
    metrics = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    fig, (ax_path, ax_metrics) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax_path.plot(old[:, 0], old[:, 1], "--", label="OLD proposal")
    ax_path.plot(new[:, 0], new[:, 1], "--", label="NEW proposal")
    ax_path.plot(linear[:, 0], linear[:, 1], label="Linear crossfade")
    ax_path.plot(spline[:, 0], spline[:, 1], label="Cubic Hermite")
    ax_path.plot(
        graph[:, 0],
        graph[:, 1],
        linewidth=2.5,
        label="SE(2) graph",
    )
    ax_path.set_title("Reconciled trajectories")
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(loc="lower right")

    metric_columns = [
        "translational_jerk_rms",
        "rotation_step_rms",
        "body_motion_smoothness",
    ]
    metric_labels = ["Trans. jerk", "Rotation step", "Body motion"]
    normalized = metrics[metric_columns].to_numpy(dtype=float, copy=True)
    normalized /= normalized[0]

    x = np.arange(len(metric_columns))
    width = 0.24
    for index, method in enumerate(metrics["method"]):
        bars = ax_metrics.bar(
            x + (index - 1) * width,
            normalized[index],
            width,
            label=method,
        )
        ax_metrics.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

    ax_metrics.axhline(1.0, color="black", linestyle=":", alpha=0.5)
    ax_metrics.set_xticks(x, metric_labels)
    ax_metrics.set_ylabel("Metric ratio (Linear crossfade = 1.0)")
    ax_metrics.set_title("Normalized smoothness metrics (lower is better)")
    ax_metrics.grid(True, axis="y", alpha=0.3)
    ax_metrics.legend()

    fig.suptitle("Experiment 02: Smoothness Baseline Comparison")
    fig.tight_layout()
    figure_path = OUTPUT_DIR / "figure.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print("\n=== Experiment 02: Smoothness Comparison ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

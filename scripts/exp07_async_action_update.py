from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from action_chunk_graph.baselines import cubic_hermite_crossfade
from action_chunk_graph.execution import (
    ExecutionConfig,
    aligned_new_reference,
    assemble_continue_old,
    assemble_hard_switch,
    assemble_local_transition,
    build_committed_prefix,
    local_transition_windows,
)
from action_chunk_graph.geometry import wrap_angle
from action_chunk_graph.metrics import (
    body_motion_smoothness,
    mean_rotational_deviation,
    mean_translational_deviation,
    rotational_increment_rms,
    translational_jerk_rms,
)
from action_chunk_graph.optimizer import GraphConfig, optimize_reconciled_trajectory
from action_chunk_graph.scenarios import make_async_goal_change_scenario


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "exp07_async_action_update"

DT = 0.1
CONTROL_FREQUENCY_HZ = 10.0
NUM_POSES = 31
OBSERVATION_STEP = 8
COMMIT_HORIZON_STEPS = 1
OPTIMIZATION_WINDOW_POSES = 7
LATENCY_STEPS = [0, 2, 4, 6]
SEED = 42
REPRESENTATIVE_LATENCY = 4
PREFIX_TOLERANCE = 1e-10

METHOD_ORDER = [
    "Continue OLD",
    "Hard switch",
    "Local cubic Hermite",
    "Local SE(2) graph",
]
COLORS = {
    "Continue OLD": "tab:blue",
    "Hard switch": "tab:red",
    "Local cubic Hermite": "tab:green",
    "Local SE(2) graph": "tab:purple",
}
GRAPH_CONFIG = GraphConfig(
    lambda_old=2.0,
    lambda_new=2.0,
    lambda_smooth=25.0,
    lambda_collision=0.0,
    rotation_scale=0.5,
    max_nfev=160,
    lambda_terminal_new=1200.0,
)


def timed_call(function, *args):
    start = perf_counter()
    value = function(*args)
    return value, 1000.0 * (perf_counter() - start)


def _position_velocity(trajectory, start, stop):
    return (trajectory[stop, :2] - trajectory[start, :2]) / DT


def boundary_metrics(trajectory, modification_step, window_poses, anchor):
    end_step = modification_step + window_poses - 1
    incoming = _position_velocity(
        trajectory, modification_step - 1, modification_step
    )
    outgoing = _position_velocity(
        trajectory, modification_step, modification_step + 1
    )
    end_incoming = _position_velocity(trajectory, end_step - 1, end_step)
    end_outgoing = _position_velocity(trajectory, end_step, end_step + 1)
    return {
        "transition_start_position_jump": float(
            np.linalg.norm(trajectory[modification_step, :2] - anchor[:2])
        ),
        "transition_start_rotation_jump": float(
            abs(wrap_angle(trajectory[modification_step, 2] - anchor[2]))
        ),
        "transition_start_velocity_mismatch": float(
            np.linalg.norm(outgoing - incoming)
        ),
        "transition_end_velocity_mismatch": float(
            np.linalg.norm(end_outgoing - end_incoming)
        ),
    }


def evaluate(method, trajectory, old, new, config, runtime_ms, result=None):
    modification_step = config.modification_step
    committed = build_committed_prefix(old, config)
    immutable_stop = modification_step
    prefix_position_error = np.linalg.norm(
        trajectory[:immutable_stop, :2] - committed[:immutable_stop, :2],
        axis=1,
    )
    prefix_rotation_error = np.abs(
        wrap_angle(
            trajectory[:immutable_stop, 2] - committed[:immutable_stop, 2]
        )
    )
    aligned_start = modification_step - config.observation_step
    executed_after_modify = trajectory[modification_step:]
    new_after_modify = new[aligned_start:]
    if executed_after_modify.shape != new_after_modify.shape:
        raise RuntimeError("Executed trajectory and aligned NEW suffix differ.")

    row = {
        "method": method,
        "latency_steps": config.inference_latency_steps,
        "latency_seconds": config.inference_latency_steps * config.dt,
        "observation_step": config.observation_step,
        "new_ready_step": config.new_ready_step,
        "modification_step": modification_step,
        "commit_horizon_steps": config.commit_horizon_steps,
        "optimization_window_poses": config.optimization_window_poses,
        "committed_prefix_max_position_error": float(
            np.max(prefix_position_error, initial=0.0)
        ),
        "committed_prefix_max_rotation_error": float(
            np.max(prefix_rotation_error, initial=0.0)
        ),
        **boundary_metrics(
            trajectory,
            modification_step,
            config.optimization_window_poses,
            committed[-1],
        ),
        "translational_jerk_rms": translational_jerk_rms(
            trajectory, dt=config.dt
        ),
        "rotational_increment_rms": rotational_increment_rms(trajectory),
        "body_motion_smoothness": body_motion_smoothness(trajectory),
        "mean_new_position_deviation_after_modify": (
            mean_translational_deviation(executed_after_modify, new_after_modify)
        ),
        "mean_new_rotation_deviation_after_modify": (
            mean_rotational_deviation(executed_after_modify, new_after_modify)
        ),
        "final_position_error_to_new": float(
            np.linalg.norm(trajectory[-1, :2] - new[-1, :2])
        ),
        "final_rotation_error_to_new": float(
            abs(wrap_angle(trajectory[-1, 2] - new[-1, 2]))
        ),
        "generation_runtime_ms": runtime_ms,
        "optimizer_success": None,
        "optimizer_cost": np.nan,
        "optimizer_nfev": np.nan,
    }
    if result is not None:
        row.update(
            {
                "optimizer_success": bool(result.success),
                "optimizer_cost": float(result.cost),
                "optimizer_nfev": int(result.nfev),
            }
        )
    return row


def run_latency(old, new, latency_steps):
    config = ExecutionConfig(
        dt=DT,
        observation_step=OBSERVATION_STEP,
        inference_latency_steps=latency_steps,
        commit_horizon_steps=COMMIT_HORIZON_STEPS,
        optimization_window_poses=OPTIMIZATION_WINDOW_POSES,
        inference_behavior="continue_old",
    )
    (continue_old, continue_runtime) = timed_call(
        assemble_continue_old, old, new, config
    )
    (hard_switch, hard_runtime) = timed_call(
        assemble_hard_switch, old, new, config
    )
    old_window, new_window = local_transition_windows(old, new, config)
    (hermite_window, hermite_runtime) = timed_call(
        cubic_hermite_crossfade, old_window, new_window
    )
    hermite = assemble_local_transition(old, new, hermite_window, config)
    (graph_output, graph_runtime) = timed_call(
        optimize_reconciled_trajectory,
        old_window,
        new_window,
        None,
        GRAPH_CONFIG,
    )
    graph_window, graph_result = graph_output
    graph = assemble_local_transition(old, new, graph_window, config)
    if not graph_result.success:
        raise RuntimeError(
            f"Graph optimizer failed at latency {latency_steps}: "
            f"{graph_result.message}"
        )

    methods = {
        "Continue OLD": (continue_old, continue_runtime, None),
        "Hard switch": (hard_switch, hard_runtime, None),
        "Local cubic Hermite": (hermite, hermite_runtime, None),
        "Local SE(2) graph": (graph, graph_runtime, graph_result),
    }
    rows = []
    for method, (trajectory, runtime_ms, result) in methods.items():
        if not np.all(np.isfinite(trajectory)):
            raise RuntimeError(f"{method} produced non-finite values.")
        row = evaluate(
            method, trajectory, old, new, config, runtime_ms, result
        )
        if (
            row["committed_prefix_max_position_error"] > PREFIX_TOLERANCE
            or row["committed_prefix_max_rotation_error"] > PREFIX_TOLERANCE
        ):
            raise RuntimeError(f"{method} modified the committed prefix.")
        rows.append(row)
    return rows, methods, config


def make_figure(metrics, representative, old, new):
    methods, config = representative
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_path, ax_jerk, ax_tracking, ax_timeline = axes.flat
    new_global = aligned_new_reference(new, config)

    ax_path.plot(old[:, 0], old[:, 1], "--", color="black", label="OLD")
    ax_path.plot(
        new_global[:, 0], new_global[:, 1], "--", color="tab:orange", label="NEW"
    )
    for method in METHOD_ORDER:
        trajectory = methods[method][0]
        ax_path.plot(
            trajectory[:, 0], trajectory[:, 1], color=COLORS[method], label=method
        )
    modification = config.modification_step
    graph = methods["Local SE(2) graph"][0]
    window_stop = modification + config.optimization_window_poses
    ax_path.scatter(
        graph[modification, 0], graph[modification, 1], marker="o", color="black",
        zorder=5, label="Modification point"
    )
    ax_path.plot(
        graph[modification:window_stop, 0],
        graph[modification:window_stop, 1],
        linewidth=5,
        alpha=0.25,
        color=COLORS["Local SE(2) graph"],
        label="Optimization window",
    )
    ax_path.set_title(f"Representative trajectories (latency={REPRESENTATIVE_LATENCY})")
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(fontsize=8)

    for method in METHOD_ORDER:
        group = metrics[metrics["method"] == method]
        ax_jerk.plot(
            group["latency_seconds"],
            group["translational_jerk_rms"],
            marker="o",
            color=COLORS[method],
            label=method,
        )
        ax_tracking.plot(
            group["latency_seconds"],
            group["mean_new_position_deviation_after_modify"],
            marker="o",
            color=COLORS[method],
            label=f"{method} mean",
        )
        ax_tracking.plot(
            group["latency_seconds"],
            group["final_position_error_to_new"],
            marker="x",
            linestyle=":",
            color=COLORS[method],
            label=f"{method} final",
        )
    ax_jerk.set_title("Latency vs translational jerk")
    ax_jerk.set_xlabel("Inference latency (s)")
    ax_jerk.set_ylabel("Jerk RMS")
    ax_jerk.set_yscale("log")
    ax_jerk.grid(True, alpha=0.3)
    ax_jerk.legend(fontsize=8)
    ax_tracking.set_title("Latency vs NEW mean/final tracking error")
    ax_tracking.set_xlabel("Inference latency (s)")
    ax_tracking.set_ylabel("Position error")
    ax_tracking.grid(True, alpha=0.3)
    ax_tracking.legend(fontsize=7, ncols=2)

    ax_timeline.axvspan(0, config.observation_step, alpha=0.15, label="Executed")
    ax_timeline.axvspan(
        config.observation_step,
        config.new_ready_step,
        alpha=0.15,
        color="tab:orange",
        label="Inference",
    )
    ax_timeline.axvspan(
        config.new_ready_step,
        config.modification_step,
        alpha=0.15,
        color="tab:red",
        label="Committed",
    )
    ax_timeline.axvspan(
        config.modification_step,
        config.modification_step + config.optimization_window_poses - 1,
        alpha=0.18,
        color="tab:purple",
        label="Local optimization",
    )
    for step, label in (
        (config.observation_step, "observation"),
        (config.new_ready_step, "new ready"),
        (config.modification_step, "modify"),
    ):
        ax_timeline.axvline(step, color="black", linestyle=":")
        ax_timeline.text(step + 0.2, 0.55, label, rotation=90, va="center")
    ax_timeline.set_xlim(0, len(methods["Local SE(2) graph"][0]) - 1)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.set_xlabel("Global execution step")
    ax_timeline.set_title("Explicit execution timeline")
    ax_timeline.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Experiment 07: Async Action-Chunk Update "
        f"({CONTROL_FREQUENCY_HZ:.0f} Hz control)"
    )
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old, new = make_async_goal_change_scenario(
        num_poses=NUM_POSES,
        observation_step=OBSERVATION_STEP,
        seed=SEED,
    )
    rows = []
    representative = None
    for latency_steps in LATENCY_STEPS:
        latency_rows, methods, config = run_latency(old, new, latency_steps)
        rows.extend(latency_rows)
        if latency_steps == REPRESENTATIVE_LATENCY:
            representative = (methods, config)

    metrics = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    figure = make_figure(metrics, representative, old, new)
    figure_path = OUTPUT_DIR / "figure.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print("\n=== Experiment 07: Async Action-Chunk Update ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

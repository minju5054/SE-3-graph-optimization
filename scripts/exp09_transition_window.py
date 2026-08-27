from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from action_chunk_graph.baselines import cubic_hermite_crossfade
from action_chunk_graph.execution import (
    ExecutionConfig,
    aligned_new_reference,
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
    steps_to_aligned_position_tolerance,
    transition_velocity_mismatches,
    translational_jerk_rms,
)
from action_chunk_graph.optimizer import GraphConfig, optimize_reconciled_trajectory
from action_chunk_graph.scenarios import make_async_goal_change_scenario


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "exp09_transition_window"

DT = 0.1
NUM_POSES = 31
OBSERVATION_STEP = 8
INFERENCE_LATENCY_STEPS = 4
COMMIT_HORIZON_STEPS = 1
TRANSITION_WINDOW_POSES = [3, 5, 7, 10, 15]
NEW_POSITION_TOLERANCE = 0.05
SEED = 42
PREFIX_TOLERANCE = 1e-10

METHOD_ORDER = ["Hard switch", "Local cubic Hermite", "Local SE(2) graph"]
WINDOW_METHODS = ["Local cubic Hermite", "Local SE(2) graph"]
COLORS = {
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
    max_nfev=200,
    lambda_terminal_new=1200.0,
)


def timed_call(function, *args):
    start = perf_counter()
    value = function(*args)
    return value, 1000.0 * (perf_counter() - start)


def evaluate(
    method,
    trajectory,
    old,
    new,
    config,
    window_poses,
    window_applies,
    runtime_ms,
    optimizer_result=None,
):
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
    start_new = modification_step - config.observation_step
    executed_after_modify = trajectory[modification_step:]
    new_after_modify = new[start_new:]
    metric_window = window_poses if window_applies else 1
    start_velocity, end_velocity = transition_velocity_mismatches(
        trajectory,
        modification_step,
        metric_window,
        dt=config.dt,
    )
    reaction_steps = steps_to_aligned_position_tolerance(
        trajectory,
        new,
        config.observation_step,
        modification_step,
        NEW_POSITION_TOLERANCE,
    )
    row = {
        "method": method,
        "transition_window_applies": window_applies,
        "transition_window_poses": window_poses,
        "transition_window_seconds": (
            (window_poses - 1) * config.dt if window_applies else 0.0
        ),
        "new_position_tolerance": NEW_POSITION_TOLERANCE,
        "observation_step": config.observation_step,
        "new_ready_step": config.new_ready_step,
        "modification_step": modification_step,
        "transition_start_position_jump": float(
            np.linalg.norm(
                trajectory[modification_step, :2] - committed[-1, :2]
            )
        ),
        "transition_start_rotation_jump": float(
            abs(
                wrap_angle(
                    trajectory[modification_step, 2] - committed[-1, 2]
                )
            )
        ),
        "transition_start_velocity_mismatch": start_velocity,
        "transition_end_velocity_mismatch": end_velocity,
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
        "steps_to_reach_new_tolerance": (
            reaction_steps if reaction_steps is not None else np.nan
        ),
        "time_or_steps_to_reach_new_tolerance": (
            reaction_steps if reaction_steps is not None else np.nan
        ),
        "reaction_metric_unit": "steps",
        "time_to_reach_new_tolerance_seconds": (
            reaction_steps * config.dt if reaction_steps is not None else np.nan
        ),
        "generation_runtime_ms": runtime_ms,
        "committed_prefix_max_position_error": float(
            np.max(prefix_position_error, initial=0.0)
        ),
        "committed_prefix_max_rotation_error": float(
            np.max(prefix_rotation_error, initial=0.0)
        ),
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
    if (
        row["committed_prefix_max_position_error"] > PREFIX_TOLERANCE
        or row["committed_prefix_max_rotation_error"] > PREFIX_TOLERANCE
    ):
        raise RuntimeError(f"{method} modified the committed prefix.")
    return row


def make_config(window_poses):
    return ExecutionConfig(
        dt=DT,
        observation_step=OBSERVATION_STEP,
        inference_latency_steps=INFERENCE_LATENCY_STEPS,
        commit_horizon_steps=COMMIT_HORIZON_STEPS,
        optimization_window_poses=window_poses,
        inference_behavior="continue_old",
    )


def run_experiment(old, new):
    rows = []
    trajectories = {}
    reference_config = make_config(TRANSITION_WINDOW_POSES[0])
    hard_switch, hard_runtime = timed_call(
        assemble_hard_switch, old, new, reference_config
    )
    rows.append(
        evaluate(
            "Hard switch",
            hard_switch,
            old,
            new,
            reference_config,
            window_poses=0,
            window_applies=False,
            runtime_ms=hard_runtime,
        )
    )
    trajectories[("Hard switch", 0)] = hard_switch

    for window_poses in TRANSITION_WINDOW_POSES:
        config = make_config(window_poses)
        old_window, new_window = local_transition_windows(old, new, config)
        if len(old_window) != window_poses or len(new_window) != window_poses:
            raise RuntimeError("Transition window was silently clipped.")

        hermite_window, hermite_runtime = timed_call(
            cubic_hermite_crossfade, old_window, new_window
        )
        hermite = assemble_local_transition(
            old, new, hermite_window, config
        )
        graph_output, graph_runtime = timed_call(
            optimize_reconciled_trajectory,
            old_window,
            new_window,
            None,
            GRAPH_CONFIG,
        )
        graph_window, graph_result = graph_output
        if not graph_result.success:
            raise RuntimeError(
                f"Graph failed for window {window_poses}: {graph_result.message}"
            )
        graph = assemble_local_transition(old, new, graph_window, config)

        for method, trajectory, runtime_ms, result in (
            (
                "Local cubic Hermite",
                hermite,
                hermite_runtime,
                None,
            ),
            ("Local SE(2) graph", graph, graph_runtime, graph_result),
        ):
            rows.append(
                evaluate(
                    method,
                    trajectory,
                    old,
                    new,
                    config,
                    window_poses,
                    True,
                    runtime_ms,
                    result,
                )
            )
            trajectories[(method, window_poses)] = trajectory
    return pd.DataFrame(rows), trajectories, reference_config


def make_figure(metrics, trajectories, old, new, config):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_path, ax_jerk, ax_reaction, ax_runtime = axes.flat
    aligned_new = aligned_new_reference(new, config)
    ax_path.plot(old[:, 0], old[:, 1], "--", color="black", label="OLD")
    ax_path.plot(
        aligned_new[:, 0],
        aligned_new[:, 1],
        "--",
        color="tab:orange",
        label="NEW",
    )
    for window_poses, linestyle in ((3, ":"), (7, "-."), (15, "-")):
        for method in WINDOW_METHODS:
            trajectory = trajectories[(method, window_poses)]
            ax_path.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=COLORS[method],
                linestyle=linestyle,
                label=f"{method}, W={window_poses}",
            )
    ax_path.scatter(
        old[config.modification_step, 0],
        old[config.modification_step, 1],
        color="black",
        zorder=5,
        label="Modification point",
    )
    ax_path.set_title("Representative local transitions")
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(fontsize=7)

    hard_row = metrics[metrics["method"] == "Hard switch"].iloc[0]
    window_metrics = metrics[metrics["transition_window_applies"]]
    for method in WINDOW_METHODS:
        group = window_metrics[window_metrics["method"] == method]
        ax_jerk.plot(
            group["transition_window_poses"],
            group["translational_jerk_rms"],
            marker="o",
            color=COLORS[method],
            label=method,
        )
        ax_reaction.plot(
            group["transition_window_poses"],
            group["steps_to_reach_new_tolerance"],
            marker="o",
            color=COLORS[method],
            label=f"{method} reaction",
        )
        ax_runtime.plot(
            group["transition_window_poses"],
            group["generation_runtime_ms"],
            marker="o",
            color=COLORS[method],
            label=method,
        )
    ax_jerk.axhline(
        hard_row["translational_jerk_rms"],
        color=COLORS["Hard switch"],
        linestyle=":",
        label="Hard switch reference",
    )
    ax_jerk.set_title("Window size vs jerk")
    ax_jerk.set_xlabel("Transition window poses")
    ax_jerk.set_ylabel("Translational jerk RMS")
    ax_jerk.grid(True, alpha=0.3)
    ax_jerk.legend(fontsize=8)

    tracking_axis = ax_reaction.twinx()
    for method in WINDOW_METHODS:
        group = window_metrics[window_metrics["method"] == method]
        tracking_axis.plot(
            group["transition_window_poses"],
            group["mean_new_position_deviation_after_modify"],
            marker="x",
            linestyle=":",
            color=COLORS[method],
            label=f"{method} tracking",
        )
    ax_reaction.set_title("NEW reaction and tracking")
    ax_reaction.set_xlabel("Transition window poses")
    ax_reaction.set_ylabel("Steps to 0.05 m tolerance")
    tracking_axis.set_ylabel("Mean NEW position deviation")
    ax_reaction.grid(True, alpha=0.3)
    reaction_handles, reaction_labels = ax_reaction.get_legend_handles_labels()
    tracking_handles, tracking_labels = tracking_axis.get_legend_handles_labels()
    ax_reaction.legend(
        reaction_handles + tracking_handles,
        reaction_labels + tracking_labels,
        fontsize=7,
    )

    ax_runtime.axhline(
        hard_row["generation_runtime_ms"],
        color=COLORS["Hard switch"],
        linestyle=":",
        label="Hard switch reference",
    )
    ax_runtime.set_title("Window size vs generation runtime")
    ax_runtime.set_xlabel("Transition window poses")
    ax_runtime.set_ylabel("Runtime (ms)")
    ax_runtime.set_yscale("log")
    ax_runtime.grid(True, alpha=0.3)
    ax_runtime.legend(fontsize=8)

    fig.suptitle("Experiment 09: Transition Window Ablation")
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old, new = make_async_goal_change_scenario(
        num_poses=NUM_POSES,
        observation_step=OBSERVATION_STEP,
        seed=SEED,
    )
    metrics, trajectories, config = run_experiment(old, new)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    figure = make_figure(metrics, trajectories, old, new, config)
    figure_path = OUTPUT_DIR / "figure.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print("\n=== Experiment 09: Transition Window Ablation ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

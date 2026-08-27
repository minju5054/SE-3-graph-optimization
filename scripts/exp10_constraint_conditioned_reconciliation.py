from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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
    mean_translational_deviation,
    polyline_collision,
    polyline_minimum_clearance,
    polyline_safety_margin_violation,
    rotational_increment_rms,
    transition_velocity_mismatches,
    translational_jerk_rms,
)
from action_chunk_graph.optimizer import GraphConfig, optimize_reconciled_trajectory
from action_chunk_graph.scenarios import (
    make_async_goal_change_scenario,
    make_constraint_severity_scenario_suite,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "exp10_constraint_conditioned"

DT = 0.1
NUM_POSES = 31
OBSERVATION_STEP = 8
INFERENCE_LATENCY_STEPS = 2
COMMIT_HORIZON_STEPS = 1
TRANSITION_WINDOW_POSES = 10
SEED = 42
PREFIX_TOLERANCE = 1e-10

METHOD_ORDER = [
    "Hard switch",
    "Local cubic Hermite",
    "Local graph without collision",
    "Local graph with segment collision",
]
COLORS = {
    "Hard switch": "tab:red",
    "Local cubic Hermite": "tab:green",
    "Local graph without collision": "tab:purple",
    "Local graph with segment collision": "tab:brown",
}
SEVERITY_ORDER = ["low", "medium_low", "medium", "medium_high", "high"]
SEVERITY_RANK = {name: index + 1 for index, name in enumerate(SEVERITY_ORDER)}

GRAPH_WITHOUT_COLLISION = GraphConfig(
    lambda_old=2.0,
    lambda_new=2.0,
    lambda_smooth=25.0,
    lambda_collision=0.0,
    rotation_scale=0.5,
    max_nfev=300,
    collision_factor="segments",
    lambda_terminal_new=1200.0,
)
GRAPH_WITH_COLLISION = GraphConfig(
    lambda_old=2.0,
    lambda_new=2.0,
    lambda_smooth=25.0,
    lambda_collision=3000.0,
    rotation_scale=0.5,
    max_nfev=600,
    collision_factor="segments",
    lambda_terminal_new=1200.0,
)


def make_config():
    return ExecutionConfig(
        dt=DT,
        observation_step=OBSERVATION_STEP,
        inference_latency_steps=INFERENCE_LATENCY_STEPS,
        commit_horizon_steps=COMMIT_HORIZON_STEPS,
        optimization_window_poses=TRANSITION_WINDOW_POSES,
        inference_behavior="continue_old",
    )


def timed_call(function, *args):
    start = perf_counter()
    value = function(*args)
    return value, 1000.0 * (perf_counter() - start)


def build_unconstrained_methods(old, new, config):
    hard, hard_runtime = timed_call(assemble_hard_switch, old, new, config)
    old_window, new_window = local_transition_windows(old, new, config)
    hermite_window, hermite_runtime = timed_call(
        cubic_hermite_crossfade, old_window, new_window
    )
    hermite = assemble_local_transition(old, new, hermite_window, config)
    graph_output, graph_runtime = timed_call(
        optimize_reconciled_trajectory,
        old_window,
        new_window,
        None,
        GRAPH_WITHOUT_COLLISION,
    )
    graph_window, graph_result = graph_output
    if not graph_result.success:
        raise RuntimeError(f"Unconstrained graph failed: {graph_result.message}")
    graph = assemble_local_transition(old, new, graph_window, config)
    return {
        "Hard switch": (hard, hard_runtime, None),
        "Local cubic Hermite": (hermite, hermite_runtime, None),
        "Local graph without collision": (
            graph,
            graph_runtime,
            graph_result,
        ),
    }, old_window, new_window


def evaluate(
    regime,
    severity,
    severity_rank,
    method,
    trajectory,
    old,
    new,
    config,
    obstacle,
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
    metric_window = 1 if method == "Hard switch" else config.optimization_window_poses
    start_velocity, end_velocity = transition_velocity_mismatches(
        trajectory,
        modification_step,
        metric_window,
        dt=config.dt,
    )
    start_new = modification_step - config.observation_step
    executed_after_modify = trajectory[modification_step:]
    new_after_modify = new[start_new:]

    if obstacle is None:
        collision_value = None
        clearance = np.nan
        margin_violation = np.nan
        prefix_collision = None
        prefix_clearance = np.nan
        new_clearance = np.nan
        obstacle_radius = np.nan
        obstacle_margin = np.nan
    else:
        collision_value = polyline_collision(trajectory, obstacle)
        clearance = polyline_minimum_clearance(trajectory, obstacle)
        margin_violation = polyline_safety_margin_violation(
            trajectory, obstacle
        )
        prefix_collision = polyline_collision(committed, obstacle)
        prefix_clearance = polyline_minimum_clearance(committed, obstacle)
        new_clearance = polyline_minimum_clearance(new, obstacle)
        obstacle_radius = obstacle["radius"]
        obstacle_margin = obstacle["margin"]

    row = {
        "regime": regime,
        "constraint_severity": severity,
        "constraint_severity_rank": severity_rank,
        "obstacle_radius": obstacle_radius,
        "obstacle_margin": obstacle_margin,
        "method": method,
        "observation_step": config.observation_step,
        "new_ready_step": config.new_ready_step,
        "modification_step": modification_step,
        "transition_window_poses": config.optimization_window_poses,
        "transition_window_seconds": (
            (config.optimization_window_poses - 1) * config.dt
        ),
        "prefix_collision": prefix_collision,
        "prefix_polyline_minimum_clearance": prefix_clearance,
        "new_proposal_polyline_minimum_clearance": new_clearance,
        "polyline_collision": collision_value,
        "polyline_minimum_clearance": clearance,
        "polyline_safety_margin_violation": margin_violation,
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
        "final_position_error_to_new": float(
            np.linalg.norm(trajectory[-1, :2] - new[-1, :2])
        ),
        "generation_runtime_ms": runtime_ms,
        "committed_prefix_max_position_error": float(
            np.max(prefix_position_error, initial=0.0)
        ),
        "committed_prefix_max_rotation_error": float(
            np.max(prefix_rotation_error, initial=0.0)
        ),
        "optimizer_success": None,
        "optimizer_nfev": np.nan,
        "optimizer_cost": np.nan,
    }
    if optimizer_result is not None:
        row.update(
            {
                "optimizer_success": bool(optimizer_result.success),
                "optimizer_nfev": int(optimizer_result.nfev),
                "optimizer_cost": float(optimizer_result.cost),
            }
        )
    if (
        row["committed_prefix_max_position_error"] > PREFIX_TOLERANCE
        or row["committed_prefix_max_rotation_error"] > PREFIX_TOLERANCE
    ):
        raise RuntimeError(f"{regime}/{severity}/{method} changed the prefix.")
    if obstacle is not None:
        if prefix_collision or prefix_clearance <= obstacle["margin"]:
            raise RuntimeError(f"Unsafe committed prefix for severity {severity}.")
        if new_clearance <= obstacle["margin"]:
            raise RuntimeError(f"Unsafe NEW proposal for severity {severity}.")
    return row


def run_experiment():
    config = make_config()
    rows = []
    representatives = {}

    benign_old, benign_new = make_async_goal_change_scenario(
        num_poses=NUM_POSES,
        observation_step=OBSERVATION_STEP,
        seed=SEED,
    )
    benign_methods, _, _ = build_unconstrained_methods(
        benign_old, benign_new, config
    )
    for method, (trajectory, runtime_ms, result) in benign_methods.items():
        rows.append(
            evaluate(
                "benign",
                "none",
                0,
                method,
                trajectory,
                benign_old,
                benign_new,
                config,
                None,
                runtime_ms,
                result,
            )
        )
    representatives["benign"] = {
        "old": benign_old,
        "new": benign_new,
        "trajectories": {
            method: values[0] for method, values in benign_methods.items()
        },
    }

    severity_scenarios = make_constraint_severity_scenario_suite(
        num_poses=NUM_POSES,
        observation_step=OBSERVATION_STEP,
        seed=SEED,
    )
    constrained_old = severity_scenarios[0]["old"]
    constrained_new = severity_scenarios[0]["new"]
    constrained_methods, old_window, new_window = build_unconstrained_methods(
        constrained_old, constrained_new, config
    )
    for case in severity_scenarios:
        severity = case["constraint_severity"]
        obstacle = case["obstacle"]
        methods = dict(constrained_methods)
        collision_output, collision_runtime = timed_call(
            optimize_reconciled_trajectory,
            old_window,
            new_window,
            obstacle,
            GRAPH_WITH_COLLISION,
        )
        collision_window, collision_result = collision_output
        if not collision_result.success:
            raise RuntimeError(
                f"Collision graph failed for {severity}: "
                f"{collision_result.message}"
            )
        collision_graph = assemble_local_transition(
            constrained_old,
            constrained_new,
            collision_window,
            config,
        )
        methods["Local graph with segment collision"] = (
            collision_graph,
            collision_runtime,
            collision_result,
        )
        for method in METHOD_ORDER:
            trajectory, runtime_ms, result = methods[method]
            rows.append(
                evaluate(
                    "constrained",
                    severity,
                    SEVERITY_RANK[severity],
                    method,
                    trajectory,
                    constrained_old,
                    constrained_new,
                    config,
                    obstacle,
                    runtime_ms,
                    result,
                )
            )
        if severity == "high":
            representatives["constrained"] = {
                "old": constrained_old,
                "new": constrained_new,
                "obstacle": obstacle,
                "trajectories": {
                    method: values[0] for method, values in methods.items()
                },
            }
    return pd.DataFrame(rows), representatives, config


def plot_trajectories(axis, representative, config, title, obstacle=None):
    old = representative["old"]
    new = representative["new"]
    aligned_new = aligned_new_reference(new, config)
    axis.plot(old[:, 0], old[:, 1], "--", color="black", label="OLD")
    axis.plot(
        aligned_new[:, 0],
        aligned_new[:, 1],
        "--",
        color="tab:orange",
        label="NEW",
    )
    for method, trajectory in representative["trajectories"].items():
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=COLORS[method],
            linewidth=(
                2.7 if method == "Local graph with segment collision" else 1.6
            ),
            label=method,
        )
    if obstacle is not None:
        center = obstacle["center"]
        axis.add_patch(
            Circle(center, obstacle["radius"], color="tab:blue", alpha=0.25)
        )
        axis.add_patch(
            Circle(
                center,
                obstacle["radius"] + obstacle["margin"],
                fill=False,
                linestyle=":",
                color="black",
            )
        )
    axis.scatter(
        old[config.modification_step, 0],
        old[config.modification_step, 1],
        color="black",
        zorder=5,
        label="Modification point",
    )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=6)


def make_figure(metrics, representatives, config):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    (
        ax_benign,
        ax_constrained,
        ax_clearance,
        ax_collision,
        ax_jerk,
        ax_runtime,
    ) = axes.flat
    plot_trajectories(
        ax_benign,
        representatives["benign"],
        config,
        "Benign regime",
    )
    constrained_rep = representatives["constrained"]
    plot_trajectories(
        ax_constrained,
        constrained_rep,
        config,
        "Constrained regime: high severity",
        constrained_rep["obstacle"],
    )

    constrained = metrics[metrics["regime"] == "constrained"]
    for method in METHOD_ORDER:
        group = constrained[constrained["method"] == method]
        ax_clearance.plot(
            group["constraint_severity_rank"],
            group["polyline_minimum_clearance"],
            marker="o",
            color=COLORS[method],
            label=method,
        )
        ax_collision.plot(
            group["constraint_severity_rank"],
            group["polyline_collision"].astype(int),
            marker="o",
            color=COLORS[method],
            label=method,
        )
        ax_jerk.plot(
            group["constraint_severity_rank"],
            group["translational_jerk_rms"],
            marker="o",
            color=COLORS[method],
            label=method,
        )
    for axis in (ax_clearance, ax_collision, ax_jerk):
        axis.set_xticks(range(1, 6), SEVERITY_ORDER, rotation=20)
        axis.grid(True, alpha=0.3)
    ax_clearance.axhline(0.0, color="black", linestyle=":", label="collision")
    ax_clearance.axhline(
        0.12, color="tab:blue", linestyle=":", label="safety margin"
    )
    ax_clearance.set_title("Constraint severity vs clearance")
    ax_clearance.set_ylabel("Polyline minimum clearance")
    ax_clearance.legend(fontsize=6)
    ax_collision.set_title("Collision by severity")
    ax_collision.set_ylabel("Collision (0/1)")
    ax_collision.set_yticks([0, 1], ["safe", "collision"])
    ax_collision.set_ylim(-0.15, 1.15)
    ax_collision.legend(fontsize=6)
    ax_jerk.set_title("Constraint severity vs jerk")
    ax_jerk.set_ylabel("Translational jerk RMS")
    ax_jerk.legend(fontsize=6)

    runtime_rows = []
    for regime in ("benign", "constrained"):
        regime_group = metrics[metrics["regime"] == regime]
        for method in METHOD_ORDER:
            values = regime_group.loc[
                regime_group["method"] == method, "generation_runtime_ms"
            ]
            if len(values):
                runtime_rows.append(
                    {
                        "label": f"{regime[0].upper()}:{method}",
                        "runtime": values.mean(),
                        "color": COLORS[method],
                    }
                )
    positions = np.arange(len(runtime_rows))
    bars = ax_runtime.bar(
        positions,
        [row["runtime"] for row in runtime_rows],
        color=[row["color"] for row in runtime_rows],
    )
    ax_runtime.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax_runtime.set_xticks(
        positions,
        [row["label"] for row in runtime_rows],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    ax_runtime.set_yscale("log")
    ax_runtime.set_ylabel("Mean generation runtime (ms)")
    ax_runtime.set_title("Runtime cost by regime/method")
    ax_runtime.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Experiment 10: Constraint-Conditioned Reconciliation")
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, representatives, config = run_experiment()
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    figure = make_figure(metrics, representatives, config)
    figure_path = OUTPUT_DIR / "figure.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print("\n=== Experiment 10: Constraint-Conditioned Reconciliation ===")
    print(metrics.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from action_chunk_graph.baselines import (
    cubic_hermite_crossfade,
    linear_crossfade,
)
from action_chunk_graph.metrics import (
    body_motion_smoothness,
    collision,
    minimum_clearance,
    polyline_collision,
    polyline_minimum_clearance,
    polyline_safety_margin_violation,
    safety_margin_violation,
    translational_jerk_rms,
)
from action_chunk_graph.optimizer import GraphConfig, optimize_reconciled_trajectory
from action_chunk_graph.scenarios import make_collision_scenario_suite


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "exp03_collision"

SEED = 42
NUM_SCENARIOS = 12
NUM_POSES = 21
DT = 0.1

GRAPH_WITHOUT_COLLISION = GraphConfig(
    lambda_old=2.0,
    lambda_new=2.0,
    lambda_smooth=25.0,
    lambda_collision=0.0,
    rotation_scale=0.5,
    max_nfev=120,
)
GRAPH_WITH_COLLISION = GraphConfig(
    lambda_old=2.0,
    lambda_new=2.0,
    lambda_smooth=25.0,
    lambda_collision=1200.0,
    rotation_scale=0.5,
    max_nfev=120,
)

METHOD_ORDER = [
    "Linear crossfade",
    "Cubic Hermite",
    "Graph without collision",
    "Graph with collision",
]
COLORS = {
    "Linear crossfade": "tab:green",
    "Cubic Hermite": "tab:red",
    "Graph without collision": "tab:purple",
    "Graph with collision": "tab:brown",
}


def timed_call(function, *args, **kwargs):
    start = perf_counter()
    value = function(*args, **kwargs)
    runtime_ms = 1000.0 * (perf_counter() - start)
    return value, runtime_ms


def evaluate(
    scenario_name,
    method,
    trajectory,
    obstacle,
    old_clearance,
    new_clearance,
    runtime_ms,
    optimizer_result=None,
):
    center = np.asarray(obstacle["center"], dtype=float)
    row = {
        "scenario": scenario_name,
        "method": method,
        "obstacle_x": center[0],
        "obstacle_y": center[1],
        "obstacle_radius": obstacle["radius"],
        "safety_margin": obstacle["margin"],
        "old_polyline_min_clearance": old_clearance,
        "new_polyline_min_clearance": new_clearance,
        "sampled_collision": collision(trajectory, obstacle),
        "polyline_collision": polyline_collision(trajectory, obstacle),
        "sampled_min_clearance": minimum_clearance(trajectory, obstacle),
        "polyline_min_clearance": polyline_minimum_clearance(
            trajectory, obstacle
        ),
        "sampled_safety_margin_violation": safety_margin_violation(
            trajectory, obstacle
        ),
        "polyline_safety_margin_violation": (
            polyline_safety_margin_violation(trajectory, obstacle)
        ),
        "translational_jerk_rms": translational_jerk_rms(
            trajectory, dt=DT
        ),
        "body_motion_smoothness": body_motion_smoothness(trajectory),
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


def summarize(per_scenario):
    rows = []
    for method in METHOD_ORDER:
        group = per_scenario[per_scenario["method"] == method]
        optimizer_success = group["optimizer_success"].dropna()
        rows.append(
            {
                "method": method,
                "num_scenarios": len(group),
                "sampled_collision_rate": group["sampled_collision"].mean(),
                "polyline_collision_rate": group["polyline_collision"].mean(),
                "mean_sampled_min_clearance": group[
                    "sampled_min_clearance"
                ].mean(),
                "mean_polyline_min_clearance": group[
                    "polyline_min_clearance"
                ].mean(),
                "worst_polyline_min_clearance": group[
                    "polyline_min_clearance"
                ].min(),
                "safety_margin_violation_rate": (
                    group["polyline_safety_margin_violation"] > 0.0
                ).mean(),
                "mean_sampled_safety_margin_violation": group[
                    "sampled_safety_margin_violation"
                ].mean(),
                "mean_safety_margin_violation": group[
                    "polyline_safety_margin_violation"
                ].mean(),
                "mean_translational_jerk_rms": group[
                    "translational_jerk_rms"
                ].mean(),
                "mean_body_motion_smoothness": group[
                    "body_motion_smoothness"
                ].mean(),
                "mean_generation_runtime_ms": group[
                    "generation_runtime_ms"
                ].mean(),
                "optimizer_success_rate": (
                    optimizer_success.astype(bool).mean()
                    if len(optimizer_success)
                    else np.nan
                ),
                "mean_optimizer_nfev": group["optimizer_nfev"].mean(),
            }
        )
    return pd.DataFrame(rows)


def make_figure(summary, per_scenario, representative):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_path, ax_collision, ax_clearance, ax_jerk = axes.flat

    old = representative["old"]
    new = representative["new"]
    obstacle = representative["obstacle"]
    trajectories = representative["trajectories"]

    ax_path.plot(old[:, 0], old[:, 1], "--", color="tab:blue", label="OLD")
    ax_path.plot(new[:, 0], new[:, 1], "--", color="tab:orange", label="NEW")
    for method in METHOD_ORDER:
        trajectory = trajectories[method]
        ax_path.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=COLORS[method],
            linewidth=2.4 if method == "Graph with collision" else 1.7,
            label=method,
        )

    center = obstacle["center"]
    ax_path.add_patch(
        Circle(center, obstacle["radius"], alpha=0.25, color="tab:blue")
    )
    ax_path.add_patch(
        Circle(
            center,
            obstacle["radius"] + obstacle["margin"],
            fill=False,
            linestyle=":",
            color="black",
        )
    )
    ax_path.set_title(f"Representative: {representative['scenario']}")
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(fontsize=8, loc="lower right")

    colors = [COLORS[method] for method in METHOD_ORDER]
    collision_values = summary.set_index("method").loc[
        METHOD_ORDER, "polyline_collision_rate"
    ]
    bars = ax_collision.bar(METHOD_ORDER, collision_values, color=colors)
    ax_collision.bar_label(bars, fmt="%.2f", padding=3)
    ax_collision.set_ylim(0.0, 1.1)
    ax_collision.set_ylabel("Collision rate")
    ax_collision.set_title("Polyline collision rate")
    ax_collision.tick_params(axis="x", rotation=20)
    ax_collision.grid(True, axis="y", alpha=0.3)

    clearance_data = [
        per_scenario.loc[
            per_scenario["method"] == method, "polyline_min_clearance"
        ].to_numpy()
        for method in METHOD_ORDER
    ]
    boxplot = ax_clearance.boxplot(
        clearance_data,
        tick_labels=METHOD_ORDER,
        patch_artist=True,
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax_clearance.axhline(0.0, color="black", linestyle=":")
    ax_clearance.set_ylabel("Minimum obstacle clearance")
    ax_clearance.set_title("Clearance distribution")
    ax_clearance.tick_params(axis="x", rotation=20)
    ax_clearance.grid(True, axis="y", alpha=0.3)

    jerk_values = summary.set_index("method").loc[
        METHOD_ORDER, "mean_translational_jerk_rms"
    ]
    jerk_ratios = jerk_values / jerk_values.iloc[0]
    bars = ax_jerk.bar(METHOD_ORDER, jerk_ratios, color=colors)
    ax_jerk.bar_label(bars, fmt="%.2f", padding=3)
    ax_jerk.axhline(1.0, color="black", linestyle=":", alpha=0.6)
    ax_jerk.set_ylabel("Mean jerk ratio (Linear = 1.0)")
    ax_jerk.set_title("Smoothness cost of collision avoidance")
    ax_jerk.tick_params(axis="x", rotation=20)
    ax_jerk.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Experiment 03: Collision Ablation ({NUM_SCENARIOS} scenarios, seed={SEED})"
    )
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = make_collision_scenario_suite(
        num_scenarios=NUM_SCENARIOS,
        num_poses=NUM_POSES,
        seed=SEED,
    )

    rows = []
    representative = None
    for scenario in scenarios:
        old = scenario["old"]
        new = scenario["new"]
        obstacle = scenario["obstacle"]
        scenario_name = scenario["scenario"]

        old_clearance = polyline_minimum_clearance(old, obstacle)
        new_clearance = polyline_minimum_clearance(new, obstacle)
        if old_clearance < obstacle["margin"] or new_clearance < obstacle["margin"]:
            raise RuntimeError(f"Unsafe proposal generated for {scenario_name}.")

        linear, linear_runtime = timed_call(linear_crossfade, old, new)
        spline, spline_runtime = timed_call(cubic_hermite_crossfade, old, new)
        graph_without_output, graph_without_runtime = timed_call(
            optimize_reconciled_trajectory,
            old,
            new,
            None,
            GRAPH_WITHOUT_COLLISION,
        )
        graph_without, result_without = graph_without_output
        graph_with_output, graph_with_runtime = timed_call(
            optimize_reconciled_trajectory,
            old,
            new,
            obstacle,
            GRAPH_WITH_COLLISION,
        )
        graph_with, result_with = graph_with_output

        if not result_without.success or not result_with.success:
            raise RuntimeError(f"Optimizer failed for {scenario_name}.")

        methods = {
            "Linear crossfade": (linear, linear_runtime, None),
            "Cubic Hermite": (spline, spline_runtime, None),
            "Graph without collision": (
                graph_without,
                graph_without_runtime,
                result_without,
            ),
            "Graph with collision": (
                graph_with,
                graph_with_runtime,
                result_with,
            ),
        }
        for method, (trajectory, runtime_ms, result) in methods.items():
            if not np.all(np.isfinite(trajectory)):
                raise RuntimeError(
                    f"{method} produced non-finite values for {scenario_name}."
                )
            rows.append(
                evaluate(
                    scenario_name,
                    method,
                    trajectory,
                    obstacle,
                    old_clearance,
                    new_clearance,
                    runtime_ms,
                    result,
                )
            )

        if representative is None:
            representative = {
                **scenario,
                "trajectories": {
                    method: values[0] for method, values in methods.items()
                },
            }

    per_scenario = pd.DataFrame(rows)
    summary = summarize(per_scenario)

    per_scenario_path = OUTPUT_DIR / "per_scenario.csv"
    metrics_path = OUTPUT_DIR / "metrics.csv"
    per_scenario.to_csv(per_scenario_path, index=False)
    summary.to_csv(metrics_path, index=False)

    fig = make_figure(summary, per_scenario, representative)
    figure_path = OUTPUT_DIR / "figure.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print("\n=== Experiment 03: Collision Ablation ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {per_scenario_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

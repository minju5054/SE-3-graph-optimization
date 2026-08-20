from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
OUTPUT_DIR = ROOT / "outputs" / "exp04_collision_weight"

SEED = 42
NUM_SCENARIOS = 12
NUM_POSES = 21
DT = 0.1
COLLISION_WEIGHTS = [0.0, 10.0, 30.0, 100.0, 300.0, 1200.0, 5000.0, 20000.0]
SAFETY_TOLERANCE = 1e-6


def make_config(collision_weight):
    return GraphConfig(
        lambda_old=2.0,
        lambda_new=2.0,
        lambda_smooth=25.0,
        lambda_collision=collision_weight,
        rotation_scale=0.5,
        max_nfev=120,
    )


def evaluate(scenario_name, collision_weight, trajectory, obstacle, result, runtime_ms):
    return {
        "scenario": scenario_name,
        "lambda_collision": collision_weight,
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
        "optimizer_success": bool(result.success),
        "optimizer_status": int(result.status),
        "optimizer_message": str(result.message),
        "optimizer_cost": float(result.cost),
        "optimizer_nfev": int(result.nfev),
        "optimizer_optimality": float(result.optimality),
        "optimization_runtime_ms": runtime_ms,
    }


def summarize(per_scenario):
    rows = []
    for collision_weight in COLLISION_WEIGHTS:
        group = per_scenario[
            per_scenario["lambda_collision"] == collision_weight
        ]
        rows.append(
            {
                "lambda_collision": collision_weight,
                "num_scenarios": len(group),
                "optimizer_success_rate": group["optimizer_success"].mean(),
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
                "sampled_safety_satisfaction_rate": (
                    group["sampled_safety_margin_violation"] <= SAFETY_TOLERANCE
                ).mean(),
                "polyline_safety_satisfaction_rate": (
                    group["polyline_safety_margin_violation"] <= SAFETY_TOLERANCE
                ).mean(),
                "mean_sampled_safety_margin_violation": group[
                    "sampled_safety_margin_violation"
                ].mean(),
                "mean_polyline_safety_margin_violation": group[
                    "polyline_safety_margin_violation"
                ].mean(),
                "mean_translational_jerk_rms": group[
                    "translational_jerk_rms"
                ].mean(),
                "mean_body_motion_smoothness": group[
                    "body_motion_smoothness"
                ].mean(),
                "mean_optimizer_nfev": group["optimizer_nfev"].mean(),
                "mean_optimization_runtime_ms": group[
                    "optimization_runtime_ms"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def make_figure(summary):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_collision, ax_margin, ax_smoothness, ax_runtime = axes.flat

    x = np.arange(len(COLLISION_WEIGHTS))
    labels = [f"{weight:g}" for weight in COLLISION_WEIGHTS]

    ax_collision.plot(
        x,
        summary["sampled_collision_rate"],
        "o-",
        label="Sampled poses",
    )
    ax_collision.plot(
        x,
        summary["polyline_collision_rate"],
        "s--",
        label="Polyline",
    )
    ax_collision.set_ylim(-0.05, 1.05)
    ax_collision.set_ylabel("Collision rate")
    ax_collision.set_title("Collision removal threshold")
    ax_collision.legend()
    ax_collision.grid(True, alpha=0.3)

    ax_margin.plot(
        x,
        summary["mean_sampled_safety_margin_violation"],
        "o-",
        label="Sampled poses",
    )
    ax_margin.plot(
        x,
        summary["mean_polyline_safety_margin_violation"],
        "s--",
        label="Polyline",
    )
    ax_margin.set_yscale("log")
    ax_margin.set_ylabel("Mean safety-margin violation")
    ax_margin.set_title("Soft-penalty and discretization floor")
    ax_margin.legend()
    ax_margin.grid(True, which="both", alpha=0.3)

    baseline_jerk = summary.loc[0, "mean_translational_jerk_rms"]
    baseline_body = summary.loc[0, "mean_body_motion_smoothness"]
    ax_smoothness.plot(
        x,
        summary["mean_translational_jerk_rms"] / baseline_jerk,
        "o-",
        label="Translational jerk",
    )
    ax_smoothness.plot(
        x,
        summary["mean_body_motion_smoothness"] / baseline_body,
        "s--",
        label="Body motion",
    )
    ax_smoothness.axhline(1.0, color="black", linestyle=":", alpha=0.6)
    ax_smoothness.set_ylabel("Ratio to lambda=0")
    ax_smoothness.set_title("Smoothness trade-off")
    ax_smoothness.legend()
    ax_smoothness.grid(True, alpha=0.3)

    ax_runtime.plot(
        x,
        summary["mean_optimization_runtime_ms"],
        "o-",
        color="tab:brown",
        label="Runtime",
    )
    ax_runtime.set_ylabel("Mean runtime (ms)")
    ax_runtime.set_title("Optimization cost")
    ax_runtime.grid(True, alpha=0.3)
    ax_iterations = ax_runtime.twinx()
    ax_iterations.plot(
        x,
        summary["mean_optimizer_nfev"],
        "s--",
        color="tab:gray",
        label="Function evaluations",
    )
    ax_iterations.set_ylabel("Mean function evaluations")
    lines = ax_runtime.get_lines() + ax_iterations.get_lines()
    ax_runtime.legend(lines, [line.get_label() for line in lines], loc="best")

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=25)
        axis.set_xlabel("lambda_collision")

    fig.suptitle(
        f"Experiment 04: Collision-Weight Sweep ({NUM_SCENARIOS} scenarios, seed={SEED})"
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
    for collision_weight in COLLISION_WEIGHTS:
        config = make_config(collision_weight)
        for scenario in scenarios:
            obstacle = scenario["obstacle"]
            old_clearance = polyline_minimum_clearance(
                scenario["old"], obstacle
            )
            new_clearance = polyline_minimum_clearance(
                scenario["new"], obstacle
            )
            if (
                old_clearance < obstacle["margin"]
                or new_clearance < obstacle["margin"]
            ):
                raise RuntimeError(
                    f"Unsafe proposal generated for {scenario['scenario']}."
                )

            start = perf_counter()
            trajectory, result = optimize_reconciled_trajectory(
                scenario["old"],
                scenario["new"],
                obstacle,
                config,
            )
            runtime_ms = 1000.0 * (perf_counter() - start)

            if not np.all(np.isfinite(trajectory)):
                raise RuntimeError(
                    f"Non-finite trajectory at lambda={collision_weight:g}, "
                    f"{scenario['scenario']}."
                )

            rows.append(
                evaluate(
                    scenario["scenario"],
                    collision_weight,
                    trajectory,
                    obstacle,
                    result,
                    runtime_ms,
                )
            )

    per_scenario = pd.DataFrame(rows)
    summary = summarize(per_scenario)

    per_scenario_path = OUTPUT_DIR / "per_scenario.csv"
    metrics_path = OUTPUT_DIR / "metrics.csv"
    per_scenario.to_csv(per_scenario_path, index=False)
    summary.to_csv(metrics_path, index=False)

    fig = make_figure(summary)
    figure_path = OUTPUT_DIR / "figure.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print("\n=== Experiment 04: Collision-Weight Sweep ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {per_scenario_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

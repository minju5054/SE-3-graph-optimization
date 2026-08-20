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
OUTPUT_DIR = ROOT / "outputs" / "exp05_segment_collision"

SEED = 42
NUM_SCENARIOS = 12
NUM_POSES = 21
DT = 0.1
COLLISION_WEIGHTS = [30.0, 100.0, 300.0, 1200.0]
FACTOR_MODES = ["nodes", "segments"]
FACTOR_LABELS = {
    "nodes": "Node-only factor",
    "segments": "Segment-aware factor",
}
COLORS = {"nodes": "tab:blue", "segments": "tab:orange"}


def make_config(collision_weight, factor_mode):
    return GraphConfig(
        lambda_old=2.0,
        lambda_new=2.0,
        lambda_smooth=25.0,
        lambda_collision=collision_weight,
        collision_factor=factor_mode,
        rotation_scale=0.5,
        max_nfev=120,
    )


def evaluate(
    scenario_name,
    factor_mode,
    collision_weight,
    trajectory,
    obstacle,
    result,
    runtime_ms,
):
    return {
        "scenario": scenario_name,
        "collision_factor": factor_mode,
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
    for factor_mode in FACTOR_MODES:
        for collision_weight in COLLISION_WEIGHTS:
            group = per_scenario[
                (per_scenario["collision_factor"] == factor_mode)
                & (per_scenario["lambda_collision"] == collision_weight)
            ]
            sampled_violation = group["sampled_safety_margin_violation"]
            polyline_violation = group["polyline_safety_margin_violation"]
            rows.append(
                {
                    "collision_factor": factor_mode,
                    "lambda_collision": collision_weight,
                    "num_scenarios": len(group),
                    "optimizer_success_rate": group[
                        "optimizer_success"
                    ].mean(),
                    "sampled_collision_rate": group[
                        "sampled_collision"
                    ].mean(),
                    "polyline_collision_rate": group[
                        "polyline_collision"
                    ].mean(),
                    "mean_sampled_min_clearance": group[
                        "sampled_min_clearance"
                    ].mean(),
                    "mean_polyline_min_clearance": group[
                        "polyline_min_clearance"
                    ].mean(),
                    "worst_polyline_min_clearance": group[
                        "polyline_min_clearance"
                    ].min(),
                    "mean_sampled_safety_margin_violation": (
                        sampled_violation.mean()
                    ),
                    "mean_polyline_safety_margin_violation": (
                        polyline_violation.mean()
                    ),
                    "mean_discretization_gap": (
                        polyline_violation - sampled_violation
                    ).mean(),
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
    ax_violation, ax_gap, ax_jerk, ax_runtime = axes.flat
    x = np.arange(len(COLLISION_WEIGHTS))
    labels = [f"{weight:g}" for weight in COLLISION_WEIGHTS]

    for factor_mode in FACTOR_MODES:
        group = summary[summary["collision_factor"] == factor_mode]
        color = COLORS[factor_mode]
        label = FACTOR_LABELS[factor_mode]
        ax_violation.plot(
            x,
            group["mean_polyline_safety_margin_violation"],
            "o-",
            color=color,
            label=f"{label}: polyline",
        )
        ax_violation.plot(
            x,
            group["mean_sampled_safety_margin_violation"],
            "s--",
            color=color,
            alpha=0.75,
            label=f"{label}: samples",
        )
        ax_gap.plot(
            x,
            group["mean_discretization_gap"],
            "o-",
            color=color,
            label=label,
        )
        ax_jerk.plot(
            x,
            group["mean_translational_jerk_rms"],
            "o-",
            color=color,
            label=label,
        )
        ax_runtime.plot(
            x,
            group["mean_optimization_runtime_ms"],
            "o-",
            color=color,
            label=label,
        )

    ax_violation.set_yscale("symlog", linthresh=1e-4)
    ax_violation.set_ylabel("Mean safety-margin violation")
    ax_violation.set_title("Safety at samples and along segments")
    ax_violation.legend(fontsize=8)
    ax_violation.grid(True, which="both", alpha=0.3)

    ax_gap.set_ylabel("Polyline violation - sampled violation")
    ax_gap.set_title("Discretization gap")
    ax_gap.legend()
    ax_gap.grid(True, alpha=0.3)

    ax_jerk.set_ylabel("Mean translational jerk RMS")
    ax_jerk.set_title("Smoothness trade-off")
    ax_jerk.legend()
    ax_jerk.grid(True, alpha=0.3)

    ax_runtime.set_ylabel("Mean runtime (ms)")
    ax_runtime.set_title("Optimization runtime")
    ax_runtime.legend()
    ax_runtime.grid(True, alpha=0.3)

    for axis in axes.flat:
        axis.set_xticks(x, labels)
        axis.set_xlabel("lambda_collision")

    fig.suptitle(
        f"Experiment 05: Node vs Segment Collision Factor "
        f"({NUM_SCENARIOS} scenarios, seed={SEED})"
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
    for factor_mode in FACTOR_MODES:
        for collision_weight in COLLISION_WEIGHTS:
            config = make_config(collision_weight, factor_mode)
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

                if not result.success:
                    raise RuntimeError(
                        f"Optimizer failed for {factor_mode}, "
                        f"lambda={collision_weight:g}, {scenario['scenario']}: "
                        f"{result.message}"
                    )
                if not np.all(np.isfinite(trajectory)):
                    raise RuntimeError(
                        f"Non-finite result for {factor_mode}, "
                        f"lambda={collision_weight:g}, {scenario['scenario']}."
                    )

                rows.append(
                    evaluate(
                        scenario["scenario"],
                        factor_mode,
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

    print("\n=== Experiment 05: Node vs Segment Collision Factor ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {per_scenario_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

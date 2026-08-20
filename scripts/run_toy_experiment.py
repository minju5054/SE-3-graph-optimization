from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from action_chunk_graph.baselines import linear_crossfade
from action_chunk_graph.metrics import (
    body_motion_smoothness,
    collision,
    minimum_clearance,
    rotational_increment_rms,
    safety_margin_violation,
    translational_jerk_rms,
)
from action_chunk_graph.optimizer import GraphConfig, optimize_reconciled_trajectory
from action_chunk_graph.scenarios import make_collision_stress_scenario


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(name, trajectory, obstacle):
    return {
        "method": name,
        "collision": collision(trajectory, obstacle),
        "min_clearance": minimum_clearance(trajectory, obstacle),
        "safety_margin_violation": safety_margin_violation(trajectory, obstacle),
        "translational_jerk_rms": translational_jerk_rms(trajectory),
        "rotation_step_rms": rotational_increment_rms(trajectory),
        "body_motion_smoothness": body_motion_smoothness(trajectory),
    }


def main():
    old, new, obstacle = make_collision_stress_scenario(num_poses=21)
    linear = linear_crossfade(old, new)

    config = GraphConfig(
        lambda_old=2.0,
        lambda_new=2.0,
        lambda_smooth=25.0,
        lambda_collision=1200.0,
        rotation_scale=0.5,
        max_nfev=120,
    )

    graph, result = optimize_reconciled_trajectory(
        old=old,
        new=new,
        obstacle=obstacle,
        config=config,
    )

    rows = [
        evaluate("OLD proposal", old, obstacle),
        evaluate("NEW proposal", new, obstacle),
        evaluate("Linear crossfade", linear, obstacle),
        evaluate("SE(2) graph", graph, obstacle),
    ]

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "metrics.csv"
    df.to_csv(csv_path, index=False)

    print("\n=== Metrics ===")
    print(df.to_string(index=False))
    print("\n=== Optimizer ===")
    print(f"success : {result.success}")
    print(f"cost    : {result.cost:.6f}")
    print(f"nfev    : {result.nfev}")
    print(f"message : {result.message}")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(old[:, 0], old[:, 1], "--", label="OLD proposal")
    ax.plot(new[:, 0], new[:, 1], "--", label="NEW proposal")
    ax.plot(linear[:, 0], linear[:, 1], label="Linear crossfade")
    ax.plot(graph[:, 0], graph[:, 1], linewidth=2.5, label="SE(2) graph")

    center = obstacle["center"]
    obstacle_circle = Circle(
        center,
        obstacle["radius"],
        alpha=0.25,
        label="Obstacle",
    )
    safety_circle = Circle(
        center,
        obstacle["radius"] + obstacle["margin"],
        fill=False,
        linestyle=":",
        label="Safety boundary",
    )
    ax.add_patch(obstacle_circle)
    ax.add_patch(safety_circle)

    stride = 3
    ax.quiver(
        graph[::stride, 0],
        graph[::stride, 1],
        np.cos(graph[::stride, 2]),
        np.sin(graph[::stride, 2]),
        angles="xy",
        scale_units="xy",
        scale=5.0,
        width=0.004,
    )

    ax.set_title("Toy Action-Chunk Reconciliation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    figure_path = OUTPUT_DIR / "toy_comparison.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

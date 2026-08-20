from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from action_chunk_graph.geometry import se2_relative_log, wrap_angle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "exp01_orientation"

START_DEG = 179.0
END_DEG = -179.0
NUM_SAMPLES = 101


def cumulative_rotation(wrapped_angles):
    increments = wrap_angle(np.diff(wrapped_angles))
    return np.concatenate([[0.0], np.cumsum(np.abs(increments))])


def summarize(name, residual_type, interpolation_delta, wrapped_angles):
    increments = wrap_angle(np.diff(wrapped_angles))
    endpoint_error = wrap_angle(wrapped_angles[-1] - np.deg2rad(END_DEG))
    midpoint = wrapped_angles[len(wrapped_angles) // 2]

    return {
        "method": name,
        "residual_type": residual_type,
        "signed_delta_deg": np.rad2deg(interpolation_delta),
        "total_rotation_deg": np.rad2deg(np.sum(np.abs(increments))),
        "max_step_deg": np.rad2deg(np.max(np.abs(increments))),
        "midpoint_wrapped_deg": np.rad2deg(midpoint),
        "endpoint_error_deg": abs(np.rad2deg(endpoint_error)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alpha = np.linspace(0.0, 1.0, NUM_SAMPLES)
    start = np.deg2rad(START_DEG)
    end = np.deg2rad(END_DEG)

    # Raw Euclidean subtraction interprets the boundary crossing as -358 degrees.
    naive_delta = end - start
    naive_unwrapped = start + alpha * naive_delta
    naive_wrapped = wrap_angle(naive_unwrapped)

    # Explicit angle wrapping selects the physically shortest +2 degree rotation.
    shortest_delta = float(wrap_angle(end - start))
    shortest_unwrapped = start + alpha * shortest_delta
    shortest_wrapped = wrap_angle(shortest_unwrapped)

    # The rotational component of Log(T_old^-1 T_new) gives the same +2 degrees.
    old_pose = np.array([0.0, 0.0, start])
    new_pose = np.array([0.0, 0.0, end])
    se2_log = se2_relative_log(old_pose, new_pose)
    se2_delta = float(se2_log[2])
    se2_unwrapped = start + alpha * se2_delta
    se2_wrapped = wrap_angle(se2_unwrapped)

    if not np.allclose(se2_log[:2], 0.0, atol=1e-12):
        raise RuntimeError("Pure rotation unexpectedly produced translation residuals.")
    if not np.isclose(se2_delta, shortest_delta, atol=1e-12):
        raise RuntimeError("SE(2) Log did not recover the shortest rotation.")

    rows = [
        summarize(
            "Naive Euclidean",
            "raw theta_new - theta_old",
            naive_delta,
            naive_wrapped,
        ),
        summarize(
            "Shortest-angle",
            "wrap(theta_new - theta_old)",
            shortest_delta,
            shortest_wrapped,
        ),
        summarize(
            "SE(2) Log",
            "Log(T_old^-1 T_new)[rotation]",
            se2_delta,
            se2_wrapped,
        ),
    ]

    metrics = pd.DataFrame(rows)
    metrics_path = OUTPUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    fig, (ax_angle, ax_distance) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax_angle.plot(alpha, np.rad2deg(naive_unwrapped), label="Naive Euclidean")
    ax_angle.plot(
        alpha,
        np.rad2deg(shortest_unwrapped),
        linewidth=3.0,
        label="Shortest-angle",
    )
    ax_angle.plot(
        alpha,
        np.rad2deg(se2_unwrapped),
        "--",
        linewidth=2.0,
        label="SE(2) Log",
    )
    ax_angle.axhline(180.0, color="black", linestyle=":", alpha=0.5)
    ax_angle.axhline(-180.0, color="black", linestyle=":", alpha=0.5)
    ax_angle.set_ylabel("Continuous viewing angle (deg)")
    ax_angle.set_title("Experiment 01: Orientation Wrap (179° to -179°)")
    ax_angle.grid(True, alpha=0.3)
    ax_angle.legend()

    ax_distance.plot(
        alpha,
        np.rad2deg(cumulative_rotation(naive_wrapped)),
        label="Naive Euclidean",
    )
    ax_distance.plot(
        alpha,
        np.rad2deg(cumulative_rotation(shortest_wrapped)),
        linewidth=3.0,
        label="Shortest-angle",
    )
    ax_distance.plot(
        alpha,
        np.rad2deg(cumulative_rotation(se2_wrapped)),
        "--",
        linewidth=2.0,
        label="SE(2) Log",
    )
    ax_distance.set_xlabel("Interpolation progress")
    ax_distance.set_ylabel("Cumulative absolute rotation (deg)")
    ax_distance.grid(True, alpha=0.3)
    ax_distance.legend()
    ax_distance.annotate(
        "358°",
        xy=(1.0, 358.0),
        xytext=(-8, -4),
        textcoords="offset points",
        ha="right",
        va="top",
    )
    ax_distance.annotate(
        "2° (Shortest-angle / SE(2) Log)",
        xy=(1.0, 2.0),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
    )

    fig.tight_layout()
    figure_path = OUTPUT_DIR / "figure.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print("\n=== Experiment 01: Orientation Wrap ===")
    print(metrics.to_string(index=False))
    print(f"\nSE(2) Log residual: {se2_log}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from action_chunk_graph.multi_update import (
    POLICIES,
    generate_episode_distribution,
    rollout_episode,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "exp11_execution_decision"

DT = 0.1
TOTAL_STEPS = 120
CHUNK_NUM_POSES = 31
COMMIT_HORIZON_STEPS = 1
TRANSITION_WINDOW_POSES = 7
NEW_POSITION_TOLERANCE = 0.05
NUM_EPISODES = 30
BASE_SEED = 42

POLICY_LABELS = {
    "always_continue_hard": "Continue + hard",
    "always_continue_hermite": "Continue + Hermite",
    "always_continue_graph": "Continue + graph",
    "always_hold_graph": "Hold + graph",
    "context_conditioned": "Context-conditioned",
}
POLICY_COLORS = {
    "always_continue_hard": "tab:red",
    "always_continue_hermite": "tab:green",
    "always_continue_graph": "tab:purple",
    "always_hold_graph": "tab:blue",
    "context_conditioned": "tab:orange",
}
METHOD_COLORS = {
    "hard_switch": "tab:red",
    "local_hermite": "tab:green",
    "local_graph": "tab:purple",
    "replan_required": "black",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 11: context-conditioned execution decisions."
    )
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    return parser.parse_args()


def summarize(event_frame, episode_frame):
    rows = []
    for policy in POLICIES:
        events = event_frame[event_frame["policy"] == policy]
        episodes = episode_frame[episode_frame["policy"] == policy]
        rows.append(
            {
                "policy": policy,
                "episodes": len(episodes),
                "events": len(events),
                "episode_collision_rate": episodes["any_collision"].mean(),
                "mean_collision_count": episodes["collision_count"].mean(),
                "mean_total_safety_margin_violation": episodes[
                    "total_safety_margin_violation"
                ].mean(),
                "mean_task_progress": episodes["total_task_progress"].mean(),
                "mean_final_goal_error": episodes["final_goal_error"].mean(),
                "mean_episode_jerk_rms": episodes[
                    "episode_translational_jerk_rms"
                ].mean(),
                "mean_hold_duration_seconds": episodes[
                    "total_hold_duration_seconds"
                ].mean(),
                "mean_new_response_delay_seconds": episodes[
                    "mean_new_response_delay_seconds"
                ].mean(),
                "mean_graph_calls": episodes["number_graph_calls"].mean(),
                "mean_replan_required_events": episodes[
                    "number_replan_required_events"
                ].mean(),
                "mean_total_computation_runtime_ms": episodes[
                    "total_computation_runtime_ms"
                ].mean(),
                "mean_deadline_miss_count": episodes[
                    "deadline_miss_count"
                ].mean(),
                "pre_new_collision_event_rate": events[
                    "collision_before_new_ready"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def choose_representative(results):
    candidates = []
    for (policy, episode_id), result in results.items():
        if policy != "context_conditioned":
            continue
        methods = {item["transition_method"] for item in result.timeline}
        behaviors = {item["inference_behavior"] for item in result.timeline}
        score = len(methods) + len(behaviors)
        candidates.append((score, -episode_id, result))
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def plot_results(event_frame, episode_frame, summary, representative, spec):
    figure, axes = plt.subplots(2, 3, figsize=(17, 10.5))
    trajectory = representative.trajectory

    axis = axes[0, 0]
    axis.plot(trajectory[:, 0], trajectory[:, 1], color="tab:orange", linewidth=2.2)
    for event, item in zip(spec.events, representative.timeline):
        obstacle = event.obstacle
        if obstacle is not None:
            axis.add_patch(
                Circle(
                    obstacle["center"],
                    obstacle["radius"],
                    color="tab:red",
                    alpha=0.25,
                )
            )
            axis.add_patch(
                Circle(
                    obstacle["center"],
                    obstacle["radius"] + obstacle["margin"],
                    fill=False,
                    linestyle="--",
                    color="tab:red",
                    alpha=0.55,
                )
            )
        observation = item["observation_step"]
        ready = item["new_ready_step"]
        axis.scatter(*trajectory[observation, :2], marker="o", color="black", s=32)
        axis.scatter(*trajectory[ready, :2], marker="x", color="tab:blue", s=45)
        axis.annotate(
            f"E{event.event_id}: {item['transition_method']}",
            trajectory[observation, :2],
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_title("Representative context-conditioned episode")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.axis("equal")
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    for lane, item in enumerate(representative.timeline):
        observation = item["observation_step"] * DT
        ready = item["new_ready_step"] * DT
        modification = item["modification_step"] * DT
        behavior_color = (
            "tab:blue" if item["inference_behavior"] == "hold_pose" else "tab:cyan"
        )
        axis.barh(lane, ready - observation, left=observation, color=behavior_color)
        axis.barh(
            lane,
            modification - ready,
            left=ready,
            color="0.65",
        )
        axis.scatter(
            modification,
            lane,
            color=METHOD_COLORS[item["transition_method"]],
            s=55,
            zorder=3,
        )
        axis.text(
            modification + 0.08,
            lane,
            item["transition_method"].replace("local_", ""),
            va="center",
            fontsize=8,
        )
    axis.set_yticks(range(len(representative.timeline)))
    axis.set_yticklabels([f"event {i}" for i in range(len(representative.timeline))])
    axis.set_xlabel("episode time [s]")
    axis.set_title("Observation-to-ready decisions")
    axis.grid(axis="x", alpha=0.25)

    axis = axes[0, 2]
    for _, row in summary.iterrows():
        axis.scatter(
            row["mean_task_progress"],
            row["episode_collision_rate"],
            s=50 + 80 * row["mean_total_safety_margin_violation"],
            color=POLICY_COLORS[row["policy"]],
            label=POLICY_LABELS[row["policy"]],
        )
    axis.set_xlabel("mean task progress [m]")
    axis.set_ylabel("episodes with collision")
    axis.set_title("Safety vs progress")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    x_positions = np.arange(len(POLICIES))
    width = 0.38
    axis.bar(
        x_positions - width / 2,
        summary["mean_new_response_delay_seconds"],
        width,
        label="response delay [s]",
        color="tab:blue",
    )
    jerk_axis = axis.twinx()
    jerk_axis.bar(
        x_positions + width / 2,
        summary["mean_episode_jerk_rms"],
        width,
        label="jerk RMS",
        color="tab:orange",
        alpha=0.75,
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], rotation=25, ha="right")
    axis.set_ylabel("response delay [s]")
    jerk_axis.set_ylabel("translational jerk RMS")
    axis.set_title("Responsiveness and smoothness")
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 1]
    bars = axis.bar(
        x_positions,
        summary["mean_total_computation_runtime_ms"],
        color=[POLICY_COLORS[p] for p in POLICIES],
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], rotation=25, ha="right")
    axis.set_ylabel("mean total runtime per episode [ms]")
    axis.set_title("Computation (labels: mean graph calls)")
    axis.grid(axis="y", alpha=0.25)
    for bar, calls in zip(bars, summary["mean_graph_calls"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{calls:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    axis = axes[1, 2]
    context = event_frame[
        (event_frame["policy"] == "context_conditioned")
        & event_frame["predicted_old_min_clearance_before_new_ready"].notna()
    ]
    for method, group in context.groupby("transition_method_selected"):
        axis.scatter(
            group["predicted_old_min_clearance_before_new_ready"],
            group["old_new_geometric_disagreement"],
            color=METHOD_COLORS[method],
            label=method,
            alpha=0.72,
            s=32,
        )
    axis.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    axis.set_xlabel("predicted OLD clearance before ready [m]")
    axis.set_ylabel("OLD/NEW SE(2) disagreement")
    axis.set_title("Context-conditioned transition choices")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)

    figure.suptitle(
        "Experiment 11 — Context-Conditioned Execution Decision\n"
        f"{summary['episodes'].iloc[0]} episodes, {len(event_frame) // len(POLICIES)} external events",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return figure


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = generate_episode_distribution(
        num_episodes=args.episodes,
        base_seed=args.seed,
        total_steps=TOTAL_STEPS,
    )

    event_rows = []
    episode_rows = []
    results = {}
    for policy in POLICIES:
        for spec in specs:
            result = rollout_episode(
                spec,
                policy,
                dt=DT,
                chunk_num_poses=CHUNK_NUM_POSES,
                transition_window_poses=TRANSITION_WINDOW_POSES,
                commit_horizon_steps=COMMIT_HORIZON_STEPS,
                position_tolerance=NEW_POSITION_TOLERANCE,
            )
            results[(policy, spec.episode_id)] = result
            event_rows.extend(result.event_rows)
            episode_rows.append(result.episode_row)

    event_frame = pd.DataFrame(event_rows)
    episode_frame = pd.DataFrame(episode_rows)
    summary = summarize(event_frame, episode_frame)
    decision_counts = (
        event_frame.groupby(
            [
                "policy",
                "inference_behavior_selected",
                "transition_method_selected",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="event_count")
    )

    if event_frame["new_used_before_new_ready"].any():
        raise AssertionError("Causality invariant failed: premature NEW usage.")
    if event_frame["committed_prefix_max_error"].max() > 1e-12:
        raise AssertionError("Committed-prefix invariant failed.")
    if not np.isfinite(episode_frame["total_task_progress"]).all():
        raise AssertionError("Non-finite episode progress.")

    event_frame.to_csv(OUTPUT_DIR / "event_metrics.csv", index=False)
    episode_frame.to_csv(OUTPUT_DIR / "episode_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    decision_counts.to_csv(OUTPUT_DIR / "decision_counts.csv", index=False)

    representative = choose_representative(results)
    representative_spec = specs[representative.episode_row["episode_id"]]
    figure = plot_results(
        event_frame,
        episode_frame,
        summary,
        representative,
        representative_spec,
    )
    figure.savefig(OUTPUT_DIR / "figure.png", dpi=180)
    plt.close(figure)

    print(summary.to_string(index=False))
    print("\nContext-conditioned decisions:")
    print(
        decision_counts[
            decision_counts["policy"] == "context_conditioned"
        ].to_string(index=False)
    )
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

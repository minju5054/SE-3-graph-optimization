from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from .baselines import cubic_hermite_crossfade
from .decision import (
    DecisionThresholds,
    InferenceDecisionFeatures,
    TransitionDecisionFeatures,
    choose_inference_behavior,
    choose_transition_method,
    validate_graph_candidate,
)
from .geometry import heading_from_xy, se2_relative_log, wrap_angle
from .metrics import (
    body_motion_smoothness,
    polyline_collision,
    polyline_minimum_clearance,
    polyline_safety_margin_violation,
    rotational_increment_rms,
    transition_velocity_mismatches,
    translational_jerk_rms,
)
from .optimizer import GraphConfig, optimize_reconciled_trajectory


POLICIES = (
    "always_continue_hard",
    "always_continue_hermite",
    "always_continue_graph",
    "always_hold_graph",
    "context_conditioned",
)


@dataclass(frozen=True)
class ExternalUpdateEvent:
    event_id: int
    observation_step: int
    latency_steps: int
    goal_y: float
    obstacle_center_x: float
    obstacle_center_y: float
    obstacle_radius: float
    obstacle_margin: float
    obstacle_present: bool
    detour_sign: int

    @property
    def obstacle(self):
        if not self.obstacle_present:
            return None
        return {
            "center": np.array(
                [self.obstacle_center_x, self.obstacle_center_y], dtype=float
            ),
            "radius": self.obstacle_radius,
            "margin": self.obstacle_margin,
        }


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    seed: int
    total_steps: int
    events: tuple


@dataclass
class RolloutResult:
    trajectory: np.ndarray
    event_rows: list
    episode_row: dict
    timeline: list


DEFAULT_GRAPH_CONFIG = GraphConfig(
    lambda_old=2.0,
    lambda_new=2.0,
    lambda_smooth=25.0,
    lambda_collision=3000.0,
    rotation_scale=0.5,
    max_nfev=350,
    collision_factor="segments",
    lambda_terminal_new=1200.0,
)


def generate_episode_spec(episode_id, seed, total_steps=120):
    """Generate four reproducible external events, independent of policy state."""
    rng = np.random.default_rng(seed)
    observation_steps = [int(rng.integers(12, 17))]
    for _ in range(3):
        observation_steps.append(
            observation_steps[-1] + int(rng.integers(21, 29))
        )
    if observation_steps[-1] + 9 >= total_steps:
        raise ValueError("total_steps is too short for the generated schedule.")

    goal_y = 0.0
    events = []
    for event_id, observation_step in enumerate(observation_steps):
        goal_y = float(
            np.clip(goal_y + rng.uniform(-0.85, 0.85), -1.45, 1.45)
        )
        nominal_x = 0.12 * observation_step
        obstacle_present = bool(rng.random() < 0.72)
        obstacle_center_y = float(rng.uniform(-0.48, 0.48))
        events.append(
            ExternalUpdateEvent(
                event_id=event_id,
                observation_step=observation_step,
                latency_steps=int(rng.integers(1, 9)),
                goal_y=goal_y,
                obstacle_center_x=float(nominal_x + rng.uniform(0.35, 1.35)),
                obstacle_center_y=obstacle_center_y,
                obstacle_radius=float(rng.uniform(0.18, 0.34)),
                obstacle_margin=float(rng.uniform(0.12, 0.20)),
                obstacle_present=obstacle_present,
                detour_sign=1 if goal_y >= obstacle_center_y else -1,
            )
        )
    return EpisodeSpec(
        episode_id=episode_id,
        seed=seed,
        total_steps=total_steps,
        events=tuple(events),
    )


def generate_episode_distribution(num_episodes=30, base_seed=42, total_steps=120):
    return [
        generate_episode_spec(index, base_seed + index, total_steps)
        for index in range(num_episodes)
    ]


def episode_spec_as_dict(spec):
    """Stable serialization helper used by reproducibility tests."""
    return asdict(spec)


def make_action_chunk(current_pose, event, num_poses=31, step_distance=0.12):
    """Create an observation-relative proposal with an optional smooth detour."""
    current_pose = np.asarray(current_pose, dtype=float)
    u = np.linspace(0.0, 1.0, num_poses)
    x = current_pose[0] + step_distance * np.arange(num_poses)
    smoothstep = 3.0 * u**2 - 2.0 * u**3
    y = current_pose[1] + smoothstep * (event.goal_y - current_pose[1])

    obstacle = event.obstacle
    if obstacle is not None and x[0] < event.obstacle_center_x < x[-1]:
        progress = (event.obstacle_center_x - x[0]) / (x[-1] - x[0])
        base_at_obstacle = float(np.interp(progress, u, y))
        safe_offset = event.obstacle_radius + event.obstacle_margin + 0.18
        target_y = event.obstacle_center_y + event.detour_sign * safe_offset
        shape = np.sin(np.pi * u) ** 2
        shape *= np.exp(-0.5 * ((u - progress) / 0.17) ** 2)
        scale = float(np.interp(progress, u, shape))
        if scale > 1e-9:
            y += (target_y - base_at_obstacle) * shape / scale

    theta = heading_from_xy(x, y)
    chunk = np.column_stack([x, y, theta])
    chunk[0] = current_pose
    return chunk


def _clearance(trajectory, obstacle):
    if obstacle is None:
        return np.inf
    return polyline_minimum_clearance(trajectory, obstacle)


def _collision(trajectory, obstacle):
    return False if obstacle is None else polyline_collision(trajectory, obstacle)


def _margin_violation(trajectory, obstacle):
    if obstacle is None:
        return 0.0
    return polyline_safety_margin_violation(trajectory, obstacle)


def _first_risk_step(trajectory, obstacle):
    if obstacle is None:
        return None
    for index in range(1, len(trajectory)):
        if _clearance(trajectory[index - 1 : index + 1], obstacle) < obstacle["margin"]:
            return index
    return None


def old_prediction_features(
    old_chunk,
    old_remaining_steps,
    latency_steps,
    obstacle,
    commit_horizon_steps,
):
    """Compute Stage-A features without constructing or reading a NEW chunk."""
    prediction_steps = latency_steps
    executable = min(prediction_steps, old_remaining_steps)
    prefix = old_chunk[: executable + 1]
    if executable < prediction_steps:
        prefix = np.vstack(
            [
                prefix,
                np.repeat(
                    prefix[-1][None, :], prediction_steps - executable, axis=0
                ),
            ]
        )
    return InferenceDecisionFeatures(
        latency_steps=latency_steps,
        commit_horizon_steps=commit_horizon_steps,
        old_remaining_steps=old_remaining_steps,
        predicted_old_min_clearance=_clearance(prefix, obstacle),
        obstacle_margin=0.0 if obstacle is None else obstacle["margin"],
        time_to_predicted_risk_steps=_first_risk_step(prefix, obstacle),
    )


def geometric_disagreement(old, new, start_index=0, rotation_scale=0.5):
    count = min(len(old) - start_index, len(new) - start_index)
    if count <= 0:
        return 0.0
    errors = []
    for index in range(start_index, start_index + count):
        residual = se2_relative_log(old[index], new[index])
        residual[2] *= rotation_scale
        errors.append(np.linalg.norm(residual))
    return float(np.mean(errors))


def _apply_inference_behavior(
    trajectory,
    old_chunk,
    observation_step,
    required_steps,
    old_remaining_steps,
    behavior,
    hold_mask,
):
    for local_step in range(1, required_steps + 1):
        global_step = observation_step + local_step
        can_continue = local_step <= old_remaining_steps
        should_continue = behavior in {"continue_old", "continue_then_hold"}
        if should_continue and can_continue:
            trajectory[global_step] = old_chunk[local_step]
        else:
            trajectory[global_step] = trajectory[global_step - 1]
            hold_mask[global_step] = True


def _assemble_hard(local_prefix, new, modification_local):
    return np.vstack([local_prefix[:-1], new[modification_local:]])


def _assemble_transition(local_prefix, transition, new, modification_local):
    suffix_start = modification_local + len(transition)
    return np.vstack([local_prefix[:-1], transition, new[suffix_start:]])


def _build_candidates(
    trajectory,
    old_chunk,
    new,
    observation_step,
    modification_step,
    window_poses,
    obstacle,
):
    modification_local = modification_step - observation_step
    local_prefix = trajectory[observation_step : modification_step + 1].copy()
    old_window = old_chunk[
        modification_local : modification_local + window_poses
    ].copy()
    if len(old_window) < window_poses:
        old_window = np.vstack(
            [
                old_window,
                np.repeat(
                    old_window[-1][None, :], window_poses - len(old_window), axis=0
                ),
            ]
        )
    old_window[0] = trajectory[modification_step]
    new_window = new[modification_local : modification_local + window_poses]
    if len(new_window) != window_poses:
        raise ValueError("NEW chunk does not cover the transition window.")

    hard = _assemble_hard(local_prefix, new, modification_local)
    hermite_start = perf_counter()
    hermite_window = cubic_hermite_crossfade(old_window, new_window)
    hermite_runtime_ms = 1000.0 * (perf_counter() - hermite_start)
    hermite = _assemble_transition(
        local_prefix, hermite_window, new, modification_local
    )
    return {
        "hard_switch": hard,
        "local_hermite": hermite,
        "old_window": old_window,
        "new_window": new_window,
        "hermite_runtime_ms": hermite_runtime_ms,
        "direct_min_clearance": _clearance(hard[modification_local:], obstacle),
        "hermite_min_clearance": _clearance(
            hermite[modification_local:], obstacle
        ),
    }


def _run_graph(
    candidates,
    local_prefix,
    new,
    modification_local,
    obstacle,
    graph_config,
    graph_optimizer,
):
    start = perf_counter()
    graph_window, result = graph_optimizer(
        candidates["old_window"],
        candidates["new_window"],
        obstacle,
        graph_config,
    )
    runtime_ms = 1000.0 * (perf_counter() - start)
    graph = _assemble_transition(local_prefix, graph_window, new, modification_local)
    suffix = graph[modification_local:]
    clearance = _clearance(suffix, obstacle)
    validation = validate_graph_candidate(
        optimizer_success=bool(result.success),
        collision_free=not _collision(suffix, obstacle),
        minimum_clearance=clearance,
        obstacle_margin=0.0 if obstacle is None else obstacle["margin"],
    )
    return graph, result, runtime_ms, clearance, validation


def _nan_if_infinite(value):
    return np.nan if not np.isfinite(value) else float(value)


def rollout_episode(
    spec,
    policy,
    *,
    dt=0.1,
    chunk_num_poses=31,
    transition_window_poses=7,
    commit_horizon_steps=1,
    position_tolerance=0.05,
    graph_config=None,
    graph_optimizer=optimize_reconciled_trajectory,
):
    """Roll out one policy without sharing robot state with other policies."""
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    if transition_window_poses < 2:
        raise ValueError("transition_window_poses must be at least two.")
    if graph_config is None:
        graph_config = DEFAULT_GRAPH_CONFIG

    storage_steps = spec.total_steps + chunk_num_poses
    x = 0.12 * np.arange(storage_steps)
    trajectory = np.column_stack([x, np.zeros(storage_steps), np.zeros(storage_steps)])
    hold_mask = np.zeros(storage_steps, dtype=bool)
    plan_valid_until = chunk_num_poses - 1
    records = []
    timeline = []

    for event_index, event in enumerate(spec.events):
        observation_step = event.observation_step
        new_ready_step = observation_step + event.latency_steps
        modification_step = new_ready_step + commit_horizon_steps
        modification_local = modification_step - observation_step
        if modification_local + transition_window_poses > chunk_num_poses:
            raise ValueError("Latency and window exceed the action chunk horizon.")

        old_chunk = trajectory[
            observation_step : observation_step + chunk_num_poses
        ].copy()
        old_remaining_steps = max(0, plan_valid_until - observation_step)
        prefix_before = trajectory[: observation_step + 1].copy()

        # Stage A is deliberately completed with OLD-only features. NEW is not
        # constructed until the inference-time behavior has been selected.
        inference_features = old_prediction_features(
            old_chunk,
            old_remaining_steps,
            event.latency_steps,
            event.obstacle,
            commit_horizon_steps,
        )
        if policy == "always_hold_graph":
            inference_behavior = "hold_pose"
            inference_reason = "fixed_always_hold"
        elif policy == "context_conditioned":
            inference_decision = choose_inference_behavior(inference_features)
            inference_behavior = inference_decision.action
            inference_reason = inference_decision.reason
        else:
            inference_behavior = (
                "continue_old"
                if old_remaining_steps >= modification_local
                else "continue_then_hold"
            )
            inference_reason = (
                "fixed_always_continue"
                if inference_behavior == "continue_old"
                else "old_horizon_exhausted_then_hold"
            )
        _apply_inference_behavior(
            trajectory,
            old_chunk,
            observation_step,
            modification_local,
            old_remaining_steps,
            inference_behavior,
            hold_mask,
        )
        committed_before_transition = trajectory[
            observation_step:modification_step
        ].copy()

        # Stage B begins only once NEW is ready. It is observation-relative and
        # therefore may be stale relative to a policy that continued moving.
        new = make_action_chunk(trajectory[observation_step], event, chunk_num_poses)
        obstacle = event.obstacle
        candidates = _build_candidates(
            trajectory,
            old_chunk,
            new,
            observation_step,
            modification_step,
            transition_window_poses,
            obstacle,
        )
        local_prefix = trajectory[observation_step : modification_step + 1].copy()
        direct_position_jump = float(
            np.linalg.norm(
                new[modification_local, :2] - trajectory[modification_step, :2]
            )
        )
        direct_rotation_jump = float(
            abs(
                wrap_angle(
                    new[modification_local, 2]
                    - trajectory[modification_step, 2]
                )
            )
        )
        transition_features = TransitionDecisionFeatures(
            direct_position_jump=direct_position_jump,
            direct_rotation_jump=direct_rotation_jump,
            direct_min_clearance=candidates["direct_min_clearance"],
            hermite_min_clearance=candidates["hermite_min_clearance"],
            obstacle_margin=0.0 if obstacle is None else obstacle["margin"],
        )

        graph_called = False
        graph_result = None
        graph_runtime_ms = 0.0
        optimizer_clearance = np.nan
        if policy == "always_continue_hard":
            transition_method = "hard_switch"
            transition_reason = "fixed_hard_switch"
            selected_plan = candidates["hard_switch"]
            generation_runtime_ms = 0.0
        elif policy == "always_continue_hermite":
            transition_method = "local_hermite"
            transition_reason = "fixed_hermite"
            selected_plan = candidates["local_hermite"]
            generation_runtime_ms = candidates["hermite_runtime_ms"]
        else:
            if policy == "context_conditioned":
                transition_decision = choose_transition_method(
                    transition_features, DecisionThresholds()
                )
                requested_method = transition_decision.action
                transition_reason = transition_decision.reason
            else:
                requested_method = "local_graph"
                transition_reason = "fixed_graph"

            if requested_method == "hard_switch":
                transition_method = requested_method
                selected_plan = candidates[requested_method]
                generation_runtime_ms = 0.0
            elif requested_method == "local_hermite":
                transition_method = requested_method
                selected_plan = candidates[requested_method]
                generation_runtime_ms = candidates["hermite_runtime_ms"]
            else:
                graph_called = True
                (
                    graph_plan,
                    graph_result,
                    graph_runtime_ms,
                    optimizer_clearance,
                    graph_validation,
                ) = _run_graph(
                    candidates,
                    local_prefix,
                    new,
                    modification_local,
                    obstacle,
                    graph_config,
                    graph_optimizer,
                )
                generation_runtime_ms = graph_runtime_ms
                transition_method = graph_validation.action
                transition_reason = (
                    f"{transition_reason}; {graph_validation.reason}"
                )
                if transition_method == "local_graph":
                    selected_plan = graph_plan
                else:
                    selected_plan = local_prefix.copy()
                    hold_suffix = np.repeat(
                        trajectory[modification_step][None, :],
                        chunk_num_poses - len(local_prefix) + 1,
                        axis=0,
                    )
                    selected_plan = np.vstack([local_prefix[:-1], hold_suffix])
                    hold_mask[
                        modification_step : observation_step + chunk_num_poses
                    ] = True

        if len(selected_plan) != chunk_num_poses:
            raise AssertionError("Selected local plan must retain the chunk horizon.")
        trajectory[
            observation_step : observation_step + chunk_num_poses
        ] = selected_plan
        trajectory[observation_step + chunk_num_poses :] = selected_plan[-1]
        committed_after_transition = trajectory[
            observation_step:modification_step
        ]
        committed_position_error = np.linalg.norm(
            committed_after_transition[:, :2]
            - committed_before_transition[:, :2],
            axis=1,
        )
        committed_rotation_error = np.abs(
            wrap_angle(
                committed_after_transition[:, 2]
                - committed_before_transition[:, 2]
            )
        )
        committed_prefix_max_error = float(
            max(
                np.max(committed_position_error, initial=0.0),
                np.max(committed_rotation_error, initial=0.0),
            )
        )
        if committed_prefix_max_error > 1e-12:
            raise AssertionError("A committed action was modified.")
        if not np.array_equal(trajectory[: observation_step + 1], prefix_before):
            raise AssertionError("An executed prefix was modified.")
        plan_valid_until = observation_step + chunk_num_poses - 1

        predicted_clearance = inference_features.predicted_old_min_clearance
        row = {
            "episode_id": spec.episode_id,
            "seed": spec.seed,
            "event_id": event.event_id,
            "policy": policy,
            "observation_step": observation_step,
            "new_ready_step": new_ready_step,
            "modification_step": modification_step,
            "latency_steps": event.latency_steps,
            "latency_seconds": event.latency_steps * dt,
            "old_remaining_steps": old_remaining_steps,
            "predicted_old_min_clearance_before_new_ready": _nan_if_infinite(
                predicted_clearance
            ),
            "predicted_old_safety_margin_violation_before_new_ready": float(
                max(0.0, inference_features.obstacle_margin - predicted_clearance)
                if np.isfinite(predicted_clearance)
                else 0.0
            ),
            "time_to_predicted_risk_steps": inference_features.time_to_predicted_risk_steps,
            "old_new_geometric_disagreement": geometric_disagreement(
                old_chunk, new, modification_local
            ),
            "direct_switch_position_jump": direct_position_jump,
            "direct_switch_rotation_jump": direct_rotation_jump,
            "hermite_candidate_minimum_clearance": _nan_if_infinite(
                candidates["hermite_min_clearance"]
            ),
            "hermite_candidate_safety_margin_violation": float(
                max(
                    0.0,
                    transition_features.obstacle_margin
                    - candidates["hermite_min_clearance"],
                )
                if np.isfinite(candidates["hermite_min_clearance"])
                else 0.0
            ),
            "new_proposal_minimum_clearance": _nan_if_infinite(
                _clearance(new, obstacle)
            ),
            "new_proposal_safety_margin_violation": _margin_violation(
                new, obstacle
            ),
            "aligned_new_suffix_minimum_clearance": _nan_if_infinite(
                _clearance(new[modification_local:], obstacle)
            ),
            "aligned_new_suffix_safety_margin_violation": _margin_violation(
                new[modification_local:], obstacle
            ),
            "inference_behavior_selected": inference_behavior,
            "transition_method_selected": transition_method,
            "decision_reason": f"{inference_reason}; {transition_reason}",
            "graph_called": graph_called,
            "replan_required": transition_method == "replan_required",
            "generation_runtime_ms": generation_runtime_ms,
            "optimizer_runtime_ms": graph_runtime_ms,
            "optimizer_success": (
                bool(graph_result.success) if graph_result is not None else np.nan
            ),
            "optimizer_nfev": (
                int(graph_result.nfev) if graph_result is not None else np.nan
            ),
            "optimizer_cost": (
                float(graph_result.cost) if graph_result is not None else np.nan
            ),
            "optimizer_candidate_minimum_clearance": optimizer_clearance,
            "deadline_miss": graph_runtime_ms > 1000.0 * dt,
            "new_used_before_new_ready": False,
            "committed_prefix_max_error": committed_prefix_max_error,
            "observation_pose_x": float(trajectory[observation_step, 0]),
            "observation_pose_y": float(trajectory[observation_step, 1]),
            "obstacle_present": event.obstacle_present,
            "obstacle_center_x": event.obstacle_center_x,
            "obstacle_center_y": event.obstacle_center_y,
            "obstacle_radius": event.obstacle_radius,
            "obstacle_margin": event.obstacle_margin,
            "new_goal_y": event.goal_y,
            "_new": new,
            "_next_observation_step": (
                spec.events[event_index + 1].observation_step
                if event_index + 1 < len(spec.events)
                else spec.total_steps - 1
            ),
        }
        records.append(row)
        timeline.append(
            {
                "event_id": event.event_id,
                "observation_step": observation_step,
                "new_ready_step": new_ready_step,
                "modification_step": modification_step,
                "inference_behavior": inference_behavior,
                "transition_method": transition_method,
            }
        )

    event_rows = []
    for row in records:
        obstacle = None
        if row["obstacle_present"]:
            obstacle = {
                "center": np.array(
                    [row["obstacle_center_x"], row["obstacle_center_y"]]
                ),
                "radius": row["obstacle_radius"],
                "margin": row["obstacle_margin"],
            }
        observation_step = row["observation_step"]
        ready_step = row["new_ready_step"]
        modification_step = row["modification_step"]
        end_step = min(row["_next_observation_step"], spec.total_steps - 1)
        full_segment = trajectory[observation_step : end_step + 1]
        before_segment = trajectory[observation_step : min(ready_step, end_step) + 1]
        after_segment = trajectory[min(ready_step, end_step) : end_step + 1]
        inference_segment = trajectory[observation_step : modification_step + 1]

        transition_window = (
            1
            if row["transition_method_selected"] == "hard_switch"
            else transition_window_poses
        )
        metric_end = modification_step + transition_window
        if metric_end < len(trajectory):
            start_mismatch, end_mismatch = transition_velocity_mismatches(
                trajectory,
                modification_step,
                transition_window,
                dt=dt,
            )
        else:
            start_mismatch, end_mismatch = np.nan, np.nan

        new = row.pop("_new")
        row.pop("_next_observation_step")
        reaction_steps = np.nan
        tracking_errors = []
        for global_step in range(modification_step, end_step + 1):
            local_step = global_step - observation_step
            if local_step >= len(new):
                break
            error = float(
                np.linalg.norm(trajectory[global_step, :2] - new[local_step, :2])
            )
            tracking_errors.append(error)
            if np.isnan(reaction_steps) and error <= position_tolerance:
                reaction_steps = global_step - modification_step

        row.update(
            {
                "collision_before_new_ready": _collision(before_segment, obstacle),
                "collision_after_new_ready": _collision(after_segment, obstacle),
                "safety_margin_violation": _margin_violation(full_segment, obstacle),
                "minimum_clearance": _nan_if_infinite(
                    _clearance(full_segment, obstacle)
                ),
                "progress_during_inference": float(
                    np.linalg.norm(
                        inference_segment[-1, :2] - inference_segment[0, :2]
                    )
                ),
                "reaction_steps_to_new": reaction_steps,
                "reaction_seconds_to_new": reaction_steps * dt,
                "transition_start_position_jump": float(
                    np.linalg.norm(
                        trajectory[modification_step, :2]
                        - trajectory[modification_step - 1, :2]
                    )
                ),
                "transition_start_rotation_jump": float(
                    abs(
                        wrap_angle(
                            trajectory[modification_step, 2]
                            - trajectory[modification_step - 1, 2]
                        )
                    )
                ),
                "transition_start_velocity_mismatch": start_mismatch,
                "transition_end_velocity_mismatch": end_mismatch,
                "local_translational_jerk_rms": translational_jerk_rms(
                    full_segment, dt=dt
                ),
                "mean_new_tracking_deviation": (
                    float(np.mean(tracking_errors)) if tracking_errors else np.nan
                ),
            }
        )
        event_rows.append(row)

    executed = trajectory[: spec.total_steps]
    collision_count = sum(
        bool(row["collision_before_new_ready"])
        or bool(row["collision_after_new_ready"])
        for row in event_rows
    )
    violations = np.array(
        [row["safety_margin_violation"] for row in event_rows], dtype=float
    )
    reaction_values = np.array(
        [row["reaction_seconds_to_new"] for row in event_rows], dtype=float
    )
    target_y = spec.events[-1].goal_y
    target_x = 0.12 * (spec.total_steps - 1)
    episode_row = {
        "policy": policy,
        "episode_id": spec.episode_id,
        "seed": spec.seed,
        "any_collision": collision_count > 0,
        "collision_count": collision_count,
        "total_safety_margin_violation": float(np.sum(violations)),
        "mean_safety_margin_violation": float(np.mean(violations)),
        "total_task_progress": float(executed[-1, 0] - executed[0, 0]),
        "final_goal_error": float(
            np.linalg.norm(executed[-1, :2] - np.array([target_x, target_y]))
        ),
        "episode_translational_jerk_rms": translational_jerk_rms(executed, dt=dt),
        "episode_rotational_increment_rms": rotational_increment_rms(executed),
        "episode_body_motion_smoothness": body_motion_smoothness(executed),
        "total_hold_duration_seconds": float(np.sum(hold_mask[: spec.total_steps]) * dt),
        "mean_new_response_delay_seconds": (
            float(np.nanmean(reaction_values))
            if np.any(np.isfinite(reaction_values))
            else np.nan
        ),
        "number_hard_switches": sum(
            row["transition_method_selected"] == "hard_switch" for row in event_rows
        ),
        "number_hermite_transitions": sum(
            row["transition_method_selected"] == "local_hermite"
            for row in event_rows
        ),
        "number_graph_calls": sum(bool(row["graph_called"]) for row in event_rows),
        "number_graph_transitions": sum(
            row["transition_method_selected"] == "local_graph" for row in event_rows
        ),
        "number_replan_required_events": sum(
            bool(row["replan_required"]) for row in event_rows
        ),
        "total_computation_runtime_ms": float(
            sum(row["generation_runtime_ms"] for row in event_rows)
        ),
        "deadline_miss_count": sum(bool(row["deadline_miss"]) for row in event_rows),
    }
    return RolloutResult(executed, event_rows, episode_row, timeline)

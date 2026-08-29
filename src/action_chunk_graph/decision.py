from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionThresholds:
    direct_position_jump: float = 0.04
    direct_rotation_jump: float = 0.08
    safety_tolerance: float = 0.002


@dataclass(frozen=True)
class InferenceDecisionFeatures:
    latency_steps: int
    commit_horizon_steps: int
    old_remaining_steps: int
    predicted_old_min_clearance: float
    obstacle_margin: float
    time_to_predicted_risk_steps: int | None


@dataclass(frozen=True)
class TransitionDecisionFeatures:
    direct_position_jump: float
    direct_rotation_jump: float
    direct_min_clearance: float
    hermite_min_clearance: float
    obstacle_margin: float


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str


def choose_inference_behavior(features):
    """Choose using OLD-only predictive quantities available at observation."""
    required_steps = features.latency_steps + features.commit_horizon_steps
    if features.predicted_old_min_clearance < features.obstacle_margin:
        return Decision("hold_pose", "predicted_old_safety_margin_violation")
    if features.old_remaining_steps < required_steps:
        return Decision("continue_then_hold", "insufficient_old_horizon")
    return Decision("continue_old", "predicted_old_safe_through_new_ready")


def choose_transition_method(features, thresholds=None):
    """Select the simplest safe candidate after NEW becomes available."""
    if thresholds is None:
        thresholds = DecisionThresholds()
    required_clearance = features.obstacle_margin - thresholds.safety_tolerance
    direct_small = (
        features.direct_position_jump <= thresholds.direct_position_jump
        and features.direct_rotation_jump <= thresholds.direct_rotation_jump
    )
    if direct_small and features.direct_min_clearance >= required_clearance:
        return Decision("hard_switch", "direct_jump_small_and_safe")
    if features.hermite_min_clearance >= required_clearance:
        return Decision("local_hermite", "hermite_candidate_safe")
    return Decision("local_graph", "hermite_candidate_violates_safety")


def validate_graph_candidate(
    optimizer_success,
    collision_free,
    minimum_clearance,
    obstacle_margin,
    thresholds=None,
):
    """Accept graph output only after independent geometry validation."""
    if thresholds is None:
        thresholds = DecisionThresholds()
    required_clearance = obstacle_margin - thresholds.safety_tolerance
    if not optimizer_success:
        return Decision("replan_required", "graph_optimizer_failed")
    if not collision_free:
        return Decision("replan_required", "graph_candidate_collision")
    if minimum_clearance < required_clearance:
        return Decision("replan_required", "graph_candidate_margin_violation")
    return Decision("local_graph", "graph_candidate_validated")

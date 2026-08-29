from dataclasses import fields
from inspect import signature
from types import SimpleNamespace
import unittest

import numpy as np

from action_chunk_graph.decision import (
    InferenceDecisionFeatures,
    TransitionDecisionFeatures,
    choose_inference_behavior,
    choose_transition_method,
    validate_graph_candidate,
)
from action_chunk_graph.multi_update import (
    EpisodeSpec,
    ExternalUpdateEvent,
    POLICIES,
    episode_spec_as_dict,
    generate_episode_spec,
    old_prediction_features,
    rollout_episode,
)


class ExecutionDecisionTest(unittest.TestCase):
    def setUp(self):
        self.spec = generate_episode_spec(0, 42)

    def test_episode_generation_is_deterministic_and_has_multiple_events(self):
        first = generate_episode_spec(3, 45)
        second = generate_episode_spec(3, 45)
        self.assertEqual(episode_spec_as_dict(first), episode_spec_as_dict(second))
        self.assertEqual(len(first.events), 4)
        self.assertTrue(
            all(
                left.observation_step < right.observation_step
                for left, right in zip(first.events, first.events[1:])
            )
        )

    def test_inference_decision_has_no_new_or_hidden_label_input(self):
        names = {field.name for field in fields(InferenceDecisionFeatures)}
        self.assertFalse(any("new" in name for name in names))
        self.assertFalse(any("hidden" in name or "label" in name for name in names))
        stage_a_inputs = set(signature(old_prediction_features).parameters)
        self.assertNotIn("event", stage_a_inputs)
        self.assertFalse(any("new" in name for name in stage_a_inputs))
        result = rollout_episode(self.spec, "context_conditioned")
        self.assertFalse(any(row["new_used_before_new_ready"] for row in result.event_rows))

    def test_executed_prefix_is_immutable_for_every_policy(self):
        for policy in POLICIES:
            result = rollout_episode(self.spec, policy)
            self.assertEqual(
                max(row["committed_prefix_max_error"] for row in result.event_rows),
                0.0,
            )

    def test_hold_preserves_pose_during_inference(self):
        result = rollout_episode(self.spec, "always_hold_graph")
        first = self.spec.events[0]
        segment = result.trajectory[
            first.observation_step : first.observation_step + first.latency_steps + 2
        ]
        expected = np.repeat(segment[0][None, :], len(segment), axis=0)
        np.testing.assert_allclose(segment, expected, atol=1e-12)

    def test_continue_advances_old_when_horizon_is_available(self):
        result = rollout_episode(self.spec, "always_continue_hard")
        first = self.spec.events[0]
        self.assertGreater(
            np.linalg.norm(
                result.trajectory[first.observation_step + 1, :2]
                - result.trajectory[first.observation_step, :2]
            ),
            0.0,
        )

    def test_old_horizon_exhaustion_continues_then_holds(self):
        event = ExternalUpdateEvent(
            event_id=0,
            observation_step=29,
            latency_steps=8,
            goal_y=0.4,
            obstacle_center_x=5.0,
            obstacle_center_y=1.0,
            obstacle_radius=0.2,
            obstacle_margin=0.15,
            obstacle_present=False,
            detour_sign=1,
        )
        spec = EpisodeSpec(99, 99, 75, (event,))
        result = rollout_episode(spec, "context_conditioned")
        row = result.event_rows[0]
        self.assertEqual(row["old_remaining_steps"], 1)
        self.assertEqual(row["inference_behavior_selected"], "continue_then_hold")
        held = result.trajectory[event.observation_step + 1 : event.observation_step + 9]
        expected = np.repeat(held[0][None, :], len(held) - 1, axis=0)
        np.testing.assert_allclose(held[1:], expected, atol=1e-12)

    def test_decision_outputs_cover_valid_cascade_actions(self):
        direct = choose_transition_method(
            TransitionDecisionFeatures(0.01, 0.01, 0.3, 0.3, 0.15)
        )
        hermite = choose_transition_method(
            TransitionDecisionFeatures(0.2, 0.2, 0.3, 0.3, 0.15)
        )
        graph = choose_transition_method(
            TransitionDecisionFeatures(0.2, 0.2, 0.05, 0.05, 0.15)
        )
        replan = validate_graph_candidate(False, False, -0.1, 0.15)
        self.assertEqual(
            {direct.action, hermite.action, graph.action, replan.action},
            {"hard_switch", "local_hermite", "local_graph", "replan_required"},
        )

    def test_stage_a_selects_risk_hold_and_safe_continue(self):
        risky = InferenceDecisionFeatures(4, 1, 20, 0.05, 0.15, 2)
        safe = InferenceDecisionFeatures(4, 1, 20, 0.30, 0.15, None)
        self.assertEqual(choose_inference_behavior(risky).action, "hold_pose")
        self.assertEqual(choose_inference_behavior(safe).action, "continue_old")

    def test_next_event_starts_from_each_policy_actual_state(self):
        continue_result = rollout_episode(self.spec, "always_continue_hard")
        hold_result = rollout_episode(self.spec, "always_hold_graph")
        for result in (continue_result, hold_result):
            second = self.spec.events[1]
            row = result.event_rows[1]
            self.assertAlmostEqual(
                row["observation_pose_x"], result.trajectory[second.observation_step, 0]
            )
            self.assertAlmostEqual(
                row["observation_pose_y"], result.trajectory[second.observation_step, 1]
            )
        self.assertFalse(np.allclose(continue_result.trajectory, hold_result.trajectory))

    def test_event_and_episode_metrics_are_finite_where_required(self):
        result = rollout_episode(self.spec, "context_conditioned")
        for row in result.event_rows:
            for key in (
                "old_new_geometric_disagreement",
                "direct_switch_position_jump",
                "direct_switch_rotation_jump",
                "progress_during_inference",
                "local_translational_jerk_rms",
                "mean_new_tracking_deviation",
            ):
                self.assertTrue(np.isfinite(row[key]), key)
        for key in (
            "total_task_progress",
            "final_goal_error",
            "episode_translational_jerk_rms",
            "total_hold_duration_seconds",
            "total_computation_runtime_ms",
        ):
            self.assertTrue(np.isfinite(result.episode_row[key]), key)

    def test_graph_failure_holds_instead_of_executing_invalid_candidate(self):
        def failing_optimizer(old, new, obstacle, config):
            invalid = new.copy()
            invalid[:, :2] = 1e6
            return invalid, SimpleNamespace(
                success=False,
                nfev=1,
                cost=1.0,
            )

        result = rollout_episode(
            self.spec,
            "always_continue_graph",
            graph_optimizer=failing_optimizer,
        )
        first_row = result.event_rows[0]
        self.assertEqual(first_row["transition_method_selected"], "replan_required")
        modification = first_row["modification_step"]
        following = result.trajectory[modification : self.spec.events[1].observation_step + 1]
        expected = np.repeat(following[0][None, :], len(following), axis=0)
        np.testing.assert_allclose(following, expected, atol=1e-12)
        self.assertLess(np.max(np.abs(following[:, :2])), 1e3)


if __name__ == "__main__":
    unittest.main()

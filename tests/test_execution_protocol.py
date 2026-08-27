import unittest

import numpy as np

from action_chunk_graph.baselines import cubic_hermite_crossfade
from action_chunk_graph.execution import (
    ExecutionConfig,
    assemble_hard_switch,
    assemble_local_transition,
    build_committed_prefix,
    local_transition_windows,
    new_local_index,
)
from action_chunk_graph.metrics import (
    polyline_minimum_clearance,
    steps_to_aligned_position_tolerance,
)
from action_chunk_graph.optimizer import GraphConfig, optimize_reconciled_trajectory
from action_chunk_graph.scenarios import (
    make_async_goal_change_scenario,
    make_constraint_severity_scenario_suite,
)


class ExecutionProtocolTest(unittest.TestCase):
    def setUp(self):
        self.old, self.new = make_async_goal_change_scenario(
            num_poses=31, observation_step=8, seed=42
        )

    def test_index_calculation(self):
        config = ExecutionConfig(
            observation_step=8,
            inference_latency_steps=3,
            commit_horizon_steps=1,
        )
        self.assertEqual(config.new_ready_step, 11)
        self.assertEqual(config.modification_step, 12)
        self.assertEqual(new_local_index(config.modification_step, config), 4)

    def test_committed_prefix_invariant_for_transition_methods(self):
        config = ExecutionConfig(inference_latency_steps=3)
        committed = build_committed_prefix(self.old, config)
        old_window, new_window = local_transition_windows(
            self.old, self.new, config
        )
        hermite_window = cubic_hermite_crossfade(old_window, new_window)
        graph_window, result = optimize_reconciled_trajectory(
            old_window,
            new_window,
            config=GraphConfig(lambda_terminal_new=1200.0),
        )
        self.assertTrue(result.success)
        trajectories = [
            assemble_hard_switch(self.old, self.new, config),
            assemble_local_transition(
                self.old, self.new, hermite_window, config
            ),
            assemble_local_transition(
                self.old, self.new, graph_window, config
            ),
        ]
        for trajectory in trajectories:
            np.testing.assert_array_equal(
                trajectory[: config.modification_step],
                committed[: config.modification_step],
            )

    def test_latency_zero_alignment(self):
        config = ExecutionConfig(
            observation_step=8,
            inference_latency_steps=0,
            commit_horizon_steps=0,
        )
        self.assertEqual(config.new_ready_step, config.observation_step)
        self.assertEqual(config.modification_step, config.observation_step)
        self.assertEqual(new_local_index(config.modification_step, config), 0)
        switched = assemble_hard_switch(self.old, self.new, config)
        np.testing.assert_array_equal(
            switched[config.modification_step], self.new[0]
        )

    def test_hold_pose_during_inference(self):
        config = ExecutionConfig(
            observation_step=8,
            inference_latency_steps=6,
            commit_horizon_steps=0,
            inference_behavior="hold_pose",
        )
        prefix = build_committed_prefix(self.old, config)
        expected = np.repeat(
            self.old[config.observation_step][None, :],
            config.inference_latency_steps + 1,
            axis=0,
        )
        np.testing.assert_array_equal(
            prefix[config.observation_step : config.new_ready_step + 1],
            expected,
        )

    def test_local_window_does_not_modify_outside_prefix(self):
        config = ExecutionConfig(inference_latency_steps=4)
        committed = build_committed_prefix(self.old, config)
        old_window, new_window = local_transition_windows(
            self.old, self.new, config
        )
        transition = cubic_hermite_crossfade(old_window, new_window)
        trajectory = assemble_local_transition(
            self.old, self.new, transition, config
        )
        np.testing.assert_array_equal(
            trajectory[: config.modification_step],
            committed[: config.modification_step],
        )
        suffix_start = (
            new_local_index(config.modification_step, config)
            + config.optimization_window_poses
        )
        np.testing.assert_array_equal(
            trajectory[
                config.modification_step + config.optimization_window_poses :
            ],
            self.new[suffix_start:],
        )

    def test_terminal_anchor_disabled_regression(self):
        old_window = self.old[:9]
        new_window = self.new[:9]
        default_trajectory, default_result = optimize_reconciled_trajectory(
            old_window, new_window, config=GraphConfig()
        )
        disabled_trajectory, disabled_result = optimize_reconciled_trajectory(
            old_window,
            new_window,
            config=GraphConfig(lambda_terminal_new=0.0),
        )
        np.testing.assert_array_equal(default_trajectory, disabled_trajectory)
        self.assertEqual(default_result.nfev, disabled_result.nfev)
        self.assertEqual(default_result.cost, disabled_result.cost)

    def test_configured_transition_windows_and_raw_new_suffix(self):
        for window_poses in (3, 5, 7, 10, 15):
            config = ExecutionConfig(
                observation_step=8,
                inference_latency_steps=4,
                commit_horizon_steps=1,
                optimization_window_poses=window_poses,
            )
            old_window, new_window = local_transition_windows(
                self.old, self.new, config
            )
            self.assertEqual(len(old_window), window_poses)
            self.assertEqual(len(new_window), window_poses)
            transition = cubic_hermite_crossfade(old_window, new_window)
            trajectory = assemble_local_transition(
                self.old, self.new, transition, config
            )
            committed = build_committed_prefix(self.old, config)
            np.testing.assert_array_equal(
                trajectory[: config.modification_step],
                committed[: config.modification_step],
            )
            suffix_start = (
                new_local_index(config.modification_step, config)
                + window_poses
            )
            np.testing.assert_array_equal(
                trajectory[config.modification_step + window_poses :],
                self.new[suffix_start:],
            )

    def test_reaction_metric_uses_observation_aligned_new(self):
        config = ExecutionConfig(
            observation_step=8,
            inference_latency_steps=4,
            commit_horizon_steps=1,
            optimization_window_poses=5,
        )
        trajectory = assemble_hard_switch(self.old, self.new, config)
        steps = steps_to_aligned_position_tolerance(
            trajectory,
            self.new,
            config.observation_step,
            config.modification_step,
            tolerance=0.0,
        )
        self.assertEqual(steps, 0)

    def test_constraint_suite_has_safe_prefix_and_new_proposals(self):
        config = ExecutionConfig(
            observation_step=8,
            inference_latency_steps=2,
            commit_horizon_steps=1,
            optimization_window_poses=10,
        )
        scenarios = make_constraint_severity_scenario_suite()
        self.assertEqual(
            [case["constraint_severity"] for case in scenarios],
            ["low", "medium_low", "medium", "medium_high", "high"],
        )
        for case in scenarios:
            prefix = build_committed_prefix(case["old"], config)
            obstacle = case["obstacle"]
            self.assertGreater(
                polyline_minimum_clearance(prefix, obstacle),
                obstacle["margin"],
            )
            self.assertGreater(
                polyline_minimum_clearance(case["new"], obstacle),
                obstacle["margin"],
            )


if __name__ == "__main__":
    unittest.main()

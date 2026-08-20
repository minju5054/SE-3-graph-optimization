import unittest

import numpy as np

from action_chunk_graph.geometry import (
    matrix_to_pose6,
    pose6_to_matrix,
    se3_exp,
    se3_log,
    se3_relative_log,
    so3_left_jacobian,
    so3_left_jacobian_inverse,
)


class SE3GeometryTest(unittest.TestCase):
    def test_exp_log_random_round_trip(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(0.0, np.pi - 1e-4)
            twist = np.concatenate([rng.uniform(-2.0, 2.0, size=3), axis * angle])
            reconstructed = se3_exp(se3_log(se3_exp(twist)))
            np.testing.assert_allclose(reconstructed, se3_exp(twist), atol=1e-10)

    def test_relative_log_reconstructs_target(self):
        rng = np.random.default_rng(7)
        for _ in range(500):
            pose_a = np.concatenate(
                [rng.uniform(-2.0, 2.0, size=3), rng.uniform(-1.0, 1.0, size=3)]
            )
            pose_b = np.concatenate(
                [rng.uniform(-2.0, 2.0, size=3), rng.uniform(-1.0, 1.0, size=3)]
            )
            reconstructed = pose6_to_matrix(pose_a) @ se3_exp(
                se3_relative_log(pose_a, pose_b)
            )
            np.testing.assert_allclose(
                reconstructed, pose6_to_matrix(pose_b), atol=1e-10
            )

    def test_jacobian_inverse_near_zero(self):
        rotation_vector = np.array([1e-10, -2e-10, 3e-10])
        identity = so3_left_jacobian_inverse(
            rotation_vector
        ) @ so3_left_jacobian(rotation_vector)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-12)

    def test_jacobian_inverse_near_pi(self):
        rotation_vector = np.array([np.pi - 1e-9, 0.0, 0.0])
        identity = so3_left_jacobian_inverse(
            rotation_vector
        ) @ so3_left_jacobian(rotation_vector)
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)

    def test_rotation_wrap_uses_shortest_relative_rotation(self):
        pose_a = np.array([0.0, 0.0, 0.0, np.deg2rad(179.0), 0.0, 0.0])
        pose_b = np.array([0.0, 0.0, 0.0, np.deg2rad(-179.0), 0.0, 0.0])
        relative = se3_relative_log(pose_a, pose_b)
        self.assertAlmostEqual(np.rad2deg(relative[3]), 2.0, places=10)
        np.testing.assert_allclose(relative[:3], 0.0, atol=1e-12)

    def test_pose_matrix_round_trip(self):
        pose = np.array([1.0, -2.0, 0.5, 0.2, -0.3, 0.4])
        reconstructed = pose6_to_matrix(matrix_to_pose6(pose6_to_matrix(pose)))
        np.testing.assert_allclose(reconstructed, pose6_to_matrix(pose), atol=1e-12)


if __name__ == "__main__":
    unittest.main()

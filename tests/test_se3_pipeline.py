import unittest

import numpy as np

from action_chunk_graph.baselines import (
    se3_euclidean_crossfade,
    se3_geodesic_crossfade,
)
from action_chunk_graph.geometry import se3_relative_log
from action_chunk_graph.metrics import (
    spatial_polyline_collision,
    spatial_polyline_minimum_clearance,
)
from action_chunk_graph.scenarios import make_se3_collision_scenario


class SE3PipelineTest(unittest.TestCase):
    def setUp(self):
        self.old, self.new, self.obstacle = make_se3_collision_scenario(
            num_poses=9
        )

    def test_proposals_are_safe_but_crossfades_collide(self):
        self.assertGreater(
            spatial_polyline_minimum_clearance(self.old, self.obstacle),
            self.obstacle["margin"],
        )
        self.assertGreater(
            spatial_polyline_minimum_clearance(self.new, self.obstacle),
            self.obstacle["margin"],
        )
        self.assertTrue(
            spatial_polyline_collision(
                se3_euclidean_crossfade(self.old, self.new), self.obstacle
            )
        )
        self.assertTrue(
            spatial_polyline_collision(
                se3_geodesic_crossfade(self.old, self.new), self.obstacle
            )
        )

    def test_crossfades_preserve_boundary_poses(self):
        for crossfade in (
            se3_euclidean_crossfade(self.old, self.new),
            se3_geodesic_crossfade(self.old, self.new),
        ):
            np.testing.assert_allclose(crossfade[0], self.old[0], atol=1e-12)
            np.testing.assert_allclose(crossfade[-1], self.new[-1], atol=1e-12)

    def test_geodesic_midpoint_uses_shortest_rotation(self):
        midpoint = len(self.old) // 2
        euclidean = se3_euclidean_crossfade(self.old, self.new)
        geodesic = se3_geodesic_crossfade(self.old, self.new)
        euclidean_distance = np.rad2deg(
            np.linalg.norm(
                se3_relative_log(self.old[midpoint], euclidean[midpoint])[3:]
            )
        )
        geodesic_distance = np.rad2deg(
            np.linalg.norm(
                se3_relative_log(self.old[midpoint], geodesic[midpoint])[3:]
            )
        )
        self.assertAlmostEqual(euclidean_distance, 170.0, places=10)
        self.assertAlmostEqual(geodesic_distance, 10.0, places=10)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import numpy as np


PROMETA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROMETA_DIR))

from analyze_tsa_clustering import (  # noqa: E402
    between_variance_fraction,
    build_representations,
    clustering_metric_rows,
    nearest_assignments,
    normalized_entropy,
    prototype_distance_matrix,
)
from require_rtx3090 import validate_gpu_names  # noqa: E402


class ClusteringAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.slices = [
            {
                "name": "classifier.weight",
                "start": 0,
                "end": 2,
                "block": "classifier",
            },
            {
                "name": "tokenizer.gate_logits",
                "start": 2,
                "end": 102,
                "block": "tokenizer.gate_logits",
            },
        ]

    def test_balanced_embedding_matches_block_mean_distance(self):
        vectors = np.zeros((1, 102), dtype=np.float32)
        prototypes = np.ones((1, 102), dtype=np.float32)
        representations, _ = build_representations(
            np.concatenate([vectors, prototypes], axis=0),
            self.slices,
            gate_weight=1.0,
        )
        balanced_squared_distance = np.sum(
            (representations["balanced"][0] - representations["balanced"][1]) ** 2
        )
        distances, _ = prototype_distance_matrix(
            vectors,
            prototypes,
            self.slices,
            "block_mean_l2",
            1.0,
        )
        self.assertAlmostEqual(balanced_squared_distance, 2.0, places=6)
        self.assertAlmostEqual(distances.item(), 2.0, places=6)

    def test_gate_weight_zero_ignores_gate(self):
        vector = np.zeros((1, 102), dtype=np.float32)
        prototype = np.zeros((1, 102), dtype=np.float32)
        prototype[:, 2:] = 10.0
        distances, _ = prototype_distance_matrix(
            vector,
            prototype,
            self.slices,
            "block_mean_l2",
            0.0,
        )
        self.assertAlmostEqual(distances.item(), 0.0, places=6)

    def test_nearest_assignments_return_margin(self):
        labels, distances, margins = nearest_assignments(
            np.asarray([[1.0, 3.0], [5.0, 2.0]])
        )
        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_allclose(distances, [1.0, 2.0])
        np.testing.assert_allclose(margins, [2.0, 3.0])

    def test_cluster_structure_metrics(self):
        values = np.asarray([
            [-2.0, 0.0],
            [-1.8, 0.1],
            [2.0, 0.0],
            [1.8, -0.1],
        ])
        labels = np.asarray([0, 0, 1, 1])
        self.assertGreater(between_variance_fraction(values, labels), 0.99)
        self.assertAlmostEqual(normalized_entropy(labels), 1.0)

    def test_empty_expected_group_is_reported(self):
        values = np.asarray([
            [-2.0, 0.0],
            [-1.8, 0.1],
            [2.0, 0.0],
            [1.8, -0.1],
        ])
        rows = clustering_metric_rows(
            {"collapsed": np.asarray([0, 0, 1, 1])},
            {"global": values},
            np.arange(4),
            permutations=0,
            seed=7,
            expected_cluster_count=3,
        )
        self.assertEqual(rows[0]["cluster_sizes"], "[2, 2, 0]")
        self.assertEqual(rows[0]["min_cluster_fraction"], 0.0)

    def test_rtx3090_guard(self):
        valid, _ = validate_gpu_names(["NVIDIA GeForce RTX 3090"])
        invalid, _ = validate_gpu_names(["NVIDIA A100-SXM4-80GB"])
        missing, _ = validate_gpu_names([])
        self.assertTrue(valid)
        self.assertFalse(invalid)
        self.assertFalse(missing)


if __name__ == "__main__":
    unittest.main()

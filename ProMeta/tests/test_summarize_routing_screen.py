import json
import sys
import tempfile
import unittest
from pathlib import Path


PROMETA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROMETA_DIR))

from summarize_routing_screen import (  # noqa: E402
    build_run_rows,
    build_variant_rows,
    latest_run_files,
)


class RoutingSummaryTests(unittest.TestCase):
    def test_routing_summary_uses_best_validation_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = (
                Path(tmpdir)
                / "H_epoch_bounded"
                / "benchmark_results"
                / "support_4"
            )
            result_dir.mkdir(parents=True)
            result_path = result_dir / "TSA-ProMeta_seed42_fixture.json"
            data = {
                "experiment_name": "H_epoch_bounded",
                "support_size": 4,
                "seed": 42,
                "config": {
                    "tsa_routing_schedule": "epoch_snapshot",
                    "tsa_switch_threshold": 0.05,
                    "tsa_min_group_fraction": 0.05,
                    "tsa_max_group_fraction": 0.5,
                },
                "summary_metrics": {"auroc": 0.71, "auprc": 0.51},
                "history": {
                    "val_auroc": [0.60, 0.72, 0.68],
                    "val_auprc": [0.40, 0.52, 0.49],
                    "tsa_group_usage": [[2, 2], [3, 1], [2, 2]],
                    "tsa_group_switch_rate": [None, 0.10, 0.20],
                    "tsa_hysteresis_retention_rate": [0.0, 0.30, 0.20],
                    "tsa_forced_rebalance_rate": [0.0, 0.05, 0.0],
                    "tsa_mean_assignment_margin": [1.0, 2.0, 1.5],
                    "tsa_group_drift": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    "tsa_mean_block_distances": [
                        {},
                        {"classifier": 1.0},
                        {},
                    ],
                    "tsa_block_nearest_agreement": [
                        {},
                        {"classifier": 0.8},
                        {},
                    ],
                    "tsa_block_task_variance": [
                        {},
                        {"classifier": 1.0},
                        {},
                    ],
                },
            }
            result_path.write_text(json.dumps(data), encoding="utf-8")

            runs = latest_run_files(tmpdir)
            run_rows = build_run_rows(runs)
            variant_rows = build_variant_rows(run_rows)

        self.assertEqual(len(run_rows), 1)
        self.assertEqual(run_rows[0]["best_epoch"], 2)
        self.assertAlmostEqual(run_rows[0]["best_val_auroc"], 0.72)
        self.assertAlmostEqual(run_rows[0]["switch_rate"], 0.10)
        self.assertEqual(len(variant_rows), 1)
        self.assertAlmostEqual(variant_rows[0]["mean_test_auroc"], 0.71)


if __name__ == "__main__":
    unittest.main()

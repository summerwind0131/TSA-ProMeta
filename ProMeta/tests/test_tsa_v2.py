import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset


PROMETA_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROMETA_DIR))

try:
    import torchmetrics  # noqa: F401
except ModuleNotFoundError:
    torchmetrics_stub = types.ModuleType("torchmetrics")
    torchmetrics_stub.functional = SimpleNamespace()
    sys.modules["torchmetrics"] = torchmetrics_stub

from main import (  # noqa: E402
    apply_switch_hysteresis,
    assign_tsa_group_params_from_vectors,
    build_epoch_assignment_map,
    build_model_checkpoint,
    capture_selector_snapshot,
    compute_tsa_distances,
    enforce_group_size_constraints,
    estimate_task_vector,
    flatten_current_group_params,
    get_param_slices,
    load_model_checkpoint,
    select_tsa_group,
    summarize_epoch_assignments,
    train_step_v2,
)
from model import FocalLoss, ProphetBioGateModel  # noqa: E402


def make_config():
    return SimpleNamespace(
        tsa_enable=True,
        tsa_param_keys=["classifier", "tokenizer.gate_logits"],
        num_task_groups=2,
        embed_dim=4,
        num_heads=2,
        num_layers=1,
        dropout_rate=0.0,
        inner_lr=0.01,
        inner_step=1,
        outer_lr=1e-3,
        focal_alpha=0.75,
        focal_gamma=2.0,
        l1_lambda=0.0,
        tsa_selector_steps=1,
        tsa_selector_source="frozen_warmup",
        tsa_assignment_source="current_group",
        tsa_distance_mode="global_l2",
        tsa_gate_distance_weight=1.0,
        tsa_selector_l1_lambda=0.0,
        tsa_assignment_metric="l2",
        tsa_routing_schedule="epoch_snapshot",
        tsa_switch_threshold=0.05,
        tsa_min_group_fraction=0.05,
        tsa_max_group_fraction=0.50,
    )


def make_model(config=None):
    config = config or make_config()
    pathway_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    model = ProphetBioGateModel(
        num_features=6,
        config=config,
        pathway_mask=pathway_mask,
        unknown_indices=[],
    )
    model.eval()
    return model


def make_support():
    torch.manual_seed(7)
    support_x = torch.randn(4, 6)
    support_y = torch.tensor([1.0, 1.0, 0.0, 0.0])
    return support_x, support_y


def initialize_assignment_metadata(model, config, support_x, support_y):
    capture_selector_snapshot(model)
    slices = get_param_slices(model, config)
    criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)
    vector = estimate_task_vector(
        model,
        support_x,
        support_y,
        config,
        criterion,
        torch.device("cpu"),
    )
    group_vectors = torch.stack([vector, vector + 10.0]).numpy()
    assign_tsa_group_params_from_vectors(model, group_vectors, slices)
    model.tsa_param_slices = slices
    model.tsa_vector_mean = torch.zeros_like(vector)
    model.tsa_vector_std = torch.ones_like(vector)
    model.tsa_centroids = torch.stack([vector, vector + 10.0])
    model.tsa_initial_group_vectors = torch.tensor(group_vectors)
    model.tsa_cluster_counts = [1, 1]
    model.tsa_cluster_terms = {"0": ["a"], "1": ["b"]}
    return vector


class TinyTaskDataset(Dataset):
    def __init__(self):
        support_x, support_y = make_support()
        self.rows = []
        for idx in range(2):
            query_x = torch.randn(4, 6)
            query_y = torch.tensor([1.0, 0.0, 1.0, 0.0])
            self.rows.append((
                query_x,
                query_y,
                support_x + idx * 0.1,
                support_y,
                f"disease_{idx}",
            ))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class TsaV2Tests(unittest.TestCase):
    def test_frozen_selector_is_stable_while_live_selector_moves(self):
        config = make_config()
        model = make_model(config)
        support_x, support_y = make_support()
        criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)
        capture_selector_snapshot(model)

        frozen_before = estimate_task_vector(
            model, support_x, support_y, config, criterion, torch.device("cpu")
        )
        with torch.no_grad():
            model.classifier.weight.add_(3.0)
            model.alphas["classifier_weight"].mul_(4.0)
        frozen_after = estimate_task_vector(
            model, support_x, support_y, config, criterion, torch.device("cpu")
        )
        self.assertTrue(torch.equal(frozen_before, frozen_after))

        config.tsa_selector_source = "live_model"
        live_after = estimate_task_vector(
            model, support_x, support_y, config, criterion, torch.device("cpu")
        )
        self.assertFalse(torch.allclose(frozen_before, live_after))

    def test_block_distance_balances_parameter_counts(self):
        vector = torch.zeros(102)
        prototypes = torch.ones(1, 102)
        slices = [
            {"name": "classifier.weight", "start": 0, "end": 2, "block": "classifier"},
            {
                "name": "tokenizer.gate_logits",
                "start": 2,
                "end": 102,
                "block": "tokenizer.gate_logits",
            },
        ]

        global_distance = compute_tsa_distances(
            vector, prototypes, slices, "global_l2", 1.0
        )
        balanced_distance = compute_tsa_distances(
            vector, prototypes, slices, "block_mean_l2", 1.0
        )
        no_gate_distance = compute_tsa_distances(
            vector, prototypes, slices, "block_mean_l2", 0.0
        )

        self.assertAlmostEqual(global_distance.item(), 102.0)
        self.assertAlmostEqual(balanced_distance.item(), 2.0)
        self.assertAlmostEqual(no_gate_distance.item(), 1.0)

    def test_checkpoint_round_trip_preserves_selector_and_assignment(self):
        config = make_config()
        model = make_model(config)
        support_x, support_y = make_support()
        initialize_assignment_metadata(model, config, support_x, support_y)
        criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)
        expected = select_tsa_group(
            model, support_x, support_y, config, criterion, torch.device("cpu")
        )

        args = SimpleNamespace(support_size=4, max_support_size=32)
        checkpoint = build_model_checkpoint(model, config, args)
        restored = make_model(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tsa.pt"
            torch.save(checkpoint, path)
            load_model_checkpoint(
                restored,
                str(path),
                torch.device("cpu"),
                strict=True,
                load_metadata=True,
            )

        actual = select_tsa_group(
            restored, support_x, support_y, config, criterion, torch.device("cpu")
        )
        self.assertEqual(expected["group"], actual["group"])
        self.assertAlmostEqual(expected["distance"], actual["distance"], places=6)
        self.assertAlmostEqual(expected["margin"], actual["margin"], places=6)
        for name in model.tsa_selector_params:
            self.assertTrue(torch.equal(
                model.tsa_selector_params[name],
                restored.tsa_selector_params[name],
            ))

    def test_current_group_matches_initial_centroid_before_training(self):
        config = make_config()
        model = make_model(config)
        support_x, support_y = make_support()
        initialize_assignment_metadata(model, config, support_x, support_y)
        criterion = FocalLoss(alpha=0.5, gamma=config.focal_gamma)

        config.tsa_assignment_source = "fixed_centroid"
        fixed = select_tsa_group(
            model, support_x, support_y, config, criterion, torch.device("cpu")
        )
        config.tsa_assignment_source = "current_group"
        dynamic = select_tsa_group(
            model, support_x, support_y, config, criterion, torch.device("cpu")
        )

        self.assertEqual(fixed["group"], dynamic["group"])
        self.assertAlmostEqual(fixed["distance"], dynamic["distance"], places=6)
        self.assertAlmostEqual(fixed["margin"], dynamic["margin"], places=5)

    def test_only_selected_group_receives_query_gradient(self):
        config = make_config()
        model = make_model(config)
        support_x, support_y = make_support()
        initialize_assignment_metadata(model, config, support_x, support_y)
        model.train()

        query_x = torch.randn(4, 6)
        query_y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        batch = (
            query_x.unsqueeze(0),
            query_y.unsqueeze(0),
            support_x.unsqueeze(0),
            support_y.unsqueeze(0),
            ["disease"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)
        *_, assignments = train_step_v2(
            model, batch, optimizer, config, torch.device("cpu")
        )

        self.assertEqual(assignments[0]["group"], 0)
        selected = model.get_tsa_group_param(0, "classifier.weight")
        unselected = model.get_tsa_group_param(1, "classifier.weight")
        self.assertIsNotNone(selected.grad)
        self.assertIsNone(unselected.grad)
        self.assertIsNotNone(model.shortcut_proj.weight.grad)
        self.assertIsNotNone(model.alphas["classifier_weight"].grad)

    def test_assignment_diagnostics_track_switches(self):
        assignments = [
            {
                "disease_term": "a",
                "group": 1,
                "distance": 0.2,
                "margin": 0.4,
                "hysteresis_retained": True,
                "forced_rebalance": False,
                "block_distances": {"classifier": 0.1},
                "block_nearest_groups": {"classifier": 1},
            },
            {
                "disease_term": "b",
                "group": 0,
                "distance": 0.4,
                "margin": 0.2,
                "hysteresis_retained": False,
                "forced_rebalance": True,
                "block_distances": {"classifier": 0.3},
                "block_nearest_groups": {"classifier": 1},
            },
        ]
        first = summarize_epoch_assignments(assignments, None, 3)
        second = summarize_epoch_assignments(assignments, {"a": 0, "b": 0}, 3)

        self.assertIsNone(first["switch_rate"])
        self.assertEqual(first["group_counts"], [1, 1, 0])
        self.assertAlmostEqual(first["mean_distance"], 0.3)
        self.assertAlmostEqual(second["switch_rate"], 0.5)
        self.assertAlmostEqual(second["hysteresis_retention_rate"], 0.5)
        self.assertAlmostEqual(second["forced_rebalance_rate"], 0.5)
        self.assertAlmostEqual(
            second["mean_block_distances"]["classifier"],
            0.2,
        )
        self.assertAlmostEqual(
            second["block_nearest_agreement"]["classifier"],
            0.5,
        )
        self.assertTrue(second["details"][0]["switched"])

    def test_switch_hysteresis_requires_relative_improvement(self):
        distances = torch.tensor([
            [1.0, 0.96],
            [1.0, 0.80],
        ]).numpy()
        selected, nearest, retained, improvements = apply_switch_hysteresis(
            distances,
            ["small_gain", "large_gain"],
            {"small_gain": 0, "large_gain": 0},
            switch_threshold=0.05,
        )

        self.assertEqual(nearest.tolist(), [1, 1])
        self.assertEqual(selected.tolist(), [0, 1])
        self.assertEqual(retained.tolist(), [True, False])
        self.assertAlmostEqual(improvements[0], 0.04, places=6)
        self.assertAlmostEqual(improvements[1], 0.20, places=6)

    def test_group_size_constraints_prevent_collapse(self):
        distances = torch.full((10, 3), 10.0)
        distances[:, 0] = 0.0
        distances[8, 1] = 0.0
        distances[9, 2] = 0.0
        initial = torch.tensor([0] * 8 + [1, 2]).numpy()

        adjusted, forced, counts = enforce_group_size_constraints(
            distances.numpy(),
            initial,
            min_count=2,
            max_count=5,
        )

        self.assertEqual(sum(counts), 10)
        self.assertTrue(all(2 <= count <= 5 for count in counts))
        self.assertGreaterEqual(int(forced.sum()), 3)
        self.assertEqual(len(adjusted), 10)

    def test_epoch_assignment_map_overrides_live_nearest_group(self):
        config = make_config()
        model = make_model(config)
        support_x, support_y = make_support()
        initialize_assignment_metadata(model, config, support_x, support_y)
        model.train()

        query_x = torch.randn(4, 6)
        query_y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        batch = (
            query_x.unsqueeze(0),
            query_y.unsqueeze(0),
            support_x.unsqueeze(0),
            support_y.unsqueeze(0),
            ["disease"],
        )
        frozen_map = {
            "disease": {
                "group": 1,
                "distance": 1.0,
                "margin": 0.5,
            }
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=config.outer_lr)
        *_, assignments = train_step_v2(
            model,
            batch,
            optimizer,
            config,
            torch.device("cpu"),
            assignment_map=frozen_map,
        )

        self.assertEqual(assignments[0]["group"], 1)
        self.assertIsNone(
            model.get_tsa_group_param(0, "classifier.weight").grad
        )
        self.assertIsNotNone(
            model.get_tsa_group_param(1, "classifier.weight").grad
        )

    def test_epoch_assignment_builder_routes_every_task_once(self):
        config = make_config()
        model = make_model(config)
        support_x, support_y = make_support()
        initialize_assignment_metadata(model, config, support_x, support_y)
        loader = DataLoader(TinyTaskDataset(), batch_size=2, shuffle=True)

        assignment_map, details, diagnostics = build_epoch_assignment_map(
            model,
            loader,
            config,
            torch.device("cpu"),
            previous_assignments=None,
        )

        self.assertEqual(set(assignment_map), {"disease_0", "disease_1"})
        self.assertEqual(len(details), 2)
        self.assertEqual(sum(diagnostics["group_counts"]), 2)
        self.assertEqual(diagnostics["min_group_count"], 1)
        self.assertEqual(diagnostics["max_group_count"], 1)
        self.assertIn("classifier", diagnostics["block_task_variance"])


if __name__ == "__main__":
    unittest.main()

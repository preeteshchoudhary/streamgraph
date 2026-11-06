"""Checks for mistakes that would invalidate a temporal graph experiment."""

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from streamgraph.config import split_masks, validate_config
from streamgraph.data import FILES, export_snapshot, load_dataset
from streamgraph.experiments import fit_source_scaler, select_threshold


CONFIG = {"feature_view": "all", "seed": 42, "splits": {
    "source": [1, 16], "adaptation": [17, 24],
    "validation": [25, 34], "test": [35, 49]}}


class ResearchBoundaries(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        ids = np.array([101, 102, 103, 104, 201, 202, 301, 302])
        steps = np.array([1, 1, 1, 1, 25, 25, 35, 35])
        x = np.zeros((8, 165), dtype=np.float32)
        x[:, 0] = np.arange(8) * 10
        x[6:, 0] = 1_000_000  # Obvious future-only values must not enter a scaler fit.
        pd.DataFrame(np.column_stack([ids, steps, x])).to_csv(self.root / FILES[0], header=False, index=False)
        # Deliberately shuffled: matching by row position would silently corrupt the labels.
        classes = pd.DataFrame({"txId": ids, "class": ["1", "2", "unknown", "2", "1", "2", "1", "2"]})
        classes.iloc[::-1].to_csv(self.root / FILES[2], index=False)
        pd.DataFrame({"txId1": [101, 103, 201, 301], "txId2": [102, 104, 202, 302]}).to_csv(self.root / FILES[1], index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def data(self):
        return load_dataset(self.root, expected_release=False)

    def test_shuffled_label_rows_are_joined_by_id_and_unknown_stays_unknown(self):
        data = self.data()
        labels = dict(zip(data.transaction_ids, data.labels))
        self.assertEqual(labels[101], 1)
        self.assertEqual(labels[102], 0)
        self.assertEqual(labels[103], -1)

    def test_supervised_training_excludes_unknown_and_all_later_periods(self):
        data = self.data()
        masks = split_masks(data.time_steps, data.labels, CONFIG)
        self.assertEqual(set(data.transaction_ids[masks["train"]]), {101, 102, 104})
        self.assertFalse((masks["train"] & masks["validation"]).any())
        self.assertFalse((masks["train"] & masks["test"]).any())
        self.assertEqual(set(data.transaction_ids[masks["test_known"]]), {301, 302})

    def test_scaler_ignores_future_values_and_unknown_source_label_rows(self):
        data = self.data()
        masks = split_masks(data.time_steps, data.labels, CONFIG)
        scaler = fit_source_scaler(data.features, masks["train"])
        self.assertAlmostEqual(scaler.mean_[0], (0 + 10 + 30) / 3)
        self.assertEqual(scaler.n_samples_seen_, 3)

    def test_snapshot_reindexes_edges_without_importing_future_nodes(self):
        data = self.data()
        output = self.root / "snapshot.npz"
        export_snapshot(data, 1, output)
        with np.load(output, allow_pickle=False) as snapshot:
            self.assertEqual(set(snapshot["transaction_ids"]), {101, 102, 103, 104})
            self.assertEqual(snapshot["edge_index"].tolist(), [[0, 2], [1, 3]])
            self.assertEqual(snapshot["y"].tolist(), [1, 0, -1, 0])
            self.assertEqual(snapshot["known_mask"].tolist(), [True, True, False, True])

    def test_missing_edge_endpoint_is_rejected(self):
        pd.DataFrame({"txId1": [101], "txId2": [999]}).to_csv(self.root / FILES[1], index=False)
        with self.assertRaisesRegex(ValueError, "without features"):
            self.data()

    def test_cross_snapshot_edge_is_rejected(self):
        pd.DataFrame({"txId1": [101], "txId2": [201]}).to_csv(self.root / FILES[1], index=False)
        with self.assertRaisesRegex(ValueError, "cross-time-step"):
            self.data()

    def test_unknown_label_spelling_is_not_silently_treated_as_licit(self):
        classes = pd.read_csv(self.root / FILES[2], dtype={"class": "string"})
        classes.loc[0, "class"] = "not-reviewed"
        classes.to_csv(self.root / FILES[2], index=False)
        with self.assertRaisesRegex(ValueError, "Unexpected class"):
            self.data()

    def test_overlapping_time_ranges_are_rejected(self):
        config = copy.deepcopy(CONFIG)
        config["splits"]["validation"] = [24, 34]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_config(config)

    def test_time_and_transaction_id_are_not_predictor_features(self):
        data = self.data()
        self.assertEqual(data.features.shape, (8, 165))
        self.assertEqual(data.feature_view("local").shape, (8, 93))
        self.assertEqual(data.features[0, 0], 0)

    def test_threshold_uses_probability_order_for_illicit_class(self):
        threshold = select_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertAlmostEqual(threshold, 0.8)


if __name__ == "__main__":
    unittest.main()

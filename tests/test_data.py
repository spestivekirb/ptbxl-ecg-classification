import unittest

import pandas as pd

from src.data import (
    DIAGNOSTIC_CLASSES,
    compute_pos_weights,
    encode_labels,
    split_ptbxl_metadata,
)


class DataTests(unittest.TestCase):
    def test_encode_labels_uses_declared_class_order(self):
        self.assertEqual(encode_labels(["STTC", "MI"]), [0, 1, 1, 0, 0])
        self.assertEqual(len(encode_labels([])), len(DIAGNOSTIC_CLASSES))

    def test_split_ptbxl_metadata_uses_official_folds(self):
        metadata = pd.DataFrame(
            {
                "strat_fold": [1, 8, 9, 10],
                "labels": [[1], [1], [1], [1]],
            }
        )
        splits = split_ptbxl_metadata(metadata)
        self.assertEqual(len(splits.train), 2)
        self.assertEqual(len(splits.val), 1)
        self.assertEqual(len(splits.test), 1)

    def test_compute_pos_weights_uses_negative_positive_ratio(self):
        metadata = pd.DataFrame({"labels": [[1, 0], [1, 1], [0, 0], [0, 0]]})
        weights = compute_pos_weights(metadata)
        self.assertEqual(weights.tolist(), [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()

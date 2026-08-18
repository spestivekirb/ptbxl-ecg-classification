import unittest
from unittest.mock import patch

import numpy as np

from src.preprocessing import compute_dataset_stats, normalize_signal


class PreprocessingTests(unittest.TestCase):
    def test_normalize_signal_applies_per_lead_statistics(self):
        signal = np.array([[1.0, 3.0], [2.0, 6.0]])
        result = normalize_signal(
            signal,
            mean=np.array([2.0, 4.0]),
            std=np.array([1.0, 2.0]),
        )
        np.testing.assert_allclose(result, [[-1.0, 1.0], [-1.0, 1.0]])

    @patch("src.preprocessing.load_ecg")
    def test_compute_dataset_stats_aggregates_records(self, load_ecg):
        load_ecg.side_effect = [
            np.array([[0.0, 2.0], [2.0, 4.0]]),
            np.array([[4.0, 6.0], [6.0, 8.0]]),
        ]
        mean, std = compute_dataset_stats(["first", "second"])
        np.testing.assert_allclose(mean, [3.0, 5.0])
        np.testing.assert_allclose(std, [np.sqrt(5.0), np.sqrt(5.0)])

    def test_compute_dataset_stats_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            compute_dataset_stats([])


if __name__ == "__main__":
    unittest.main()

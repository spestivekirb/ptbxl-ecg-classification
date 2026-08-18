import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import evaluate_multilabel, optimize_f1_thresholds
from src.training import fit


class TrainingAndEvaluationTests(unittest.TestCase):
    def test_fit_records_history_and_writes_best_checkpoint(self):
        torch.manual_seed(1)
        signals = torch.randn(12, 2, 4)
        labels = (signals.mean(dim=(1, 2)) > 0).float().unsqueeze(1)
        loader = DataLoader(TensorDataset(signals, labels), batch_size=4)
        model = nn.Sequential(nn.Flatten(), nn.Linear(8, 1))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            result = fit(
                model,
                loader,
                loader,
                criterion,
                optimizer,
                num_epochs=2,
                device=torch.device("cpu"),
                checkpoint_path=checkpoint,
                early_stopping_patience=2,
                verbose=False,
            )

            self.assertEqual(len(result.history.train_loss), 2)
            self.assertIn(result.best_epoch, [1, 2])
            self.assertTrue(checkpoint.is_file())

    def test_evaluate_multilabel_returns_expected_metrics(self):
        labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        probabilities = np.array(
            [[0.9, 0.1], [0.2, 0.8], [0.7, 0.9], [0.1, 0.2]]
        )
        result = evaluate_multilabel(
            labels, probabilities, class_names=["A", "B"]
        )

        self.assertEqual(result.predictions.shape, labels.shape)
        self.assertAlmostEqual(result.overall.loc["Macro F1", "Score"], 1.0)
        self.assertTrue((result.per_class["AUROC"] == 1.0).all())

    def test_optimize_f1_thresholds_operates_per_class(self):
        labels = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        probabilities = np.array(
            [[0.4, 0.1], [0.3, 0.8], [0.2, 0.7], [0.1, 0.6]]
        )
        thresholds = optimize_f1_thresholds(
            labels, probabilities, candidates=[0.25, 0.5, 0.75]
        )
        np.testing.assert_allclose(thresholds, [0.25, 0.5])


if __name__ == "__main__":
    unittest.main()

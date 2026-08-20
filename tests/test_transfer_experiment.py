import unittest

from transfer_experiment import (
    choose_threshold_for_recall,
    metrics_at_threshold,
    stratified_train_val_split,
)


class TransferExperimentTests(unittest.TestCase):
    def test_stratified_split_preserves_both_classes(self):
        targets = [0] * 10 + [1] * 10
        indices = list(range(20))
        fit, val = stratified_train_val_split(indices, targets, 0.2, 2026)

        self.assertEqual(len(val), 4)
        self.assertEqual(sum(targets[i] == 0 for i in val), 2)
        self.assertEqual(sum(targets[i] == 1 for i in val), 2)
        self.assertTrue(set(fit).isdisjoint(set(val)))

    def test_threshold_changes_recall(self):
        truth = [0, 0, 1, 1]
        probabilities = [0.10, 0.60, 0.45, 0.90]

        high = metrics_at_threshold(truth, probabilities, 0.5)
        low = metrics_at_threshold(truth, probabilities, 0.4)

        self.assertEqual(high["crack_recall"], 0.5)
        self.assertEqual(low["crack_recall"], 1.0)

    def test_choose_highest_threshold_meeting_target_recall(self):
        truth = [0, 0, 1, 1]
        probabilities = [0.10, 0.60, 0.45, 0.90]

        threshold, metrics = choose_threshold_for_recall(
            truth, probabilities, target_recall=1.0
        )

        self.assertEqual(threshold, 0.45)
        self.assertEqual(metrics["crack_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()

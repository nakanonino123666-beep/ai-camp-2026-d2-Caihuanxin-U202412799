from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import SmallCNN
from train import balanced_split_indices, confusion_counts


class ModelTests(unittest.TestCase):
    def test_cnn_output_has_two_class_scores(self):
        model = SmallCNN()
        output = model(torch.zeros(4, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (4, 2))


class SplitTests(unittest.TestCase):
    def test_balanced_split_uses_both_classes(self):
        # Labels test split logic only; course evidence uses downloaded images.
        train, test = balanced_split_indices([0] * 10 + [1] * 10, 8, 2026)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(test), 4)
        self.assertEqual({0, 1}, {0 if index < 10 else 1 for index in test})

    def test_false_negative_count(self):
        result = confusion_counts([0, 1, 1], [0, 0, 1])
        self.assertEqual(result["false_negative_cracks"], 1)
        self.assertEqual(result["crack_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()

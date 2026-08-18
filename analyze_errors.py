"""Analyze all CNN errors on the same fixed split as train.py.

Reuses train.py's dataset loading, split and training route so every
number here can be reproduced from `python train.py --model cnn`
plus this script. Writes runs/error-analysis.csv.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset

from train import (
    LABELS,
    balanced_split_indices,
    load_real_dataset,
    set_seed,
)

DATA_ROOT = Path("data/raw")
MAX_PER_CLASS = 600
BATCH_SIZE = 64
SEED = 2026
EPOCHS = 2


def image_stats(path: str) -> dict[str, float]:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    gradient_y, gradient_x = np.gradient(array)
    edge = float(np.sqrt(gradient_x**2 + gradient_y**2).mean())
    return {
        "brightness": round(float(array.mean()), 4),
        "contrast": round(float(array.std()), 4),
        "edge_density": round(edge, 6),
    }


def train_model() -> tuple[nn.Module, list[int], list[int]]:
    from models import SmallCNN

    set_seed(SEED)
    dataset = load_real_dataset(DATA_ROOT)
    train_indices, test_indices = balanced_split_indices(
        dataset.targets, MAX_PER_CLASS, SEED
    )
    train_loader = DataLoader(
        Subset(dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True
    )
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(EPOCHS):
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
    return model, test_indices, [int(v) for v in dataset.targets]


def main() -> int:
    model, test_indices, targets = train_model()
    dataset = load_real_dataset(DATA_ROOT)
    test_loader = DataLoader(
        Subset(dataset, test_indices), batch_size=BATCH_SIZE, shuffle=False
    )
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for images, _ in test_loader:
            predictions.extend(int(v) for v in model(images).argmax(dim=1).tolist())

    rows = []
    for index, predicted_value in zip(test_indices, predictions):
        true_value = targets[index]
        if true_value != predicted_value:
            path = dataset.samples[index][0]
            rows.append(
                {
                    "path": path,
                    "true": LABELS[true_value],
                    "predicted": LABELS[predicted_value],
                    "error_type": "false_negative"
                    if true_value == 1
                    else "false_positive",
                    **image_stats(path),
                }
            )

    output = Path("runs/error-analysis.csv")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    false_negatives = [r for r in rows if r["error_type"] == "false_negative"]
    false_positives = [r for r in rows if r["error_type"] == "false_positive"]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4)

    print(f"wrote {output}: {len(rows)} errors "
          f"({len(false_negatives)} false negatives, "
          f"{len(false_positives)} false positives)")
    print(
        "FN  mean brightness/contrast/edge: "
        f"{mean([r['brightness'] for r in false_negatives])} / "
        f"{mean([r['contrast'] for r in false_negatives])} / "
        f"{mean([r['edge_density'] for r in false_negatives])}"
    )
    print(
        "FP  mean brightness/contrast/edge: "
        f"{mean([r['brightness'] for r in false_positives])} / "
        f"{mean([r['contrast'] for r in false_positives])} / "
        f"{mean([r['edge_density'] for r in false_positives])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

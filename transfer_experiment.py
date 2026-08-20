"""Day 2 extension: transfer learning + recall-oriented threshold selection.

This file leaves the original course pipeline untouched.
It reuses the same Kaggle Surface Crack Detection images and held-out test split,
then adds:
1) ImageNet-pretrained ResNet18;
2) train-only data augmentation;
3) frozen-backbone warm-up + optional full fine-tuning;
4) validation-based threshold selection to prioritize crack recall;
5) an ablation mode: same ResNet18 trained from scratch.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from train import (
    LABELS,
    balanced_split_indices,
    confusion_counts,
    find_class_root,
    save_error_grid,
    set_seed,
    verify_real_data,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_datasets(root: Path) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Create train/eval views over identical files with different transforms."""
    class_root = find_class_root(root)

    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = datasets.ImageFolder(class_root, transform=train_transform)
    eval_dataset = datasets.ImageFolder(class_root, transform=eval_transform)

    expected = {"Negative": 0, "Positive": 1}
    if train_dataset.class_to_idx != expected or eval_dataset.class_to_idx != expected:
        raise ValueError(f"Expected class mapping {expected}")

    return train_dataset, eval_dataset


def stratified_train_val_split(
    indices: list[int],
    targets: list[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split original training indices into fit/validation sets by class."""
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")

    rng = random.Random(seed)
    fit_indices: list[int] = []
    val_indices: list[int] = []

    for class_index in (0, 1):
        class_indices = [i for i in indices if int(targets[i]) == class_index]
        rng.shuffle(class_indices)
        n_val = max(1, int(round(len(class_indices) * val_fraction)))
        val_indices.extend(class_indices[:n_val])
        fit_indices.extend(class_indices[n_val:])

    rng.shuffle(fit_indices)
    rng.shuffle(val_indices)
    return fit_indices, val_indices


def build_resnet18(pretrained: bool = True) -> nn.Module:
    """Build two-class ResNet18, optionally using ImageNet pretrained weights."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze backbone while keeping final classifier trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = trainable
    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * len(images)

    return total_loss / len(loader.dataset)


def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    model.eval()
    truth: list[int] = []
    crack_probabilities: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            truth.extend(int(v) for v in labels.tolist())
            crack_probabilities.extend(float(v) for v in probabilities.cpu().tolist())

    return truth, crack_probabilities


def metrics_at_threshold(
    truth: list[int],
    crack_probabilities: list[float],
    threshold: float,
) -> dict[str, object]:
    predicted = [
        1 if probability >= threshold else 0
        for probability in crack_probabilities
    ]
    return confusion_counts(truth, predicted)


def choose_threshold_for_recall(
    truth: list[int],
    crack_probabilities: list[float],
    target_recall: float,
) -> tuple[float, dict[str, object]]:
    """Choose highest validation threshold that still meets target crack recall."""
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")

    candidates = sorted(set([0.0, 1.0, *crack_probabilities]))
    feasible: list[tuple[float, dict[str, object]]] = []

    for threshold in candidates:
        metrics = metrics_at_threshold(truth, crack_probabilities, threshold)
        if float(metrics["crack_recall"]) >= target_recall:
            feasible.append((threshold, metrics))

    if not feasible:
        return 0.5, metrics_at_threshold(truth, crack_probabilities, 0.5)

    return max(feasible, key=lambda item: item[0])


def errors_from_predictions(
    dataset: datasets.ImageFolder,
    indices: list[int],
    truth: list[int],
    probabilities: list[float],
    threshold: float,
) -> list[dict[str, object]]:
    predicted = [1 if p >= threshold else 0 for p in probabilities]

    return [
        {
            "path": dataset.samples[index][0],
            "true": LABELS[true_value],
            "predicted": LABELS[predicted_value],
            "crack_probability": probability,
        }
        for index, true_value, predicted_value, probability in zip(
            indices, truth, predicted, probabilities
        )
        if true_value != predicted_value
    ]


def save_recall_threshold_curve(
    truth: list[int],
    probabilities: list[float],
    output: Path,
) -> None:
    thresholds = [i / 100 for i in range(0, 101, 2)]
    recalls = [
        float(metrics_at_threshold(truth, probabilities, threshold)["crack_recall"])
        for threshold in thresholds
    ]

    figure = plt.figure(figsize=(7, 4))
    plt.plot(thresholds, recalls)
    plt.xlabel("crack probability threshold")
    plt.ylabel("crack recall")
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25)
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-per-class", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--frozen-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Ablation: same ResNet18 architecture with random initialization.",
    )
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()

    if args.check_data:
        result = verify_real_data(args.data)
        print("REAL DATA CHECK PASSED")
        print(f"class_root: {result['class_root']}")
        print(f"counts: {result['counts']}")
        return 0

    set_seed(args.seed)
    train_dataset, eval_dataset = make_datasets(args.data)

    original_train_indices, test_indices = balanced_split_indices(
        train_dataset.targets,
        args.max_per_class,
        args.seed,
    )

    fit_indices, val_indices = stratified_train_val_split(
        original_train_indices,
        train_dataset.targets,
        args.val_fraction,
        args.seed + 1,
    )

    fit_loader = DataLoader(
        Subset(train_dataset, fit_indices),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(eval_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        Subset(eval_dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(pretrained=not args.no_pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss()
    started = time.perf_counter()

    losses: list[dict[str, float | int | str]] = []

    if args.frozen_epochs:
        set_backbone_trainable(model, False)
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

        for epoch in range(args.frozen_epochs):
            loss = train_epoch(model, fit_loader, optimizer, loss_fn, device)
            losses.append({"phase": "frozen", "epoch": epoch + 1, "loss": loss})

    if args.finetune_epochs:
        set_backbone_trainable(model, True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(args.finetune_epochs):
            loss = train_epoch(model, fit_loader, optimizer, loss_fn, device)
            losses.append({"phase": "finetune", "epoch": epoch + 1, "loss": loss})

    val_truth, val_probabilities = collect_probabilities(model, val_loader, device)
    chosen_threshold, val_metrics = choose_threshold_for_recall(
        val_truth,
        val_probabilities,
        args.target_recall,
    )

    test_truth, test_probabilities = collect_probabilities(model, test_loader, device)

    default_metrics = metrics_at_threshold(
        test_truth,
        test_probabilities,
        0.5,
    )
    recall_oriented_metrics = metrics_at_threshold(
        test_truth,
        test_probabilities,
        chosen_threshold,
    )

    errors = errors_from_predictions(
        eval_dataset,
        test_indices,
        test_truth,
        test_probabilities,
        chosen_threshold,
    )

    model_name = (
        "resnet18_pretrained"
        if not args.no_pretrained
        else "resnet18_scratch"
    )

    result = {
        "dataset": "Surface Crack Detection",
        "source": "Kaggle arunrk7/surface-crack-detection",
        "experiment": model_name,
        "device": str(device),
        "seed": args.seed,
        "max_per_class": args.max_per_class,
        "fit_images": len(fit_indices),
        "validation_images": len(val_indices),
        "test_images": len(test_indices),
        "augmentation": [
            "Resize(224,224)",
            "RandomHorizontalFlip",
            "RandomRotation(8deg)",
            "ColorJitter(brightness=0.15, contrast=0.15)",
            "ImageNet normalization",
        ],
        "training": {
            "frozen_epochs": args.frozen_epochs,
            "finetune_epochs": args.finetune_epochs,
            "losses": losses,
        },
        "threshold_selection": {
            "target_validation_crack_recall": args.target_recall,
            "chosen_threshold": chosen_threshold,
            "validation_metrics": val_metrics,
        },
        "test_at_threshold_0_5": default_metrics,
        "test_at_recall_oriented_threshold": recall_oriented_metrics,
        "first_errors": errors[:12],
        "elapsed_seconds": time.perf_counter() - started,
    }

    output_dir = Path("runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / f"{model_name}.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    save_error_grid(
        errors,
        output_dir / f"{model_name}-errors.png",
    )

    save_recall_threshold_curve(
        val_truth,
        val_probabilities,
        output_dir / f"{model_name}-validation-recall-threshold.png",
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

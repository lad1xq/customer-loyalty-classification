"""Run the reproducible model comparison from the command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loyalty_segmentation.data import prepare_data
from loyalty_segmentation.model import OneVsRestLogisticRegression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the CSV")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    return parser.parse_args()


def metrics(targets, predictions) -> tuple[float, float, float]:
    return (
        accuracy_score(targets, predictions),
        precision_score(targets, predictions, average="macro", zero_division=0),
        recall_score(targets, predictions, average="macro", zero_division=0),
    )


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data)
    prepared = prepare_data(frame)

    models = {
        "Manual GD": OneVsRestLogisticRegression(
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            optimizer="gd",
        ),
        "Manual Nesterov": OneVsRestLogisticRegression(
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            optimizer="nesterov",
        ),
        "sklearn LogisticRegression": LogisticRegression(max_iter=1000),
    }

    print(f"Dataset: {frame.shape[0]} rows, {frame.shape[1]} columns")
    print(f"Train/test: {len(prepared.y_train)}/{len(prepared.y_test)}")
    print()
    print(f"{'Model':30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")

    for name, model in models.items():
        model.fit(prepared.x_train, prepared.y_train)
        prediction = model.predict(prepared.x_test)
        accuracy, precision, recall = metrics(prepared.y_test, prediction)
        print(f"{name:30} {accuracy:10.4f} {precision:10.4f} {recall:10.4f}")


if __name__ == "__main__":
    main()


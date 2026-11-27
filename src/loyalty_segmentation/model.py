"""Manual one-vs-rest logistic regression optimizers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class OneVsRestLogisticRegression:
    """Multiclass logistic regression trained with full-batch updates.

    A binary logistic model is fitted for every class. Final predictions use
    the largest class score. ``optimizer`` may be ``"gd"`` or ``"nesterov"``.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        epochs: int = 200,
        optimizer: str = "gd",
        momentum: float = 0.9,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if optimizer not in {"gd", "nesterov"}:
            raise ValueError("optimizer must be 'gd' or 'nesterov'")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1)")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.momentum = momentum

    @staticmethod
    def _sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(values, -30, 30)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _loss(
        self,
        features: NDArray[np.float64],
        targets: NDArray[np.int64],
        weights: NDArray[np.float64],
        bias: NDArray[np.float64],
    ) -> float:
        losses = []
        for class_index in range(len(self.classes_)):
            binary_target = (targets == class_index).astype(float)
            probability = self._sigmoid(
                features @ weights[:, class_index] + bias[class_index]
            )
            probability = np.clip(probability, 1e-15, 1 - 1e-15)
            class_loss = -np.mean(
                binary_target * np.log(probability)
                + (1 - binary_target) * np.log(1 - probability)
            )
            losses.append(class_loss)
        return float(np.mean(losses))

    def fit(
        self, features: ArrayLike, targets: ArrayLike
    ) -> "OneVsRestLogisticRegression":
        x = np.asarray(features, dtype=float)
        y_original = np.asarray(targets)

        if x.ndim != 2:
            raise ValueError("features must be a two-dimensional array")
        if y_original.ndim != 1 or len(y_original) != len(x):
            raise ValueError("targets must contain one value per row")

        self.classes_, y = np.unique(y_original, return_inverse=True)
        n_samples, n_features = x.shape
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("at least two target classes are required")

        self.weights_ = np.zeros((n_features, n_classes), dtype=float)
        self.bias_ = np.zeros(n_classes, dtype=float)
        velocity_weights = np.zeros_like(self.weights_)
        velocity_bias = np.zeros_like(self.bias_)
        self.loss_history_: list[float] = []

        for _ in range(self.epochs):
            if self.optimizer == "nesterov":
                lookahead_weights = self.weights_ + self.momentum * velocity_weights
                lookahead_bias = self.bias_ + self.momentum * velocity_bias
            else:
                lookahead_weights = self.weights_
                lookahead_bias = self.bias_

            gradient_weights = np.zeros_like(self.weights_)
            gradient_bias = np.zeros_like(self.bias_)

            for class_index in range(n_classes):
                binary_target = (y == class_index).astype(float)
                probability = self._sigmoid(
                    x @ lookahead_weights[:, class_index]
                    + lookahead_bias[class_index]
                )
                error = probability - binary_target
                gradient_weights[:, class_index] = x.T @ error / n_samples
                gradient_bias[class_index] = np.mean(error)

            if self.optimizer == "nesterov":
                velocity_weights = (
                    self.momentum * velocity_weights
                    - self.learning_rate * gradient_weights
                )
                velocity_bias = (
                    self.momentum * velocity_bias
                    - self.learning_rate * gradient_bias
                )
                self.weights_ += velocity_weights
                self.bias_ += velocity_bias
            else:
                self.weights_ -= self.learning_rate * gradient_weights
                self.bias_ -= self.learning_rate * gradient_bias

            self.loss_history_.append(
                self._loss(x, y, self.weights_, self.bias_)
            )

        return self

    def decision_function(self, features: ArrayLike) -> NDArray[np.float64]:
        self._check_is_fitted()
        x = np.asarray(features, dtype=float)
        return x @ self.weights_ + self.bias_

    def predict_proba(self, features: ArrayLike) -> NDArray[np.float64]:
        scores = self.decision_function(features)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=1, keepdims=True)

    def predict(self, features: ArrayLike) -> NDArray:
        class_indices = np.argmax(self.decision_function(features), axis=1)
        return self.classes_[class_indices]

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "weights_"):
            raise RuntimeError("fit must be called before prediction")


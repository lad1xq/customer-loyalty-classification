import numpy as np
import pytest
from sklearn.datasets import make_classification

from loyalty_segmentation.model import OneVsRestLogisticRegression


@pytest.mark.parametrize("optimizer", ["gd", "nesterov"])
def test_model_learns_multiclass_problem(optimizer):
    features, targets = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        class_sep=1.8,
        random_state=42,
    )
    model = OneVsRestLogisticRegression(
        optimizer=optimizer,
        learning_rate=0.1,
        epochs=150,
    )

    model.fit(features, targets)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features[:10])

    assert np.mean(predictions == targets) > 0.75
    assert probabilities.shape == (10, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_invalid_optimizer_is_rejected():
    with pytest.raises(ValueError, match="optimizer"):
        OneVsRestLogisticRegression(optimizer="unknown")


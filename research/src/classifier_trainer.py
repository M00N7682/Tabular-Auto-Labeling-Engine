"""Final classifier training and evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import ClassifierConfig
from src.evaluation import compute_metrics


def train_final_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: ClassifierConfig,
) -> ClassifierMixin:
    """Train the final classifier using the aggregated training data."""

    model = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        max_iter=config.n_estimators,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: ClassifierMixin,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """Compute primary classification metrics."""

    preds = model.predict(X_test)
    return compute_metrics(y_test, preds)


def save_classifier(model: ClassifierMixin, path: Path) -> None:
    """Serialize the trained classifier to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

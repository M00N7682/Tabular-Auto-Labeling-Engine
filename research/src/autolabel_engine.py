"""Few-shot auto-labeling utilities leveraging synthetic augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import ClassifierConfig


@dataclass(slots=True)
class PseudoLabelResult:
    labels: np.ndarray
    confidences: np.ndarray


def train_base_classifier(
    X_labeled: np.ndarray,
    y_labeled: np.ndarray,
    config: ClassifierConfig,
) -> ClassifierMixin:
    """Train the baseline classifier used for pseudo-labeling."""

    model = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        max_iter=config.n_estimators,
        random_state=config.random_state,
    )
    model.fit(X_labeled, y_labeled)
    return model


def pseudo_label_unlabeled(
    classifier: ClassifierMixin,
    X_unlabeled: np.ndarray,
) -> PseudoLabelResult:
    """Infer pseudo-labels and associated confidences for unlabeled samples."""

    proba = classifier.predict_proba(X_unlabeled)
    confidences = proba.max(axis=1)
    labels = classifier.classes_[np.argmax(proba, axis=1)]
    return PseudoLabelResult(labels=labels, confidences=confidences)


def build_autolabeled_dataset(
    df_labeled: pd.DataFrame,
    df_unlabeled: pd.DataFrame,
    pseudo_result: PseudoLabelResult,
    threshold: float,
    target_col: str = "Conversion",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return concatenated training data and filtered pseudo-labeled subset."""

    df_unlabeled = df_unlabeled.copy()
    df_unlabeled[target_col] = pseudo_result.labels
    df_unlabeled["pseudo_confidence"] = pseudo_result.confidences
    accepted = df_unlabeled[df_unlabeled["pseudo_confidence"] >= threshold]
    combined = pd.concat([df_labeled, accepted.drop(columns=["pseudo_confidence"])]).reset_index(drop=True)
    return combined, accepted

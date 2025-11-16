"""Evaluation helpers for reporting classifier performance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return common classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """Persist evaluation metrics to a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))

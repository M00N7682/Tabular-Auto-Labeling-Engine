"""Evaluation helpers for reporting classifier performance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return common classification metrics."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    unique_labels = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    if unique_labels.size > 2:
        average = "weighted"
        return {
            "accuracy": accuracy_score(y_true_arr, y_pred_arr),
            "precision": precision_score(y_true_arr, y_pred_arr, average=average, zero_division=0),
            "recall": recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0),
            "f1": f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0),
        }

    return {
        "accuracy": accuracy_score(y_true_arr, y_pred_arr),
        "precision": precision_score(y_true_arr, y_pred_arr, zero_division=0),
        "recall": recall_score(y_true_arr, y_pred_arr, zero_division=0),
        "f1": f1_score(y_true_arr, y_pred_arr, zero_division=0),
    }


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """Persist evaluation metrics to a JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))

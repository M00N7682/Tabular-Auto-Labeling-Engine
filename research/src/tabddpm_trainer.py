"""Placeholder TabDDPM trainer module to be replaced with a real implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.config import TabDDPMConfig


@dataclass(slots=True)
class TabDDPMModel:
    config: TabDDPMConfig
    num_features: int
    training_embeddings: np.ndarray = field(repr=False)
    labels: np.ndarray = field(repr=False)

    def sample(
        self,
        num_samples: int,
        class_condition: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draw bootstrap samples from stored embeddings to mimic class conditioning."""

        rng = np.random.default_rng(self.config.embedding_dim)

        if class_condition is None:
            indices = rng.choice(self.training_embeddings.shape[0], size=num_samples, replace=True)
            return self.training_embeddings[indices]

        conditions = np.atleast_1d(class_condition)
        # Assuming binary target; generalised to unique labels present in training data.
        selected_rows = []
        for cls in conditions:
            mask = self.labels == cls
            available = np.where(mask)[0]
            if available.size == 0:
                raise ValueError(f"No samples available for class {cls!r} in training data")
            indices = rng.choice(available, size=num_samples, replace=True)
            selected_rows.append(self.training_embeddings[indices])
        return np.vstack(selected_rows)


def train_tabddpm(
    X: np.ndarray,
    y: np.ndarray,
    config: TabDDPMConfig,
) -> TabDDPMModel:
    """Train the TabDDPM placeholder and return a lightweight model object."""

    # TODO: integrate real TabDDPM training implementation.
    return TabDDPMModel(
        config=config,
        num_features=X.shape[1],
        training_embeddings=X,
        labels=y,
    )


def save_tabddpm(model: TabDDPMModel, path: Path) -> None:
    """Persist the placeholder model configuration and metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "num_features": model.num_features,
        "config": model.config.__dict__,
    }
    path.write_text(json.dumps(payload, indent=2))


def load_tabddpm(path: Path, config: TabDDPMConfig) -> TabDDPMModel:
    """Load a placeholder TabDDPM model from disk."""

    data = json.loads(path.read_text())
    return TabDDPMModel(
        config=config,
        num_features=data["num_features"],
        training_embeddings=np.empty((0, data["num_features"])),
        labels=np.empty(0),
    )

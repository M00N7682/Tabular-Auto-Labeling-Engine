"""Placeholder TabDDPM trainer module to be replaced with a real implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import TabDDPMConfig


@dataclass(slots=True)
class TabDDPMModel:
    config: TabDDPMConfig
    num_features: int

    def sample(
        self,
        num_samples: int,
        class_condition: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return synthetic feature vectors sampled from a simple Gaussian prior."""

        rng = np.random.default_rng(self.config.embedding_dim)
        samples = rng.normal(size=(num_samples, self.num_features))
        if class_condition is not None:
            return samples
        return samples


def train_tabddpm(
    X: np.ndarray,
    y: np.ndarray,
    config: TabDDPMConfig,
) -> TabDDPMModel:
    """Train the TabDDPM placeholder and return a lightweight model object."""

    _ = X, y  # TODO: integrate real TabDDPM training
    return TabDDPMModel(config=config, num_features=X.shape[1])


def save_tabddpm(model: TabDDPMModel, path: Path) -> None:
    """Persist the placeholder model configuration and metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"num_features": model.num_features, "config": model.config.__dict__}
    path.write_text(json.dumps(payload))


def load_tabddpm(path: Path, config: TabDDPMConfig) -> TabDDPMModel:
    """Load a placeholder TabDDPM model from disk."""

    data = json.loads(path.read_text())
    return TabDDPMModel(config=config, num_features=data["num_features"])

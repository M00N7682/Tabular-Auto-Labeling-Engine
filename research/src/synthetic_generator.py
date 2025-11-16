"""Synthetic data generation utilities wrapping the TabDDPM model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.preprocessing import PreprocessorArtifacts, inverse_transform
from src.tabddpm_trainer import TabDDPMModel


def generate_synthetic_data(
    model: TabDDPMModel,
    artifacts: PreprocessorArtifacts,
    n_per_class: int,
    classes: Iterable[int],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate class-conditional synthetic samples and optionally persist them."""

    frames: list[pd.DataFrame] = []
    for cls in classes:
        encoded = model.sample(num_samples=n_per_class, class_condition=np.array([cls]))
        decoded = inverse_transform(artifacts, encoded)
        df_cls = decoded.copy()
        df_cls["Conversion"] = cls
        frames.append(df_cls)

    synthetic_df = pd.concat(frames, ignore_index=True)

    target_path = output_path or CONFIG.synthetic_output_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(target_path, index=False)
    return synthetic_df

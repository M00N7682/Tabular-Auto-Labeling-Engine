"""Synthetic data generation utilities wrapping the TabDDPM model."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.tabddpm_trainer import TabDDPMModel


def generate_synthetic_data(
    model: TabDDPMModel,
    n_per_class: int,
    classes: Iterable[int],
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Generate class-conditional synthetic samples and optionally persist them."""

    frames: list[pd.DataFrame] = []
    for cls in classes:
        df_cls = model.sample_class(cls=cls, num_samples=n_per_class)
        frames.append(df_cls)

    synthetic_df = pd.concat(frames, ignore_index=True)

    if output_path is None:
        raise ValueError("output_path must be provided for synthetic data persistence")

    target_path = output_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(target_path, index=False)
    return synthetic_df

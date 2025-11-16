"""Utilities for loading tabular datasets used in the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with basic sanity logging."""

    csv_path = Path(path)
    if not csv_path.exists():
        msg = f"Input data file not found: {csv_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(csv_path)
    logger.info("Loaded dataset %s with shape %s", csv_path.name, df.shape)
    missing_target = df[df.columns[-1]].isna().sum()
    logger.info("Detected %d rows with missing target labels", missing_target)
    return df

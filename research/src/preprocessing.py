"""Preprocessing helpers for splitting and encoding tabular data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(slots=True)
class PreprocessorArtifacts:
    transformer: ColumnTransformer
    feature_names: List[str]
    categorical_columns: List[str]
    numeric_columns: List[str]
    original_columns: List[str]


def split_labeled_unlabeled(
    df: pd.DataFrame,
    target_col: str = "Conversion",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into labeled and unlabeled partitions."""

    if target_col not in df.columns:
        msg = f"Target column '{target_col}' not found in DataFrame"
        raise KeyError(msg)

    mask_labeled = df[target_col].notna()
    df_labeled = df.loc[mask_labeled].copy()
    df_unlabeled = df.loc[~mask_labeled].copy()
    return df_labeled, df_unlabeled


def _infer_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical = [col for col in df.columns if df[col].dtype == "object"]
    numeric = [col for col in df.columns if col not in categorical]
    return categorical, numeric


def fit_preprocessor(
    df: pd.DataFrame,
    categorical_cols: Optional[Iterable[str]] = None,
    numeric_cols: Optional[Iterable[str]] = None,
) -> PreprocessorArtifacts:
    """Fit a ColumnTransformer that handles numeric scaling and categorical encoding."""

    cats, nums = _infer_column_types(df)
    categorical_columns = list(categorical_cols) if categorical_cols is not None else cats
    numeric_columns = list(numeric_cols) if numeric_cols is not None else nums

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    transformer.fit(df)

    feature_names: List[str] = []
    if numeric_columns:
        feature_names.extend(numeric_columns)
    if categorical_columns:
        ohe = transformer.named_transformers_["categorical"]
        feature_names.extend(list(ohe.get_feature_names_out(categorical_columns)))

    return PreprocessorArtifacts(
        transformer=transformer,
        feature_names=feature_names,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        original_columns=list(df.columns),
    )


def transform(artifacts: PreprocessorArtifacts, df: pd.DataFrame) -> np.ndarray:
    """Apply the fitted transformer to a dataframe of features."""

    return artifacts.transformer.transform(df)


def inverse_transform(artifacts: PreprocessorArtifacts, array: np.ndarray) -> np.ndarray:
    """Best-effort inverse transformation of preprocessed features."""

    return artifacts.transformer.inverse_transform(array)

"""End-to-end orchestration for the TabDDPM auto-labeling pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.autolabel_engine import (
    PseudoLabelResult,
    build_autolabeled_dataset,
    pseudo_label_unlabeled,
    train_base_classifier,
)
from src.classifier_trainer import evaluate_classifier, save_classifier, train_final_classifier
from src.config import CONFIG, PipelineConfig
from src.data_loader import load_data
from src.evaluation import save_metrics
from src.preprocessing import (
    PreprocessorArtifacts,
    fit_preprocessor,
    split_labeled_unlabeled,
    transform,
)
from src.synthetic_generator import generate_synthetic_data
from src.tabddpm_trainer import save_tabddpm, train_tabddpm

logger = logging.getLogger(__name__)


def _persist_preprocessor(artifacts: PreprocessorArtifacts, path: Path) -> None:
    joblib.dump(artifacts, path)


def _prepare_features(
    artifacts: PreprocessorArtifacts,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    return transform(artifacts, df[feature_cols])


def _maybe_simulate_unlabeled(
    df: pd.DataFrame,
    target_col: str,
    ratio: float,
    min_labeled: int,
    seed: int,
) -> pd.DataFrame:
    """Mask a fraction of labels to create a semi-supervised split when needed."""

    if not 0 < ratio < 1:
        return df

    df_simulated = df.copy()
    if df_simulated[target_col].isna().any():
        return df_simulated
    rng = np.random.default_rng(seed)
    labeled_mask = df_simulated[target_col].notna()
    grouped = df_simulated.loc[labeled_mask].groupby(target_col)

    for cls_value, group in grouped:
        total = len(group)
        if total <= min_labeled:
            continue

        desired_unlabeled = int(np.floor(total * ratio))
        max_unlabeled = total - min_labeled
        num_unlabeled = min(desired_unlabeled, max_unlabeled)
        if num_unlabeled <= 0:
            continue

        indices = rng.choice(group.index.to_numpy(), size=num_unlabeled, replace=False)
        df_simulated.loc[indices, target_col] = pd.NA

    return df_simulated


def run_full_pipeline(config: PipelineConfig = CONFIG) -> Dict[str, float]:
    """Execute the complete pipeline and return evaluation metrics."""

    logging.basicConfig(level=logging.INFO)

    df = load_data(config.data_path)
    target_col = "Conversion"
    df = _maybe_simulate_unlabeled(
        df,
        target_col=target_col,
        ratio=config.simulate_unlabeled_ratio,
        min_labeled=config.min_labeled_per_class,
        seed=config.random_seed,
    )
    df_labeled, df_unlabeled = split_labeled_unlabeled(df, target_col=target_col)
    logger.info(
        "Semi-supervised split → labeled: %d rows, unlabeled: %d rows",
        len(df_labeled),
        len(df_unlabeled),
    )

    feature_cols = [col for col in df_labeled.columns if col != target_col]
    artifacts = fit_preprocessor(df_labeled[feature_cols])
    _persist_preprocessor(artifacts, config.preprocessor_path)

    X_labeled = _prepare_features(artifacts, df_labeled, feature_cols)
    y_labeled = df_labeled[target_col].to_numpy()

    tabddpm_model = train_tabddpm(X_labeled, y_labeled, config.tabddpm)
    save_tabddpm(tabddpm_model, config.tabddpm_checkpoint_dir / "tabddpm_model.json")

    classes = sorted(df_labeled[target_col].unique())
    df_synth = generate_synthetic_data(
        tabddpm_model,
        artifacts,
        config.synthetic_samples_per_class,
        classes,
        config.synthetic_output_path,
    )

    df_labeled_aug = pd.concat([df_labeled, df_synth], ignore_index=True)
    X_labeled_aug = _prepare_features(artifacts, df_labeled_aug, feature_cols)
    y_labeled_aug = df_labeled_aug[target_col].to_numpy()

    base_classifier = train_base_classifier(X_labeled_aug, y_labeled_aug, config.classifier)

    pseudo_result: PseudoLabelResult | None = None
    if not df_unlabeled.empty:
        X_unlabeled = _prepare_features(artifacts, df_unlabeled, feature_cols)
        pseudo_result = pseudo_label_unlabeled(base_classifier, X_unlabeled)
        combined_df, accepted_df = build_autolabeled_dataset(
            df_labeled_aug,
            df_unlabeled,
            pseudo_result,
            config.confidence_threshold,
            target_col=target_col,
        )
        accepted_df.to_csv(config.auto_labeled_output_path, index=False)
    else:
        combined_df = df_labeled_aug
        accepted_df = pd.DataFrame(columns=df_labeled_aug.columns)
        accepted_df.to_csv(config.auto_labeled_output_path, index=False)

    combined_df.to_csv(config.final_train_path, index=False)

    X_final = _prepare_features(artifacts, combined_df, feature_cols)
    y_final = combined_df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y_final,
        test_size=0.2,
        random_state=config.random_seed,
        stratify=y_final,
    )

    final_classifier = train_final_classifier(X_train, y_train, config.classifier)
    save_classifier(final_classifier, config.classifier_checkpoint_dir / "classifier.joblib")

    metrics = evaluate_classifier(final_classifier, X_test, y_test)
    save_metrics(metrics, config.metrics_path)
    logger.info("Pipeline completed with metrics: %s", metrics)
    return metrics

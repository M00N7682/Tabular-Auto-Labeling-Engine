"""Experiment utilities for sweeping configuration parameters."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

from src.config import CONFIG, PipelineConfig
from src.pipeline import run_full_pipeline

_EXPERIMENTS_DIR = (CONFIG.metrics_path.parent / "experiments").resolve()


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _with_output_suffix(config: PipelineConfig, suffix: str) -> PipelineConfig:
    """Create a new config pointing to isolated artifact paths."""

    base_outputs = CONFIG.synthetic_output_path.parent
    base_models = CONFIG.tabddpm_checkpoint_dir.parent
    exp_outputs = base_outputs / "experiments" / suffix
    exp_models = base_models / "experiments" / suffix

    return replace(
        config,
        synthetic_output_path=exp_outputs / CONFIG.synthetic_output_path.name,
        auto_labeled_output_path=exp_outputs / CONFIG.auto_labeled_output_path.name,
        final_train_path=exp_outputs / CONFIG.final_train_path.name,
        metrics_path=_EXPERIMENTS_DIR / f"metrics_{suffix}.json",
        preprocessor_path=exp_models / f"preprocessor_{suffix}.joblib",
        tabddpm_checkpoint_dir=exp_models / "tabddpm",
        classifier_checkpoint_dir=exp_models / "classifier",
    )


def run_threshold_sweep(
    thresholds: Iterable[float],
    base_config: PipelineConfig = CONFIG,
) -> List[Dict[str, float]]:
    """Run the pipeline across multiple confidence thresholds."""

    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, float]] = []
    run_id = _timestamp()

    for threshold in thresholds:
        suffix = f"thr_{threshold:.2f}_{run_id}"
        config = _with_output_suffix(base_config, suffix)
        config = replace(config, confidence_threshold=threshold)
        metrics = run_full_pipeline(config)
        record: Dict[str, float] = {"threshold": threshold, **metrics}
        results.append(record)

    summary_path = _EXPERIMENTS_DIR / f"threshold_sweep_{run_id}.json"
    summary_path.write_text(json.dumps(results, indent=2))
    return results


def run_unlabeled_ratio_sweep(
    ratios: Iterable[float],
    base_config: PipelineConfig = CONFIG,
) -> List[Dict[str, float]]:
    """Run the pipeline across different simulated unlabeled ratios."""

    _EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, float]] = []
    run_id = _timestamp()

    for ratio in ratios:
        suffix = f"ratio_{ratio:.2f}_{run_id}"
        config = _with_output_suffix(base_config, suffix)
        config = replace(config, simulate_unlabeled_ratio=ratio)
        metrics = run_full_pipeline(config)
        record: Dict[str, float] = {"simulate_unlabeled_ratio": ratio, **metrics}
        results.append(record)

    summary_path = _EXPERIMENTS_DIR / f"ratio_sweep_{run_id}.json"
    summary_path.write_text(json.dumps(results, indent=2))
    return results

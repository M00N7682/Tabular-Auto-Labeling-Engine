"""Central configuration definitions for the TabDDPM auto-labeling pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TabDDPMConfig:
    """Hyperparameters for the TabDDPM model placeholder."""

    epochs: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-3
    noise_steps: int = 1000
    embedding_dim: int = 128
    device: str = "cpu"


@dataclass(frozen=True)
class ClassifierConfig:
    """Hyperparameters for the baseline classifier."""

    model_type: str = "xgboost"
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 200
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    random_state: int = 42


@dataclass(frozen=True)
class PipelineConfig:
    """Container for all pipeline-level configuration values."""

    random_seed: int = 42
    data_path: Path = PROJECT_ROOT / "data" / "digital_marketing_campaign_dataset_few_label.csv"
    synthetic_output_path: Path = PROJECT_ROOT / "outputs" / "synthetic_data.csv"
    auto_labeled_output_path: Path = PROJECT_ROOT / "outputs" / "auto_labeled_data.csv"
    final_train_path: Path = PROJECT_ROOT / "outputs" / "train_final.csv"
    metrics_path: Path = PROJECT_ROOT / "reports" / "metrics.json"
    preprocessor_path: Path = PROJECT_ROOT / "models" / "preprocessor.joblib"
    tabddpm_checkpoint_dir: Path = PROJECT_ROOT / "models" / "tabddpm"
    classifier_checkpoint_dir: Path = PROJECT_ROOT / "models" / "classifier"
    confidence_threshold: float = 0.9
    min_labeled_per_class: int = 50
    synthetic_samples_per_class: int = 200
    simulate_unlabeled_ratio: float = 0.7
    tabddpm: TabDDPMConfig = field(default_factory=TabDDPMConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)

    def model_metadata(self) -> Dict[str, str]:
        return {
            "tabddpm_dir": str(self.tabddpm_checkpoint_dir),
            "classifier_dir": str(self.classifier_checkpoint_dir),
        }


CONFIG = PipelineConfig()

"""TabDDPM trainer module backed by SDV's diffusion synthesizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
 
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TableDiffusionSynthesizer

from src.config import TabDDPMConfig


@dataclass(slots=True)
class TabDDPMModel:
    synthesizer: TableDiffusionSynthesizer
    metadata: SingleTableMetadata
    target_col: str
    config: TabDDPMConfig

    def sample_class(self, cls: object, num_samples: int) -> pd.DataFrame:
        """Sample rows conditioned on the target class."""

        conditions = pd.DataFrame({self.target_col: [cls] * num_samples})
        try:
            generated = self.synthesizer.sample_remaining_columns(conditions=conditions)
        except AttributeError as exc:
            try:
                generated = self.synthesizer.sample_conditions(conditions)
            except AttributeError as second_exc:
                raise RuntimeError("Current SDV version lacks conditional sampling support") from second_exc
        generated[self.target_col] = conditions[self.target_col].to_numpy()
        return generated


def train_tabddpm(
    df_labeled: pd.DataFrame,
    target_col: str,
    config: TabDDPMConfig,
) -> TabDDPMModel:
    """Train an SDV diffusion synthesizer on the labeled dataset."""

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df=df_labeled)

    synthesizer = TableDiffusionSynthesizer(
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        epochs=config.epochs,
        batch_size=config.batch_size,
    )
    synthesizer.fit(df_labeled)

    return TabDDPMModel(
        synthesizer=synthesizer,
        metadata=metadata,
        target_col=target_col,
        config=config,
    )


def save_tabddpm(model: TabDDPMModel, directory: Path) -> None:
    """Persist the trained synthesizer, metadata, and config."""

    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "tabddpm_model.pkl"
    metadata_path = directory / "metadata.json"
    manifest_path = directory / "manifest.json"

    model.synthesizer.save(str(model_path))
    model.metadata.to_json(filepath=str(metadata_path))

    manifest: Dict[str, object] = {
        "target_col": model.target_col,
        "config": model.config.__dict__,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))


def load_tabddpm(directory: Path, config: TabDDPMConfig) -> TabDDPMModel:
    """Load a diffusion synthesizer and associated metadata from disk."""

    model_path = directory / "tabddpm_model.pkl"
    metadata_path = directory / "metadata.json"
    manifest_path = directory / "manifest.json"

    if not model_path.exists() or not metadata_path.exists() or not manifest_path.exists():
        msg = f"Incomplete TabDDPM checkpoint in {directory}" 
        raise FileNotFoundError(msg)

    metadata = SingleTableMetadata.load_from_json(filepath=str(metadata_path))
    synthesizer = TableDiffusionSynthesizer.load(str(model_path))
    manifest = json.loads(manifest_path.read_text())

    return TabDDPMModel(
        config=config,
        synthesizer=synthesizer,
        metadata=metadata,
        target_col=manifest["target_col"],
    )

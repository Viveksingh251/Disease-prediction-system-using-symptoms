from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ModelArtifacts:
    model: Any
    # Feature columns order used during training (binary/multi-hot)
    feature_columns: List[str]
    # Label order used during training
    label_map: List[str]



_MODEL_ARTIFACTS: Optional[ModelArtifacts] = None


def set_artifacts(artifacts: ModelArtifacts) -> None:
    global _MODEL_ARTIFACTS
    _MODEL_ARTIFACTS = artifacts


def get_artifacts() -> Optional[ModelArtifacts]:
    return _MODEL_ARTIFACTS


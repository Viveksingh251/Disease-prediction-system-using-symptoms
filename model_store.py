from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelArtifacts:
    model: Any
    vocabulary: list[str]
    label_map: list[str]


_MODEL_ARTIFACTS: Optional[ModelArtifacts] = None


def set_artifacts(artifacts: ModelArtifacts) -> None:
    global _MODEL_ARTIFACTS
    _MODEL_ARTIFACTS = artifacts


def get_artifacts() -> Optional[ModelArtifacts]:
    return _MODEL_ARTIFACTS


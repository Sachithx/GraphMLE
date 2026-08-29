from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ValueType(str, Enum):
    DATA = "data"
    FEATURES = "features"
    PREDICTIONS = "predictions"
    SUBMISSION = "submission"


@dataclass(frozen=True)
class DataBundle:
    frames: dict[str, pd.DataFrame]
    official_splits: dict[str, list[tuple[Any, ...]]]
    data_dir: Path
    starter_kit_dir: Path


@dataclass(frozen=True)
class FeatureBundle:
    name: str
    frames: dict[str, pd.DataFrame]
    categorical_columns: tuple[str, ...]
    data: DataBundle


@dataclass(frozen=True)
class PredictionBundle:
    name: str
    scores: dict[str, np.ndarray]
    data: DataBundle


@dataclass(frozen=True)
class SubmissionArtifact:
    path: Path
    split: str
    checker_output: str
    predictions: PredictionBundle


@dataclass
class ExecutionContext:
    data_dir: Path
    starter_kit_dir: Path
    output_dir: Path
    seed: int = 0
    cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir.expanduser().resolve()
        self.starter_kit_dir = self.starter_kit_dir.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()

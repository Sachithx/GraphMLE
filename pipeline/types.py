from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


class ValueType(str, Enum):
    DATA = "data"
    FEATURES = "features"
    PREDICTIONS = "predictions"
    SUBMISSION = "submission"


class TemporalScope(str, Enum):
    SAME_ROW = "same_row"
    STRICTLY_EARLIER = "strictly_earlier"
    STATIC = "static"


class SourceLog(str, Enum):
    STANDARD = "standard"
    RANDOMIZED = "randomized"
    SIDE_FILE = "side_file"


@dataclass(frozen=True)
class FeatureLineage:
    sources: frozenset[str]
    temporal_scope: TemporalScope
    source_log: SourceLog = SourceLog.STANDARD
    max_source_date: int | None = None


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
    provenance: Mapping[str, FeatureLineage] = field(default_factory=dict)


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
    run_log_path: Path | None = None
    leakage_guard_enabled: bool = True

    def __post_init__(self) -> None:
        self.data_dir = self.data_dir.expanduser().resolve()
        self.starter_kit_dir = self.starter_kit_dir.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        if self.run_log_path is not None:
            self.run_log_path = self.run_log_path.expanduser().resolve()

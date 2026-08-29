from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from guards.leakage import LeakageError, LeakageGuard
from pipeline.execute import execute_graph
from pipeline.graph import OperatorGraph
from pipeline.registry import OperatorRegistry, OperatorSpec
from pipeline.types import (
    DataBundle,
    ExecutionContext,
    FeatureBundle,
    FeatureLineage,
    SourceLog,
    TemporalScope,
    ValueType,
)


def tiny_data(tmp_path: Path) -> DataBundle:
    frames = {
        split: pd.DataFrame(
            {
                "row_id": [0, 1],
                "user_id": ["u", "u"],
                "long_view": [0, 1],
                "is_click": [0, 1],
            }
        )
        for split in ("train", "valid", "test")
    }
    return DataBundle(frames, {}, tmp_path, tmp_path)


def feature_bundle(
    data: DataBundle, column: str, lineage: FeatureLineage | None
) -> FeatureBundle:
    frames = {
        split: pd.DataFrame({"row_id": [0, 1], column: [0.0, 1.0]})
        for split in ("train", "valid", "test")
    }
    provenance = {} if lineage is None else {column: lineage}
    return FeatureBundle("candidate", frames, (), data, provenance)


def test_same_row_outcome_is_rejected_and_logged(tmp_path: Path) -> None:
    log_path = tmp_path / "run_log.jsonl"
    guard = LeakageGuard(log_path)
    bundle = feature_bundle(
        tiny_data(tmp_path),
        "click_copy",
        FeatureLineage(frozenset({"is_click"}), TemporalScope.SAME_ROW),
    )

    with pytest.raises(LeakageError, match="same-row"):
        guard.check_feature_bundle(bundle, node_id="leaky")

    record = json.loads(log_path.read_text().strip())
    assert record["event"] == "leakage_rejection"
    assert record["layer"] == "static"
    assert record["node_id"] == "leaky"


def test_prior_date_outcome_aggregate_is_allowed(tmp_path: Path) -> None:
    guard = LeakageGuard(tmp_path / "run_log.jsonl")
    bundle = feature_bundle(
        tiny_data(tmp_path),
        "prior_click_rate",
        FeatureLineage(frozenset({"is_click"}), TemporalScope.STRICTLY_EARLIER),
    )
    guard.check_feature_bundle(bundle, node_id="history")


def test_unknown_provenance_is_rejected(tmp_path: Path) -> None:
    guard = LeakageGuard(tmp_path / "run_log.jsonl")
    with pytest.raises(LeakageError, match="missing lineage"):
        guard.check_feature_bundle(
            feature_bundle(tiny_data(tmp_path), "mystery", None), node_id="generated"
        )


def test_auxiliary_targets_are_explicitly_allowed_as_labels(tmp_path: Path) -> None:
    guard = LeakageGuard(tmp_path / "run_log.jsonl")
    guard.check_auxiliary_targets(["is_click", "is_like", "play_time_ms"])


def test_randomized_log_must_be_restricted_to_training_dates(tmp_path: Path) -> None:
    guard = LeakageGuard(tmp_path / "run_log.jsonl")
    leaking = feature_bundle(
        tiny_data(tmp_path),
        "randomized_prior",
        FeatureLineage(
            frozenset({"video_id"}),
            TemporalScope.STRICTLY_EARLIER,
            SourceLog.RANDOMIZED,
            max_source_date=20220428,
        ),
    )
    with pytest.raises(LeakageError, match="randomized-exposure"):
        guard.check_feature_bundle(leaking, node_id="randomized")

    safe = feature_bundle(
        tiny_data(tmp_path),
        "randomized_train_prior",
        FeatureLineage(
            frozenset({"video_id"}),
            TemporalScope.STRICTLY_EARLIER,
            SourceLog.RANDOMIZED,
            max_source_date=20220421,
        ),
    )
    guard.check_feature_bundle(safe, node_id="randomized_train_only")


def test_empirical_tripwire_rejects_implausible_primary(tmp_path: Path) -> None:
    guard = LeakageGuard(tmp_path / "run_log.jsonl")
    with pytest.raises(LeakageError, match="0.80"):
        guard.check_metrics({"primary": 0.81}, node_id="model")


def test_execution_guard_stops_before_model_training(tmp_path: Path) -> None:
    data = tiny_data(tmp_path)
    model_called = False

    def source(inputs, params, ctx):
        return data

    def leaky(inputs, params, ctx):
        return feature_bundle(
            data,
            "same_row_click",
            FeatureLineage(frozenset({"is_click"}), TemporalScope.SAME_ROW),
        )

    def model(inputs, params, ctx):
        nonlocal model_called
        model_called = True
        raise AssertionError("model must not run")

    registry = OperatorRegistry()
    registry.register(OperatorSpec("data.synthetic", (), ValueType.DATA, source))
    registry.register(
        OperatorSpec(
            "features.leaky", (ValueType.DATA,), ValueType.FEATURES, leaky
        )
    )
    registry.register(
        OperatorSpec(
            "model.marker", (ValueType.FEATURES,), ValueType.PREDICTIONS, model
        )
    )
    graph = OperatorGraph.from_dict(
        {
            "nodes": [
                {"id": "load", "type": "data.synthetic"},
                {"id": "leaky", "type": "features.leaky", "inputs": ["load"]},
                {"id": "model", "type": "model.marker", "inputs": ["leaky"]},
            ]
        }
    )

    with pytest.raises(LeakageError):
        execute_graph(
            graph,
            ExecutionContext(tmp_path, tmp_path, tmp_path / "run"),
            registry,
        )
    assert not model_called

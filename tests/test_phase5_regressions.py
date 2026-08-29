from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.run import AgentRunner, RunConfig
from pipeline.execute import execute_graph
from pipeline.graph import OperatorGraph
from pipeline.registry import OperatorRegistry, OperatorSpec
from pipeline.types import (
    DataBundle,
    ExecutionContext,
    PredictionBundle,
    ValueType,
)


def hypothesis(hypothesis_id: str, expected_delta: float, node_type: str) -> dict:
    return {
        "id": hypothesis_id,
        "target_node": "model",
        "rationale": "bounded validation regression",
        "method_source": "unit test",
        "expected_delta": expected_delta,
        "expected_cost_minutes": 1,
        "patch": {
            "op": "add_node",
            "replace_node": "model",
            "node": {
                "id": "model",
                "type": node_type,
                "params": {"seed": 0},
                "inputs": ["raw", "duration", "temporal"],
            },
        },
    }


def valid_hypothesis(hypothesis_id: str, expected_delta: float) -> dict:
    return {
        "id": hypothesis_id,
        "target_node": "model",
        "rationale": "valid bounded proposal",
        "method_source": "unit test",
        "expected_delta": expected_delta,
        "expected_cost_minutes": 1,
        "patch": {
            "op": "replace_params",
            "node": "model",
            "params": {"seed": 7},
        },
    }


def run_config(hypotheses: list[dict], reproposals: int) -> RunConfig:
    return RunConfig.model_validate(
        {
            "seed_graph": "configs/pipeline_seed.json",
            "loop": {
                "max_iterations": 1,
                "max_wall_clock_s": 30,
                "iteration_timeout_s": 5,
                "memory_limit_mb": 512,
                "convergence_window": 1,
                "convergence_delta": 0.002,
                "significance_threshold": 0.0016,
                "confirm_small_deltas": False,
                "validation_reproposal_attempts": reproposals,
                "repair_attempts": 1,
            },
            "evaluation": {"mode": "synthetic", "baseline_primary": 0.6016},
            "llm": {"mode": "canned", "hypotheses": hypotheses},
        }
    )


def test_unknown_type_reproposes_without_becoming_a_convergence_delta(
    tmp_path: Path,
) -> None:
    config = run_config(
        [
            hypothesis("invalid_1", 0.02, "model.not_registered"),
            valid_hypothesis("valid_1", 0.01),
        ],
        reproposals=1,
    )
    run_dir = tmp_path / "unknown_type"
    summary = AgentRunner(config, run_dir=run_dir).run()
    record = json.loads((run_dir / "run_log.jsonl").read_text())

    assert summary["stop_reason"] == "iteration_cap"
    assert summary["executed_iterations"] == 1
    assert summary["rejected_proposals"] == 1
    assert record["executed"] is True
    assert record["proposal_rejections"][0]["hypothesis_id"] == "invalid_1"
    assert record["delta_vs_incumbent"] == pytest.approx(0.01)


def test_three_consecutive_invalid_proposals_do_not_terminate_the_run(
    tmp_path: Path,
) -> None:
    config = run_config(
        [
            hypothesis("invalid_1", 0.04, "model.unknown_one"),
            hypothesis("invalid_2", 0.03, "model.unknown_two"),
            hypothesis("invalid_3", 0.02, "model.unknown_three"),
            valid_hypothesis("valid_after_three", 0.01),
        ],
        reproposals=3,
    )
    run_dir = tmp_path / "three_invalid"
    summary = AgentRunner(config, run_dir=run_dir).run()
    record = json.loads((run_dir / "run_log.jsonl").read_text())

    assert summary["stop_reason"] == "iteration_cap"
    assert summary["executed_iterations"] == 1
    assert summary["rejected_proposals"] == 3
    assert record["hypothesis"]["id"] == "valid_after_three"
    assert record["executed"] is True
    assert len(record["proposal_rejections"]) == 3


def test_phase5_metric_firewall_scores_validation_only(
    tmp_path: Path, monkeypatch
) -> None:
    frames = {
        split: pd.DataFrame(
            {"row_id": [0, 1], "user_id": ["u", "u"], "long_view": [0, 1]}
        )
        for split in ("train", "valid", "test")
    }
    data = DataBundle(frames, {}, tmp_path, tmp_path)
    prediction = PredictionBundle(
        "synthetic",
        {split: np.array([0.1, 0.9]) for split in frames},
        data,
    )
    registry = OperatorRegistry()
    registry.register(
        OperatorSpec(
            "model.synthetic",
            (),
            ValueType.PREDICTIONS,
            lambda inputs, params, ctx: prediction,
        )
    )
    graph = OperatorGraph.from_dict(
        {"nodes": [{"id": "model", "type": "model.synthetic"}]}
    )
    scored_splits: list[str] = []

    def fake_evaluate(user_ids, labels, scores, starter_kit_dir):
        del user_ids, labels, scores, starter_kit_dir
        scored_splits.append("called")
        return {"gauc": 0.7, "ndcg5": 0.6, "primary": 0.65}

    monkeypatch.setattr("pipeline.execute.evaluate_official", fake_evaluate)
    output = tmp_path / "firewall"
    result = execute_graph(
        graph,
        ExecutionContext(
            tmp_path,
            tmp_path,
            output,
            metric_splits=("valid",),
        ),
        registry,
    )

    assert list(result.metrics) == ["valid"]
    assert len(scored_splits) == 1
    assert "test" not in json.loads((output / "metrics.json").read_text())["metrics"]

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from guards.leakage import LeakageError
from guards.sandbox import run_in_sandbox
from pipeline.execute import execute_graph
from pipeline.graph import OperatorGraph
from pipeline.registry import OperatorRegistry, OperatorSpec
from pipeline.types import (
    DataBundle,
    ExecutionContext,
    FeatureBundle,
    FeatureLineage,
    PredictionBundle,
    TemporalScope,
    ValueType,
)

from .significance import decide_significance


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_data(root: Path) -> DataBundle:
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
    return DataBundle(frames, {}, root, root)


def _leaky_registry(data: DataBundle, marker: dict[str, bool]) -> OperatorRegistry:
    def source(inputs, params, ctx):
        return data

    def leaky_feature(inputs, params, ctx):
        frames = {
            split: pd.DataFrame(
                {
                    "row_id": frame["row_id"].to_numpy(),
                    "same_row_click": frame["is_click"].to_numpy(),
                }
            )
            for split, frame in data.frames.items()
        }
        return FeatureBundle(
            "deliberately_leaky",
            frames,
            (),
            data,
            {
                "same_row_click": FeatureLineage(
                    frozenset({"is_click"}), TemporalScope.SAME_ROW
                )
            },
        )

    def model(inputs, params, ctx):
        marker["model_called"] = True
        return PredictionBundle(
            "marker",
            {split: np.zeros(len(frame)) for split, frame in data.frames.items()},
            data,
        )

    registry = OperatorRegistry()
    registry.register(OperatorSpec("data.synthetic", (), ValueType.DATA, source))
    registry.register(
        OperatorSpec(
            "features.leaky_same_row",
            (ValueType.DATA,),
            ValueType.FEATURES,
            leaky_feature,
        )
    )
    registry.register(
        OperatorSpec(
            "model.marker", (ValueType.FEATURES,), ValueType.PREDICTIONS, model
        )
    )
    return registry


def run_gate(output_dir: Path, config_dir: Path = ROOT / "configs") -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = {"model_called": False}
    leakage_log = output_dir / "leaky" / "run_log.jsonl"
    try:
        execute_graph(
            OperatorGraph.from_path(config_dir / "phase3_leaky.json"),
            ExecutionContext(
                ROOT,
                ROOT,
                output_dir / "leaky",
                run_log_path=leakage_log,
            ),
            _leaky_registry(_synthetic_data(ROOT), marker),
        )
    except LeakageError as exc:
        leakage_record = exc.record
    else:
        raise RuntimeError("deliberately leaky graph was not rejected")
    if marker["model_called"]:
        raise RuntimeError("leaky graph reached model training")

    broken_dir = output_dir / "broken" / "iterations" / "000"
    sandbox_result = run_in_sandbox(
        [
            sys.executable,
            "-m",
            "pipeline.execute",
            str(config_dir / "phase3_broken.json"),
            "--output-dir",
            str(output_dir / "broken" / "execution"),
        ],
        workdir=ROOT,
        stdout_path=broken_dir / "stdout.log",
        timeout_s=30,
        memory_limit_mb=4096,
    )
    broken_output = sandbox_result.stdout_path.read_text()
    if sandbox_result.status != "failed" or "unknown node type" not in broken_output:
        raise RuntimeError(
            f"sandbox did not capture the broken graph correctly: {sandbox_result.status}"
        )

    large_delta = decide_significance(0.6000, 0.60161)
    confirmed_small_delta = decide_significance(
        0.6000,
        0.6005,
        seed_scores=[0.6005, 0.6002, 0.5999],
    )
    rejected_noise = decide_significance(0.6000, 0.6005)
    if not large_delta.accepted or not confirmed_small_delta.accepted or rejected_noise.accepted:
        raise RuntimeError("significance acceptance examples did not produce expected decisions")

    sandbox_payload = asdict(sandbox_result)
    sandbox_payload["stdout_path"] = str(sandbox_result.stdout_path)
    sandbox_payload["result_path"] = str(sandbox_result.result_path)
    summary = {
        "status": "passed",
        "leakage": {
            "rejected_before_training": True,
            "model_called": marker["model_called"],
            "log_path": str(leakage_log),
            "record": leakage_record,
        },
        "sandbox": sandbox_payload,
        "significance": {
            "large_delta": asdict(large_delta),
            "confirmed_small_delta": asdict(confirmed_small_delta),
            "rejected_noise": asdict(rejected_noise),
        },
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 3 acceptance gate")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "phase3")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    args = parser.parse_args()
    run_gate(args.output_dir, args.config_dir)


if __name__ == "__main__":
    main()

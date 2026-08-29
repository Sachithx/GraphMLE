from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline.execute import execute_graph
from pipeline.graph import OperatorGraph
from pipeline.registry import default_registry
from pipeline.types import ExecutionContext, SubmissionArtifact


ROOT = Path(__file__).resolve().parents[1]


def run_gate(
    config_dir: Path,
    data_dir: Path,
    starter_kit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    paths = sorted(config_dir.glob("phase2_*.json"))
    if len(paths) != 3:
        raise RuntimeError(f"expected exactly three Phase 2 graphs, found {len(paths)}")
    registry = default_registry()
    shared_cache: dict[str, Any] = {}
    results = []
    for path in paths:
        graph = OperatorGraph.from_path(path)
        graph.validate(registry)
        graph_output = output_dir / path.stem
        result = execute_graph(
            graph,
            ExecutionContext(
                data_dir,
                starter_kit_dir,
                graph_output,
                cache=shared_cache,
            ),
            registry,
        )
        if not isinstance(result.terminal, SubmissionArtifact):
            raise RuntimeError(f"graph {path.name} did not produce a submission")
        record = {
            "graph": path.name,
            "metrics": result.metrics,
            "submission": str(result.terminal.path),
            "submission_check": result.terminal.checker_output,
            "wall_clock_s": result.wall_clock_s,
        }
        results.append(record)
        print(json.dumps(record, indent=2))
    primary_scores = [round(item["metrics"]["valid"]["primary"], 6) for item in results]
    if len(set(primary_scores)) != 3:
        raise RuntimeError(f"Phase 2 graphs did not produce three distinct scores: {primary_scores}")
    summary = {"status": "passed", "primary_scores": primary_scores, "graphs": results}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 2 acceptance gate")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "configs")
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data" / "kuairand-pure" / "data"
    )
    parser.add_argument(
        "--starter-kit", type=Path, default=ROOT / "data" / "starter-kit"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "phase2")
    args = parser.parse_args()
    run_gate(args.config_dir, args.data_dir, args.starter_kit, args.output_dir)


if __name__ == "__main__":
    main()

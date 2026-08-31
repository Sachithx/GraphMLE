from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eval.official import evaluate_official
from guards.leakage import LeakageGuard

from .graph import OperatorGraph
from .registry import OperatorRegistry, default_registry
from .types import (
    DataBundle,
    ExecutionContext,
    FeatureBundle,
    PredictionBundle,
    SubmissionArtifact,
    ValueType,
)


@dataclass(frozen=True)
class ExecutionResult:
    graph: OperatorGraph
    outputs: dict[str, Any]
    metrics: dict[str, dict[str, float]]
    terminal: Any
    wall_clock_s: float


_RUNTIME_TYPES = {
    ValueType.DATA: DataBundle,
    ValueType.FEATURES: FeatureBundle,
    ValueType.PREDICTIONS: PredictionBundle,
    ValueType.SUBMISSION: SubmissionArtifact,
}


def op_submit_rank(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> SubmissionArtifact:
    if len(inputs) != 1 or not isinstance(inputs[0], PredictionBundle):
        raise TypeError("submit.rank requires one PredictionBundle")
    predictions = inputs[0]
    split = str(params.get("split", "test"))
    if split not in ("valid", "test"):
        raise ValueError("submission split must be valid or test")
    filename = str(params.get("filename", "submission.csv"))
    if Path(filename).name != filename:
        raise ValueError("submission filename must not contain a directory")
    output = ctx.output_dir / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = predictions.data.official_splits[split]
    scores = np.asarray(predictions.scores[split], dtype=float)
    if len(scores) != len(rows) or not np.isfinite(scores).all():
        raise ValueError("submission scores have the wrong length or contain NaN/Inf")
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "user_id", "video_id", "score"])
        for row_id, (row, score) in enumerate(zip(rows, scores)):
            writer.writerow([row_id, row[1], row[2], f"{float(score):.12g}"])
    command = [
        sys.executable,
        str(ctx.starter_kit_dir / "submit.py"),
        str(output),
        "--check",
        "--split",
        split,
        "--data_dir",
        str(ctx.data_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=ctx.starter_kit_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=int(params.get("check_timeout_s", 300)),
    )
    if completed.returncode:
        raise RuntimeError(f"official submission check failed:\n{completed.stdout}")
    return SubmissionArtifact(output, split, completed.stdout.strip(), predictions)


def _prediction_from_terminal(terminal: Any) -> PredictionBundle | None:
    if isinstance(terminal, PredictionBundle):
        return terminal
    if isinstance(terminal, SubmissionArtifact):
        return terminal.predictions
    return None


def _dependency_levels(order: tuple[Any, ...]) -> list[list[Any]]:
    """Group a topological order into levels of mutually independent nodes.

    Nodes in the same level have no path between them, so they may run
    concurrently. A seed bag puts its replicas in one level, which is where the
    parallelism actually pays.
    """
    depth: dict[str, int] = {}
    levels: dict[int, list[Any]] = {}
    for node in order:
        level = max((depth[i] + 1 for i in node.inputs), default=0)
        depth[node.id] = level
        levels.setdefault(level, []).append(node)
    return [levels[key] for key in sorted(levels)]


def _prewarm_shared_state(ctx: ExecutionContext) -> None:
    """Populate caches whose first population is not thread-safe.

    The starter-kit loader mutates sys.path and sys.modules, and the official
    evaluator memoises a module load. Racing either from worker threads is
    unsafe, so both are forced once on the main thread before any parallel work.
    """
    from .ops_models import _load_kit_runtime

    try:
        _load_kit_runtime(ctx)
    except Exception:
        # Absent in synthetic fixtures; parallel execution then has nothing to race.
        pass


def execute_graph(
    graph: OperatorGraph,
    ctx: ExecutionContext,
    registry: OperatorRegistry | None = None,
) -> ExecutionResult:
    registry = registry or default_registry()
    order = graph.validate(registry)
    leakage_guard = (
        LeakageGuard(ctx.run_log_path or ctx.output_dir / "run_log.jsonl")
        if ctx.leakage_guard_enabled
        else None
    )
    start = time.monotonic()
    outputs: dict[str, Any] = {}
    guard_lock = threading.Lock()
    max_workers = max(1, int(getattr(ctx, "max_parallel_nodes", 1)))

    def run_node(node: Any) -> tuple[str, Any]:
        spec = registry[node.type]
        value = spec.function(
            [outputs[input_id] for input_id in node.inputs], dict(node.params), ctx
        )
        expected = _RUNTIME_TYPES[spec.output_type]
        if not isinstance(value, expected):
            raise TypeError(
                f"operator {node.id} returned {type(value).__name__}, expected {expected.__name__}"
            )
        if leakage_guard is not None and isinstance(value, FeatureBundle):
            # The guard appends to a shared log, so serialise the check itself.
            with guard_lock:
                leakage_guard.check_feature_bundle(value, node_id=node.id)
        return node.id, value

    if max_workers > 1:
        _prewarm_shared_state(ctx)
        for level in _dependency_levels(order):
            if len(level) == 1:
                node_id, value = run_node(level[0])
                outputs[node_id] = value
                continue
            with ThreadPoolExecutor(max_workers=min(max_workers, len(level))) as pool:
                for node_id, value in pool.map(run_node, level):
                    outputs[node_id] = value
    else:
        for node in order:
            node_id, value = run_node(node)
            outputs[node_id] = value
    submit_nodes = [node for node in order if node.type.startswith("submit.")]
    if len(submit_nodes) > 1:
        raise ValueError("graph must not contain multiple submission nodes")
    terminal_node = submit_nodes[0] if submit_nodes else order[-1]
    terminal = outputs[terminal_node.id]
    prediction = _prediction_from_terminal(terminal)
    metrics: dict[str, dict[str, float]] = {}
    if prediction is not None:
        for split in ctx.metric_splits:
            frame = prediction.data.frames[split]
            metrics[split] = evaluate_official(
                frame["user_id"].tolist(),
                frame["long_view"].to_numpy(),
                prediction.scores[split],
                starter_kit_dir=ctx.starter_kit_dir,
            )
            if leakage_guard is not None and split == "valid":
                leakage_guard.check_metrics(metrics[split], node_id=prediction.name)
    elapsed = time.monotonic() - start
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    (ctx.output_dir / "graph.json").write_text(json.dumps(graph.to_dict(), indent=2))
    (ctx.output_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "wall_clock_s": elapsed}, indent=2)
    )
    return ExecutionResult(graph, outputs, metrics, terminal, elapsed)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Execute one typed operator graph")
    parser.add_argument("graph", type=Path)
    parser.add_argument(
        "--data-dir", type=Path, default=root / "data" / "kuairand-pure" / "data"
    )
    parser.add_argument(
        "--starter-kit", type=Path, default=root / "data" / "starter-kit"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "runs" / "manual")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metric-splits",
        nargs="+",
        choices=("valid", "test"),
        default=("valid", "test"),
        help="labeled splits to score; Phase 5 passes only valid",
    )
    parser.add_argument(
        "--max-parallel-nodes",
        type=int,
        default=1,
        help="run mutually independent nodes concurrently; 1 is strictly sequential",
    )
    args = parser.parse_args()
    result = execute_graph(
        OperatorGraph.from_path(args.graph),
        ExecutionContext(
            args.data_dir,
            args.starter_kit,
            args.output_dir,
            args.seed,
            metric_splits=tuple(args.metric_splits),
            max_parallel_nodes=args.max_parallel_nodes,
        ),
    )
    print(json.dumps({"metrics": result.metrics, "wall_clock_s": result.wall_clock_s}, indent=2))


if __name__ == "__main__":
    main()

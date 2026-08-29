from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Protocol

from pipeline.graph import OperatorGraph
from pipeline.registry import OperatorRegistry

from .llm import StructuredClient, TokenUsage
from .propose import (
    EXAMPLE_GRAPHS,
    Hypothesis,
    apply_hypothesis,
    llm_hypothesis_schema,
)


Metrics = dict[str, float]
Evaluator = Callable[[OperatorGraph], Metrics]


class RepairProvider(Protocol):
    def repair(self, graph: OperatorGraph, error: str, attempt: int) -> OperatorGraph: ...


class CannedRepairProvider:
    """Offline repair used by the acceptance gate; removes only the test sentinel."""

    usage = TokenUsage()

    def repair(self, graph: OperatorGraph, error: str, attempt: int) -> OperatorGraph:
        del error, attempt
        raw = graph.to_dict()
        changed = False
        for node in raw["nodes"]:
            if node["params"].pop("induce_failure", None) is not None:
                changed = True
        if not changed:
            raise RuntimeError("canned repair has no recognized failure sentinel")
        return OperatorGraph.from_dict(raw)


class LLMRepairProvider:
    def __init__(
        self,
        client: StructuredClient,
        registry: OperatorRegistry,
        failing_hypothesis: Hypothesis,
        generated_feature_dir: Path | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.failing_hypothesis = failing_hypothesis
        self.generated_feature_dir = generated_feature_dir
        self.schema = llm_hypothesis_schema(registry)
        self.usage = TokenUsage()

    def repair(self, graph: OperatorGraph, error: str, attempt: int) -> OperatorGraph:
        payload = {
            "attempt": attempt,
            "traceback_or_error": error,
            "failing_patch": self.failing_hypothesis.patch.model_dump(),
            "failing_graph": graph.to_dict(),
            "operator_catalog": self.registry.catalog(),
            "valid_graph_examples": EXAMPLE_GRAPHS,
        }
        result = self.client.parse(
            self.schema,
            instructions=(
                "Repair the failing bounded graph patch. Return one of the five allowed "
                "patch operations, make the smallest change that addresses the traceback, "
                "and do not add unrelated experiments. Use only types and parameters in "
                "operator_catalog and respect all input/output signatures."
            ),
            input_text=json.dumps(payload, sort_keys=True),
        )
        self.usage = self.usage + result.usage
        return apply_hypothesis(
            graph,
            result.value.to_runtime(),
            self.registry,
            generated_feature_dir=self.generated_feature_dir,
        ).graph


@dataclass(frozen=True)
class RepairOutcome:
    recovered: bool
    graph: OperatorGraph
    metrics: Metrics | None
    events: list[dict[str, Any]]
    errors: list[str]


class RepairManager:
    def __init__(self, provider: RepairProvider, *, max_attempts: int = 3) -> None:
        self.provider = provider
        self.max_attempts = int(max_attempts)

    def recover(
        self,
        graph: OperatorGraph,
        initial_error: Exception,
        evaluate: Evaluator,
    ) -> RepairOutcome:
        current = graph
        error = initial_error
        errors = [f"{type(error).__name__}: {error}"]
        events: list[dict[str, Any]] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                current = self.provider.repair(current, errors[-1], attempt)
                metrics = evaluate(current)
                events.append(
                    {
                        "attempt": attempt,
                        "status": "recovered",
                        "error": errors[-1],
                    }
                )
                return RepairOutcome(True, current, metrics, events, errors)
            except Exception as exc:
                error = exc
                errors.append(f"{type(exc).__name__}: {exc}")
                events.append(
                    {
                        "attempt": attempt,
                        "status": "failed",
                        "error": errors[-1],
                    }
                )
        return RepairOutcome(False, current, None, events, errors)

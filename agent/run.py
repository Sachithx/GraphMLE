from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from eval.significance import ACCEPTANCE_THRESHOLD, decide_significance
from guards.sandbox import run_in_sandbox
from pipeline.graph import OperatorGraph
from pipeline.registry import OperatorRegistry, default_registry

from .ablate import Ablator
from .llm import OpenAIStructuredClient, TokenUsage
from .memory import AgentMemory
from .propose import (
    CannedHypothesisProposer,
    Hypothesis,
    LiveHypothesisProposer,
    ProposalResult,
    apply_hypothesis,
    graph_diff,
    topology_signature,
)
from .repair import CannedRepairProvider, LLMRepairProvider, RepairManager
from .scheduler import AgentScheduler, ArmObservation


ROOT = Path(__file__).resolve().parents[1]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoopConfig(ConfigModel):
    max_iterations: int = 50
    max_wall_clock_s: float = 21_600
    iteration_timeout_s: float = 900
    memory_limit_mb: int = 4096
    convergence_window: int = 3
    convergence_delta: float = 0.002
    significance_threshold: float = ACCEPTANCE_THRESHOLD
    confirm_small_deltas: bool = True
    repair_attempts: int = 3


class EvaluationConfig(ConfigModel):
    mode: Literal["production", "synthetic"] = "production"
    data_dir: str = "data/kuairand-pure/data"
    starter_kit_dir: str = "data/starter-kit"
    baseline_primary: float = 0.6016


class LLMConfig(ConfigModel):
    mode: Literal["live", "canned"] = "live"
    model: str = "gpt-5.4"
    max_retries: int = 3
    hypotheses: list[Hypothesis] = Field(default_factory=list)


class RunConfig(ConfigModel):
    seed_graph: str = "configs/pipeline_seed.json"
    loop: LoopConfig = Field(default_factory=LoopConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    dry_run_hypotheses: list[Hypothesis] = Field(default_factory=list)


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    wall_clock_s: float


class Evaluator(Protocol):
    def evaluate(
        self,
        graph: OperatorGraph,
        output_dir: Path,
        *,
        incumbent_primary: float,
        seed: int = 0,
    ) -> EvaluationResult: ...

    def confirm_seeds(
        self,
        graph: OperatorGraph,
        output_dir: Path,
        *,
        incumbent_primary: float,
    ) -> list[float]: ...

    def ablate(
        self, graph: OperatorGraph, node_id: str, incumbent_primary: float
    ) -> float: ...


class SyntheticEvaluator:
    """Deterministic orchestration test double; never presented as a model result."""

    def __init__(self, baseline_primary: float) -> None:
        self.baseline_primary = float(baseline_primary)

    @staticmethod
    def _metrics(primary: float) -> dict[str, float]:
        return {
            "gauc": float(primary + 0.065),
            "ndcg5": float(primary - 0.065),
            "primary": float(primary),
        }

    def evaluate(
        self,
        graph: OperatorGraph,
        output_dir: Path,
        *,
        incumbent_primary: float,
        seed: int = 0,
    ) -> EvaluationResult:
        del seed
        start = time.monotonic()
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "stdout.log").open("a") as output:
            if any(node.params.get("induce_failure") for node in graph.nodes):
                output.write("RuntimeError: induced failure for Phase 4 recovery gate\n")
                raise RuntimeError("induced failure for Phase 4 recovery gate")
            expected_delta = float(graph.meta.get("expected_delta", 0.0))
            primary = (
                self.baseline_primary
                if graph.meta.get("hypothesis_id") == "seed_fm"
                else incumbent_primary + expected_delta
            )
            metrics = self._metrics(primary)
            output.write(json.dumps({"synthetic": True, "metrics": metrics}) + "\n")
        (output_dir / "graph.json").write_text(json.dumps(graph.to_dict(), indent=2))
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return EvaluationResult(metrics, time.monotonic() - start)

    def confirm_seeds(
        self,
        graph: OperatorGraph,
        output_dir: Path,
        *,
        incumbent_primary: float,
    ) -> list[float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        delta = max(float(graph.meta.get("expected_delta", 0.0)), 0.0)
        scores = [
            incumbent_primary + delta,
            incumbent_primary - delta,
            incumbent_primary - delta,
        ]
        (output_dir / "seed_scores.json").write_text(json.dumps(scores, indent=2))
        return scores

    def ablate(
        self, graph: OperatorGraph, node_id: str, incumbent_primary: float
    ) -> float:
        node = next(node for node in graph.nodes if node.id == node_id)
        if node.type.startswith("model."):
            cost = 0.030
        elif node.type.startswith("features."):
            cost = 0.012
        elif node.type.startswith("data."):
            cost = 0.002
        else:
            cost = 0.0
        return incumbent_primary - cost


def neutralize_node(
    graph: OperatorGraph, node_id: str, registry: OperatorRegistry
) -> OperatorGraph:
    raw = graph.to_dict()
    nodes = raw["nodes"]
    target = next((node for node in nodes if node["id"] == node_id), None)
    if target is None:
        raise ValueError(f"unknown ablation node {node_id}")
    kind = str(target["type"]).split(".", 1)[0]
    if kind == "data":
        raise ValueError("data source cannot be neutralized while preserving graph types")
    if kind == "features":
        target["type"] = "features.ablation_constant"
        target["params"] = {}
    elif kind == "model":
        target["type"] = "model.ablation_constant"
        target["params"] = {}
    elif kind == "ensemble":
        if not target["inputs"]:
            raise ValueError("ensemble has no bypass input")
        bypass = target["inputs"][0]
        for node in nodes:
            node["inputs"] = [bypass if value == node_id else value for value in node["inputs"]]
        raw["nodes"] = [node for node in nodes if node["id"] != node_id]
    elif kind == "submit":
        raw["nodes"] = [node for node in nodes if node["id"] != node_id]
    else:
        raise ValueError(f"unsupported ablation node type {target['type']}")
    raw["meta"] = {**raw.get("meta", {}), "ablation_node": node_id}
    neutral = OperatorGraph.from_dict(raw)
    neutral.validate(registry)
    return neutral


class ProductionEvaluator:
    def __init__(
        self,
        *,
        root: Path,
        run_dir: Path,
        config: EvaluationConfig,
        loop: LoopConfig,
        registry: OperatorRegistry,
    ) -> None:
        self.root = root
        self.run_dir = run_dir
        self.data_dir = (root / config.data_dir).resolve()
        self.starter_kit_dir = (root / config.starter_kit_dir).resolve()
        self.loop = loop
        self.registry = registry

    def evaluate(
        self,
        graph: OperatorGraph,
        output_dir: Path,
        *,
        incumbent_primary: float,
        seed: int = 0,
    ) -> EvaluationResult:
        del incumbent_primary
        output_dir.mkdir(parents=True, exist_ok=True)
        input_graph = output_dir / "candidate_graph.json"
        input_graph.write_text(json.dumps(graph.to_dict(), indent=2))
        command = [
            str(Path(sys.executable).resolve()),
            "-m",
            "pipeline.execute",
            str(input_graph),
            "--data-dir",
            str(self.data_dir),
            "--starter-kit",
            str(self.starter_kit_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(seed),
        ]
        sandbox = run_in_sandbox(
            command,
            workdir=self.root,
            stdout_path=output_dir / "stdout.log",
            timeout_s=self.loop.iteration_timeout_s,
            memory_limit_mb=self.loop.memory_limit_mb,
        )
        if sandbox.status != "success":
            tail = (output_dir / "stdout.log").read_text(errors="replace")[-8_000:]
            raise RuntimeError(
                f"sandbox {sandbox.status} (returncode={sandbox.returncode})\n{tail}"
            )
        payload = json.loads((output_dir / "metrics.json").read_text())
        return EvaluationResult(
            {key: float(value) for key, value in payload["metrics"]["valid"].items()},
            sandbox.wall_clock_s,
        )

    def confirm_seeds(
        self,
        graph: OperatorGraph,
        output_dir: Path,
        *,
        incumbent_primary: float,
    ) -> list[float]:
        return [
            self.evaluate(
                graph,
                output_dir / f"seed_{seed}",
                incumbent_primary=incumbent_primary,
                seed=seed,
            ).metrics["primary"]
            for seed in (11, 29, 47)
        ]

    def ablate(
        self, graph: OperatorGraph, node_id: str, incumbent_primary: float
    ) -> float:
        neutral = neutralize_node(graph, node_id, self.registry)
        signature = topology_signature(graph)[:12]
        result = self.evaluate(
            neutral,
            self.run_dir / "ablations" / signature / node_id,
            incumbent_primary=incumbent_primary,
        )
        return result.metrics["primary"]


class AgentRunner:
    def __init__(
        self,
        config: RunConfig,
        *,
        run_dir: Path,
        root: Path = ROOT,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.root = root.resolve()
        self.run_dir = run_dir.resolve()
        self.registry = default_registry()
        self.seed_graph = OperatorGraph.from_path(self.root / config.seed_graph)
        self.seed_graph.validate(self.registry)
        self.dry_run = dry_run

    def _proposer(self) -> tuple[Any, Any | None]:
        mode = "canned" if self.dry_run else self.config.llm.mode
        hypotheses = (
            self.config.dry_run_hypotheses
            if self.dry_run
            else self.config.llm.hypotheses
        )
        if mode == "canned":
            if not hypotheses:
                raise ValueError("canned mode requires hypotheses")
            return CannedHypothesisProposer(hypotheses), None
        client = OpenAIStructuredClient(
            model=self.config.llm.model,
            max_retries=self.config.llm.max_retries,
        )
        return LiveHypothesisProposer(client), client

    def _evaluator(self) -> Evaluator:
        if self.config.evaluation.mode == "synthetic":
            return SyntheticEvaluator(self.config.evaluation.baseline_primary)
        return ProductionEvaluator(
            root=self.root,
            run_dir=self.run_dir,
            config=self.config.evaluation,
            loop=self.config.loop,
            registry=self.registry,
        )

    def run(self) -> dict[str, Any]:
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(f"run directory is not empty: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "interventions.jsonl").touch()
        (self.run_dir / "generated_features").mkdir()
        run_log = self.run_dir / "run_log.jsonl"
        memory = AgentMemory(self.run_dir / "memory.json")
        ablator = Ablator(self.run_dir / "ablation_cache.json")
        proposer, structured_client = self._proposer()
        evaluator = self._evaluator()
        loop = self.config.loop
        scheduler = AgentScheduler(
            max_iterations=loop.max_iterations,
            max_wall_clock_s=loop.max_wall_clock_s,
            convergence_window=loop.convergence_window,
            convergence_delta=loop.convergence_delta,
        )

        run_start = time.monotonic()
        initial = evaluator.evaluate(
            self.seed_graph,
            self.run_dir / "initial",
            incumbent_primary=self.config.evaluation.baseline_primary,
        )
        incumbent_graph = self.seed_graph
        incumbent_metrics = initial.metrics
        self._write_best(incumbent_graph, self.run_dir / "initial")
        cumulative_tokens = TokenUsage()
        recent_deltas: list[float] = []
        arm_observations: list[ArmObservation] = []
        completed = 0
        stop_reason: str | None = None

        while True:
            stop_reason = scheduler.should_stop(
                completed, time.monotonic() - run_start, recent_deltas
            )
            if stop_reason:
                break
            iteration = completed + 1
            iteration_dir = self.run_dir / "iterations" / f"{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            iteration_start = time.monotonic()
            errors: list[str] = []
            recovery_events: list[dict[str, Any]] = []
            proposal_usage = TokenUsage()
            candidate_graph = incumbent_graph
            metrics = incumbent_metrics
            diff = ""
            decision_payload: dict[str, Any]

            ablation_table = ablator.run(
                incumbent_graph,
                incumbent_metrics["primary"],
                evaluator.ablate,
            )
            proposal: ProposalResult = proposer.propose(
                graph=incumbent_graph,
                ablation_table=ablation_table,
                recent_outcomes=memory.last_outcomes(10),
                rejected_hypotheses=memory.rejected_hypothesis_ids(),
            )
            hypothesis = proposal.hypothesis
            proposal_usage = proposal.usage

            try:
                mutation = apply_hypothesis(
                    incumbent_graph,
                    hypothesis,
                    self.registry,
                    generated_feature_dir=self.run_dir / "generated_features",
                )
                candidate_graph = mutation.graph

                def evaluate_candidate(repaired: OperatorGraph) -> dict[str, float]:
                    return evaluator.evaluate(
                        repaired,
                        iteration_dir,
                        incumbent_primary=incumbent_metrics["primary"],
                    ).metrics

                try:
                    metrics = evaluate_candidate(candidate_graph)
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    if structured_client is None:
                        repair_provider = CannedRepairProvider()
                    else:
                        repair_provider = LLMRepairProvider(
                            structured_client, self.registry, hypothesis
                        )
                    outcome = RepairManager(
                        repair_provider, max_attempts=loop.repair_attempts
                    ).recover(candidate_graph, exc, evaluate_candidate)
                    errors = outcome.errors
                    recovery_events = outcome.events
                    cumulative_tokens = cumulative_tokens + getattr(
                        repair_provider, "usage", TokenUsage()
                    )
                    if not outcome.recovered or outcome.metrics is None:
                        raise RuntimeError("repair attempts exhausted")
                    candidate_graph = outcome.graph
                    metrics = outcome.metrics

                raw_delta = metrics["primary"] - incumbent_metrics["primary"]
                seed_scores = None
                if (
                    loop.confirm_small_deltas
                    and 0 < raw_delta <= loop.significance_threshold
                ):
                    seed_scores = evaluator.confirm_seeds(
                        candidate_graph,
                        iteration_dir / "significance",
                        incumbent_primary=incumbent_metrics["primary"],
                    )
                decision = decide_significance(
                    incumbent_metrics["primary"],
                    metrics["primary"],
                    seed_scores=seed_scores,
                    threshold=loop.significance_threshold,
                )
                accepted = decision.accepted
                decision_payload = asdict(decision)
            except Exception as exc:
                if not errors or errors[-1] != f"{type(exc).__name__}: {exc}":
                    errors.append(f"{type(exc).__name__}: {exc}")
                raw_delta = 0.0
                accepted = False
                metrics = incumbent_metrics
                decision_payload = {
                    "accepted": False,
                    "reason": "execution_failed",
                    "raw_delta": 0.0,
                    "threshold": loop.significance_threshold,
                    "seeds": 0,
                    "seed_mean": None,
                    "improving_seeds": None,
                }

            cumulative_tokens = cumulative_tokens + proposal_usage
            diff = graph_diff(incumbent_graph, candidate_graph)
            candidate_graph_path = iteration_dir / "graph.json"
            candidate_graph_path.write_text(json.dumps(candidate_graph.to_dict(), indent=2))
            (iteration_dir / "diff.patch").write_text(diff)
            (iteration_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (iteration_dir / "stdout.log").touch(exist_ok=True)
            iteration_wall = time.monotonic() - iteration_start
            record = {
                "iteration": iteration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "hypothesis": hypothesis.model_dump(mode="json"),
                "ablation_table": ablation_table,
                "diff": diff,
                "metrics": metrics,
                "delta_vs_incumbent": raw_delta,
                "accepted": accepted,
                "significance": decision_payload,
                "errors": errors,
                "recovery_events": recovery_events,
                "wall_clock_s": iteration_wall,
                "tokens": proposal_usage.to_log(),
                "cumulative": {
                    "wall_clock_s": time.monotonic() - run_start,
                    "tokens_in": cumulative_tokens.input_tokens,
                    "tokens_out": cumulative_tokens.output_tokens,
                },
            }
            with run_log.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            memory.record(
                hypothesis_id=hypothesis.id,
                accepted=accepted,
                target_node=hypothesis.target_node,
                delta=raw_delta,
                reason=str(decision_payload["reason"]),
            )
            arm_observations.append(ArmObservation(hypothesis.target_node, raw_delta))
            recent_deltas.append(raw_delta if accepted else 0.0)
            if accepted:
                incumbent_graph = candidate_graph
                incumbent_metrics = metrics
                self._write_best(candidate_graph, iteration_dir)
            completed += 1

        summary = {
            "status": "completed",
            "iterations": completed,
            "stop_reason": stop_reason,
            "best_metrics": incumbent_metrics,
            "wall_clock_s": time.monotonic() - run_start,
            "tokens": cumulative_tokens.to_log(),
            "interventions": sum(
                1
                for line in (self.run_dir / "interventions.jsonl").read_text().splitlines()
                if line.strip()
            ),
        }
        (self.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    def _write_best(self, graph: OperatorGraph, source_dir: Path) -> None:
        best = self.run_dir / "best"
        best.mkdir(parents=True, exist_ok=True)
        (best / "graph.json").write_text(json.dumps(graph.to_dict(), indent=2))
        for name in ("submission.csv", "metrics.json"):
            source = source_dir / name
            if source.is_file():
                shutil.copy2(source, best / name)


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()))


def run_from_config(
    config_path: Path,
    *,
    run_id: str,
    run_root: Path | None = None,
    dry_run: bool = False,
) -> tuple[dict[str, Any], Path]:
    config = load_config(config_path)
    run_dir = (run_root or ROOT / "runs") / run_id
    summary = AgentRunner(config, run_dir=run_dir, dry_run=dry_run).run()
    return summary, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unattended TechJam research agent")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "run.yaml")
    parser.add_argument("--run-id", default=datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--dry-run", action="store_true", help="use canned hypotheses and spend no LLM tokens"
    )
    args = parser.parse_args()
    summary, run_dir = run_from_config(
        args.config.resolve(), run_id=args.run_id, dry_run=args.dry_run
    )
    print(json.dumps({**summary, "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()


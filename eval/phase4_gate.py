from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.run import ROOT, run_from_config


def run_gate(output_root: Path | None = None) -> dict[str, Any]:
    root = (output_root or ROOT / "runs" / "phase4_gate").resolve()
    summary, run_dir = run_from_config(
        ROOT / "configs" / "phase4_gate.yaml",
        run_id="unattended_5_iterations",
        run_root=root,
    )
    records = [
        json.loads(line)
        for line in (run_dir / "run_log.jsonl").read_text().splitlines()
        if line.strip()
    ]
    interventions = sum(
        1
        for line in (run_dir / "interventions.jsonl").read_text().splitlines()
        if line.strip()
    )
    recoveries = sum(len(record["recovery_events"]) for record in records)
    complete_artifacts = all(
        all((run_dir / "iterations" / f"{index:03d}" / name).is_file() for name in (
            "graph.json", "diff.patch", "metrics.json", "stdout.log"
        ))
        for index in range(1, 6)
    )
    passed = (
        summary["status"] == "completed"
        and len(records) == 5
        and interventions == 0
        and recoveries >= 1
        and complete_artifacts
        and all(record["tokens"] == {"in": 0, "out": 0} for record in records)
    )
    result = {
        "status": "passed" if passed else "failed",
        "iterations": len(records),
        "interventions": interventions,
        "recovery_events": recoveries,
        "stop_reason": summary["stop_reason"],
        "run_dir": str(run_dir),
        "run_log": str(run_dir / "run_log.jsonl"),
        "complete_artifacts": complete_artifacts,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "gate_result.json").write_text(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 4 unattended-loop gate")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_gate(args.output_root), indent=2))


if __name__ == "__main__":
    main()


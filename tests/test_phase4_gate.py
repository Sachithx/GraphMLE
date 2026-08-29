from __future__ import annotations

import json
from pathlib import Path

from eval.phase4_gate import run_gate


def test_phase4_gate_runs_five_unattended_iterations_with_recovery(tmp_path: Path) -> None:
    result = run_gate(tmp_path / "phase4")
    assert result["status"] == "passed"
    assert result["iterations"] == 5
    assert result["interventions"] == 0
    assert result["recovery_events"] >= 1

    records = [
        json.loads(line)
        for line in Path(result["run_log"]).read_text().splitlines()
        if line.strip()
    ]
    assert len(records) == 5
    assert [record["iteration"] for record in records] == [1, 2, 3, 4, 5]
    required = {
        "hypothesis",
        "ablation_table",
        "diff",
        "metrics",
        "delta_vs_incumbent",
        "accepted",
        "significance",
        "errors",
        "recovery_events",
        "wall_clock_s",
        "tokens",
        "cumulative",
    }
    assert all(required <= record.keys() for record in records)
    assert any(record["recovery_events"] for record in records)
    assert all(record["tokens"] == {"in": 0, "out": 0} for record in records)
    assert (Path(result["run_dir"]) / "best" / "graph.json").exists()


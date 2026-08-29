from __future__ import annotations

from pathlib import Path

from eval.phase3_gate import run_gate


def test_phase3_gate_rejects_leakage_and_captures_failure(tmp_path: Path) -> None:
    result = run_gate(tmp_path / "phase3")
    assert result["status"] == "passed"
    assert result["leakage"]["rejected_before_training"]
    assert not result["leakage"]["model_called"]
    assert result["sandbox"]["status"] == "failed"

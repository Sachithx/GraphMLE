from __future__ import annotations

from pathlib import Path

from agent.memory import AgentMemory


def test_memory_persists_recent_and_rejected_hypotheses(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = AgentMemory(path)
    for index in range(12):
        memory.record(
            hypothesis_id=f"h_{index:03d}",
            accepted=index % 2 == 0,
            target_node="model",
            delta=index / 10_000,
            reason="test",
        )

    reloaded = AgentMemory(path)
    assert len(reloaded.last_outcomes(10)) == 10
    assert reloaded.last_outcomes(10)[0]["hypothesis_id"] == "h_002"
    assert "h_011" in reloaded.rejected_hypothesis_ids()
    assert "h_010" not in reloaded.rejected_hypothesis_ids()


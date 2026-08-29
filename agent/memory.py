from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AgentMemory:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._outcomes: list[dict[str, Any]] = []
        if self.path.is_file():
            raw = json.loads(self.path.read_text())
            self._outcomes = list(raw.get("outcomes", []))

    def record(
        self,
        *,
        hypothesis_id: str,
        accepted: bool,
        target_node: str,
        delta: float,
        reason: str,
    ) -> None:
        self._outcomes.append(
            {
                "hypothesis_id": hypothesis_id,
                "accepted": bool(accepted),
                "target_node": target_node,
                "delta": float(delta),
                "reason": reason,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"outcomes": self._outcomes}, indent=2))

    def last_outcomes(self, limit: int = 10) -> list[dict[str, Any]]:
        return [dict(item) for item in self._outcomes[-limit:]]

    def rejected_hypothesis_ids(self) -> list[str]:
        return [
            str(item["hypothesis_id"])
            for item in self._outcomes
            if not item.get("accepted", False)
        ]


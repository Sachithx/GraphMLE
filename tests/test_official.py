from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from eval.official import evaluate_official


ROOT = Path(__file__).resolve().parents[1]
STARTER_KIT = ROOT / "data" / "starter-kit"


def test_wrapper_uses_kit_evaluator_without_modifying_it() -> None:
    evaluator = STARTER_KIT / "evaluate.py"
    before = hashlib.sha256(evaluator.read_bytes()).hexdigest()

    metrics = evaluate_official(
        user_ids=["user-1", "user-1"],
        labels=[0, 1],
        scores=[0.1, 0.9],
        starter_kit_dir=STARTER_KIT,
    )

    after = hashlib.sha256(evaluator.read_bytes()).hexdigest()
    assert metrics == {"gauc": 1.0, "ndcg5": 1.0, "primary": 1.0}
    assert after == before


def test_wrapper_fails_clearly_when_evaluator_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="evaluate.py"):
        evaluate_official(["u", "u"], [0, 1], [0.0, 1.0], tmp_path)

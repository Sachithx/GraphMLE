"""Thin adapter around the starter kit's authoritative ``evaluate.py``.

The metric implementation is loaded directly from the kit and is never copied or
reimplemented here.  This module only normalizes its public result keys for the
rest of the harness.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence


DEFAULT_STARTER_KIT = Path(__file__).resolve().parents[1] / "data" / "starter-kit"


def _starter_kit_path(starter_kit_dir: str | os.PathLike[str] | None) -> Path:
    configured = starter_kit_dir or os.environ.get("TECHJAM_STARTER_KIT")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_STARTER_KIT


@lru_cache(maxsize=None)
def _load_evaluator(evaluator_path: Path) -> ModuleType:
    if not evaluator_path.is_file():
        raise FileNotFoundError(
            f"Official evaluator not found at {evaluator_path}. "
            "Place the unmodified starter kit in data/starter-kit or set "
            "TECHJAM_STARTER_KIT."
        )
    spec = importlib.util.spec_from_file_location("techjam_official_evaluate", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_official(
    user_ids: Sequence[object] | Iterable[object],
    labels: Sequence[float] | Iterable[float],
    scores: Sequence[float] | Iterable[float],
    starter_kit_dir: str | os.PathLike[str] | None = None,
) -> dict[str, float]:
    """Evaluate scores with the kit and return normalized metric names."""
    evaluator = _load_evaluator(_starter_kit_path(starter_kit_dir) / "evaluate.py")
    raw = evaluator.evaluate(user_ids, labels, scores, k=5)
    return {
        "gauc": float(raw["GAUC"]),
        "ndcg5": float(raw["nDCG@5"]),
        "primary": float(raw["primary"]),
    }

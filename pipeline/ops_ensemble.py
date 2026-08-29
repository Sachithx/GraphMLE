from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .types import ExecutionContext, PredictionBundle


def _require_predictions(inputs: list[Any]) -> list[PredictionBundle]:
    if not inputs or not all(isinstance(value, PredictionBundle) for value in inputs):
        raise TypeError("ensemble operators require PredictionBundle inputs")
    predictions: list[PredictionBundle] = inputs
    data = predictions[0].data
    if any(item.data is not data for item in predictions[1:]):
        raise ValueError("all ensemble inputs must derive from the same DataBundle")
    return predictions


def op_rank_average(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    del ctx
    predictions = _require_predictions(inputs)
    weights = np.asarray(params.get("weights", [1.0] * len(predictions)), dtype=float)
    if len(weights) != len(predictions) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("rank_average weights must be finite, positive, and match inputs")
    scores = {}
    for split in ("train", "valid", "test"):
        users = predictions[0].data.frames[split]["user_id"].astype(str)
        ranked = []
        for item in predictions:
            ranked.append(
                pd.Series(item.scores[split]).groupby(users, sort=False).rank(pct=True).to_numpy()
            )
        scores[split] = np.average(np.vstack(ranked), axis=0, weights=weights)
    return PredictionBundle("rank_average", scores, predictions[0].data)


def op_seed_bag(
    inputs: list[Any], params: dict[str, Any], ctx: ExecutionContext
) -> PredictionBundle:
    del ctx
    predictions = _require_predictions(inputs)
    weights = np.asarray(params.get("weights", [1.0] * len(predictions)), dtype=float)
    if len(weights) != len(predictions) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("seed_bag weights must be finite, positive, and match inputs")
    scores = {
        split: np.average(
            np.vstack([item.scores[split] for item in predictions]),
            axis=0,
            weights=weights,
        )
        for split in ("train", "valid", "test")
    }
    return PredictionBundle("seed_bag", scores, predictions[0].data)

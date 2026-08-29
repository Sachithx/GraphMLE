from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Sequence


BASELINE_SEED_STD = 0.0008
ACCEPTANCE_THRESHOLD = 2 * BASELINE_SEED_STD


@dataclass(frozen=True)
class SignificanceDecision:
    accepted: bool
    reason: str
    raw_delta: float
    threshold: float
    seeds: int
    seed_mean: float | None = None
    improving_seeds: int | None = None


def decide_significance(
    incumbent_primary: float,
    candidate_primary: float,
    *,
    seed_scores: Sequence[float] | None = None,
    threshold: float = ACCEPTANCE_THRESHOLD,
) -> SignificanceDecision:
    raw_delta = float(candidate_primary - incumbent_primary)
    if raw_delta - threshold > 1e-12:
        return SignificanceDecision(
            True,
            "delta_above_two_sigma",
            raw_delta,
            threshold,
            1,
        )
    if seed_scores is not None:
        if len(seed_scores) != 3:
            raise ValueError("significance reruns require exactly three seed scores")
        seed_mean = mean(float(score) for score in seed_scores)
        improving = sum(float(score) > incumbent_primary for score in seed_scores)
        if seed_mean > incumbent_primary and improving >= 2:
            return SignificanceDecision(
                True,
                "confirmed_by_three_seed_rerun",
                raw_delta,
                threshold,
                3,
                seed_mean,
                improving,
            )
        return SignificanceDecision(
            False,
            "not_significant",
            raw_delta,
            threshold,
            3,
            seed_mean,
            improving,
        )
    return SignificanceDecision(
        False,
        "not_significant",
        raw_delta,
        threshold,
        1,
    )

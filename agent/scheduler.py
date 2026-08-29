from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ArmObservation:
    arm: str
    delta: float


class AgentScheduler:
    def __init__(
        self,
        *,
        max_iterations: int,
        max_wall_clock_s: float,
        convergence_window: int = 3,
        convergence_delta: float = 0.002,
        ucb_exploration: float = 0.25,
    ) -> None:
        self.max_iterations = int(max_iterations)
        self.max_wall_clock_s = float(max_wall_clock_s)
        self.convergence_window = int(convergence_window)
        self.convergence_delta = float(convergence_delta)
        self.ucb_exploration = float(ucb_exploration)

    def order_candidates(self, candidates: Iterable[tuple[str, float]]) -> list[tuple[str, float]]:
        return sorted(candidates, key=lambda item: (-float(item[1]), item[0]))

    def select_arm(self, arms: Sequence[str], observations: Sequence[ArmObservation]) -> str:
        if not arms:
            raise ValueError("at least one scheduler arm is required")
        grouped = {arm: [] for arm in arms}
        for observation in observations:
            if observation.arm in grouped:
                grouped[observation.arm].append(float(observation.delta))
        for arm in arms:
            if not grouped[arm]:
                return arm
        total = sum(len(values) for values in grouped.values())
        scores = {
            arm: mean(values)
            + self.ucb_exploration * math.sqrt(math.log(max(total, 2)) / len(values))
            for arm, values in grouped.items()
        }
        return max(arms, key=lambda arm: (scores[arm], -arms.index(arm)))

    def should_stop(
        self,
        iteration_count: int,
        elapsed_s: float,
        recent_deltas: Sequence[float],
    ) -> str | None:
        if iteration_count >= self.max_iterations:
            return "iteration_cap"
        if elapsed_s >= self.max_wall_clock_s:
            return "wall_clock_cap"
        if len(recent_deltas) >= self.convergence_window and all(
            delta <= self.convergence_delta
            for delta in recent_deltas[-self.convergence_window :]
        ):
            return "converged"
        return None


from __future__ import annotations

from agent.scheduler import AgentScheduler, ArmObservation


def test_scheduler_orders_large_expected_effects_first() -> None:
    scheduler = AgentScheduler(max_iterations=50, max_wall_clock_s=6 * 3600)
    ordered = scheduler.order_candidates(
        [
            ("hyperparameters", 0.001),
            ("multitask", 0.009),
            ("listwise", 0.012),
        ]
    )
    assert [name for name, _ in ordered] == ["listwise", "multitask", "hyperparameters"]


def test_scheduler_uses_ucb_and_enforces_all_stop_conditions() -> None:
    scheduler = AgentScheduler(max_iterations=5, max_wall_clock_s=100)
    assert scheduler.select_arm(
        ["features", "model"],
        [ArmObservation("features", 0.002), ArmObservation("model", 0.004)],
    ) == "model"
    assert scheduler.should_stop(5, 1, [0.01]) == "iteration_cap"
    assert scheduler.should_stop(1, 101, [0.01]) == "wall_clock_cap"
    assert scheduler.should_stop(3, 5, [0.0019, -0.001, 0.001]) == "converged"
    assert scheduler.should_stop(3, 5, [0.0019, 0.0021, 0.001]) is None


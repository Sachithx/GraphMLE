from __future__ import annotations

from eval.significance import ACCEPTANCE_THRESHOLD, decide_significance


def test_single_seed_requires_more_than_two_sigma() -> None:
    assert ACCEPTANCE_THRESHOLD == 0.0016
    assert not decide_significance(0.6000, 0.6016).accepted
    assert decide_significance(0.6000, 0.60161).accepted


def test_three_seed_rerun_can_confirm_a_smaller_delta() -> None:
    decision = decide_significance(
        0.6000,
        0.6005,
        seed_scores=[0.6005, 0.6002, 0.5999],
    )
    assert decision.accepted
    assert decision.seeds == 3
    assert decision.reason == "confirmed_by_three_seed_rerun"


def test_unstable_three_seed_rerun_is_rejected() -> None:
    decision = decide_significance(
        0.6000,
        0.6005,
        seed_scores=[0.6005, 0.5999, 0.5998],
    )
    assert not decision.accepted
    assert decision.reason == "not_significant"

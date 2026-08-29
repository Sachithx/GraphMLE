from __future__ import annotations

from pathlib import Path

from eval.phase1_gate import (
    EXPECTED_ROWS,
    inspect_dataset,
    mean_metrics,
    metrics_match_baseline,
)


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_matches_official_split_sizes() -> None:
    assert inspect_dataset(
        ROOT / "data" / "starter-kit",
        ROOT / "data" / "kuairand-pure" / "data",
    ) == EXPECTED_ROWS


def test_baseline_gate_uses_published_precision() -> None:
    assert metrics_match_baseline(
        {"gauc": 0.66739, "ndcg5": 0.53571, "primary": 0.60155}
    )
    assert not metrics_match_baseline(
        {"gauc": 0.66, "ndcg5": 0.5357, "primary": 0.6016}
    )


def test_published_validation_is_the_five_seed_mean() -> None:
    observed = [
        {"gauc": 0.6671, "ndcg5": 0.5358, "primary": 0.6015},
        {"gauc": 0.6674, "ndcg5": 0.5361, "primary": 0.6018},
        {"gauc": 0.6671, "ndcg5": 0.5351, "primary": 0.6011},
        {"gauc": 0.6675, "ndcg5": 0.5355, "primary": 0.6015},
        {"gauc": 0.6679, "ndcg5": 0.5361, "primary": 0.6020},
    ]
    assert metrics_match_baseline(mean_metrics(observed))

"""Executable acceptance gate for Phase 1.

One command verifies split sizes, reproduces the official FM baseline, creates a
test submission, and delegates final validation to the starter kit's ``--check``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STARTER_KIT = ROOT / "data" / "starter-kit"
DEFAULT_DATA_DIR = ROOT / "data" / "kuairand-pure" / "data"
DEFAULT_SUBMISSION = ROOT / "runs" / "phase1" / "submission.csv"

EXPECTED_ROWS = {"train": 1_141_112, "valid": 124_909, "test": 170_588}
EXPECTED_METRICS = {
    "valid": {"gauc": 0.6674, "ndcg5": 0.5357, "primary": 0.6016},
    "test": {"gauc": 0.6610, "ndcg5": 0.5282, "primary": 0.5946},
}
_METRIC_LINE = re.compile(
    r"(?P<split>valid|test)\s+GAUC\s+(?P<gauc>[0-9.]+)\s+\|\s+"
    r"nDCG@5\s+(?P<ndcg5>[0-9.]+)\s+\|\s+primary\s+(?P<primary>[0-9.]+)"
)


def _load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Required starter-kit file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_dataset(starter_kit_dir: Path, data_dir: Path) -> dict[str, int]:
    """Load with the official loader and return exact split row counts."""
    loader = _load_module(starter_kit_dir / "data.py", "techjam_official_data")
    return {name: len(rows) for name, rows in loader.load(str(data_dir)).items()}


def metrics_match_baseline(
    metrics: dict[str, float], expected: dict[str, float] | None = None
) -> bool:
    """Compare at the four-decimal precision published by the starter kit."""
    expected = expected or EXPECTED_METRICS["valid"]
    return all(f"{metrics[key]:.4f}" == f"{value:.4f}" for key, value in expected.items())


def mean_metrics(observations: list[dict[str, float]]) -> dict[str, float]:
    if not observations:
        raise ValueError("At least one metric observation is required")
    return {
        key: sum(item[key] for item in observations) / len(observations)
        for key in ("gauc", "ndcg5", "primary")
    }


def _run(command: list[str], cwd: Path, timeout_s: int) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return completed


def _parse_metrics(output: str) -> dict[str, dict[str, float]]:
    parsed: dict[str, dict[str, float]] = {}
    for match in _METRIC_LINE.finditer(output):
        groups = match.groupdict()
        parsed[groups["split"]] = {
            key: float(groups[key]) for key in ("gauc", "ndcg5", "primary")
        }
    if set(parsed) != set(EXPECTED_METRICS):
        raise RuntimeError("Could not parse validation and test metrics from baseline.py output")
    return parsed


def run_gate(
    starter_kit_dir: Path,
    data_dir: Path,
    submission: Path,
    timeout_s: int = 1_800,
) -> dict[str, object]:
    starter_kit_dir = starter_kit_dir.resolve()
    data_dir = data_dir.resolve()
    submission = submission.resolve()

    rows = inspect_dataset(starter_kit_dir, data_dir)
    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"Split row counts mismatch: expected {EXPECTED_ROWS}, got {rows}")
    print(f"split rows verified: {rows}")

    parsed_per_seed: list[dict[str, dict[str, float]]] = []
    for seed in range(5):
        baseline = _run(
            [
                sys.executable,
                str(starter_kit_dir / "baseline.py"),
                "--model",
                "fm",
                "--seed",
                str(seed),
                "--data_dir",
                str(data_dir),
            ],
            cwd=starter_kit_dir,
            timeout_s=timeout_s,
        )
        parsed_per_seed.append(_parse_metrics(baseline.stdout))

    means = {
        split: mean_metrics([metrics[split] for metrics in parsed_per_seed])
        for split in ("valid", "test")
    }
    for split, expected in EXPECTED_METRICS.items():
        if not metrics_match_baseline(means[split], expected):
            raise RuntimeError(
                f"Official FM {split} five-seed mean mismatch: "
                f"expected {expected}, got {means[split]}"
            )
    print(f"five-seed FM means verified: {means}")

    submission.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            str(starter_kit_dir / "submit.py"),
            str(submission),
            "--make",
            "--split",
            "test",
            "--data_dir",
            str(data_dir),
        ],
        cwd=starter_kit_dir,
        timeout_s=timeout_s,
    )
    check = _run(
        [
            sys.executable,
            str(starter_kit_dir / "submit.py"),
            str(submission),
            "--check",
            "--split",
            "test",
            "--data_dir",
            str(data_dir),
        ],
        cwd=starter_kit_dir,
        timeout_s=timeout_s,
    )
    result: dict[str, object] = {
        "status": "passed",
        "rows": rows,
        "baseline_mean": means,
        "baseline_per_seed": [
            {"seed": seed, **metrics} for seed, metrics in enumerate(parsed_per_seed)
        ],
        "submission": str(submission),
        "submission_check": check.stdout.strip(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starter-kit", type=Path, default=DEFAULT_STARTER_KIT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--timeout-s", type=int, default=1_800)
    args = parser.parse_args()
    run_gate(args.starter_kit, args.data_dir, args.submission, args.timeout_s)


if __name__ == "__main__":
    main()

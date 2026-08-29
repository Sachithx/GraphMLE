from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from agent.run import ROOT, load_config


REQUIRED_DATA_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
)
REQUIRED_KIT_FILES = ("baseline.py", "data.py", "evaluate.py", "submit.py")


def run_preflight(
    config_path: Path,
    *,
    require_gpus: int = 1,
    allow_missing_api_key: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    data_dir = (ROOT / config.evaluation.data_dir).resolve()
    kit_dir = (ROOT / config.evaluation.starter_kit_dir).resolve()
    missing_data = [name for name in REQUIRED_DATA_FILES if not (data_dir / name).is_file()]
    missing_kit = [name for name in REQUIRED_KIT_FILES if not (kit_dir / name).is_file()]
    blockers: list[str] = []
    if config.evaluation.mode != "production":
        blockers.append("evaluation.mode must be production")
    if config.llm.mode != "live":
        blockers.append("llm.mode must be live")
    if missing_data:
        blockers.append(f"missing data files: {missing_data}")
    if missing_kit:
        blockers.append(f"missing starter-kit files: {missing_kit}")

    api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
    if not api_key_present and not allow_missing_api_key:
        blockers.append("OPENAI_API_KEY is missing")

    try:
        import lightgbm

        lightgbm_version = lightgbm.__version__
    except Exception as exc:
        lightgbm_version = None
        blockers.append(f"LightGBM import failed: {type(exc).__name__}: {exc}")

    gpu_names: list[str] = []
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            gpu_names = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        torch_version = None
        blockers.append(f"PyTorch import failed: {type(exc).__name__}: {exc}")
    if len(gpu_names) < require_gpus:
        blockers.append(f"requires {require_gpus} visible GPU(s), found {len(gpu_names)}")

    disk = shutil.disk_usage(ROOT)
    if disk.free < 10 * 1024**3:
        blockers.append("less than 10 GiB free disk space")

    return {
        "status": "ready" if not blockers else "blocked",
        "blockers": blockers,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "api_key_present": api_key_present,
        "proposal_model": config.llm.model,
        "repair_model": config.llm.repair_model,
        "gpus": gpu_names,
        "lightgbm": lightgbm_version,
        "torch": torch_version,
        "data_dir": str(data_dir),
        "starter_kit_dir": str(kit_dir),
        "disk_free_gib": round(disk.free / 1024**3, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate scored-run prerequisites")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "run.yaml")
    parser.add_argument("--require-gpus", type=int, default=1)
    parser.add_argument("--allow-missing-api-key", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_preflight(
        args.config.resolve(),
        require_gpus=args.require_gpus,
        allow_missing_api_key=args.allow_missing_api_key,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    raise SystemExit(0 if result["status"] == "ready" else 1)


if __name__ == "__main__":
    main()


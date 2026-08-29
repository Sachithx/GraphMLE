from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SandboxResult:
    status: str
    command: tuple[str, ...]
    returncode: int | None
    timed_out: bool
    wall_clock_s: float
    stdout_path: Path
    result_path: Path


def run_in_sandbox(
    command: Sequence[str],
    *,
    workdir: Path,
    stdout_path: Path,
    timeout_s: float = 900,
    memory_limit_mb: int | None = 4096,
    result_path: Path | None = None,
) -> SandboxResult:
    workdir = Path(workdir).resolve()
    stdout_path = Path(stdout_path).resolve()
    result_path = (
        Path(result_path).resolve()
        if result_path is not None
        else stdout_path.with_name("sandbox_result.json")
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    returncode: int | None = None
    status = "failed"
    wrapper = Path(__file__).with_name("sandbox_child.py")
    launched_command = [
        str(Path(sys.executable)),
        str(wrapper),
        str(int(memory_limit_mb or 0)),
        str(float(timeout_s)),
        "--",
        *list(command),
    ]
    with stdout_path.open("w") as output:
        try:
            completed = subprocess.run(
                launched_command,
                cwd=workdir,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s + 5,
                check=False,
            )
            returncode = completed.returncode
            timed_out = returncode == 124
            status = "timed_out" if timed_out else ("success" if returncode == 0 else "failed")
        except subprocess.TimeoutExpired:
            timed_out = True
            status = "timed_out"
            output.write(f"\n[sandbox] timeout after {timeout_s} seconds\n")
        except OSError as exc:
            output.write(f"\n[sandbox] process launch failed: {exc}\n")
    result = SandboxResult(
        status=status,
        command=tuple(str(part) for part in command),
        returncode=returncode,
        timed_out=timed_out,
        wall_clock_s=time.monotonic() - start,
        stdout_path=stdout_path,
        result_path=result_path,
    )
    payload = asdict(result)
    payload["stdout_path"] = str(result.stdout_path)
    payload["result_path"] = str(result.result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2))
    return result

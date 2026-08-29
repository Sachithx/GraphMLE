from __future__ import annotations

import json
import sys
from pathlib import Path

from guards.sandbox import run_in_sandbox


def test_broken_process_is_captured_in_stdout_log(tmp_path: Path) -> None:
    log_path = tmp_path / "stdout.log"
    result = run_in_sandbox(
        [sys.executable, "-c", "print('before crash'); raise RuntimeError('induced')"],
        workdir=tmp_path,
        stdout_path=log_path,
        timeout_s=5,
        memory_limit_mb=512,
    )

    assert result.status == "failed"
    assert result.returncode != 0
    assert "before crash" in log_path.read_text()
    assert "RuntimeError: induced" in log_path.read_text()
    recorded = json.loads(result.result_path.read_text())
    assert recorded["status"] == "failed"


def test_timeout_is_captured_and_logged(tmp_path: Path) -> None:
    result = run_in_sandbox(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        workdir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        timeout_s=0.05,
        memory_limit_mb=None,
    )
    assert result.status == "timed_out"
    assert result.timed_out
    assert "timeout" in result.stdout_path.read_text().lower()


def test_memory_cap_terminates_oversized_process(tmp_path: Path) -> None:
    result = run_in_sandbox(
        [
            sys.executable,
            "-c",
            "x = bytearray(128 * 1024 * 1024); import time; time.sleep(1)",
        ],
        workdir=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        timeout_s=5,
        memory_limit_mb=64,
    )
    assert result.status == "failed"
    assert result.returncode != 0


def test_memory_cap_measures_rss_not_large_virtual_cuda_style_mapping(tmp_path: Path) -> None:
    result = run_in_sandbox(
        [
            sys.executable,
            "-c",
            (
                "import mmap; "
                "mapping = mmap.mmap(-1, 16 * 1024 * 1024 * 1024, "
                "flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS); "
                "mapping[0] = 1; print('mapped')"
            ),
        ],
        workdir=tmp_path,
        stdout_path=tmp_path / "virtual_memory.log",
        timeout_s=5,
        memory_limit_mb=128,
    )
    assert result.status == "success"
    assert "mapped" in result.stdout_path.read_text()

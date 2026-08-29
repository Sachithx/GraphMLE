"""Fresh-process memory limiter used internally by :mod:`guards.sandbox`."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


class _ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32),
        ("syscalls_unix", ctypes.c_int32),
        ("context_switches", ctypes.c_int32),
        ("thread_count", ctypes.c_int32),
        ("running_threads", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


def _macos_rss_bytes(pid: int) -> int:
    libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    proc_pidinfo = libproc.proc_pidinfo
    proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    proc_pidinfo.restype = ctypes.c_int
    task_info = _ProcTaskInfo()
    returned = proc_pidinfo(
        pid,
        4,  # PROC_PIDTASKINFO
        0,
        ctypes.byref(task_info),
        ctypes.sizeof(task_info),
    )
    return int(task_info.resident_size) if returned == ctypes.sizeof(task_info) else 0


def _terminate_group(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _linux_process_tree(pid: int) -> set[int]:
    pending = [pid]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        children_path = Path(f"/proc/{current}/task/{current}/children")
        try:
            pending.extend(int(value) for value in children_path.read_text().split())
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return seen


def _linux_rss_bytes(pid: int) -> int:
    total_kib = 0
    for process_id in _linux_process_tree(pid):
        try:
            for line in Path(f"/proc/{process_id}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total_kib += int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total_kib * 1024


def _run_supervised(command: list[str], limit_bytes: int, timeout_s: float) -> int:
    process = subprocess.Popen(command, start_new_session=True)
    start = time.monotonic()
    while process.poll() is None:
        if time.monotonic() - start > timeout_s:
            print(
                f"[sandbox] timeout after {timeout_s} seconds",
                file=sys.stderr,
                flush=True,
            )
            _terminate_group(process)
            return 124
        if (
            sys.platform == "darwin"
            and limit_bytes
            and _macos_rss_bytes(process.pid) > limit_bytes
        ):
            print(
                f"[sandbox] memory limit exceeded: {limit_bytes // (1024 * 1024)} MB",
                file=sys.stderr,
                flush=True,
            )
            _terminate_group(process)
            return 137
        if (
            sys.platform.startswith("linux")
            and limit_bytes
            and _linux_rss_bytes(process.pid) > limit_bytes
        ):
            print(
                f"[sandbox] memory limit exceeded: {limit_bytes // (1024 * 1024)} MB RSS",
                file=sys.stderr,
                flush=True,
            )
            _terminate_group(process)
            return 137
        time.sleep(0.05)
    returncode = int(process.returncode or 0)
    return 128 + abs(returncode) if returncode < 0 else returncode


def main() -> None:
    if len(sys.argv) < 5 or sys.argv[3] != "--":
        raise SystemExit(
            "usage: sandbox_child.py MEMORY_MB TIMEOUT_S -- COMMAND [ARG ...]"
        )
    limit_bytes = int(sys.argv[1]) * 1024 * 1024
    timeout_s = float(sys.argv[2])
    command = sys.argv[4:]
    raise SystemExit(_run_supervised(command, limit_bytes, timeout_s))


if __name__ == "__main__":
    main()

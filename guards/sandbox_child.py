"""Fresh-process memory limiter used internally by :mod:`guards.sandbox`."""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import time


def _descendant_rss_kb(root_pid: int) -> int:
    completed = subprocess.run(
        ["/bin/ps", "-axo", "pid=,ppid=,rss="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    rows: dict[int, tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3:
            pid, parent, rss = (int(value) for value in fields)
            rows[pid] = (parent, rss)
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _rss) in rows.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return sum(rows.get(pid, (0, 0))[1] for pid in descendants)


def _run_with_macos_rss_cap(command: list[str], limit_bytes: int) -> int:
    process = subprocess.Popen(command)
    limit_kb = limit_bytes // 1024
    while process.poll() is None:
        if _descendant_rss_kb(process.pid) > limit_kb:
            print(
                f"[sandbox] memory limit exceeded: {limit_bytes // (1024 * 1024)} MB",
                file=sys.stderr,
                flush=True,
            )
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return 137
        time.sleep(0.05)
    return int(process.returncode or 0)


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("usage: sandbox_child.py MEMORY_MB -- COMMAND [ARG ...]")
    limit_bytes = int(sys.argv[1]) * 1024 * 1024
    command = sys.argv[3:]
    if sys.platform == "darwin":
        raise SystemExit(_run_with_macos_rss_cap(command, limit_bytes))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (OSError, ValueError) as exc:
        print(f"[sandbox] could not establish memory limit: {exc}", file=sys.stderr)
        raise SystemExit(125) from exc
    os.execvpe(command[0], command, os.environ.copy())


if __name__ == "__main__":
    main()

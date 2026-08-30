"""Generate the Track 2 submission deliverables from a scored run directory.

Produces the per-iteration run log (deliverable 3) and the results / resource
summary (deliverable 4) directly from run artifacts, so the numbers in the report
are always the ones the run actually produced.

Usage:
    python report/make_deliverables.py final_04
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = {"primary": 0.6014687563529677, "gauc": 0.6674, "ndcg5": 0.5357}
BASELINE_TEST = {"primary": 0.5946, "gauc": 0.6610, "ndcg5": 0.5282}


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def diff_summary(path: Path, limit: int = 12) -> list[str]:
    """Return the meaningful changed lines from an iteration's diff."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith(("+", "-")) and line.strip() not in ("+", "-"):
            if any(k in line for k in ("hypothesis_id", "parent", "expected_delta",
                                       "validation_evidence", "previous_best", "name")):
                continue
            out.append(line.strip())
    return out[:limit]


def main(run_id: str) -> None:
    run_dir = ROOT / "runs" / run_id
    if not run_dir.is_dir():
        raise SystemExit(f"no such run: {run_dir}")

    summary = None
    console = run_dir / "console.log"
    if console.is_file():
        text = console.read_text()
        # The summary is the last top-level JSON object in the log. Nested objects
        # mean a naive rfind("{") lands inside one, so scan line-start braces.
        starts = [i for i, ch in enumerate(text) if ch == "{" and (i == 0 or text[i - 1] == "\n")]
        for start in reversed(starts):
            try:
                summary = json.loads(text[start:].strip())
                break
            except Exception:
                continue

    lines: list[str] = []
    lines.append(f"# Run & iteration log — `{run_id}`\n")
    lines.append(
        "Generated from run artifacts by `report/make_deliverables.py`. Every metric "
        "here is the official evaluator's output on the **validation** split; the "
        "hidden test set is never read during development.\n"
    )

    seed_graph = load(run_dir / "initial" / "candidate_graph.json")
    initial = load(run_dir / "initial" / "metrics.json")
    if initial:
        m = initial.get("metrics", {}).get("valid", initial)
        lines.append("## Iteration 0 — seed (baseline reproduction)\n")
        if seed_graph:
            nodes = ", ".join(f"`{n['type']}`" for n in seed_graph.get("nodes", []))
            lines.append(f"Graph: {nodes}\n")
        lines.append(
            f"| metric | seed | official baseline | delta |\n|---|---:|---:|---:|\n"
            f"| primary | {m.get('primary', float('nan')):.6f} | {BASELINE['primary']:.6f} | "
            f"{m.get('primary', 0) - BASELINE['primary']:+.6f} |\n"
        )

    it_dir = run_dir / "iterations"
    if it_dir.is_dir():
        lines.append("\n## Autonomous iterations\n")
        lines.append("| # | hypothesis | expected | primary | GAUC | nDCG@5 | vs baseline | accepted |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        best = initial and initial.get("metrics", {}).get("valid", {}).get("primary", -1) or -1
        rows = []
        for d in sorted(it_dir.iterdir()):
            if not d.is_dir():
                continue
            metrics = load(d / "metrics.json") or {}
            m = metrics.get("metrics", {}).get("valid", metrics)
            graph = load(d / "candidate_graph.json") or {}
            meta = graph.get("meta", {})
            p = m.get("primary")
            if p is None:
                continue
            accepted = "yes" if p > best else "no"
            if p > best:
                best = p
            rows.append((d.name, meta.get("hypothesis_id", "?"), meta.get("expected_delta"),
                         p, m.get("gauc"), m.get("ndcg5"), accepted, d))
            exp = meta.get("expected_delta")
            exp_s = f"{exp:+.4f}" if isinstance(exp, (int, float)) else "—"
            lines.append(
                f"| {d.name} | `{meta.get('hypothesis_id','?')}` | {exp_s} | {p:.6f} | "
                f"{m.get('gauc', float('nan')):.6f} | {m.get('ndcg5', float('nan')):.6f} | "
                f"{p - BASELINE['primary']:+.6f} | {accepted} |"
            )

        lines.append("\n### Per-iteration detail\n")
        for name, hyp, _exp, p, _g, _n, accepted, d in rows:
            lines.append(f"#### Iteration {name} — `{hyp}` ({accepted})\n")
            changes = diff_summary(d / "diff.patch")
            if changes:
                lines.append("Graph change:\n")
                lines.append("```diff")
                lines.extend(changes)
                lines.append("```\n")
            lines.append(f"Result: validation primary **{p:.6f}** "
                         f"({p - BASELINE['primary']:+.6f} vs official baseline).\n")
            run_log = d / "run_log.jsonl"
            if run_log.is_file():
                events = [json.loads(x) for x in run_log.read_text().splitlines() if x.strip()]
                if events:
                    lines.append("Error / recovery events:\n")
                    for e in events:
                        lines.append(f"- `{e.get('event')}` on `{e.get('node_id')}`: {e.get('reason')}")
                    lines.append("")

    # Manual interventions (autonomy reporting).
    lines.append("\n## Manual interventions\n")
    iv = run_dir / "interventions.jsonl"
    entries = []
    if iv.is_file():
        entries = [json.loads(x) for x in iv.read_text().splitlines() if x.strip()]
    lines.append(f"**Count: {len(entries)}**\n")
    for e in entries:
        lines.append(f"- `{e.get('timestamp')}` — {e.get('action')}\n\n  {e.get('reason')}\n")
    if not entries:
        lines.append("No manual interventions occurred during this run.\n")

    # Resource usage (Feasibility & Practicality).
    if summary:
        lines.append("\n## Resource usage\n")
        tok = summary.get("tokens", {})
        wall = summary.get("wall_clock_s", 0)
        lines.append(f"| quantity | value |\n|---|---:|")
        lines.append(f"| LLM tokens in | {tok.get('in', 0):,} |")
        lines.append(f"| LLM tokens out | {tok.get('out', 0):,} |")
        lines.append(f"| LLM tokens total | {tok.get('in', 0) + tok.get('out', 0):,} |")
        lines.append(f"| Agent wall-clock | {wall/3600:.2f} h ({wall:,.0f} s) |")
        lines.append(f"| Iterations used | {summary.get('executed_iterations', '?')} of 50 |")
        lines.append(f"| Rejected proposals | {summary.get('rejected_proposals', 0)} |")
        lines.append(f"| Stop reason | `{summary.get('stop_reason','?')}` |")
        lines.append(f"| Manual interventions | {summary.get('interventions', len(entries))} |")
        lines.append("")
        bm = summary.get("best_metrics", {})
        if bm:
            lines.append("\n## Converged result (validation)\n")
            lines.append("| metric | agent | official baseline | absolute delta |")
            lines.append("|---|---:|---:|---:|")
            for key, label in (("gauc", "GAUC"), ("ndcg5", "nDCG@5"), ("primary", "primary")):
                if key in bm:
                    lines.append(f"| {label} | {bm[key]:.6f} | {BASELINE[key]:.4f} | "
                                 f"{bm[key] - BASELINE[key]:+.6f} |")
            lines.append("")

    out = ROOT / "report" / f"run_log_{run_id}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final_04")

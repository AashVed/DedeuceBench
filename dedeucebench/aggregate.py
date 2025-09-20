from __future__ import annotations

"""Aggregate DedeuceBench JSONL results into a one-line CSV summary.

Metrics include Success@Budget, TrapFreeRate, EffSucc (efficiency among
successful episodes), QueriesUsed, BudgetLeft, and token usage stats.
The primary scoreboard metric is Score100 = 100 * Success@Budget.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _agg(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean metrics over result rows.

    rows: list of dicts, each with keys like ok, trap_hit, queries_used,
    budget_left, tokens_* that are produced by the eval CLI.
    Returns a dict of float metrics suitable for CSV output.
    """
    n = max(1, len(rows))
    succ = sum(1.0 for r in rows if bool(r.get("ok", False))) / n
    trap_free = sum(1.0 for r in rows if not bool(r.get("trap_hit", False))) / n
    q_used = sum(float(r.get("queries_used", 0.0)) for r in rows) / n
    b_left = sum(float(r.get("budget_left", 0.0)) for r in rows) / n
    tok_in = sum(float(r.get("tokens_in", 0.0)) for r in rows) / n
    tok_out = sum(float(r.get("tokens_out", 0.0)) for r in rows) / n
    tok_total = sum(float(r.get("tokens_total", 0.0)) for r in rows) / n
    # Efficiency among successes only: fraction of budget left when successful
    succ_rows = [r for r in rows if bool(r.get("ok", False))]
    if succ_rows:
        eff_succ = sum(
            max(0.0, 1.0 - (float(r.get("queries_used", 0.0)) / max(1.0, float(r.get("queries_used", 0.0)) + float(r.get("budget_left", 0.0)))))
            for r in succ_rows
        ) / len(succ_rows)
    else:
        eff_succ = 0.0
    score100 = 100.0 * succ  # primary scoreboard = success percentage
    return {
        "Success@Budget": float(succ),
        "TrapFreeRate": float(trap_free),
        "EffSucc": float(eff_succ),
        "QueriesUsed": float(q_used),
        "BudgetLeft": float(b_left),
        "TokensIn": float(tok_in),
        "TokensOut": float(tok_out),
        "TokensTotal": float(tok_total),
        "Score100": float(score100),
    }


def main():
    ap = argparse.ArgumentParser(description="Aggregate DedeuceBench JSONL results to CSV")
    ap.add_argument("results", type=str, help="results.jsonl path")
    args = ap.parse_args()

    path = Path(args.results)
    rows = _read_jsonl(path)
    if not rows:
        return

    # Ensure all rows are from the same model (optional strictness)
    models = {str(r.get("model", "n/a")) for r in rows}
    assert len(models) == 1, f"expected a single model in results, found: {sorted(models)}"
    model = next(iter(models))
    # Dedupe seeds for an accurate split label when multiple rollouts exist per seed
    seeds = [
        int(r.get("seed", -1))
        for r in rows
        if isinstance(r.get("seed", None), (int, float)) and int(r.get("seed", -1)) >= 0
    ]
    uniq_seeds = sorted(set(seeds))
    split_label = f"{len(uniq_seeds)}x" if uniq_seeds else "unknown"

    agg = _agg(rows)
    import sys
    writer = csv.writer(sys.stdout)
    writer.writerow(["model", "split", "Score100", "Success@Budget", "TrapFreeRate", "EffSucc", "QueriesUsed", "BudgetLeft", "TokensIn", "TokensOut", "TokensTotal"])
    writer.writerow([
        model,
        split_label,
        f"{agg['Score100']:.2f}",
        f"{agg['Success@Budget']:.4f}",
        f"{agg['TrapFreeRate']:.4f}",
        f"{agg['EffSucc']:.4f}",
        f"{agg['QueriesUsed']:.2f}",
        f"{agg['BudgetLeft']:.2f}",
        f"{agg['TokensIn']:.0f}",
        f"{agg['TokensOut']:.0f}",
        f"{agg['TokensTotal']:.0f}",
    ])


if __name__ == "__main__":
    main()

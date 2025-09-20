from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import contextlib

from .env import build_dataset_from_split, make_env
from .eval_cli import main as eval_main
from .aggregate import main as agg_main


def main():
    """Run a deterministic end-to-end self-check.

    - Builds the fixed test split dataset
    - Runs the deterministic heuristic baseline
    - Aggregates results to CSV
    - Checks basic invariants (counts, determinism, expected scores)
    """
    root = Path(__file__).resolve().parents[1]
    split = root / "seeds" / "levels_dev.json"

    # Ensure we can build the dataset without error (dev split)
    ds = build_dataset_from_split(split, subset="dev")
    assert len(ds) > 0, f"expected non-empty dataset, got {len(ds)}"

    # Run eval into a temp file by simulating CLI argv
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "results.jsonl"
        argv = [
            "dedeucebench-eval",
            "--split",
            str(split),
            "--subset",
            "dev",
            "--model",
            "heuristic:none",
            "--out",
            str(out),
        ]
        with contextlib.ExitStack() as stack:
            stack.enter_context(_patch_argv(argv))
            eval_main()
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(ds), f"expected {len(ds)} rows, got {len(lines)}"
        # Verify JSON structure per row
        rec0 = json.loads(lines[0])
        for k in ["model", "ok", "trap_hit", "queries_used", "budget_left", "seed"]:
            assert k in rec0, f"missing key {k} in results row"

        # Aggregate and capture CSV
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            argv2 = ["dedeucebench-aggregate", str(out)]
            with _patch_argv(argv2):
                agg_main()
        csv_text = buf.getvalue().strip().splitlines()
        assert len(csv_text) == 2, f"aggregate CSV should have 2 lines, got {len(csv_text)}"
        header = csv_text[0].split(",")
        row = csv_text[1].split(",")
        assert header[:3] == ["model", "split", "Score100"], "unexpected header"
        # For the deterministic no-op baseline on dev: Score100=0, TrapFreeRate=1
        score100 = float(row[2])
        trapfree = float(row[4])
        assert abs(score100 - 0.0) < 1e-6, f"expected Score100=0, got {score100}"
        assert abs(trapfree - 1.0) < 1e-6, f"expected TrapFreeRate=1, got {trapfree}"
        print("DedeuceBench self-check passed.")


class _patch_argv:
    def __init__(self, argv):
        self.argv = list(argv)
        self._old = None

    def __enter__(self):
        self._old = sys.argv[:]
        sys.argv = self.argv
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.argv = self._old
        return False


if __name__ == "__main__":
    main()

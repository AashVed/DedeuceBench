from __future__ import annotations

"""Benchmark dataset builder and environment factory for DedeuceBench.

This module reads split JSONs, constructs a deterministic `datasets.Dataset`
by concatenating per-seed items, and wraps the underlying `dedeuce` env with
the canonical scoring rubric and prompting.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
import verifiers as vf
from verifiers.parsers.parser import Parser

# Import the Prime env + helpers (already available in this workspace/package)
# Compatible import: support both package (dedeuce/dedeuce.py) and single-module installs
try:
    from dedeuce.dedeuce import (
        DedeuceEnv,
        reward_dedeuce,
        metric_queries,
        metric_correct,
        metric_trap,
        metric_budget_left,
        _build_dataset as _build_dedeuce_dataset,
    )
except Exception:  # pragma: no cover - fallback for module layout
    from dedeuce import (  # type: ignore
        DedeuceEnv,
        reward_dedeuce,
        metric_queries,
        metric_correct,
        metric_trap,
        metric_budget_left,
        _build_dataset as _build_dedeuce_dataset,
    )


def _load_split(split_path: str | Path) -> Dict[str, Any]:
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset_from_split(
    split_path: str | Path,
    *,
    subset: str | None = None,
    mode: str | None = None,
    trap: bool | None = None,
    budget: int | None = None,
    n_states: int | None = None,
    target_len: int | None = None,
    variety: bool | None = None,
    feedback: bool | None = None,
) -> Dataset:
    """Build a fixed, deterministic dataset from a split JSON.

    - Accepts either a single-config JSON or a multi-split JSON with
      `{"splits": {name: cfg, ...}}` (or top-level named subsets).
    - When multiple subsets are present, `subset` is required (e.g., `easy`).
    - Each subset defines global knobs and a list of `{seed}` items.
    """
    cfg_all = _load_split(split_path)
    # Support multi-split files. Accept either {"splits": {name: cfg, ...}} or {name: cfg, ...}
    if isinstance(cfg_all, dict) and (
        "splits" in cfg_all
        or any(k in cfg_all for k in ("test", "easy", "medium", "hard"))
    ):
        if "splits" in cfg_all:
            if subset is None:
                raise ValueError(
                    "split file contains multiple subsets; pass subset name (e.g., --subset easy)"
                )
            cfg = dict(cfg_all["splits"].get(str(subset), {}))
            if not cfg:
                raise ValueError(f"subset '{subset}' not found in split file")
        else:
            # Direct top-level named subsets
            if subset is None:
                raise ValueError(
                    "split file contains multiple subsets; pass subset name (e.g., --subset easy)"
                )
            if str(subset) not in cfg_all:
                raise ValueError(f"subset '{subset}' not found in split file")
            cfg = dict(cfg_all[str(subset)])
    else:
        cfg = cfg_all
    mode = str(mode if mode is not None else cfg.get("mode", "control"))
    trap = bool(trap if trap is not None else cfg.get("trap", True))
    budget = int(budget if budget is not None else cfg.get("budget", 25))
    n_states = int(n_states if n_states is not None else cfg.get("n_states", 5))
    target_len = int(
        target_len if target_len is not None else cfg.get("target_len", 20)
    )
    variety = bool(
        variety if variety is not None else cfg.get("variety", False)
    )  # deterministic default
    enable_feedback = bool(feedback) if feedback is not None else bool(cfg.get("feedback", False))
    items = list(cfg.get("items", []))

    prompts: List[List[Dict[str, str]]] = []
    answers: List[str] = []
    for it in items:
        seed = int(it.get("seed"))
        ds = _build_dedeuce_dataset(
            n=1,
            seed=seed,
            budget=budget,
            mode=mode,
            trap=trap,
            n_states=n_states,
            target_len=target_len,
            variety=variety,
        )
        # Append the single example from this seed
        rec = ds[0]
        # Always use the original answer/meta from the env
        ans = rec["answer"]

        # Replace the prompt with a benchmark-optimized, identification-first prompt
        try:
            meta = json.loads(ans)
        except Exception:
            meta = {}
        budget_i = int(meta.get("budget", budget))
        mode_i = str(meta.get("mode", mode))
        target_len_i = int(meta.get("target_len", target_len))
        target_hash_i = str(meta.get("target_hash", ""))
        # Prefer the exact number of states from the ground-truth table; fallback to split override
        try:
            n_states_i = int(
                (meta.get("table", {}) or {}).get(
                    "n", n_states if n_states is not None else 0
                )
            )
        except Exception:
            n_states_i = int(n_states if n_states is not None else 0)
        trap_pairs = list(meta.get("trap_pairs", []))
        trap_i = bool(trap_pairs) if trap_pairs is not None else bool(trap)

        sys_msg = {
            "role": "system",
            "content": (
                "You are an autonomous tool-using agent interacting with a hidden Mealy machine (finite-state transducer).\n"
                "Objective: exactly identify the machine and submit the full transition table via submit_table(table_json).\n"
                "Return ONLY function tool calls; never output natural language. All responses must be valid JSON if any content is emitted.\n\n"
                "Benchmark focus: identification-first. Success is achieved only by exact transition-table submission via submit_table(table_json).\n\n"
                "Episode semantics:\n"
                "- Start state is 0; the hidden state updates only when you call act(symbol).\n"
                + (
                    "- Each act() consumes 1 query from the budget. In feedback mode, an incorrect submit_table consumes 1 query and returns a counterexample without ending the episode; a correct submission ends the episode and does not consume budget.\n"
                    if enable_feedback
                    else
                    "- Each act() consumes 1 query from the budget. Submissions end the episode and do not consume budget.\n"
                )
                + "Tools (use only act() and submit_table()):\n"
                "- act(symbol: 'A'|'B'|'C') -> JSON {out, budget_left, t, trap_hit, queries_used}. Each call consumes 1 query and advances the hidden state.\n"
                + ("- submit_table(table_json: string) -> JSON {ok, budget_left, queries_used, trap_hit, counterexample?}. In feedback mode, if ok=false returns a short separating input-output trace and does NOT end the episode; if ok=true, ends the episode.\n\n" if enable_feedback else "- submit_table(table_json: string) -> JSON {ok, budget_left, queries_used, trap_hit}. Ends the episode.\n\n")
                + "Tool return fields (definitions):\n"
                "- out: integer output in {0,1,2} produced by this act() step.\n"
                "- budget_left: remaining number of act() queries.\n"
                "- t: 1-based step index since the episode started (increments on each act()).\n"
                "- trap_hit: boolean; once true it remains true for the rest of the episode.\n"
                "- queries_used: total count of act() calls so far.\n\n"
                "Submit-table JSON schema (table_json string must parse to this shape):\n"
                "{\n"
                '  "n": <int total_states>,\n'
                '  "start": 0,\n'
                '  "trans": {\n'
                '    "0": { "A": [<ns:int>, <out:int 0|1|2>], "B": [<ns>, <out>], "C": [<ns>, <out>] },\n'
                '    "1": { "A": [<ns>, <out>], "B": [<ns>, <out>], "C": [<ns>, <out>] },\n'
                '    ... up to "n-1"\n'
                "  }\n"
                "}\n\n"
                "Skeleton example (for n=2 — adjust values):\n"
                '{"n":2,"start":0,"trans":{"0":{"A":[1,2],"B":[0,1],"C":[0,0]},"1":{"A":[0,0],"B":[1,1],"C":[1,2]}}}\n\n'
                "Formatting & compliance:\n"
                "- Respond only with function tool calls as per the provided tool schemas.\n"
                "- The submit_table argument must be a single JSON string (not an object) matching the schema mentioned.\n"
                "- Do NOT echo the observation or tool descriptions.\n"
                '- Ensure "trans" covers every state index 0..n-1 and each of A,B,C exactly once.'
            ),
        }
        # No ground-truth leak is ever appended in public releases.
        # Observation shown to the model; goal reflects the underlying mode for transparency
        obs = {
            "alphabet": ["A", "B", "C"],
            "budget": budget_i,
            "goal": mode_i,
            "n_states": n_states_i,
            "target_len": target_len_i,
        }
        # Provide target_hash for transparency in control mode; instruct to ignore for scoring
        if mode_i == "control":
            obs["target_hash"] = target_hash_i
        # Indicate whether traps may exist (trap-free on some splits)
        obs["trap"] = bool(trap_i)
        usr_msg = {
            "role": "user",
            "content": (
                "OBSERVATION:\n" + json.dumps(obs) + "\n\n"
                "Episode notes:\n"
                "- alphabet: allowed input symbols.\n"
                "- budget: total act() calls available this episode.\n"
                "- n_states: total number of states to cover (0..n-1), start=0.\n"
                "- trap: whether trap transitions may exist; if trap_hit becomes true, success is impossible.\n\n"
                "Task: Use act() to gather enough evidence, then call submit_table(table_json) with a complete table for all states 0..n-1 and symbols A,B,C.\n"
                + ("If your table is incorrect, submit_table will return a counterexample, consume 1 query, and allow you to continue.\n" if enable_feedback else "")
                + "Use only act() and submit_table(...). Always terminate by calling submit_table(...)."
            ),
        }
        prompts.append([sys_msg, usr_msg])
        answers.append(ans)  # contains seed + meta; target kept hashed

    return Dataset.from_dict({"prompt": prompts, "answer": answers})


def make_env(dataset: Dataset, feedback: bool = False, **kwargs) -> vf.Environment:
    """Create a Dedeuce environment with the canonical rubric for scoring.

    Also sets a budget-aware max_turns unless provided:
    - If feedback is on: max_turns = budget + max(3, min(10, 2*n_states))
    - If feedback is off: max_turns = budget + 2
    This bounds transcript length without restricting model token output.
    """
    rubric = vf.Rubric(
        funcs=[
            reward_dedeuce,
            metric_queries,
            metric_correct,
            metric_trap,
            metric_budget_left,
        ],
        weights=[1.0, 0.0, 0.0, 0.0, 0.0],
        parser=Parser(extract_fn=lambda s: s),
        parallelize_scoring=False,
    )
    # Derive per-split budget + n_states from the first item, if not explicitly set
    eff_kwargs = dict(kwargs)
    if "max_turns" not in eff_kwargs:
        try:
            if len(dataset) > 0:
                meta = json.loads(dataset[0]["answer"])  # type: ignore[index]
                budget = int(meta.get("budget", 25))
                n_states = int((meta.get("table", {}) or {}).get("n", 0))
                if bool(feedback):
                    eff_kwargs["max_turns"] = int(budget + max(3, min(10, 2 * max(0, n_states))))
                else:
                    eff_kwargs["max_turns"] = int(budget + 2)
        except Exception:
            pass
    env = DedeuceEnv(
        dataset=dataset,
        message_type="chat",
        rubric=rubric,
        submit_feedback=bool(feedback),
        **eff_kwargs,
    )
    return env

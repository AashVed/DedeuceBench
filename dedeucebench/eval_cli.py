from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .env import build_dataset_from_split, make_env
from .adapters import get_adapter
import json as _json


def _effective_feedback(split_path: Path, subset: Optional[str], cli_flag: bool) -> bool:
    """Resolve feedback setting: CLI flag wins; otherwise read from split."""
    if bool(cli_flag):
        return True
    try:
        with open(split_path, "r", encoding="utf-8") as f:
            cfg_all = _json.load(f)
        # Multi-split {"splits": {...}} or top-level named subsets or single cfg
        if isinstance(cfg_all, dict) and ("splits" in cfg_all or any(k in cfg_all for k in ("test", "easy", "medium", "hard"))):
            if "splits" in cfg_all:
                if subset is None:
                    return bool(cfg_all.get("feedback", False))
                cfg = dict(cfg_all["splits"].get(str(subset), {}))
            else:
                if subset is None:
                    return bool(cfg_all.get("feedback", False))
                cfg = dict(cfg_all.get(str(subset), {}))
        else:
            cfg = cfg_all if isinstance(cfg_all, dict) else {}
        return bool(cfg.get("feedback", False))
    except Exception:
        return False


def _deterministic_none_agent(env, answer: str, state: Dict[str, Any]) -> str:
    """A deterministic placeholder agent: submit empty macro immediately.

    This avoids traps and consumes 0 queries; it will not succeed, but yields
    fully reproducible results for CI/docs and aggregation examples.
    Returns a terse string summarizing the action.
    """
    # Attach state and call the tool directly
    env._attach_state(state)  # type: ignore[attr-defined]
    try:
        _ = env.submit_macro("", 0)
    except Exception:
        # Fall back to table submission with an empty JSON
        _ = env.submit_table("{}")
    return "submit_macro(seq='', repeat=0)"


async def _run_chat_agent(
    env,
    prompt,
    state: Dict[str, Any],
    model_spec: str,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    base_url: Optional[str] = None,
    enable_feedback: bool = False,
    max_steps: int = 64,
    debug: bool = False,
) -> str:
    """Run a provider-agnostic chat model with tool-calling to interact with the env."""
    adapter, _, _ = get_adapter(model_spec, base_url=base_url)
    # Tool schemas
    # Build tool schemas. Hide submit_macro in identification/basic mode.
    mode = str(state.get("mode", ""))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "act",
                "description": "Execute one input symbol. Each call consumes 1 query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "enum": ["A", "B", "C"]}
                    },
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_table",
                "description": "Submit the exact transition table as a JSON string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_json": {"type": "string"}
                    },
                    "required": ["table_json"],
                },
            },
        },
    ]
    if mode == "control":
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "submit_macro",
                    "description": "Submit a controller: a sequence and repeat count.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seq": {"type": "string"},
                            "repeat": {"type": "integer", "minimum": 0}
                        },
                        "required": ["seq", "repeat"],
                    },
                },
            }
        )

    # Start chat with provided prompt (system+user messages)
    messages = prompt[:]  # prompt is already a list of {role, content}
    steps = 0
    actions = []
    env._attach_state(state)  # type: ignore[attr-defined]
    # Episode-level token accounting
    tokens_in = 0
    tokens_out = 0
    tokens_total = 0
    # Bound the number of model interaction steps per episode
    max_steps = int(max(1, max_steps))
    while steps < max_steps and not bool(state.get("done", False)):
        # Prefer forcing tool calls and JSON-only content; fall back gracefully if unsupported
        # Call adapter; it handles provider-specific fallbacks
        reply = adapter.chat(
            messages,
            tools=tools,
            tool_choice="required",
            response_format={"type": "json_object"},
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if debug:
            print(
                (
                    "\nDEBUG: reply received — size(bytes)=%d, tool_call_args_len=%d"
                    % (
                        int(getattr(sys, "getsizeof", lambda x: 0)(reply) or 0),
                        int(len(str(reply.tool_calls))) if reply.tool_calls else 0,
                    )
                ),
                file=sys.stderr,
                flush=True,
            )
        # Accumulate token usage if available
        try:
            u = reply.usage or {}
            tokens_in += int(u.get("prompt_tokens", 0) or 0)
            tokens_out += int(u.get("completion_tokens", 0) or 0)
            tokens_total += int(u.get("total_tokens", 0) or 0)
        except Exception:
            pass
        # Append the assistant message (with any tool_calls) to maintain conversation state
        try:
            raw_tool_calls = reply.tool_calls or []
            assistant_msg = {
                "role": "assistant",
                "content": (reply.content or ""),
                "tool_calls": [
                    tc for tc in raw_tool_calls
                ],
            }
            messages.append(assistant_msg)
        except Exception:
            raw_tool_calls = []
            messages.append({"role": "assistant", "content": "", "tool_calls": []})

        tool_calls = raw_tool_calls
        if not tool_calls:
            # No tool call; stop after recording assistant message
            break
        for tc in tool_calls:
            # tc is a normalized dict: {id, type, function: {name, arguments}}
            try:
                fn = tc.get("function", {})
                name = str(fn.get("name", ""))
                raw_args = fn.get("arguments", "{}")
                args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else {}
            except Exception:
                name = ""
                args = {}
            # Execute tool
            if name == "act":
                sym = str(args.get("symbol", ""))
                out = env.act(sym)
                actions.append(f"act({sym})")
            elif name == "submit_table":
                tj = str(args.get("table_json", "{}"))
                out = env.submit_table(tj)
                actions.append("submit_table(.)")
            elif name == "submit_macro":
                seq = str(args.get("seq", ""))
                rep = int(args.get("repeat", 0))
                out = env.submit_macro(seq, rep)
                actions.append(f"submit_macro(len(seq)={len(seq)}, repeat={rep})")
            else:
                out = json.dumps({"error": f"unknown tool {name}"})
            # Append tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": name,
                "content": out,
            })
            steps += 1
            if bool(state.get("done", False)):
                break
        if bool(state.get("done", False)):
            break
    # Stash token usage on state for scoring/recording
    state["tokens_in"] = int(tokens_in)
    state["tokens_out"] = int(tokens_out)
    state["tokens_total"] = int(tokens_total)
    return "; ".join(actions) if actions else "no_tool_calls"


async def _score_once(
    env,
    prompt,
    answer,
    model: str,
    *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    base_url: Optional[str] = None,
    episode_id: Optional[str] = None,
    enable_feedback: bool = False,
    max_steps: int = 64,
    debug: bool = False,
) -> Dict[str, Any]:
    # Initialize per-episode state via env hook
    state: Dict[str, Any] = {"answer": answer}
    state = await env.setup_state(state)

    # Run agent by model type
    act_summary: str
    if ":" in model and not model.startswith("heuristic:"):
        act_summary = await _run_chat_agent(
            env,
            prompt,
            state,
            model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            base_url=base_url,
            enable_feedback=enable_feedback,
            max_steps=int(max_steps),
            debug=bool(debug),
        )
    else:
        act_summary = _deterministic_none_agent(env, answer, state)
        # Deterministic baseline performs no API calls
        state.setdefault("tokens_in", 0)
        state.setdefault("tokens_out", 0)
        state.setdefault("tokens_total", 0)

    # Score via rubric (completion content unused for main metrics)
    scores = await env.rubric.score_rollout(
        prompt=prompt,
        completion=act_summary,
        answer=answer,
        state=state,
        task="default",
        info={},
    )

    # Extract key fields from state for reporting
    # Provider/model decomposition (if present)
    provider = "heuristic" if model.startswith("heuristic:") else (model.split(":", 1)[0] if ":" in model else "unknown")
    model_id = model.split(":", 1)[1] if ":" in model else model
    rec = {
        "model": model,
        "provider": provider,
        "model_id": model_id,
        "act": act_summary,
        "ok": bool(state.get("ok", False)),
        "trap_hit": bool(state.get("trap_hit", False)),
        "queries_used": int(state.get("queries_used", 0)),
        "budget_left": int(state.get("budget", 0)),
        "reward": float(getattr(scores, "reward", 0.0)),
    }
    # Token usage (if available)
    rec["tokens_in"] = int(state.get("tokens_in", 0))
    rec["tokens_out"] = int(state.get("tokens_out", 0))
    rec["tokens_total"] = int(state.get("tokens_total", 0))
    # Seed is embedded in answer JSON
    try:
        meta = json.loads(answer)
        rec["seed"] = int(meta.get("seed", -1))
        rec["budget_cfg"] = int(meta.get("budget", -1))
        rec["mode"] = str(meta.get("mode", ""))
    except Exception:
        rec["seed"] = -1

    return rec


def main():
    ap = argparse.ArgumentParser(description="DedeuceBench evaluation CLI")
    ap.add_argument("--split", type=str, required=True, help="Path to split JSON (single or multi-split; e.g., seeds/levels_dev.json)")
    ap.add_argument("--model", type=str, default="heuristic:none", help="Model identifier; combine with --provider or use provider:model form")
    ap.add_argument("--provider", type=str, default=None, help="Adapter/provider to use (e.g., openai, openrouter, anthropic, gemini)")
    ap.add_argument("--subset", type=str, default=None, help="Subset name when split JSON contains multiple named subsets (e.g., test/easy/medium)")
    ap.add_argument("--rollouts", type=int, default=1, help="Repeat eval per item (for stochastic models)")
    # Optional overrides for the split config
    ap.add_argument("--budget", type=int, default=None, help="Override budget from split")
    ap.add_argument("--traps", type=int, choices=[0,1], default=None, help="Override trap flag (0/1)")
    ap.add_argument("--mode", type=str, default=None, choices=["control", "basic"], help="Override mode")
    ap.add_argument("--n-states", dest="n_states", type=int, default=None, help="Override n_states")
    ap.add_argument("--target-len", dest="target_len", type=int, default=None, help="Override target_len")
    ap.add_argument("--out", type=str, default="results.jsonl", help="Output JSONL file path")
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=None, help="Max tokens per model response (omit or <=0 to let provider decide)")
    ap.add_argument("--temperature", dest="temperature", type=float, default=0.0, help="Sampling temperature (default 0.0 for determinism)")
    ap.add_argument("--top-p", dest="top_p", type=float, default=1.0, help="Nucleus sampling top_p (default 1.0)")
    ap.add_argument(
        "--max-steps",
        dest="max_steps",
        type=int,
        default=None,
        help=(
            "Max tool-use steps per episode (guardrail). "
            "If omitted, defaults to budget + buffer (feedback:on -> budget + max(3, min(10, 2*n_states)); feedback:off -> budget + 2)."
        ),
    )
    ap.add_argument("--feedback", dest="feedback", action="store_true", help="Enable feedback in submit_table: wrong submissions return a counterexample without ending the episode")
    ap.add_argument("--base-url", dest="base_url", type=str, default=None, help="Override base URL for OpenAI-compatible endpoints (e.g., OpenRouter)")
    # Transcript saving removed in v1 for simplicity and privacy
    ap.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Verbose debug logs to stderr (provider replies, sizes)",
    )
    args = ap.parse_args()

    model_spec = str(args.model).strip()
    if args.provider:
        provider_clean = str(args.provider).strip().lower()
        if not provider_clean:
            ap.error("--provider must be non-empty when supplied")
        if "::" in model_spec:
            ap.error("Model name contains '::'; supply plain model id")
        if ':' in model_spec:
            ap.error("When using --provider, pass the bare model id without provider: e.g., --provider openrouter --model openai/gpt-5-mini")
        model_spec = f"{provider_clean}:{model_spec}"
    args.model = model_spec

    split_path = Path(args.split)
    # Resolve feedback: CLI flag wins; otherwise use split-config
    feedback_eff = _effective_feedback(Path(args.split), args.subset, bool(args.feedback))
    dataset = build_dataset_from_split(
        split_path,
        subset=args.subset,
        mode=args.mode,
        trap=(bool(args.traps) if args.traps is not None else None),
        budget=args.budget,
        n_states=args.n_states,
        target_len=args.target_len,
        feedback=bool(feedback_eff),
    )
    env = make_env(dataset, feedback=bool(feedback_eff))
    # Derive a dynamic default for max_steps if not provided on the CLI
    if args.max_steps is None:
        try:
            if len(dataset) > 0:
                meta = json.loads(dataset[0]["answer"])  # type: ignore[index]
                _budget = int(meta.get("budget", 25))
                _n_states = int((meta.get("table", {}) or {}).get("n", 0))
                if bool(feedback_eff):
                    args.max_steps = int(_budget + max(3, min(10, 2 * max(0, _n_states))))
                else:
                    args.max_steps = int(_budget + 2)
            else:
                # Fallback if dataset is empty (should not happen)
                args.max_steps = 64
        except Exception:
            # Conservative fallback
            args.max_steps = 64

    # Deterministic ordering
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def run_all():
        rows = []
        total = len(dataset) * max(1, int(args.rollouts))
        done = 0
        t0 = time.time()
        for i in range(len(dataset)):
            rec = dataset[i]
            prompt, answer = rec["prompt"], rec["answer"]
            for r in range(max(1, int(args.rollouts))):
                # Attach per-run guardrails/state hints
                row = await _score_once(
                    env,
                    prompt,
                    answer,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    base_url=args.base_url,
                    episode_id=f"i{i}_r{r}",
                    enable_feedback=bool(args.feedback),
                    max_steps=int(args.max_steps),
                    debug=bool(args.debug),
                )
                rows.append(row)
                done += 1
                # Minimal progress to stderr (does not pollute stdout or output file)
                print(f"\rProgress: {done}/{total}", end="", file=sys.stderr, flush=True)
        # Write JSONL
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        # Final newline + timing summary
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.2f}s — wrote {out_path}", file=sys.stderr)

    asyncio.run(run_all())


if __name__ == "__main__":
    main()

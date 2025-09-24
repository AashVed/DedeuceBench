## DedeuceBench

DedeuceBench is an interactive active‑learning benchmark for LLM agents over hidden Mealy machines. Each episode has alphabet `{A,B,C}` and outputs `{0,1,2}`. Agents must actively probe with tool calls under a strict query budget, then submit an exact hypothesis to succeed. Optional “trap” transitions penalize unsafe exploration.

- Environment wrapper: builds a fixed, deterministic dataset of episodes and wraps the `dedeuce` env for scoring.
- Tasks: supports `control` and `basic`, but this benchmark’s primary objective is identification via exact table submission.
- Metrics: exact success, trap‑free rate, and query efficiency; token usage is recorded when using API providers.

### At a Glance
- Tools you can call: `act(symbol)` and `submit_table(table_json)`.
- Budget: each `act()` costs 1 query; submissions don’t cost budget.
- Success: exactly recover the hidden transition table (identification-first).
- Traps: optional edges that, once hit, make success impossible.
- Feedback mode: wrong `submit_table` returns a counterexample and you can keep going.

### What It Tests
- Identification under budget: plan probes to recover the exact transition table.
- Tool use + stateful interaction: correct multi‑step tool‑calling and memory across steps.
- Safety vs. efficiency: avoid “trap” edges while minimizing queries.

### Mealy Machines (Brief)
A Mealy machine is a finite‑state transducer: each input symbol produces an output, and the next state depends on both the current state and input. In DedeuceBench, the machine is hidden. The agent can `act(symbol)` to probe behavior (consumes 1 query) and must then submit an exact transition table via `submit_table(table_json)` to succeed in the primary benchmark task.

Control mode is supported by the underlying env and the CLI, but leaderboard scoring here focuses on table identification (see Success below).

## Quick Start

Run an evaluation on the public dev split with the deterministic baseline (no real actions):

```
dedeucebench-eval \
  --split seeds/levels_dev.json \
  --subset dev \
  --model heuristic:none \
  --rollouts 1 \
  --out results.jsonl
```

- `--model` can be `heuristic:none` (deterministic baseline) or a provider spec like `openai:<model>`.
- Results stream to JSONL, one row per episode.

Aggregate to a leaderboard CSV:

```
dedeucebench-aggregate results.jsonl > leaderboard.csv
```



### Using OpenAI

```
export OPENAI_API_KEY=...  # required
pip install -e ./DedeuceBench[all]  # brings the openai client

dedeucebench-eval \
  --split seeds/levels_dev.json \
  --subset dev \
  --model openai:gpt-4o-mini \
  --rollouts 1 \
  --out results.gpt.jsonl

dedeucebench-aggregate results.gpt.jsonl > leaderboard.gpt.csv
```

### Using Anthropic

```
export ANTHROPIC_API_KEY=...
# optional: set ANTHROPIC_BASE_URL for proxies
pip install -e ./DedeuceBench[anthropic]

dedeucebench-eval \
  --split seeds/levels_dev.json \
  --subset dev \
  --model anthropic:claude-3-5-sonnet-20241022 \
  --max-tokens 512 \
  --out results.claude.jsonl

```

### Using Gemini

```
export GEMINI_API_KEY=...  # or GOOGLE_API_KEY
pip install -e ./DedeuceBench[gemini]

dedeucebench-eval \
  --split seeds/levels_dev.json \
  --subset dev \
  --model gemini:gemini-1.5-pro \
  --max-tokens 512 \
  --out results.gemini.jsonl

```

### Model Adapters (Provider-Agnostic)

The CLI uses a tiny adapter layer so you can swap providers while keeping a single chat+tools interface.

- Supported providers:
  - `openai:<model_id>` (OpenAI Chat Completions)
  - `openrouter:<route_id>` (OpenAI-compatible proxy, e.g., OpenRouter)
  - `anthropic:<model_id>` (Claude 3 models; requires `pip install -e ./DedeuceBench[anthropic]`)
  You can name models either as `provider:model` or use the new `--provider` flag with a bare `--model` (e.g., `--provider openrouter --model openai/gpt-5-mini`).

Common flags:
- `--temperature` (default 0.0 for determinism)
- `--top-p` (default 1.0)
- `--max-tokens` (per-response cap)
- `--max-steps` (guardrail on tool-using steps per episode; default 64)
- `--debug` (verbose stderr logs; off by default)

OpenRouter usage (OpenAI-compatible endpoint):

```
export OPENAI_API_KEY=...  # your OpenRouter key
export OPENAI_BASE_URL=https://openrouter.ai/api/v1

dedeucebench-eval \
  --split seeds/levels_dev.json --subset dev \
  --provider openrouter \
  --model openai/gpt-5-mini \
  --out results.openrouter.jsonl
```

Notes:
- If `OPENAI_BASE_URL` is set, you can also use `--model openai:<route_id>`; the adapter will talk to the provided base URL.
- For official leaderboard entries, prefer direct provider SDKs where possible; OpenRouter is great for quick local testing.

### Tools Available to Agents
- `act(symbol)`: execute one input symbol (costs 1 query)
- `submit_table(table_json)`: submit exact transition table. If the table is incorrect, it consumes 1 query and does not end the episode; when `--feedback` is enabled a short counterexample is returned. If the table is correct, it ends the episode and does not consume budget.
- `submit_macro(seq, repeat)`: submit controller macro (ends episode; not part of the primary leaderboard)

Episodes end only on a correct submission (or when budget is exhausted). Wrong submissions consume 1 query and the episode continues; with `--feedback` the environment also returns a short counterexample. The rubric scores success, safety (trap‑free), and efficiency. The OpenAI adapter uses Chat Completions tool‑calling with correct assistant→tool sequencing for multi‑step tool use.

Note: table submissions are now accepted up to state relabeling (isomorphism), preserving the start state. This removes unfair dependence on hidden state numbering.

## Metrics
- Success@Budget: mean(ok)
- Trap‑Free Rate: mean(1 − trap_hit)
- Queries Used: mean(queries_used)
- Budget Left: mean(budget_left)

Primary scoreboard metric:
- Score100 = 100 × Success@Budget

Auxiliary (reported, not part of Score100):
- EffSucc = mean over successful episodes of (1 − used_budget / initial_budget)

## Result Schema (JSONL)
Each line corresponds to one episode rollout. Keys produced by the CLI:

- `model`: model spec used (e.g., `heuristic:none`, `openai:gpt-4o-mini`)
- `provider`: parsed provider (e.g., `heuristic`, `openai`, `openrouter`)
- `model_id`: parsed model id (e.g., `gpt-4o-mini`, `google/gemini-1.5-pro`)
- `act`: terse action summary, or `no_tool_calls`
- `ok`: boolean success
- `trap_hit`: whether any trap was traversed
- `queries_used`: number of `act()` calls used
- `budget_left`: remaining budget on termination
- `reward`: scalar reward from the rubric
- `seed`: episode seed (from the split)
- `mode`: `control` or `basic`
- `budget_cfg`: configured budget from the split
- `tokens_in`, `tokens_out`, `tokens_total`: token usage (if available; 0 for deterministic baseline)

Note: `ok=true` is achieved when the environment sets success after a correct submission. For the leaderboard here, we evaluate identification success (exact `submit_table`). Macro submissions are recorded but not used for the primary score.



### Datasets (Hugging Face)
- Dev (public): `dataset/dedeucebench-dev/levels_dev.json` — mirrors an easy configuration (S=2, budget=25) under subset name `dev` for convenience.
- Test (leaderboard): `dataset/dedeucebench-test/levels_test.json` — includes `easy` and `medium` subsets for public evaluation. Use `--subset easy` or `--subset medium` when running locally. The official leaderboard will use the hosted `dedeucebench-test` dataset.
- Results: upload your JSONL outputs to `comfortably-dumb/dedeucebench-results` for aggregation.

HF pages
- https://huggingface.co/datasets/comfortably-dumb/dedeucebench-dev
- https://huggingface.co/datasets/comfortably-dumb/dedeucebench-test
- https://huggingface.co/datasets/comfortably-dumb/dedeucebench-results

Programmatic download (dev example)

```
pip install huggingface_hub
python - << 'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    repo_id="comfortably-dumb/dedeucebench-dev",
    filename="levels_dev.json",
    repo_type="dataset",
)
print(p)
PY

dedeucebench-eval --split /path/to/levels_dev.json --subset dev --model heuristic:none --out results.dev.jsonl
```

Programmatic download (test examples)

```
pip install huggingface_hub
python - << 'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="comfortably-dumb/dedeucebench-test",
    filename="levels_test.json",
    repo_type="dataset",
)
print(path)
PY

# Easy subset (n_states=3, budget=35)
dedeucebench-eval --split /path/to/levels_test.json --subset easy --model heuristic:none --out results.test.easy.jsonl

# Medium subset (n_states=4, budget=60)
dedeucebench-eval --split /path/to/levels_test.json --subset medium --model heuristic:none --out results.test.medium.jsonl
```

## Canonical Benchmark Prompting
The benchmark exposes a canonical, identification‑first prompt that is optimized for LLM tool‑use compliance:

- Instructs “return ONLY function tool calls.”
- Requires JSON-only outputs (no prose). The adapter also requests `response_format={"type":"json_object"}` when supported.
- Forces tool usage via `tool_choice="required"` for compliant providers.
- Emphasizes identification: “Always finish by calling submit_table(table_json).”
- Provides tool schemas and a short probe‑then‑submit strategy under the budget.
- Includes an explicit submit_table JSON schema and a compact skeleton example in the system prompt to reduce malformed submissions.
 - Observation includes `n_states` (when available) so agents can format the full table with the correct number of states.

The underlying `dedeuce` environment remains general and continues to support both `basic` and `control`. The benchmark overrides prompts to focus on exact table submission for leaderboard comparability.

## Install

Install both the underlying env and the benchmark:

```
pip install -e ./dedeuce
pip install -e ./DedeuceBench
```

Note: use `dedeuce >= 0.1.3` to match the current submit_table semantics (wrong submissions consume 1 query and do not end the episode; counterexamples are returned only with `--feedback`).

Optional extras:
- `DedeuceBench[all]` installs the OpenAI client used by the OpenAI/OpenRouter adapter.
- Anthropic/Gemini adapters require installing the matching extras; without them, the CLI will raise an informative error telling you which package to install.


## Self‑Check

Run an end‑to‑end self‑check (dataset build → eval → aggregate) to verify installation and determinism:

```
dedeucebench-selfcheck
```

## Related Work
DedeuceBench is a focused probe of active system identification and safe control for LLM agents. It is inspired by classical active automata learning (e.g., L*‑style DFA/Mealy inference under membership/equivalence queries) and is complementary to broader agent benchmarks that emphasize tool use and multi‑step decision‑making, such as:

- AgentBench and related agentic evaluations (tool use, planning, and multi‑step interactions)
- WebArena / WebShop / Mind2Web (web interaction and goal‑directed action)
- BabyAI / MiniGrid / TextWorld (instruction‑following and RL in gridworlds/text MDPs)

Unlike those, DedeuceBench offers a compact, fully deterministic, hash‑guarded ground truth and a strict query budget over a finite‑state transducer, emphasizing data‑efficient probing and safety.

## Notes
- The default CLI ships with a deterministic baseline (`heuristic:none`) that simply terminates safely to produce reproducible files. To evaluate real models, use `--model openai:<name>` or integrate another provider in `dedeucebench/eval_cli.py`.
- The benchmark wrapper builds a fixed dataset from seeds, then reuses the original `dedeuce` env and rubric for scoring.

## License
MIT

## Citation

If you use DedeuceBench, please cite it. You can cite the latest version via the concept DOI, or pin this release.

- Latest (all versions, concept DOI): 10.5281/zenodo.17166596
- This specific version (v1.0.0): 10.5281/zenodo.17166597

[![DOI (latest)](https://zenodo.org/badge/DOI/10.5281/zenodo.17166596.svg)](https://doi.org/10.5281/zenodo.17166596)
[![DOI (v1.0.0)](https://zenodo.org/badge/DOI/10.5281/zenodo.17166597.svg)](https://doi.org/10.5281/zenodo.17166597)

For reproducibility, you may also reference the GitHub release tag.


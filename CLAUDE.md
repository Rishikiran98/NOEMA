# NOEMA: instructions for Claude Code

NOEMA is a private, CPU-native reasoning engine. The goal is a product that businesses deploy
on their own infrastructure, trust with meaningful work, and pay for. The first product is
**NOEMA Engine: private codebase reasoning with independent verification** for engineering
teams. General intelligence is a long-term research ambition; commercial reliability is the
engineering priority.

**The plan lives in [docs/ROADMAP.md](docs/ROADMAP.md).** Read it before starting work.

## How to pick up work

1. Open `docs/ROADMAP.md`, find the current milestone (the first one not marked DONE), and
   take the first unchecked task whose dependencies are done. If the user names a task ID
   (e.g. `M2.3`), do that one.
2. Work on a branch named `<milestone>-<short-name>` (e.g. `m2-repo-index`). Never commit to
   `main`, never merge a pull request, never force-push `main`.
3. In the same pull request, tick the task's checkbox in `docs/ROADMAP.md` and add one line
   under "Progress log" at the bottom: date, task ID, what changed, measured result if any.
4. The pull request description answers the three sprint questions (below) and lists
   what was NOT done or NOT measured.

## Commands

```bash
python -m compileall -q cpu_first cpu_platform tests     # syntax check
python -m unittest discover -s tests -v                  # full offline suite (must stay green)
python -m examples.verify_semantic                       # model-free demo
python -m cpu_platform.hardware --no-bandwidth           # hardware profile
python -m cpu_platform.launch --model qwen3-0.6b --dry-run
```

Python 3.11+, standard library only for runtime code. Don't add a runtime dependency without
a roadmap task that justifies it; optional extras (e.g. `psutil`) must degrade gracefully.

## Hard constraints

- **Local by default.** No mandatory GPU or cloud inference. Customer source code, prompts and
  traces never leave the machine unless the user explicitly configures it.
- **Never execute model-generated code in the API process.** Code runs only through the
  sandbox interface (roadmap M3), and every result records which backend ran it. The `dev`
  backend is not a security boundary and must say so.
- **No unverified accuracy claims.** Anything not measured is labelled `NOT RUN`, never
  `PASS`. Model answers are unverified unless a named checker verified a specific claim.
  Correct arithmetic or passing tests never imply the interpretation was correct.
- **Frozen evaluation data stays frozen.** Don't run `datasets freeze --force`; tune only on
  dev splits. Runs on eval splits need explicit user approval.
- **Keep every existing test passing.** Add regression and adversarial tests with every
  change. Don't weaken or delete a test to make a change pass; if behaviour changes on
  purpose, update the test and say why in the PR.
- **No production deployment, paid infrastructure or `main` merges** without the user's approval.

## Where real models run

This cloud workspace can't download model weights. Real-model runs happen on the user's
Windows laptop (i7-8850H, 6 cores, 32 GB RAM, CPU-only llama.cpp at
`vendor\llama-cpu\llama-server.exe`). For any task that needs real inference: build and test
it offline here with `tests/fake_llama.py`, then give the user the exact PowerShell commands
to run, and record their results in the progress log. Never invent or estimate results.

## The three sprint questions

Every PR must answer: Does it make NOEMA more **capable**? More **reliable**? Closer to a
**paying customer**? If none, it isn't a priority. Progress is measured by independently
verified task success, operating cost, reliability and customer value, not by version
numbers, line counts or test counts.

## Repository map

| Path | What it is |
|---|---|
| `cpu_first/core.py` | llama.cpp client (`LocalLlama`), exact arithmetic, thinking/budget controls |
| `cpu_platform/server.py` | loopback HTTP API (`/v1/reason`, `/v1/ask`, `/v1/verify-*`, `/v1/jobs`) |
| `cpu_platform/engine.py` | legacy reasoning routes (auto, hybrid, adaptive, proof, semantic) |
| `cpu_platform/contract.py` | NOEMA Engine request/result contract 1.0 ([docs/engine-contract.md](docs/engine-contract.md)); every new engine route returns an `EngineResult` |
| `cpu_platform/general.py` | `/v1/ask` general Q&A with evidence accounting |
| `cpu_platform/search.py`, `cot_check.py` | sampled reasoning, arithmetic checker, policies |
| `cpu_platform/logic.py` | exact propositional entailment checker |
| `cpu_platform/experiment6.py`, `report.py` | paired matched-compute experiments and reports |
| `cpu_platform/datasets.py`, `provenance.py` | frozen splits, leak guard, model hashes, evidence table |
| `cpu_platform/model_adapter.py` | checks the served model's identity, context and hash before serving (`--expect-model`); model-error classes live in `cpu_first/core.py` |
| `cpu_platform/launch.py`, `hardware.py` | hardware-aware CPU-only llama.cpp launcher |
| `cpu_platform/jobs.py` | SQLite job queue |
| `tests/fake_llama.py` | offline stand-in for llama.cpp's HTTP API (not a model) |

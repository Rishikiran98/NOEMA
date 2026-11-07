# NOEMA roadmap

**Definition of success:** NOEMA is a private, CPU-native reasoning engine that businesses can
deploy on their own infrastructure, trust with meaningful work, and pay to use.

**First product: NOEMA Engine.** Private reasoning infrastructure for software engineering
teams: deploy locally, connect a private codebase, ask complex engineering questions, run
controlled verification, and get auditable answers or proposed code changes, with no
cloud-hosted inference. First customers: teams with sensitive repositories, on-premises
requirements or strict inference-cost limits.

**Why this fits the architecture:** code comes with a real checker (tests and execution), and
claims about a codebase can be checked against its source. A local model proposes; exact
execution and source checks verify; search picks what survives. The differentiator is the
**audit trail**: what was executed, what passed, and which claims were verified against the
source.

How Claude Code uses this file: see [CLAUDE.md](../CLAUDE.md). Take the first unchecked task
in the current milestone, tick it in the same PR, and add a line to the progress log.

---

## v1.0 release gates

Proposed acceptance thresholds, not capabilities NOEMA has today. Numbers marked *TBD* are
fixed in M4, once a representative workload and hardware baseline exist.

| Dimension | v1.0 target |
|---|---|
| Accuracy | Measurable improvement over the direct-model baseline on the M4 workload, held-out, with a paired confidence interval |
| Verification | Every result separates verified facts (with the checker that verified them) from unverified model claims |
| Reliability | ≥ 99.5% of requests end in a well-formed result or explicit error under supported conditions |
| Privacy | Local-only by default; documented data retention; no mandatory external inference |
| Deployment | Reproducible install on supported Windows and Linux CPU hardware |
| Security | Isolated tool execution, access controls, security tests, no unresolved critical vulnerabilities |
| Performance | Published p50/p95 latency, RAM and CPU per supported hardware profile (*TBD targets*) |
| Commercial | ≥ 3 external pilots with repeat use, ≥ 1 paying customer |

---

## Status snapshot (v0.10.0)

**Works today:** CPU-only llama.cpp launcher with hardware detection and GPU hiding; local
API; exact arithmetic and propositional-logic checkers; reasoning-text arithmetic checker
with grounding; paired matched-compute experiment harness; frozen dev/eval splits with leak
guard; provenance (model SHA256, build, RSS); Markdown reports with failure labels; SQLite
jobs; 245 offline tests.

**Measured on real hardware so far:** Qwen3-0.6B at ~28 tokens/s on an i7-8850H (6 threads);
on 5 smoke questions, free-form reasoning 5/5 vs grammar-constrained plans 1/5.

**Not yet:** any codebase capability, code execution, a real-model benchmark beyond smoke
questions, installers, auth, or external users.

---

## M0: Evidence baseline (parallel with M1; runs on the user's laptop)

Closes out the arithmetic research thread so its result informs the engine design.

- [ ] **M0.1** Run the 20-question GSM8K pilot on the frozen **dev** split and commit its `results/*.report.md`.
  *Setup (once):* `py -m pip install datasets psutil`; `py scripts\fetch_gsm8k.py --limit 200`;
  `py -m cpu_platform.datasets freeze` only if `data\splits\manifest.json` does not exist yet
  (never `--force`); then `py -m cpu_platform.datasets validate`.
  *Servers:* two model servers, each in its own PowerShell window left running:
  `py -m cpu_platform.launch --model qwen3-0.6b --port 8080` and
  `py -m cpu_platform.launch --model qwen3-8b --port 8081`.
  *Run:* `scripts\run_windows_pilot.ps1 --dataset data/splits/dev.jsonl --limit 20 --output results/m0_pilot_dev20.jsonl`
  (later arguments override the script's defaults). The report header must read `split: **dev**`, not `unregistered` or `eval`.
  *Accept:* `results/m0_pilot_dev20.report.md` committed (`.gitignore` admits `results/*.report.md`); progress log records accuracy and CPU-seconds per arm.
- [ ] **M0.2** Decide which reasoning mode NOEMA Engine uses by default (prose reasoning, native thinking, sampled + checked) based on M0.1.
  *Accept:* decision and evidence written in `docs/decisions/0001-default-reasoning-mode.md`.

## M1: Engine foundation

- [x] **M1.1** Gap audit: v0.10 against the product goal (security, API stability, execution, retrieval, packaging, observability).
  *Accept:* `docs/audit-v0.10.md` with a prioritized gap list; each gap mapped to a roadmap task or marked out of scope.
- [ ] **M1.2** Coder-model presets in `hardware.py` / `launch.py`: Qwen3-Coder-30B-A3B (MoE, ~3B active parameters per token) and a dense 7B coder as fallback; RAM-fit checks.
  Each preset declares whether the model supports thinking; thinking flags (`/no_think`, `enable_thinking`, `--reasoning`) are sent only to models that do, not to anything whose name contains `qwen3` (audit G4).
  *Accept:* `--list-models` shows them; dry-run tests; user runs one and records tokens/s in the progress log.
- [x] **M1.3** Versioned request/result contract (`EngineRequest`, `EngineResult`) with fields for evidence, verified claims, unverified claims, execution records, cost and NOT RUN items.
  *Accept:* dataclasses plus JSON schema in `docs/`; contract tests; existing endpoints unchanged.
- [x] **M1.4** Model adapter interface: llama.cpp adapter validates model identity, context size and per-request thinking before serving (audit G5); transport failures (connection, timeout, HTTP error) are reported as retryable upstream errors, internal errors are not (audit G3).
  *Accept:* clear startup errors on mismatch; tests with `fake_llama.py`, including a programming error that must not come back as 503 retryable.

- [x] **M1.5** API hardening (audit G1, G2): reject requests whose `Host` is not a loopback name or the bound address, or whose `Origin` is cross-site, with 403 (DNS-rebinding defence); a catch-all handler turns any unexpected exception into `500 {"error": "internal_error", "request_id": ...}` and logs the traceback locally without request bodies; the launcher documents or sets `--api-key` for llama.cpp.
  *Accept:* regression tests for a foreign `Host`, a cross-site `Origin`, and an injected internal exception (client gets a JSON 500, not a dropped connection); all existing tests pass.

## M2: Codebase grounding

- [ ] **M2.1** Local repository indexer: walks a repo honouring `.gitignore`; skips binaries, large files and likely secrets (`.env`, keys); chunks Python by function/class via `ast`, other files by line windows; incremental re-index by mtime and hash.
  *Accept:* indexes NOEMA itself in < 10 s on the laptop; tests for ignore rules, secret skipping and incremental updates.
- [x] **M2.2** Retrieval: BM25 keyword search over chunks plus symbol lookup (definitions, call sites). Standard library only.
  *Accept:* retrieval eval on ≥ 30 hand-written NOEMA questions with gold files; recall@5 reported.
- [x] **M2.3** Grounded codebase Q&A: retrieve, answer with `file:line` citations, return an EngineResult.
  *Accept:* `/v2/repo/ask` endpoint (or CLI) works against a fake model; real-model run recorded.
- [x] **M2.4** Citation verifier: every cited file exists, line ranges are valid, quoted code matches the source exactly; unverifiable citations are labelled, not dropped silently.
  *Accept:* adversarial tests (invented file, wrong lines, paraphrased quote, real-but-irrelevant citation flagged as "exists, relevance NOT CHECKED").
- [x] **M2.5** Prompt-injection hardening for repository content (comments or docs telling the model to ignore instructions).
  *Accept:* adversarial fixtures; repository text always passed as data, never as instructions.

## M3: Execution and verification

- [x] **M3.1** Sandbox interface (`run(command, workdir, timeout, network=False) -> ExecutionRecord`) with a `dev` backend: temporary copy of the repo, timeouts, output caps, network-disabling environment, resource limits where the OS supports them.
  *Accept:* every ExecutionRecord states its backend and that `dev` is **not a security boundary**; tests for timeout, output cap, and no writes to the original repo.
- [x] **M3.2** Container backend (Docker or WSL2): no network, read-only base, CPU/RAM limits, non-root.
  *Accept:* same contract tests as M3.1 pass; documented setup on Windows and Linux.
- [x] **M3.3** Test runner: detect and run `unittest` / `pytest`; parse pass/fail per test; record duration.
  *Accept:* works on NOEMA's own suite inside the sandbox.
- [x] **M3.4** Patch pipeline: model proposes a unified diff; validate it (applies cleanly, touches only allowed paths, size limit); apply in the sandbox copy only.
  *Accept:* adversarial diffs (path traversal, binary, huge, touching tests to make them pass) are rejected or flagged.
- [ ] **M3.5** Propose-and-test search: N candidate patches, run targeted then full tests, keep passing ones, prefer the smallest diff; stop early when one passes.
  *Accept:* matched-cost comparison against single-shot in M4.
- [ ] **M3.6** Audit report: Markdown and JSON per request, covering question, retrieved evidence, citations and their status, commands run, test results, patch, cost, and what was NOT checked.
  *Accept:* a human can reproduce every verified claim from the report alone.

## M4: Evaluation (the accuracy gate)

- [ ] **M4.1** Injected-bug benchmark: AST mutation operators (flip comparison, off-by-one, swapped operands, wrong constant, dropped condition) applied to functions covered by tests; keep only mutants the tests catch; the engine must restore passing tests without editing tests.
  *Accept:* ≥ 200 mutants from NOEMA plus ≥ 1 other open-source Python repo; frozen dev/eval split using `cpu_platform.datasets`; generator seeds recorded.
- [ ] **M4.2** Repository-QA benchmark: questions with gold files/lines and checkable answers.
  *Accept:* ≥ 50 questions, frozen split.
- [ ] **M4.3** Baselines at matched CPU-seconds: (a) model alone, no retrieval; (b) model + retrieval, no verification; (c) NOEMA full (retrieval + citation check + sandboxed tests + search).
  *Accept:* paired results with bootstrap intervals; held-out eval run once with settings frozen on dev.
- [ ] **M4.4** Performance profile per supported hardware: p50/p95 latency, peak RAM, CPU-seconds, index time.
  *Accept:* published table; sets the *TBD* release-gate numbers.

**Go/no-go after M4:** if NOEMA full doesn't beat model + retrieval at matched cost, revisit the
search and verification design before M5. The audit trail alone may still be a product, but
the claim changes.

## M5: Productization

- [ ] **M5.1** Stable versioned API plus Python SDK; deprecation policy for legacy routes.
- [ ] **M5.2** Authentication (local API keys → configurable), roles, audit log of every request.
- [ ] **M5.3** Threat model and security test suite (OWASP LLM Top 10 as reference); secrets redaction in logs and traces.
- [ ] **M5.4** Windows installer and Linux package with pinned llama.cpp build and model hashes; no PowerShell steps for end users.
- [ ] **M5.5** Monitoring: health, request metrics, error rates, resource use; opt-in only.
- [ ] **M5.6** Deployment and operations docs; upgrade and data-retention controls.

## M6: Customer validation

- [ ] **M6.1** Three external pilots on real repositories (on their hardware, their data stays local).
- [ ] **M6.2** Failure taxonomy from pilot use feeds back into M2–M4.
- [ ] **M6.3** Willingness-to-pay test; pricing and licensing model.
- [ ] **M6.4** v1.0 launch when every release gate is met.

---

## Out of scope until after v1.0

Pretraining or fine-tuning foundation models; general-intelligence claims; multi-node
clusters; public internet-facing SaaS; autonomous agents acting outside the sandbox; new
arithmetic benchmark modes not needed by M0.

## Known risks

- **Model capability:** small models may not pick correct fixes often enough for search to
  help; mitigated by MoE coder models (M1.2) and measured in M4.
- **CPU latency:** large repos and multi-candidate search can be slow; M4.4 decides the supported envelope.
- **Sandbox escape:** the `dev` backend is not isolation; nothing customer-facing ships on it (M3.2).
- **Benchmark leakage:** public code benchmarks are in training data; injected-bug and
  private-repo evaluations (M4.1, M6) are the primary evidence.
- **Overclaiming:** passing tests don't prove a fix is correct; reports say exactly what was checked.

---

## Progress log

One line per merged PR: `YYYY-MM-DD | task | change | measured result (or NOT RUN)`.

- 2026-11-03 | — | Roadmap and CLAUDE.md created | NOT RUN
- 2026-11-03 | M1.1 | `docs/audit-v0.10.md`: 20 prioritized gaps mapped to tasks; added M1.5; M0.1 instructions fixed (dev split, two servers, report commit) | 2 P0 gaps confirmed by local probe (DNS-rebinding Host accepted; internal error drops connection); no model run
- 2026-11-03 | M1.5 | Host/Origin allowlist (403) + `--allowed-host`; catch-all JSON 500 with request id; JSON 405 for PUT/PATCH/OPTIONS; opt-in `LLAMA_API_KEY` sent by all model clients, launcher records whether llama.cpp enforces it; 21 new tests (266 total) | Offline only; real llama.cpp key enforcement NOT RUN
- 2026-11-03 | M1.2 (code; task stays open) | Presets `qwen3-coder-30b-a3b` (MoE, 3.3B active) and `qwen2.5-coder-7b`; per-preset thinking flag (fixes audit G4: no `/no_think` / `--reasoning` for non-hybrid models), KV-cache size, 16k context; RAM budget max(60%, RAM − 10 GB); `--list-models` shows role, active params, memory need; 16 new tests (282 total) | tokens/s NOT RUN: awaiting laptop run
- 2026-11-03 | M1.3 | `cpu_platform/contract.py`: EngineRequest/EngineResult 1.0 (evidence, checked claims verified/refuted, unverified claims, executions, cost, NOT RUN), strict stdlib decoder + cross-field rules (dev backend never a security boundary; unmeasured cost must be in not_run); generated `docs/engine-contract-v1.schema.json`; v1 ask/logic adapters; /v1 unchanged; 28 new tests (310 total) | Offline only; no model run
- 2026-11-03 | M1.4 | `cpu_platform/model_adapter.py` + `--expect-model/--min-ctx/--expect-sha256`: served model's identity, context and file hash checked before serving, fail closed (G5); model errors classified transport (503, retryable) / rejected 4xx (502) / bad response (502) / mismatch (503), bugs no longer reported as retryable outages (G3); thinking requests to non-thinking models are 400; 29 new tests (339 total), suite 28 s → 18 s | Offline only; real llama.cpp `/props` fields and `-hf` model ids NOT RUN
- 2026-11-04 | M2.1 (code; task stays open for the laptop timing) | `cpu_platform/repo_index.py` + `ignore.py`: gitignore-compatible walker (matches `git ls-files -co --exclude-standard` on a tricky fixture and on NOEMA), no symlink/junction following, binary/non-UTF-8/large/secret (name + content) skipping without storing secrets, `ast` chunks per function/method/class header + 60-line windows, symbols, SQLite index outside the repo (0600), incremental by size/mtime then SHA256; 24 new tests (363 total) | Cloud container: NOEMA 117 files / 1,182 chunks in 0.22 s, unchanged re-index 0.005 s; laptop NOT RUN
- 2026-11-04 | M2.2 (code; task stays open for the held-out run) | `cpu_platform/retrieval.py`: code-aware BM25 (path/symbol weighting, definition bonus, docs/tests × 0.6), `defs` and `calls` (call sites with enclosing function; index schema 2 adds refs, v1 auto-upgrades); 40 frozen questions written before the code (20 dev / 20 eval, SHA256 manifest), `.noemaignore` keeps them unindexed; 22 new tests (385 total) | dev (tuned on): recall@5 0.80 plain BM25 → 0.95 frozen, recall@1 0.75, MRR 0.82; held-out eval (user-approved, run once at `978a353`): recall@5 0.95, recall@1 0.70, MRR 0.80; one miss (q25, lexical gap)
- 2026-11-04 | M2.3 (code; task stays open for the real-model run) | `cpu_platform/repo_qa.py` + `POST /v2/repo/ask` + CLI: retrieve, line-numbered untrusted excerpts (delimiters neutralised), answer with `[path:START-END]`, `EngineResult` with every excerpt as `file_span` evidence and the answer + each citation (inside/overlapping/outside the excerpts) as unverified claims; no model call when nothing is retrieved; server answers only for `--repo NAME=PATH` registrations; 26 new tests (411 total) | Fake model only; real-model run NOT RUN
- 2026-11-04 | M2.3 real run, M1.4 real check | User's laptop, Qwen3-Coder-30B-A3B Q4_K_M at `ecb99ef`: `--expect-model qwen3-coder-30b-a3b` passed on real llama.cpp; 3 questions answered in 58.7 s / 99.8 s / 158.8 s (1,766-3,424 prompt tokens); human-checked: 2 correct with exact real citations, 1 wrong basis (retrieval missed `core.py:96-97`, model cited docs and invented "a test case"); parser now reads `[E3:59]`-style citations the model used | Tokens/s split NOT MEASURED (M1.2 open); RAM not measured (working set read 0.02 GB, mmap'd weights not counted)
- 2026-11-04 | M2.4 | `cpu_platform/citation_check.py`, run by `repo_qa` on every answer: path safety (no absolute, `..`, symlink escapes), reads only files the index accepted (secret/ignored files never opened), file and line-range checks, quotes (inline exact, fenced modulo indentation) tied to the citation that follows them, results as `citation_check.file/lines/quote` checked claims; relevance always NOT CHECKED; `verify` CLI re-checks saved results; bracketed dotfile/extensionless citations now parsed; 24 new tests (435 total) | On the real laptop answers: Q1's 4 citations and 2 quotes verified (a first version falsely refuted 2 of them, caught and fixed); Q2's doc citations verify as existing with relevance NOT CHECKED, so the checker cannot catch that answer's error
- 2026-11-04 | M2.5 | `cpu_platform/injection.py` + `injection_eval.py`, wired into `repo_qa`: repository text only inside escaped `<excerpt>` blocks (tested for every fixture), chat-template tokens / `<think>` / bidi and zero-width characters made inert and visible, rule restated after the excerpts, instruction-like excerpts flagged (never dropped) and answer echoes flagged; 8 adversarial fixtures with canaries in `data/injection/` (not indexed as NOEMA code); 18 new tests (453 total) | Flags on NOEMA: 11 of 1,401 chunks, all model-directed text; fixtures 8/8 flagged (development number); stand-ins 8/8 vs 0/8; real-model attack success rate NOT RUN
- 2026-11-04 | M3.1 | `cpu_platform/sandbox.py`: `Sandbox.run(command, workdir, timeout, network) -> ExecutionRecord` + limits report; `dev` backend (requires explicit acknowledgement; every record `backend=dev`, `security_boundary=false`): temporary copy without ignored/secret/symlinked files, argv only, process-tree kill on timeout, output tail cap + flood kill, credential-free environment with dead-port proxies (network NOT blocked), POSIX rlimits (memory, file size, CPU, core); `changes()` diff; CLI; 25 new tests (478 total) | Linux: all pass (60 s sleep killed at 1 s; surviving grandchild killed, mutation-checked; 2 GiB allocation refused at 512 MiB; original repo hash unchanged); Windows NOT RUN
- 2026-11-05 | M3.3 | `cpu_platform/testrun.py`: detect pytest (config + importable) or unittest; per-test outcome/duration from a nonce-checked results file (NOEMA's unittest runner script, or pytest `--junitxml`), subtests/setUpClass/import errors, targeted runs, exit-code consistency check; documented limit: in-process forgery not detectable (pinned by a test). Fixed 3 M3.1 sandbox bugs found by the acceptance run: dead-port proxy captured loopback, copy applied `.noemaignore`, HOME/TEMP inside the copy (self-copy, index refused); 19 new tests (497 total) | NOEMA's own suite in a dev sandbox: 477 passed, 1 skipped (no `.git`), 0 failed, ~36 s; full local suite now ~81 s; Windows NOT RUN
- 2026-11-06 | M3.2 (code; task stays open for the real-Docker acceptance run) | `cpu_platform/container.py`: Docker backend (`--network none`, `--read-only` + tmpfs, single `--mount` of the sandbox folder, `--user 65534`, `--cap-drop ALL`, `no-new-privileges`, memory/CPU/pids/fsize limits, `--pull never`, `docker kill` on timeout); `security_boundary=True` only after an in-container probe (non-root, read-only root, no outbound connection, writable work area) passes, else refuses; sandbox launch refactored into backend hooks; `--backend container` on the sandbox and testrun CLIs; `tests/fake_docker.py` stand-in; docs with Windows/Linux setup; 24 new tests (521 total, 5 real-Docker skipped) | Offline through the fake: all pass (flags, every probe gate, kill on timeout, output cap, no writes, test runner); real Docker NOT RUN
- 2026-11-07 | M3.2 real run | User's laptop: Docker Desktop 4.98 (engine 29.9.2, WSL2 kernel 6.18.54.1), image `python:3.12-slim` (`sha256:4840d514c583…`), code at `4b87d43`: `py -m unittest tests.test_m32_container -v` | RealDocker 5/5 passed: isolation probe; no outbound connection, `/etc` not writable, non-root, host home not visible; timeout, output cap, no writes to the original; 1 GiB allocation refused at 256 MiB; test runner in the container. 14 fake-Docker tests skip on Windows (POSIX-only, expected). M3.2 ticked
- 2026-11-07 | fixes from the first container run of the full suite | User's laptop, `py -m cpu_platform.testrun . --backend container`: 507 passed, 2 failed, 1 error, 11 skipped (tools absent from `python:3.12-slim`), 111 s, `security_boundary: true`. Causes: Git for Windows CRLF checkout changed the frozen question file's raw SHA256 (fixed: `frozen_sha256` reads CRLF as LF, same manifest hash; `.gitattributes` `data/** -text`; regression test); a test expected output ending in exactly `OK` (now accepts `OK (skipped=N)`) | Re-run on the laptop NOT RUN yet
- 2026-11-07 | M3.4 | `cpu_platform/patch.py`: unified-diff parser (also inside ```diff fences), validation with reject/flag (path traversal, absolute/UNC, `.git`/CI/secret/start-up-hook paths, test files and pytest config protected, binary/NUL/non-UTF-8, renames/modes/symlinks, Trojan Source characters, size limits, duplicates; test-machinery tampering flagged), atomic exact-context application to the sandbox copy only (unique offsets, CRLF, no-final-newline), `changes()` cross-check; `check`/`try` CLI; 22 new tests (544 total) | Fix loop on a fixture: tests fail, patch, tests pass, original byte-identical; a test-editing "fix" is rejected; real model diffs NOT RUN
- 2026-11-07 | first full suite on Windows (host) | User's laptop at `f2f2c32`, `py -m unittest discover -s tests`: 544 tests in 396 s, 1 failure, 6 errors, 24 skipped. Causes, all fixed: (1) early 401/403/413 replies without reading the request body; Windows resets such connections, so clients got WinError 10053 instead of the response (4 errors): the server now drains up to 1 MiB, with a 5 s timeout, before replying; (2) `RepoIndex` kept its SQLite file open when refusing another repository's index, and a test leaked a connection; Windows cannot delete open files (2 errors, plus the nested-suite failure that contained them). The Windows sandbox (taskkill tree kill) and test-runner paths passed. A second laptop run, still at `f2f2c32` without these fixes, repeated them and showed one more case of the same Windows reset in two *test* fake servers (`LauncherKeyStatus`, and `fake_llama`'s 401 path) that replied before reading the body; both fixed | Re-run with the fixes: see next line
- 2026-11-07 | Windows + container verification at `78495c9` | User's laptop: `py -m unittest discover -s tests` 546 tests, **OK (skipped=24)**, 0 failures, 0 errors, 396 s (first fully green Windows run; M3.1 `taskkill` and M3.3 paths included); `py -m cpu_platform.testrun . --backend container` **535 passed, 0 failed, 0 errors**, 11 skipped (jsonschema/git/pytest absent from `python:3.12-slim`, RealDocker and nested-suite guards), 115.6 s, `security_boundary: true` | Verified on Windows host and in Docker

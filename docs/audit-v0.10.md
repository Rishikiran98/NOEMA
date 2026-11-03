# Gap audit: v0.10.0 against NOEMA Engine (roadmap M1.1)

**Date:** 2026-11-03. **Commit audited:** `fee5a7a` (main, after the roadmap merge).
**Goal audited against:** [ROADMAP.md](ROADMAP.md) definition of success and v1.0 release gates:
private codebase reasoning with independent verification, deployed on the customer's own CPU hardware.

**Method.** Read the runtime modules (`cpu_first/`, `cpu_platform/`), scripts, CI workflow, deploy
example and existing readiness docs; ran the offline suite (245 tests, OK); and ran small probes
against a live in-process `cpu_platform.server` (no model) to confirm the behaviour of two findings
(G1, G2). Every finding marked **confirmed** was reproduced; findings marked **by reading** come from
the code only. No real model was run for this audit: anything about model behaviour is NOT RUN.

## Summary

v0.10 is a sound **research harness**: exact checkers, a loopback-only model client, frozen-split
evaluation discipline, provenance, and an honest "unverified" label on model answers. It has **no
codebase capability at all** (no indexing, retrieval, citations, execution, or patches), so M2–M3 are
new construction, not refactoring. The most important pre-existing defects are two API reliability
and security gaps (G1, G2) that are cheap to fix and should be fixed before any external pilot; the
current roadmap had no task for them, so this audit adds **M1.5**.

## What already holds (keep, build on it)

- **Local by default.** `LocalLlama` (`cpu_first/core.py:164`) and `LazyLocalModel`
  (`cpu_platform/engine.py:66`) refuse any non-loopback model URL; the API refuses to bind off
  loopback (`cpu_platform/server.py:42`); the launcher binds llama.cpp to `127.0.0.1` and hides GPUs.
- **No model-generated code is executed.** Arithmetic is evaluated by an AST walker
  (`exact_arithmetic`), logic by a truth-table checker (`logic.py`). The only `subprocess` use is the
  launcher probing `llama-server --help` (`launch.py:81`). Nothing to remove before M3.
- **Honest verification labels.** `/v1/ask` returns `verified: False` with an explicit scope note
  (`general.py:146`); only exact arithmetic and propositional entailment are marked verified.
- **Evaluation discipline.** Frozen dev/eval splits with content fingerprints, fail-closed registry,
  logged eval looks (`datasets.py`, `experiment6.py:450`); evidence tables print NOT RUN.
- **Provenance.** Loaded model SHA256, llama.cpp build info and peak RSS per run (`provenance.py`).
- **Request hygiene.** Strict JSON content type, bounded body, field allowlists per route,
  constant-time key comparison, bounded concurrency with 429, no request logging.
- **Jobs.** SQLite WAL queue with idempotency, single-owner OS lock, tenant scoping, TTL.

## Prioritized gap list

Priority: **P0** blocks any external pilot; **P1** blocks the M4 accuracy gate or v1.0;
**P2** needed for v1.0 polish or later. Each gap maps to a roadmap task or is marked out of scope.

| ID | Pri | Area | Gap | Evidence | Maps to |
|---|---|---|---|---|---|
| G1 | P0 | Security | **No Host/Origin check: DNS-rebinding exposure when no API key is set (the default).** A web page the user visits can rebind its domain to 127.0.0.1 and call the API same-origin. **Confirmed:** `POST /v1/verify-logic` with `Host: attacker.example:8070` and a matching `Origin` returns 200. Once M2/M3 exist this would expose indexed source code and sandbox runs. The llama.cpp server on 8080 has the same exposure (no `--api-key`). | `server.py:96-110`, no Host validation anywhere | **M1.5** (new) |
| G2 | P0 | Reliability | **Unexpected exceptions drop the connection instead of returning an error.** `do_POST` catches only `ValueError` and `UpstreamUnavailable`; anything else propagates to `socketserver`, which prints a traceback and closes the socket. **Confirmed:** an injected `RuntimeError` in `engine.reason` gives the client `RemoteDisconnected`. Violates the release gate "well-formed result or explicit error". | `server.py:234-290` | **M1.5** (new) |
| G3 | P1 | Reliability | Model-call failures of every kind, including programming errors, are re-raised as `UpstreamUnavailable` and returned as 503 `retryable: true`. Real bugs look like transient outages and clients retry them. | `engine.py:87-90, 183-184, 214-215` (by reading) | M1.4 (adapter separates transport errors from internal errors) |
| G4 | P1 | Model | **Thinking control keys on the substring `"qwen3"`.** `Qwen3-Coder-30B-A3B` is a non-thinking instruct model but matches, so every prompt gets `/no_think` appended and `enable_thinking` sent; the launcher likewise adds `--reasoning auto`. Effect on output quality: NOT RUN. | `core.py:184-200`, `launch.py:213` | M1.2 (preset declares thinking support) and M1.4 |
| G5 | P1 | Model | The client serves whichever model is `data[0]` in `/v1/models`; it does not check identity, context size or thinking support against what the caller expects. Model hash is recorded only by the experiment harness, not by the API. | `core.py:173-177` | M1.4 |
| G6 | P1 | Model | No coder presets; largest preset is `qwen3-14b`. RAM-fit check exists but has never been applied to an MoE model, whose RAM is set by total, not active, parameters. | `hardware.py:41-47` | M1.2 |
| G7 | P1 | API stability | No common result contract: each of the six POST routes returns a different shape; errors are `{"error": code}` with an optional `detail`. No schema, no field for execution records, citations, cost or NOT RUN items. | `server.py:236-287` | M1.3 |
| G8 | P1 | API limits | Request body capped at 16 KiB (`MAX_BODY_BYTES`) and questions at 4,000 characters (`Settings.max_chars`). Sufficient for arithmetic; too small for code questions, diffs or file context. | `server.py:28`, `engine.py:59` | M1.3 (limits per route in the contract), M2.3 |
| G9 | P1 | Retrieval | No repository indexing, search, symbol lookup or citations of any kind. | absent | M2.1–M2.4 |
| G10 | P1 | Security | No prompt-injection handling. Today inputs are short user questions; once repository text is placed in prompts, it must be passed as data. | absent | M2.5 |
| G11 | P1 | Execution | No sandbox, test runner, patch validation or execution records. (Correctly absent; see "What already holds".) | absent | M3.1–M3.6 |
| G12 | P1 | Evaluation | Only GSM8K and 5 smoke questions; no code benchmark, no matched-cost code baselines. | `data/`, `docs/benchmark*.md` | M0.1, M4.1–M4.3 |
| G13 | P1 | Observability | `log_message` is a no-op and nothing else logs: no request audit trail, no error log (G2 tracebacks go only to stderr). Metrics are in-memory and reset on restart. | `server.py:80-82`, `metrics.py` | M5.2 (audit log), M5.5 (monitoring); M3.6 covers per-request audit reports |
| G14 | P2 | Security | Auth is static bearer keys from environment variables; no roles, rotation or per-request audit. Fine for loopback; not for shared deployments. | `server.py:96-104, 320-323` | M5.2 |
| G15 | P2 | Privacy | With `--jobs-db`, prompts and results are stored as plaintext; TTL enforced lazily; SQLite deletes are not secure erasure (already documented in `product-readiness.md`). No file-permission tightening on the DB. Once prompts contain customer source this matters more. | `jobs.py:4`, `product-readiness.md` | M5.3 (redaction), M5.6 (retention controls) |
| G16 | P2 | Packaging | Not reproducible: `build_cpu_llama.sh` clones llama.cpp `HEAD` (`--depth 1`) and only prints the commit; presets download via `--hf-repo` with no expected hash. No console entry points; package name `cpu-reasoning-platform` and README title differ from "NOEMA". Windows setup is several manual PowerShell steps. | `scripts/build_cpu_llama.sh:12`, `hardware.py:38-47`, `pyproject.toml` | M5.4 |
| G17 | P2 | CI | CI is manual-dispatch only (to avoid hosted-runner charges); hosted Windows/Linux runs are recorded as pending in the README. Pull requests are not automatically tested. | `.github/workflows/ci.yml:2-3` | M5.4 (release builds); running CI on PRs needs user approval of runner cost: **decision for the user** |
| G18 | P2 | API stability | Legacy reasoning routes (`/v1/reason` modes auto/hybrid/adaptive/proof/semantic) and GSM8K-specific prompts sit inside the product server. They are not part of NOEMA Engine. | `engine.py`, `server.py:203` | M5.1 (deprecation policy) |
| G19 | P2 | Performance | Only one real-hardware number (Qwen3-0.6B ≈ 28 tokens/s). No latency/RAM profile for any coder model, index time, or multi-candidate search cost. | ROADMAP status snapshot; `docs/hardware.md` gives only estimates | M1.2 (tokens/s), M4.4 |
| G20 | P2 | Docs | `README.md` is a stack of version histories (v0.10 → v0.9.1 → …) and describes a research preview, not the product. | `README.md` | M5.6 |

### Out of scope (until after v1.0, per the roadmap)

- Internet-facing TLS termination, SSO, multi-tenant SaaS (deploy behind the customer's own reverse proxy).
- Distributed or multi-host job scheduling; GPU support.
- Physical secure erasure of SQLite data (delegated to disk encryption, as already documented).
- Paraphrase-level contamination detection for GSM8K (the guard catches copies, not paraphrases;
  M4 relies on generated mutants and private repositories instead).

## Roadmap changes made with this audit

- **Added M1.5: API hardening (G1, G2).** Reject requests whose `Host` is not a loopback name or the
  configured bind address (and a cross-site `Origin`) with 403; wrap every handler in a catch-all that
  returns `500 {"error": "internal_error", "request_id": ...}` and logs the traceback locally without
  request bodies; document `--api-key` for the llama.cpp server in the launcher. Small, test-first,
  and should land before any pilot.
- G3–G5 are folded into the existing acceptance criteria of M1.2 and M1.4 (see ROADMAP).

## Not done / not measured in this audit

- No real model was run; G4's effect on answer quality is NOT RUN.
- No dependency or CVE scan (standard library only, so the surface is CPython and llama.cpp; the
  llama.cpp build is unpinned, G16).
- The browser playground (`static/index.html`) was not reviewed beyond its CSP header.
- Windows-specific behaviour (job-store locking via `msvcrt`, PowerShell scripts) was read, not run.
- Fixes for G1 and G2 are **not** part of this audit; they landed separately as M1.5 (see the ROADMAP progress log).

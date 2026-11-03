# NOEMA Engine contract 1.0 (`EngineRequest` / `EngineResult`)

Roadmap task M1.3. Code: [`cpu_platform/contract.py`](../cpu_platform/contract.py).
Generated schema: [`engine-contract-v1.schema.json`](engine-contract-v1.schema.json)
(JSON Schema 2020-12; regenerate with `python -m cpu_platform.contract --write docs/engine-contract-v1.schema.json`,
check with `--check`; a test fails if it is stale).

**Status:** the contract, validator and schema exist and are tested. No endpoint speaks it yet:
the existing `/v1` routes are unchanged, and `from_v1_ask` / `from_v1_logic` show how their
results map onto it. The first native producer is grounded repository Q&A (M2.3).

## Why this shape

Every engine answer must let a reader tell apart what was **checked** from what was merely
**said**. So there is deliberately **no top-level `verified` flag**. A model answer is
unverified unless a named checker verified a specific claim, and then only that claim,
within the stated `scope`. Correct arithmetic or passing tests never imply the
interpretation was correct, and the contract has no field that could say otherwise.

## EngineRequest

| Field | Type | Default | Notes |
|---|---|---|---|
| `contract_version` | `"1.0"` | `"1.0"` | |
| `question` | string, 1–20,000 chars | required | must not be blank |
| `task` | `ask` \| `repo_ask` \| `repo_fix` | `ask` | `repo_ask` = grounded Q&A (M2.3); `repo_fix` = propose and test a patch (M3) |
| `repo` | `{path, revision?}` or null | null | required for `repo_*`, forbidden for `ask` |
| `budget` | `{max_model_calls 1–64, max_output_tokens 1–32768, max_wall_seconds >0}` | `2, 1024, 600` | |
| `allow_execution` | bool | false | only `repo_fix` may set it |
| `thinking` | bool or null | null | null = the model adapter decides (M1.4) |
| `client_request_id` | string `[A-Za-z0-9_.:-]{1,64}` or null | null | echoed for correlation |

Requests are always decoded **strictly**: unknown fields, wrong types (`1` for `true`, `true` for
`1`), NaN/infinite numbers and out-of-range values are rejected, and every problem is reported
at once (`ContractError.problems`).

## EngineResult

| Field | Meaning |
|---|---|
| `request_id`, `task`, `engine_version`, `contract_version` | identity |
| `status` | `ok` (an answer exists; it may still be unverified) · `incomplete` (no usable answer within budget) · `error` (see `error`) |
| `answer` | text or null; required when `status` is `ok` |
| `evidence[]` | what was looked at: `input`, `model_output`, `file_span` (path + lines + SHA256), `command_output` |
| `checked_claims[]` | a specific claim, `result: verified \| refuted`, the named `checker`, its `scope`, and the `evidence_ids` it rests on (at least one) |
| `unverified_claims[]` | a claim and **why** it was not checked |
| `executions[]` | command, `backend`, `security_boundary`, `network`, exit code, timeout, duration, output tails |
| `cost` | model calls, tokens, model and total wall seconds, `cpu_seconds`, `peak_rss_bytes` (null = not measured) |
| `not_run[]` | everything not run or not measured, with a reason |
| `model` | model id and SHA256 (null if unknown) |
| `error` | `{code, message, retryable}`; present exactly when `status` is `error` |

### Rules enforced in code (beyond the schema)

1. `error` is present **exactly** when `status` is `error`; `answer` is required when `status` is `ok`.
2. Every `evidence_ids` entry in a claim must name an `evidence` or `executions` id; ids are unique across both.
3. A `file_span` has `start_line` and `end_line`, with `end_line >= start_line >= 1`.
4. An execution on the **`dev` backend can never set `security_boundary: true`** (CLAUDE.md: the
   dev backend is not a security boundary and must say so).
5. Every cost field left null (`cpu_seconds`, `peak_rss_bytes`) must appear in `not_run` as
   `cost.<field>`: unmeasured never silently becomes zero. `not_measured(cost)` builds those entries.
6. Producers call `validate_result()`, which round-trips through the decoder, so the engine
   cannot emit a result a consumer would reject.

## Versioning

`contract_version` is `"<major>.<minor>"`. A **minor** version only adds optional fields; a
reader that must accept newer minors decodes with `result_from_dict(data, strict=False)`, which
ignores unknown fields (other major versions are still rejected). Removing or changing a field
is a **major** version and a new schema file (`engine-contract-v2.schema.json`).

## Mapping from `/v1`

| v1 result | EngineResult |
|---|---|
| `/v1/ask`, route `exact_arithmetic` | one `checked_claim` (`exact_arithmetic`, verified); no unverified claims |
| `/v1/ask`, model answer | the answer is an `unverified_claim` with v1's `verification_scope` as the reason; each arithmetic equality the audit recomputed becomes its own `checked_claim` (verified or **refuted**, with the exact value) about that equation only |
| `/v1/ask` issues, code/factual domains | `not_run` entries ("code execution", "source retrieval", each v1 issue) |
| `/v1/verify-logic` | one `checked_claim` (`propositional_logic`) |
| v1 `usage` | `cost`; `cpu_seconds` and `peak_rss_bytes` are not measured there, so they are listed in `not_run` |

## Not done in M1.3

- No `/v2` endpoint and no server changes (by design: "existing endpoints unchanged").
- No patch/diff record: added as an optional field in a 1.x minor version with M3.4.
- Request size limits for a future endpoint (audit G8) are defined here (`MAX_QUESTION_CHARS`)
  but not wired into the HTTP server, whose v1 limits are unchanged.
- `/v1/reason` (the legacy GSM8K modes) has no adapter; it is slated for deprecation (M5.1).

# Propose-and-test fix search (M3.5)

Code: [`cpu_platform/fix_search.py`](../cpu_platform/fix_search.py). Built on the
[sandbox](sandbox.md) / [container](container.md), [test runner](testrun.md) and
[patch pipeline](patch.md). Model-generated code runs only in sandbox copies.

```powershell
py -m cpu_platform.fix_search data\fix_fixtures\inventory_offbyone --backend container --expect-model qwen3-coder-30b-a3b -n 4
```

## What it does

1. **Baseline:** runs the tests in a fresh sandbox (`--select` limits them) to find what fails.
   If nothing fails, the model is not called.
2. **Context for the model**, most specific first, as neutralized `<excerpt>` blocks:
   * non-test code where a failure traceback points;
   * definitions of the functions each failing test **calls** (an assertion failure has no
     traceback frame inside the code that returned the wrong value), then the module-level code
     (constants, imports) of those files;
   * the failing tests' own source (read-only);
   * a keyword search over the test names and failure messages.
3. **Up to N candidates:** call 1 at temperature 0, later calls at 0.7, each with its own seed.
   Each answer must contain one diff, which goes through the patch pipeline (tests, CI, secrets
   and start-up hooks protected). It is applied to a **fresh copy** and run against the previously
   failing tests first, then, only if those pass, against the **full suite**.
4. **Choice:** among candidates that pass the full suite, the smallest diff (changed lines),
   earliest on ties. By default the search stops at the first candidate that passes
   (`--no-stop-early` tries all N).

Each candidate ends as `no_diff`, `rejected` (policy), `does_not_apply`, `targeted_failed`,
`full_failed`, `passed` or `error` (the model call failed).

## What the `EngineResult` (task `repo_fix`) says

| Section | Contents |
|---|---|
| `executions` | every test run: `B-X1` (baseline), `C<n>T-…` (targeted), `C<n>F-…` (full), with backend and `security_boundary` |
| `checked_claims` | "before any patch, k selected tests fail"; per candidate "passes the patch policy", "applies to the repository" (with any lenient reading named), "the previously failing tests pass", "the full test suite passes", each verified or **refuted**, citing the execution or the candidate's answer |
| `unverified_claims` | "candidate n's patch fixes the bug": passing tests do not show that, and on `dev` the results could be forged |
| `not_run` | fix correctness, candidates skipped (early stop, request budget, wall-time budget), unmeasured costs |
| `answer` | the chosen diff, or null |

The request must be `task: "repo_fix"` with `allow_execution: true`. The request budget caps
the number of candidates (`max_model_calls`) and the wall time (`max_wall_seconds`).

## Fixtures

[`data/fix_fixtures/`](../data/fix_fixtures/): three small repositories, each with one bug its
tests catch (a boundary `<` vs `<=`, a wrong constant, a missing condition). Their reference
fixes live only in `tests/test_m35_fix_search.py`, never in the fixtures, and `.noemaignore` keeps
the fixtures out of NOEMA's own index.

## Measured

Offline, with a scripted stand-in model (not a model): every candidate outcome, early stopping,
smallest-diff choice, budgets, errors, contract validity, context ordering, each fixture fixed
by its reference patch through the search, and by imperfect diffs of the kind the real model wrote.
23 tests.

**Real model, first run (user's laptop, 2026-11-07, `qwen3-coder-30b-a3b`, container backend, `-n 4`):**

| Fixture | Result | Cost |
|---|---|---|
| `inventory_offbyone` | **passed**: candidate 1, `<` to `<=` (2 changed lines), targeted and full suites passed | 1 call, 658+103 tokens, 42 s model, 62 s total |
| `pricing_wrong_constant` | 4/4 candidates `does_not_apply` | 4 calls, 51 s total |
| `text_dropped_condition` | 4/4 candidates `malformed` | 4 calls, 75 s total |

Both failures were in reading the model's diff, not in the tests. The raw answers were not
captured, so the cause is inferred: the excerpts show a module's top-level lines without the
blank lines between them, so a model writing context can drop a blank line, and models often
miscount `@@` headers. The [patch reader](patch.md) is now lenient in bounded, flagged ways
(recount, `@@ @@` headers, trimmed context or whitespace-insensitive placement, one match
required), with offline tests for those exact mistakes on these fixtures. The CLI now prints
why each candidate failed. `--show-answers` prints the raw answers (untrusted, made inert for
the terminal). **Re-run after the fix: NOT RUN.**

```powershell
py -m cpu_platform.fix_search data\fix_fixtures\inventory_offbyone --backend container --expect-model qwen3-coder-30b-a3b -n 4 --show-answers
```
```powershell
py -m cpu_platform.fix_search data\fix_fixtures\pricing_wrong_constant --backend container --expect-model qwen3-coder-30b-a3b -n 4 --show-answers
```
```powershell
py -m cpu_platform.fix_search data\fix_fixtures\text_dropped_condition --backend container --expect-model qwen3-coder-30b-a3b -n 4 --show-answers
```

Each model call takes about 10 s to 1 minute here. Three hand-made fixtures say little about real
capability; M4.1's mutation benchmark is the measurement.

## Not done in M3.5

* M3.5's acceptance is the matched-cost comparison against a single attempt, which runs in M4.3.
  M3.5 stays unticked until then.
* No feedback loop: a failed candidate's test output is not fed into the next attempt
  (candidates are independent samples).
* Duplicate candidates are not removed; Python repositories only.

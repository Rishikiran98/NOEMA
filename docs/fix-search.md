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
| `answer` | the chosen change as a clean diff regenerated from the files (what was applied and tested, not the model's text), or null; the model's answer is evidence `P<n>` |

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
24 tests.

**Real model (user's laptop, 2026-11-07, `qwen3-coder-30b-a3b`, container backend, `-n 4`):**

| Fixture | First run (`a0dde09`) | After lenient diff reading (`a339b95`) |
|---|---|---|
| `inventory_offbyone` | **passed**, candidate 1 (`<` to `<=`); 1 call, 62 s | **passed**, candidate 1; 1 call, 9 s model, 13 s total |
| `pricing_wrong_constant` | 4/4 `does_not_apply`; 51 s | **passed**, candidate 1 (`0.02` to `0.2`); 1 call, 689+90 tokens, 8 s model, 17 s total |
| `text_dropped_condition` | 4/4 `malformed`; 75 s | **passed**, candidate 1 (`slug.rstrip('-')` after truncation); 1 call, 644+152 tokens, 14 s model, 18 s total |

So **3/3** passed the targeted and full suites on the first candidate, against 1/3 before. The
first run's failures were in reading the model's diff, not in the tests, and the second run's flags
confirm the cause. The model miscounted `@@` headers (`-3 +3` for a 4-line body, `-11 +13` for
`-10 +12`), and it wrote one blank line where the file has two. The excerpts show top-level
lines without the blank lines between them. The [patch reader](patch.md) now recounts and places
such hunks leniently, with bounds and flags. Both are reported in the "applies" claim.

That run's printed patch was still the model's own text (wrong counts, one blank line missing),
which `git apply` would refuse. Since then the reported patch is regenerated from the files:
the change that was actually applied and tested, as a clean diff with 3 context lines. Offline
tests check that it applies strictly and gives the same result. Real-model re-run of that
change: NOT RUN (it does not affect what is tested).

Passing tests do not make a fix correct (the slug fix strips dashes only after truncation, which
is the only place they can arise here; that is a reading, not a check). Three hand-made fixtures
say little about real capability, and model time varies (42 s for a first call, 8 to 14 s once the
server is warm). M4.1's mutation benchmark is the measurement.

```powershell
py -m cpu_platform.fix_search data\fix_fixtures\pricing_wrong_constant --backend container --expect-model qwen3-coder-30b-a3b -n 4 --show-answers
```

`--show-answers` prints the raw answer of each candidate that did not pass (untrusted, made inert
for the terminal).

## Not done in M3.5

* M3.5's acceptance is the matched-cost comparison against a single attempt, which runs in M4.3.
  M3.5 stays unticked until then.
* No feedback loop: a failed candidate's test output is not fed into the next attempt
  (candidates are independent samples).
* Duplicate candidates are not removed; Python repositories only.

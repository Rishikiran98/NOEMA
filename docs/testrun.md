# Test runner (M3.3)

Code: [`cpu_platform/testrun.py`](../cpu_platform/testrun.py). Runs a repository's tests inside a
[sandbox](sandbox.md) and reports each test's outcome and duration.

```python
from cpu_platform.sandbox import Sandbox, DevBackend
from cpu_platform.testrun import run_tests
with Sandbox("path/to/repo", DevBackend(acknowledge_not_isolated=True)) as box:
    run = run_tests(box)                                  # detect and run everything
    run = run_tests(box, select=["tests.test_x.Case"])    # targeted run (M3.5: targeted, then full)
    run.status          # passed | failed | error | timed_out
    run.counts          # passed / failed / error / skipped / xfail / xpass / total
    run.tests[0]        # id, outcome, duration_s, message (last 2,000 characters)
    run.record          # the sandbox ExecutionRecord (backend, security_boundary, exit code, ...)
```

```powershell
py -m cpu_platform.testrun . --i-understand-dev-is-not-isolated            # non-passing tests only
py -m cpu_platform.testrun . --i-understand-dev-is-not-isolated --details  # every test
```

## Detection

**pytest** when the repository configures it (`pytest.ini`, `pyproject.toml [tool.pytest…]`,
`setup.cfg [tool:pytest]`, `tox.ini [pytest]`, or a `conftest.py`) **and** it can be imported by
the interpreter used in the sandbox. Otherwise **unittest**, discovering `test*.py` under `tests/`
(or the repository root), the way `python -m unittest discover -s tests` does.
`framework=` overrides detection. NOEMA does not depend on pytest. It uses whatever the repository's
interpreter has.

## How per-test results are obtained

Console output is capped (8,000 characters per stream), so results come from a file:

* **unittest:** NOEMA writes a small runner script into the sandbox's temp folder. It records every
  test's outcome and duration as JSON, including subtests (any failing subtest fails the test),
  `setUpClass` errors and modules that fail to import.
* **pytest:** its built-in `--junitxml` report, parsed with the standard library (xfail is
  recognised).

Each run uses a fresh random nonce. A results file without it (stale, or planted by the code) is
rejected. Outcomes are cross-checked against the exit code, and a contradiction is reported as
`results inconsistent` with status `error`. A run that exits before writing results, collects
nothing, or times out is `error` / `timed_out`, never `passed`.

## What a passing run does NOT show

* **Forgery:** results are reported by the process the code under test runs in. Code that is
  actively malicious can read the nonce and write a clean report. A test pins this limitation
  (`test_in_process_forgery_is_NOT_detected_documented_limit`), so the claim stays true. Isolation
  (M3.2) is what would address it.
* **Correctness:** passing tests show that those tests passed, not that the code is correct, and
  not that the tests test the right thing (CLAUDE.md). Every `TestRun` carries this as `trust_note`.

## Measured

NOEMA's own suite inside a `dev` sandbox (Linux, cloud container): **477 passed, 1 skipped** (the
git comparison test; the copy has no `.git`), 0 failed, about 36 s, with every test reported
individually and the copy unmodified afterwards. This is the M3.3 acceptance test. It runs in the
normal suite, and skips itself when already inside a sandbox, to avoid recursion.

Getting there exposed three bugs in the M3.1 sandbox, now fixed:
1. The dead-port proxy variables also captured **loopback** traffic (`no_proxy` was empty), so every
   test talking to a local fake server failed. Loopback now bypasses the proxy.
2. The copy applied `.noemaignore`, so `data/retrieval` and `data/injection` were missing.
   `.noemaignore` only governs NOEMA's index; the copy now follows `.gitignore` only.
3. HOME/TEMP lived **inside** the copy. A sandbox started inside a sandbox could copy itself, and
   an index created in TEMP was refused as "inside the repository". They now live beside the copy.

Windows (user's laptop, 2026-11-07): the M3.3 tests passed except the nested full-suite run, which
failed on two Windows-only bugs elsewhere (unclosed SQLite files), since fixed. Re-run NOT RUN yet. Pytest projects are tested with
pytest 9.2 installed for the tests only.

## Not done in M3.3

* No tamper-resistant reporting (needs M3.2 isolation).
* No coverage measurement; no flaky-test detection or reruns.
* Other ecosystems (JavaScript, Go, …) are not supported.

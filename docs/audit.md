# Audit reports (M3.6)

Code: [`cpu_platform/audit.py`](../cpu_platform/audit.py). One Markdown and one JSON report per
request, built from the request and its [`EngineResult`](engine-contract.md). The JSON is the
complete record; the Markdown is the same content for people.

```powershell
py -m cpu_platform.fix_search data\fix_fixtures\pricing_wrong_constant --backend container --expect-model qwen3-coder-30b-a3b -n 4 --audit audits
```
```powershell
py -m cpu_platform.repo_qa . "How does NOEMA keep secrets out of the repository index?" --expect-model qwen3-coder-30b-a3b --audit audits
```

Each writes `audits\<request_id>.audit.json` and `.audit.md`. From saved files:
`py -m cpu_platform.audit render RESULT.json --request REQUEST.json --out audits`.

## What a report contains

| Section | From |
|---|---|
| Request | question, task, repository path and revision, budget, whether execution was allowed |
| Repository state | tree SHA-256 of the files a sandbox copies, file count, git commit and whether the tree was modified (evidence `repo`, taken when the request ran) |
| Answer, Patch | the answer; for `repo_fix`, the change that was applied and tested, as a clean diff |
| Checked claims | every verified **and refuted** claim, its checker and scope, and **how to reproduce it** |
| Citations | each citation and what its check established |
| Unverified claims | what no checker verified, and why |
| Test results | per run: framework, selection, patch, status, counts, every non-passing test |
| Commands run | backend, security boundary, network, exit code, time-out, duration, command, output tails |
| Evidence | every excerpt, with source, lines, SHA-256 and the step that produced it |
| Cost | model calls, tokens, model and total time; CPU seconds and peak RAM say NOT MEASURED |
| NOT checked | the result's `not_run`, plus claims the report cannot support (below) |

## How a claim is reproduced

Each checked claim has steps, commands and the expected output. `<REPO>` is the repository in
the recorded state: `py -m cpu_platform.audit digest <REPO>` must print the same `tree_sha256`.
Files the commands need, such as a patch or a model answer, are written out of the report with
`py -m cpu_platform.audit save REPORT.json ID --out FILE`.

| Checker | Command | Expected |
|---|---|---|
| `testrun.*`, no patch (baseline) | `py -m cpu_platform.testrun <REPO> --backend container [--select …]` | exactly the recorded tests fail |
| `testrun.*`, with candidate n | `py -m cpu_platform.patch try Dn.diff --repo <REPO> --backend container [--select …]` | `applied` true; `tests.status` passed (or not, if refuted) |
| `patch.validate` | `py -m cpu_platform.patch check Pn.txt --repo <REPO>` | `ok` true or false |
| `patch.apply` | `py -m cpu_platform.patch try Pn.txt --repo <REPO> --backend container --no-tests` | `applied` true or false |
| `citation_check.file` / `.lines` / `.quote` | `py -m cpu_platform.audit span <REPO> FILE START END [--quote … --kind …]` | `exists`, `in_range` and the span's SHA-256, or `quote_found` |
| `injection.*`, `exact_arithmetic`, `cot_check.arithmetic`, `propositional_logic` | manual steps | stated in the report |

Test claims also give the plain equivalent (`git apply`, then `python -m unittest …` or
`python -m pytest …`) for checking NOEMA without NOEMA's tools, in a disposable copy only: the
patch is model output. Without Docker, `--backend dev --i-understand-dev-is-not-isolated`
works but is NOT a security boundary.

A claim the report cannot support says `reproducible: false` with the reason, and appears under
"NOT checked". That happens when an excerpt was cut at the report's 4,000-character limit (a
longer answer or patch, unless it is the full answer), a test run's selection or outcomes did not
fit, a quote was longer than the claim keeps (300 characters), or a checker has no recipe.

## What reproducing does NOT show

Only what the checker showed. Passing tests do not make a fix correct and can be forged by
malicious code under test ([testrun.md](testrun.md)). A quote existing in a span does not make
the citation relevant. A different repository state, or flaky tests, can give a different result.
That is why the state is recorded.

## Measured

Offline, 13 tests. A scripted reader that sees only the saved JSON report, saves files with
`audit save`, runs the printed commands (substituting only the documented dev backend) and
compares the output with the expectations reproduces all of these:
- every claim of a `fix_search` run: 11 claims covering a policy rejection, a patch that does
  not apply, a failing candidate, and a leniently read passing one;
- every citation claim of a `repo_qa` run, including refuted files, line ranges and quotes.

Changing the repository afterwards breaks the affected reproductions and changes the digest.
Also tested: unknown checkers, missing repository state, cut-off test summaries, Markdown fences
around backticks, and span paths outside the repository.

**By hand, user's laptop (Windows, 2026-11-07, `4311770`):** a real-model `fix_search` report
(`pricing_wrong_constant`, `qwen3-coder-30b-a3b`, container backend). Each command printed in the
report was run and gave the expected output:
- `audit digest`: 3 files, git commit `4311770`, clean;
- `testrun`: exactly `test_tax.VatTests.test_gross` and `test_vat_part` failed;
- `patch check P1.txt`: `ok` true, flag `recounted`;
- `patch try P1.txt --no-tests`: `applied` true, flags `recounted` and `fuzzy_context`;
- `patch try D1.diff`: `applied` true with **no** flags (the clean diff), 3/3 tests passed.

All five checks reproduced. The digest was not compared with the report's value in the pasted
output, and the targeted (`--select`) claim was not repeated separately.

Not measured:
- The cost of the repository digest. Every file is read and hashed once per request, so on
  large repositories it adds time that has not been measured.

## Not done in M3.6

* No signing: a report can be edited after the fact. Hashes tie claims to content, not to NOEMA.
* No report from the HTTP API yet (CLI only); `/v2/repo/ask` records the repository state, so a
  client can build one with `audit render`.
* Revisions other than the working tree are not supported, as elsewhere.

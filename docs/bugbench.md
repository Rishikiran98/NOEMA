# Injected-bug benchmark (M4.1)

Code: [`cpu_platform/mutate.py`](../cpu_platform/mutate.py) (operators) and
[`cpu_platform/bugbench.py`](../cpu_platform/bugbench.py) (build, freeze, check, run). Data:
[`data/bugbench/`](../data/bugbench/). The task is the [fix search](fix-search.md) task: restore
passing tests without editing tests.

## How mutants are made

1. **Base repositories.**
   - `noema`: NOEMA itself at a pinned commit, materialized with `git archive`. Six modules are
     not mutated: `sandbox`, `container`, `launch`, `server`, `hardware` and `jobs`. They manage
     processes, sandboxes, ports or files on the host, and the build runs mutants on the dev
     backend.
   - `more_itertools`: more-itertools 10.5.0 (MIT), vendored unchanged with its tests.
     [`PROVENANCE.json`](../data/bugbench/repos/more_itertools/PROVENANCE.json) records the
     source URL and the sdist's SHA-256.
2. **Sites.** Every function body is parsed with `ast`. Each candidate is a minimal edit of
   the source text, so the diff is the bug and nothing else.

   | Operator | Edit |
   |---|---|
   | `flip_comparison` | `<` to `>=`, `==` to `!=`, `in` to `not in`, `is` to `is not`, and back |
   | `off_by_one` | `<` to `<=` and similar boundaries; an integer in a comparison, index, slice, `range()` or `+`/`-` changed by one |
   | `swapped_operands` | the operands of `-`, `/`, `//`, `%`, `**`, `<<`, `>>` swapped |
   | `wrong_constant` | `True`/`False` swapped; a float, or an integer of magnitude 2 or more, times 10; a short string that is compared, passed to a string method or returned, changed |
   | `dropped_condition` | one operand of `and`/`or` removed, or an `if` guard that only returns, raises, breaks or continues deleted |

   Docstrings, decorators, annotations, default values and f-strings are never mutated. Every
   mutant compiles and parses to a different AST.
3. **Scope.** The test modules that mention the function's name (1 to 3 modules; names
   starting with `__` or shorter than 3 characters are skipped). The scope must pass on the base
   within 20 s.
4. **Keep only caught mutants.** The scope runs with the mutant in a sandbox copy. A mutant is
   kept only if the run fails, with at least one failing test and no time-out. Surviving mutants
   are dropped, so equivalent mutants never enter the benchmark.
5. **Order and balance.** The seed (20261108) shuffles the functions. Each function gets at most 2
   sites, picked each time from the operator used least so far. The build stops at the target
   per repository.

Each mutant records its id, repository, file, function, operator, the bug and reference-fix
diffs, the SHA-256 of the original file, its test scope and the tests that caught it. The
reference fix is never shown to the engine.

## Split and freezing

dev and eval are split **by function**, seeded: all mutants of a function go to the same side,
so tuning on dev cannot learn a function that appears in eval. `manifest.json` records:
- the generator version, the seed and the build backend;
- each base (commit or tree SHA-256);
- the build statistics;
- the SHA-256 of `mutants.jsonl` and the dev and eval ids.

`build` refuses to overwrite a frozen benchmark without `--force`; don't use `--force`
(CLAUDE.md). `check` verifies the hashes and splits, that every bug applies strictly and its
reference fix restores the file, and with `--run-tests N` that a sample is still caught. A run
on eval needs `--i-have-approval-for-eval` and is logged in `data/bugbench/eval_runs.jsonl`
(`cpu_platform.datasets.log_eval_run`).

## Running it (laptop, real model)

```powershell
py -m cpu_platform.bugbench status
```
```powershell
py -m cpu_platform.bugbench check
```
```powershell
py -m cpu_platform.bugbench run --split dev --limit 20 --backend container --expect-model qwen3-coder-30b-a3b -n 4 --out bugbench_dev.jsonl
```

For each mutant, `run`:
1. materializes the base and applies the bug;
2. runs `fix_search` with the mutant's scope as the baseline and the "full" suite;
3. records solved or not, whether the chosen patch restores the original file exactly, the
   candidates and the cost.

The summary gives solved/attempted with a Wilson 95% interval overall, per repository and per
operator. A mutant whose tests do not fail on that machine (for example, because a test is
skipped in the container image) is reported under `not_caught_here` and left out of the rate.
`--audit DIR` writes an [audit report](audit.md) per mutant. `--resume` keeps the mutants
already in the results file (dev only) and retries those a model error cut short;
`summarize RESULTS` recomputes the summary from a results file, including a partial one.

The summary counts how candidates ended (`candidate_outcomes`). For candidates whose patch
applied but whose tests failed, `failed_candidates` splits why:
- `not_fixed`: the originally failing tests still fail and nothing else broke;
- `broke_others`: the original failures pass, but other tests in scope now fail;
- `both`;
- `untargeted`: how many of these could not run the failing tests on their own (doctests).

Each row keeps every candidate's applied diff and these test lists (`candidate_details`), so
failures can be studied without calling the model again. Rows written before 2026-11-15 lack
them and count as `no_detail`.

## What a solved mutant does NOT show

- The tests in scope pass, nothing more. A patch can pass without restoring the original
  (`exact_restore` is reported, but not scored: a different correct fix is fine). It can also
  pass while being wrong where the tests don't look.
- The "full" suite in a benchmark run is the mutant's scope, not the repository's whole suite.
- Contamination: more-itertools and NOEMA are public. The mutants are new, but a model may know
  the surrounding code.
- Single-function, single-edit bugs are easier than real bugs; this measures a floor of capability.

## Measured

**The frozen benchmark (built 2026-11-07, cloud container, dev backend, seed 20261108):**
277 mutants, `mutants.jsonl` SHA-256 `8d4f0331…afe52`. dev has 139 mutants and eval 138.

| Base | Kept / tested | Survived (dropped) | Timed out (dropped) | flip | off-by-one | swap | constant | dropped cond. | Build time |
|---|---|---|---|---|---|---|---|---|---|
| NOEMA at `8e833cf` | 117 / 193 | 76 | 0 | 36 | 20 | 23 | 21 | 17 | 642 s |
| more-itertools 10.5.0 | 160 / 179 | 15 | 4 | 48 | 35 | 24 | 18 | 35 | 583 s |

NOEMA's candidates ran out before the target of 160. Only functions whose name appears in 1 to
3 test modules qualify, and 39% of NOEMA's tested mutants survived its tests, against 8% for
more-itertools. That survival rate is itself a finding: NOEMA's tests miss many small changes
to its own code.

Offline tests (stand-ins, not models):
- every operator, and the edits it must not make;
- building on a small repository: only caught mutants of tested functions are kept, the build
  is deterministic and frozen, and a tampered file is refused;
- splits never share a function;
- the runner: an oracle that returns the reference fix solves its mutant with an exact restore,
  and a null model solves none;
- the eval guard and log;
- the frozen benchmark itself: counts, a pristine vendored base, every bug and fix applies, and
  a sample is still caught.

**Real model on dev: NOT RUN.** Eval: not run, and it needs the user's approval.

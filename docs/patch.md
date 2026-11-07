# Patch pipeline (M3.4)

Code: [`cpu_platform/patch.py`](../cpu_platform/patch.py). Takes a model-proposed unified diff,
validates it, and applies it **only to a sandbox copy**. The function takes a `Sandbox`, never a
path. Standard library only.

```python
from cpu_platform import patch as pt
p = pt.parse(model_answer)                     # raw diff, or ```diff fenced blocks in prose
check = pt.validate(p, box.root)               # problems: reject | flag
result = pt.apply(box, p)                      # validates again, then applies atomically
run = run_tests(box)                           # M3.3
```

```powershell
py -m cpu_platform.patch check fix.diff --repo .
py -m cpu_platform.patch try fix.diff --repo . --backend container      # apply in a sandbox, then run the tests
```

## What is rejected (the patch is not applied)

| Code | When |
|---|---|
| `path` | absolute, `..`, drive letter or UNC path, or not normalized (`app/./x.py`) |
| `denied_path` | `.git/`, CI configuration (`.github/`, `.gitlab-ci.yml`, …), `.noemaignore`, and files Python runs by itself at start-up: `sitecustomize.py`, `usercustomize.py`, `*.pth` (they could rig every test run) |
| `secret_path` | likely-secret files (`.env`, keys, …) |
| `touches_tests` | test files (`tests/`, `test_*.py`, `*_test.py`), `conftest.py`, `pytest.ini`, `tox.ini`, `noxfile.py`, and `setup.cfg`/`pyproject.toml` when they hold a pytest section. The engine must make tests pass **without editing them** |
| `binary` | git binary patches, "Binary files … differ", NUL bytes, non-UTF-8 added text |
| `rename`, `mode_change` | renames, copies, chmods, symlinks: refused, never silently skipped |
| `invisible_characters` | bidi/zero-width characters in added lines (a Trojan Source patch) |
| `too_many_files`, `too_large` | more than 10 files, 400 changed lines or 200 kB (configurable in `PatchPolicy`) |
| `exists`, `missing`, `duplicate`, `symlink` | creates an existing file, changes a missing one, names a file twice, targets a symlink |
| `not_allowed` | outside `PatchPolicy.allow` globs, when set |
| `malformed`, `empty`, `does_not_apply` | bad hunks, no diff, or context that does not match the copy |

## What is flagged (applied, but reported)

* `touches_tests` when `PatchPolicy(allow_test_changes=True)`: allowed, never silent.
* `test_machinery`: added code **outside** tests that reaches into the test machinery
  (`os._exit`, assigning `sys.exit`, patching `unittest`/`pytest`/`builtins`, `sys.modules[...]`).
* `other_changes`: after applying, the copy differs in files the patch did not name.

## How it applies

Every hunk's context must match the copy **exactly**. A single unambiguous line offset is allowed
(and reported); fuzz is not, and context that matches in more than one place is refused. Every
file is computed before any file is written, so a patch either applies completely or not at all.
Line endings (CRLF), missing final newlines (`\ No newline at end of file`), new files and
deletions are handled. Afterwards `Sandbox.changes()` must match the patch.

## What a passing `try` does NOT show

That the fix is correct. It shows that this patch applied and that these tests then passed, in a
process the code under test could tamper with (see [testrun.md](testrun.md)). The tests may not
cover the bug. Protecting test files stops the most direct way of "passing" without fixing.

## Measured

22 tests, all passing (Linux): every reject and flag code above, the full loop on a small
repository (tests fail, fix applied in the sandbox, tests pass, original byte-identical), atomic
refusal when a later file does not apply, offsets and ambiguity, CRLF and no-final-newline files,
and the CLI refusing a patch that "fixes" a test by editing its expected value. Building it found
and fixed a bug: a newly created file lost its final newline.

Windows and real model-generated diffs: NOT RUN.

## Not done in M3.4

* Renames, binary files, permission changes (refused rather than supported).
* No semantic checks on the change itself (for example "does this weaken an assertion in source
  code?"), beyond the test-machinery flag.
* Candidate search and choosing among patches is M3.5.

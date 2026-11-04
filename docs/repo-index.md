# Repository index (M2.1)

Code: [`cpu_platform/repo_index.py`](../cpu_platform/repo_index.py) (walker, skipping, chunking,
storage) and [`cpu_platform/ignore.py`](../cpu_platform/ignore.py) (gitignore matching).
Standard library only; nothing leaves the machine.

```powershell
py -m cpu_platform.repo_index index C:\path\to\repo     # build or update
py -m cpu_platform.repo_index stats C:\path\to\repo     # what is indexed and what was skipped
```

## What gets indexed

Every regular file in the working tree that is **not**:

| Excluded | How |
|---|---|
| ignored | `.gitignore` at every level, `.git/info/exclude`, `.noemaignore` (same syntax, for things you want out of NOEMA but in git), and defaults: `.git/ .hg/ .svn/ __pycache__/ node_modules/ .venv/ venv/ .tox/ .mypy_cache/ .pytest_cache/ .ruff_cache/ .noema/ *.egg-info/` (a repository can re-include these with `!`) |
| a symlink or Windows junction | never followed, since it can point outside the repository; counted as `symlink` |
| binary | a NUL byte in the first 8 KiB |
| not UTF-8 | counted as `not_utf8` |
| too large | over 1 MiB by default (`--max-file-bytes`) |
| a likely secret, by name | `.env`, `.env.*`, `*.env`, `*.pem/.key/.p12/.pfx/.jks/.keystore/.kdbx/.ppk/.asc/.gpg/.ovpn`, `id_rsa*` and other SSH keys, `.netrc .npmrc .pypirc .pgpass .htpasswd .git-credentials`, `credentials` (and `credentials.json/yaml/…`), `secrets.*`, `*service-account*.json`, `terraform.tfvars`, `*.tfstate`. Files ending `.example/.sample/.template/.dist` are allowed. |
| a likely secret, by content | private-key blocks, AWS access key ids, GitHub, Slack, Google API, Stripe live and `sk-…` style API keys |

For skipped files only the **reason** is stored, never the content or the matched text.
The detector is deliberately conservative: a test fixture containing a realistic fake key is
skipped too, and shows up in `stats` under `secret_content:<kind>`. It is a safeguard against
indexing obvious credentials, **not** a secret scanner: it will miss secrets in unfamiliar
formats.

The walker's file set matches `git ls-files --cached --others --exclude-standard` on a
fixture of tricky patterns and on NOEMA itself (tested whenever git is installed).

## Chunks

* **Python** (`ast`): one chunk per top-level function (decorators included; nested
  functions stay inside their parent), per method (`Class.method`), and per class header
  (docstring and attributes between methods). Module-level code (imports, constants,
  `__main__`) goes into line windows. Definitions longer than 200 lines are split into
  windows that keep the symbol name. A file that does not parse falls back to windows and
  is marked `python_parse_error`.
* **Everything else:** 60-line windows overlapping by 10 lines; blank windows are dropped.

Each chunk stores its exact text, 1-based `start_line`/`end_line` and SHA256, so a later
citation check (M2.4) can compare quoted code with the source. Python definitions are also
stored as symbols (name, kind, file, line span), and call sites and imports as refs (name,
line, enclosing function, callee text) for [retrieval](retrieval.md). Index schema 2 added
refs; a schema-1 index is upgraded automatically by re-reading every file.

## Incremental updates

On each `index` run: a file with unchanged size and mtime is not read; a file whose SHA256
is unchanged is not re-chunked; changed files are re-chunked; deleted, newly ignored and
newly secret files leave the index (their chunks and symbols are deleted).

## Storage and privacy

The index is a SQLite file at `~/.noema/index/<repo-name>-<hash>.sqlite` (`NOEMA_HOME`
overrides `~/.noema`; `--db` sets a path). It **contains your source code in plain text**.
It is refused inside the repository it indexes, holds exactly one repository, is created
owner-only (`0600`) on Linux/macOS, and carries a schema version that is checked on open.
SQLite's `-wal`/`-shm` side files follow the process umask; on Windows, permissions come from
the folder's ACLs. Delete the file to remove the index.

## Measured

| Where | Files | Chunks | Symbols | Full index | Unchanged re-index |
|---|---|---|---|---|---|
| Cloud container (NOEMA, 2026-11-04) | 117 | 1,182 | 955 | 0.22 s | 0.005 s |
| User's laptop | NOT RUN | | | | |

## Not done in M2.1

* No answering (M2.3), no citation checks (M2.4). Search is in [retrieval](retrieval.md) (M2.2).
* No language-aware chunking or call sites beyond Python.
* Not tested on Windows (junctions, ACLs) or on very large repositories.

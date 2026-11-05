# Sandbox (M3.1)

Code: [`cpu_platform/sandbox.py`](../cpu_platform/sandbox.py). The only way NOEMA runs a command
against a repository. Model-generated code never runs in the API process (CLAUDE.md).

```python
from cpu_platform.sandbox import Sandbox, DevBackend
with Sandbox("path/to/repo", DevBackend(acknowledge_not_isolated=True)) as box:
    result = box.run(["python", "-m", "unittest"], workdir=".", timeout=120, network=False)
    result.record      # contract ExecutionRecord: backend, security_boundary, exit code, output tails
    result.limits      # what was enforced, and what was NOT
    box.changes()      # created / modified / deleted files in the copy
```

```powershell
py -m cpu_platform.sandbox run . --i-understand-dev-is-not-isolated -- py -m unittest tests.test_m13_contract
```

## The `dev` backend is NOT a security boundary

It runs the command on this machine, as the current user. The command can read anything that
user can read and can reach the network. Every `ExecutionRecord` it produces has
`backend: "dev"` and `security_boundary: false`. The contract refuses a `dev` record that claims
otherwise. Using it requires `acknowledge_not_isolated=True` (CLI:
`--i-understand-dev-is-not-isolated`), so nothing reaches it by accident. Never use it for
untrusted code on a machine holding data you care about. Isolation is M3.2 (container backend).

## What `dev` does enforce

| | Behaviour |
|---|---|
| copy | the command runs in a temporary copy: files ignored by `.gitignore`, likely secrets (by name, e.g. `.env`, keys) and symlinks are **not copied**; `.noemaignore` is not applied (it only governs NOEMA's index, and tests may need those files); the original repository is never written |
| command | an argument list only, never a shell string; `workdir` must be a directory inside the copy |
| timeout | the whole process tree is killed (POSIX process group; Windows `taskkill /T`); `exit_code` is null and `timed_out` true |
| output | the last 8,000 characters of each stream are kept (`output_truncated` set); a stream over 10 MiB kills the command |
| environment | only `PATH` and a few locale/system variables pass through; credentials (API keys, tokens, cloud, NOEMA and `LLAMA_API_KEY` secrets) are dropped; HOME/TEMP are folders **beside** the copy (inside the sandbox, not inside the repository); proxy variables point at a dead port (a hint, not a block) while loopback (`localhost`, `127.0.0.1`, `::1`) bypasses it; `PYTHONNOUSERSITE`, `PIP_NO_INDEX` |
| limits (POSIX) | address space 2 GiB, file size 256 MiB, CPU seconds = timeout + 1, no core dumps |
| limits (Windows) | **NOT APPLIED**: reported as such in `limits` |
| network | **NOT blocked**: `limits.network_blocked` is always false for `dev` |

`changes()` compares the copy with what was copied (for M3.4: which files a patch or test run touched).

## Measured

Linux (cloud container): all 25 M3.1 tests pass, including:
* a 60-second sleep killed at 1 s;
* a grandchild process that outlived its parent killed with the tree. A mutation test confirms the
  test fails if only the parent is killed;
* a 2 MiB output flood killed;
* a 2 GiB allocation refused under a 512 MiB limit;
* NOEMA's own contract tests passing inside a copy while the original's hash stays unchanged.

**Windows: NOT RUN.** On the laptop:

```powershell
py -m unittest tests.test_m31_sandbox -v
```

The POSIX-only tests (resource limits, process-group kill) skip there. The rest exercise the
`taskkill` path.

## Not done in M3.1

* No isolation: container backend (no network, read-only base, CPU/RAM limits, non-root) is M3.2.
* No Windows resource limits (would need Job Objects via `ctypes`).
* The copy is a full file copy per sandbox; no copy-on-write. Not measured on large repositories.

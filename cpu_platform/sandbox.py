"""Sandbox interface for running commands against a repository (roadmap M3.1).

    with Sandbox(repo, DevBackend(acknowledge_not_isolated=True)) as box:
        result = box.run(["python", "-m", "unittest"], timeout=120)
        result.record          # contract.ExecutionRecord: backend, security_boundary, exit code, output tails
        result.limits          # which limits were actually applied
        box.changes()          # files the command created, modified or deleted in the copy

Only the ``dev`` backend exists so far. **It is not a security boundary.** It runs the command
as the current user, on the host, with access to anything that user can read, and it cannot
block the network. What it does do:

* runs in a **temporary copy** of the repository, made with the index's rules: ignored files,
  likely secrets (by name) and symlinks are not copied, and the original is never written;
* takes the command as an **argument list**, never a shell string; the working directory must
  stay inside the copy;
* enforces a **timeout** that kills the whole process tree, and **caps output** (keeps the
  tail; a command that floods output is killed);
* gives the command a **minimal environment**: HOME/TEMP inside the sandbox, no credentials
  (API keys, tokens, cloud and NOEMA secrets are dropped), proxy variables pointed at a dead
  port, which is a hint to well-behaved clients, not a block;
* applies **POSIX resource limits** where available (address space, CPU seconds, file size,
  no core dumps); on Windows they are not applied, and the result says so.

Every ``ExecutionRecord`` carries ``backend="dev"`` and ``security_boundary=False``, and
``limits`` spells out what was and was not enforced. Model-generated code must never run in
the API process (CLAUDE.md); it runs only through this interface. A real isolation backend
(container, no network, non-root) is M3.2.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

from .contract import ExecutionRecord, MAX_OUTPUT_TAIL_CHARS
from .repo_index import DEFAULT_MAX_FILE_BYTES, iter_files, secret_name_reason

DEV_WARNING = ("dev backend is NOT a security boundary: the command runs on this machine as the current "
               "user, can read anything that user can read, and network access is NOT blocked")
DEFAULT_TIMEOUT_S = 120
DEFAULT_OUTPUT_TAIL = 8_000                 # characters kept per stream (contract max 16,000)
DEFAULT_OUTPUT_LIMIT = 10 * 1024 * 1024     # bytes per stream before the command is killed
DEFAULT_MEMORY_BYTES = 2 * 1024 ** 3
DEFAULT_FILE_BYTES = 256 * 1024 ** 2
DEAD_PROXY = "http://127.0.0.1:9"           # discard port: connections fail fast

# Environment variables passed through (everything else is dropped).
ENV_ALLOW = ("PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS",
             "PROCESSOR_ARCHITECTURE", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "PYTHONIOENCODING")


class SandboxError(ValueError):
    """The request cannot be run safely (bad command, workdir escape, no acknowledgement)."""


@dataclass
class SandboxResult:
    record: ExecutionRecord
    workdir: str                            # relative to the sandbox copy
    limits: dict                            # what was enforced, and what was not
    warnings: list[str] = field(default_factory=list)
    output_killed: bool = False             # killed for exceeding the output limit


class _Tail:
    """Drain a stream fully, keep only the last ``keep`` bytes, count the rest."""

    def __init__(self, stream, keep: int, limit: int, on_limit):
        self.buffer: deque[bytes] = deque()
        self.kept = 0
        self.total = 0
        self.keep, self.limit, self.on_limit = keep, limit, on_limit
        self.thread = threading.Thread(target=self._drain, args=(stream,), daemon=True)
        self.thread.start()

    def _drain(self, stream):
        for chunk in iter(lambda: stream.read(65536), b""):
            self.total += len(chunk)
            self.buffer.append(chunk)
            self.kept += len(chunk)
            while self.kept - len(self.buffer[0]) >= self.keep:
                self.kept -= len(self.buffer.popleft())
            if self.total > self.limit:
                self.on_limit()
        stream.close()

    def text(self, max_chars: int) -> tuple[str, bool]:
        data = b"".join(self.buffer)[-self.keep:]
        text = data.decode("utf-8", errors="replace")
        truncated = self.total > len(data) or len(text) > max_chars
        return text[-max_chars:], truncated


class DevBackend:
    """Runs commands on the host in a temporary copy. NOT a security boundary."""
    name = "dev"
    security_boundary = False

    def __init__(self, *, acknowledge_not_isolated: bool = False, memory_bytes: int = DEFAULT_MEMORY_BYTES,
                 file_bytes: int = DEFAULT_FILE_BYTES):
        if not acknowledge_not_isolated:
            raise SandboxError("the dev backend is not isolated; pass acknowledge_not_isolated=True to use it "
                               "(never for untrusted code on a machine with data you care about)")
        self.memory_bytes, self.file_bytes = memory_bytes, file_bytes

    def limits(self, timeout: float) -> dict:
        posix = os.name == "posix"
        try:
            import resource  # noqa: F401
            have_resource = posix
        except ImportError:
            have_resource = False
        return {
            "backend": self.name, "security_boundary": False,
            "timeout_s": timeout, "kills_process_tree": True,
            "network_blocked": False,
            "network_note": "dev cannot block the network; proxy variables point at a dead port",
            "filesystem": "temporary copy; the original repository is never written; the host is NOT protected",
            "memory_bytes": self.memory_bytes if have_resource else "NOT APPLIED (no POSIX rlimits here)",
            "file_size_bytes": self.file_bytes if have_resource else "NOT APPLIED",
            "cpu_seconds": "timeout + 1" if have_resource else "NOT APPLIED",
            "core_dumps": "disabled" if have_resource else "NOT APPLIED",
        }

    def prepare(self, sandbox: "Sandbox") -> None:
        """Called once when the sandbox copy exists. Nothing to do for dev."""

    def python(self) -> str:
        return sys.executable

    def run_warnings(self, network: bool) -> list[str]:
        out = [DEV_WARNING]
        if network:
            out.append("network was requested; the dev backend does not restrict it anyway")
        return out

    def launch(self, sandbox: "Sandbox", command: list[str], cwd: Path, env: dict, timeout: float, run_id: str):
        """(argv, cwd, env, popen kwargs) for this run."""
        posix = os.name == "posix"
        kwargs = {"start_new_session": True, "preexec_fn": self.preexec(timeout)} if posix else \
            {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        return list(command), cwd, env, kwargs

    def kill(self, proc: subprocess.Popen, run_id: str) -> None:
        _kill_tree(proc)

    def preexec(self, timeout: float):
        if os.name != "posix":
            return None
        memory, file_bytes, cpu = self.memory_bytes, self.file_bytes, int(timeout) + 1

        def apply():   # runs in the child between fork and exec
            import resource
            for limit, value in ((resource.RLIMIT_AS, memory), (resource.RLIMIT_FSIZE, file_bytes),
                                 (resource.RLIMIT_CPU, cpu), (resource.RLIMIT_CORE, 0)):
                try:
                    soft, hard = resource.getrlimit(limit)
                    cap = value if hard == resource.RLIM_INFINITY else min(value, hard)
                    resource.setrlimit(limit, (cap, hard))
                except (ValueError, OSError):
                    pass
        return apply


def copy_selection(src: Path):
    """Repository-relative paths a sandbox copies (ignored, secret-named, symlinked and
    oversized files are left out). Also defines the repository fingerprint in audit reports."""
    # .gitignore rules apply; .noemaignore does not (it only keeps files out of NOEMA's
    # index, and a repository's tests may need them). HOME/TEMP live beside the copy, never
    # inside it, so a sandbox started inside a sandbox cannot copy itself.
    for item in iter_files(Path(src), extra_ignore_file=None):
        if item[0] == "<skip>":
            continue
        rel, info = item
        if secret_name_reason(rel.rsplit("/", 1)[-1]) or info.st_size > DEFAULT_MAX_FILE_BYTES * 16:
            continue
        yield rel


def _copy_repository(src: Path, dst: Path) -> dict[str, str]:
    """Copy the files ``copy_selection`` names. Returns {relative path: sha256} of what was copied."""
    copied = {}
    for rel in copy_selection(src):
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, target)
        copied[rel] = hashlib.sha256(target.read_bytes()).hexdigest()
    return copied


def _snapshot(root: Path) -> dict[str, str]:
    out = {}
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            rel = path.relative_to(root).as_posix()
            if "__pycache__" in rel.split("/"):
                continue
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


class Sandbox:
    """A temporary copy of ``repo`` and a backend to run commands in it."""

    def __init__(self, repo: str | Path, backend, *, keep: bool = False):
        self.repo = Path(repo).resolve()
        if not self.repo.is_dir():
            raise SandboxError(f"not a directory: {repo}")
        self.backend = backend
        self.keep = keep
        self._tmp = Path(tempfile.mkdtemp(prefix="noema-sandbox-"))
        self.root = self._tmp / "repo"
        self.root.mkdir()
        self.copied = _copy_repository(self.repo, self.root)
        # HOME and TEMP are siblings of the copy: the copy holds only the repository's files,
        # and temporary files the code creates are not "inside the repository".
        self.home = self._tmp / "home"
        self.tmpdir = self._tmp / "tmp"
        self.home.mkdir()
        self.tmpdir.mkdir()
        self._runs = 0
        try:
            backend.prepare(self)
        except BaseException:
            self.close()
            raise

    @property
    def base(self) -> Path:
        """The sandbox folder: the repository copy plus HOME and TEMP."""
        return self._tmp

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if not self.keep:
            shutil.rmtree(self._tmp, ignore_errors=True)

    def changes(self) -> dict[str, list[str]]:
        """Files created, modified or deleted in the copy since it was made."""
        now = _snapshot(self.root)
        return {"created": sorted(set(now) - set(self.copied)),
                "modified": sorted(p for p in now if p in self.copied and now[p] != self.copied[p]),
                "deleted": sorted(set(self.copied) - set(now))}

    def _workdir(self, workdir: str) -> Path:
        target = (self.root / workdir).resolve()
        if not target.is_relative_to(self.root.resolve()) or not target.is_dir():
            raise SandboxError(f"workdir must be a directory inside the sandbox copy: {workdir!r}")
        return target

    def environment(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k.upper() in ENV_ALLOW}
        home, tmp = str(self.home), str(self.tmpdir)
        env.update({"HOME": home, "USERPROFILE": home, "TMPDIR": tmp, "TEMP": tmp, "TMP": tmp,
                    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                    "NOEMA_SANDBOX": self.backend.name, "NO_COLOR": "1",
                    "http_proxy": DEAD_PROXY, "https_proxy": DEAD_PROXY, "HTTP_PROXY": DEAD_PROXY,
                    "HTTPS_PROXY": DEAD_PROXY, "ALL_PROXY": DEAD_PROXY, "all_proxy": DEAD_PROXY,
                    # Loopback is not "the network": local test servers must stay reachable.
                    "no_proxy": "localhost,127.0.0.1,::1", "NO_PROXY": "localhost,127.0.0.1,::1",
                    "PIP_NO_INDEX": "1"})
        return env

    def run(self, command: list[str], workdir: str = ".", timeout: float = DEFAULT_TIMEOUT_S, network: bool = False,
            *, output_tail: int = DEFAULT_OUTPUT_TAIL, output_limit: int = DEFAULT_OUTPUT_LIMIT) -> SandboxResult:
        if isinstance(command, (str, bytes)) or not command or not all(isinstance(a, str) and a for a in command):
            raise SandboxError("command must be a non-empty list of non-empty strings (no shell)")
        if any("\x00" in a for a in command):
            raise SandboxError("command arguments must not contain NUL")
        if not 0 < timeout <= 3600:
            raise SandboxError("timeout must be between 0 and 3600 seconds")
        output_tail = max(1, min(output_tail, MAX_OUTPUT_TAIL_CHARS))
        cwd = self._workdir(workdir)
        self._runs += 1
        limits = self.backend.limits(timeout)
        warnings = self.backend.run_warnings(network)
        run_id = f"X{self._runs}"
        argv, run_cwd, run_env, kwargs = self.backend.launch(self, list(command), cwd, self.environment(), timeout,
                                                             run_id)
        started = time.perf_counter()
        flood = threading.Event()
        try:
            proc = subprocess.Popen(argv, cwd=run_cwd, env=run_env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        except OSError as exc:
            record = ExecutionRecord(id=f"X{self._runs}", command=list(command), backend=self.backend.name,
                                     security_boundary=self.backend.security_boundary, network=network,
                                     exit_code=None, timed_out=False, duration_s=0.0,
                                     stderr_tail=f"could not start: {exc}"[:output_tail])
            return SandboxResult(record, os.path.relpath(cwd, self.root), limits, warnings)
        out = _Tail(proc.stdout, output_tail * 4, output_limit, flood.set)
        err = _Tail(proc.stderr, output_tail * 4, output_limit, flood.set)
        timed_out = False
        deadline = started + timeout
        while proc.poll() is None:
            if flood.is_set():
                break
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            time.sleep(0.02)
        if proc.poll() is None:
            self.backend.kill(proc, run_id)
        exit_code = None if (timed_out or flood.is_set()) else proc.returncode
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        out.thread.join(timeout=5)
        err.thread.join(timeout=5)
        stdout, t1 = out.text(output_tail)
        stderr, t2 = err.text(output_tail)
        record = ExecutionRecord(id=f"X{self._runs}", command=list(command), backend=self.backend.name,
                                 security_boundary=self.backend.security_boundary, network=network,
                                 exit_code=exit_code, timed_out=timed_out,
                                 duration_s=round(time.perf_counter() - started, 3),
                                 stdout_tail=stdout, stderr_tail=stderr,
                                 output_truncated=t1 or t2 or flood.is_set())
        if flood.is_set():
            warnings.append(f"killed: output exceeded {output_limit} bytes on a stream")
        return SandboxResult(record, os.path.relpath(cwd, self.root).replace("\\", "/"), limits, warnings,
                             output_killed=flood.is_set())


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and everything it started."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    else:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    try:
        proc.kill()
    except OSError:
        pass


def make_backend(name: str, acknowledge_dev: bool = False, image: str | None = None):
    """Backend by name, for command lines: 'dev' needs the acknowledgement; 'container' probes."""
    if name == "container":
        from .container import DEFAULT_IMAGE, ContainerBackend
        return ContainerBackend(image=image or DEFAULT_IMAGE)
    return DevBackend(acknowledge_not_isolated=acknowledge_dev)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a command in a sandbox copy of a repository (dev backend: "
                                                 "NOT a security boundary)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("path")
    p.add_argument("--backend", choices=("dev", "container"), default="dev")
    p.add_argument("--image", help="container image (container backend; never pulled implicitly)")
    p.add_argument("--i-understand-dev-is-not-isolated", action="store_true", dest="ack",
                   help="required for --backend dev")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    p.add_argument("--workdir", default=".")
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after "--" is the command, untouched by option parsing.
    command = argv[argv.index("--") + 1:] if "--" in argv else []
    head = argv[:argv.index("--")] if "--" in argv else argv
    args = parser.parse_args(head)
    if not command:
        parser.error("give the command after --, e.g. -- python -m unittest")
    try:
        with Sandbox(args.path, make_backend(args.backend, args.ack, args.image)) as box:
            result = box.run(command, args.workdir, args.timeout)
            changes = box.changes()
    except SandboxError as exc:
        parser.error(str(exc))
    from dataclasses import asdict
    print(json.dumps({"record": asdict(result.record), "limits": result.limits, "warnings": result.warnings,
                      "changes": changes}, indent=2))
    return 0 if result.record.exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

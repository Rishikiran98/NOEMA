"""Container sandbox backend (roadmap M3.2): runs sandbox commands inside a Docker container.

    from cpu_platform.sandbox import Sandbox
    from cpu_platform.container import ContainerBackend
    with Sandbox(repo, ContainerBackend(image="python:3.12-slim")) as box:
        box.run(["python", "-m", "unittest"])

Works with Docker Engine on Linux and Docker Desktop on Windows/macOS (which runs Linux
containers through WSL2 or a VM). Each command runs in a fresh container with:

* ``--network none``: no network interfaces except loopback;
* ``--read-only`` root filesystem, a size-capped ``/tmp`` tmpfs, and only the sandbox
  folder (repository copy, HOME, TEMP) mounted, at ``/sandbox``;
* ``--user 65534:65534`` (nobody), ``--cap-drop ALL``, ``--security-opt no-new-privileges``;
* ``--memory``/``--memory-swap``, ``--cpus``, ``--pids-limit`` and a file-size ulimit;
* ``--pull never``: no image is ever downloaded implicitly. Pull it once, deliberately.

**The isolation claim is earned, not assumed.** When the sandbox is prepared, a probe runs
inside a container with exactly these flags and must show: not root, root filesystem not
writable, an outbound connection impossible, and the work area writable. Only then does
the backend report ``security_boundary=True``. If any check fails, or Docker or the image is
missing, it raises ``SandboxError`` and runs nothing.

What it does NOT protect against: kernel or container-runtime escapes, a misconfigured
Docker daemon (for example one exposing its socket), or side channels. Docker Desktop's VM
adds a layer on Windows/macOS; on Linux the container shares the host kernel.
"""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from secrets import token_hex

from .sandbox import SandboxError, _kill_tree

DEFAULT_IMAGE = "python:3.12-slim"
CONTAINER_BASE = "/sandbox"
NOBODY = "65534:65534"

PROBE = r'''
import json, os, socket
report = {"uid": os.getuid()}
try:
    open("/noema_probe_root", "w").close()
    report["root_writable"] = True
except OSError:
    report["root_writable"] = False
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3).close()
    report["network"] = True
except OSError:
    report["network"] = False
try:
    path = os.path.join(os.environ.get("TMPDIR", "/sandbox/tmp"), "noema_probe")
    open(path, "w").write("ok")
    os.remove(path)
    report["work_writable"] = True
except OSError:
    report["work_writable"] = False
print("NOEMA_PROBE " + json.dumps(report))
'''


class ContainerBackend:
    """Docker backend. ``security_boundary`` becomes True only after a passing probe."""
    name = "container"

    def __init__(self, *, image: str = DEFAULT_IMAGE, docker: str = "docker", memory_bytes: int = 2 * 1024 ** 3,
                 cpus: float = 2.0, pids: int = 256, file_bytes: int = 256 * 1024 ** 2,
                 tmpfs_bytes: int = 256 * 1024 ** 2, probe_timeout: float = 60):
        if not image or any(c.isspace() for c in image) or image.startswith("-"):
            raise SandboxError(f"invalid image name {image!r}")
        self.image, self.docker = image, docker
        self.memory_bytes, self.cpus, self.pids = memory_bytes, cpus, pids
        self.file_bytes, self.tmpfs_bytes, self.probe_timeout = file_bytes, tmpfs_bytes, probe_timeout
        self.security_boundary = False
        self.probe: dict | None = None

    # ------------------------------------------------------------------ helpers

    def _docker(self, *args: str, timeout: float = 30) -> subprocess.CompletedProcess:
        exe = shutil.which(self.docker) or self.docker
        try:
            return subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise SandboxError(f"'{self.docker}' not found: install Docker Engine (Linux) or Docker Desktop "
                               "(Windows/macOS)") from None
        except subprocess.TimeoutExpired:
            raise SandboxError(f"'{self.docker} {args[0]}' did not answer within {timeout:.0f} s") from None

    def _to_container(self, sandbox, value: str) -> str:
        """Rewrite host paths under the sandbox folder to their /sandbox equivalents."""
        for host in {str(sandbox.base), str(sandbox.base.resolve())}:
            if host in value:
                head, _, tail = value.partition(host)
                return head + CONTAINER_BASE + tail.replace("\\", "/")
        return value

    def run_flags(self, sandbox, name: str, workdir: str, env: dict[str, str]) -> list[str]:
        source = str(sandbox.base.resolve())
        if "," in source or '"' in source:
            raise SandboxError(f"sandbox path cannot be mounted safely: {source!r}")
        # --mount is unambiguous with Windows drive letters (C:\...), unlike -v host:container.
        mount = f"type=bind,source={source},target={CONTAINER_BASE}"
        flags = ["run", "--rm", "--name", name, "--pull", "never", "--network", "none", "--read-only",
                 "--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.tmpfs_bytes}",
                 "--user", NOBODY, "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                 "--pids-limit", str(self.pids), "--memory", str(self.memory_bytes),
                 "--memory-swap", str(self.memory_bytes), "--cpus", str(self.cpus),
                 "--ulimit", f"fsize={self.file_bytes}", "--ulimit", "core=0",
                 "--mount", mount, "-w", workdir]
        for key, value in sorted(env.items()):
            flags += ["-e", f"{key}={value}"]
        return flags + [self.image]

    def _environment(self, env: dict[str, str]) -> dict[str, str]:
        keep = {k: v for k, v in env.items() if k in ("LANG", "LC_ALL", "LC_CTYPE", "TZ", "PYTHONIOENCODING")}
        keep.update({"HOME": f"{CONTAINER_BASE}/home", "TMPDIR": f"{CONTAINER_BASE}/tmp",
                     "TEMP": f"{CONTAINER_BASE}/tmp", "TMP": f"{CONTAINER_BASE}/tmp",
                     "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                     "PIP_NO_INDEX": "1", "NO_COLOR": "1", "NOEMA_SANDBOX": self.name})
        return keep

    # ------------------------------------------------------------------ backend interface

    def python(self) -> str:
        return "python"

    def prepare(self, sandbox) -> None:
        info = self._docker("info", "--format", "{{.ServerVersion}}")
        if info.returncode != 0:
            raise SandboxError("Docker engine not reachable (is Docker Desktop / dockerd running?): "
                               + (info.stderr or info.stdout).strip()[:300])
        if self._docker("image", "inspect", self.image).returncode != 0:
            raise SandboxError(f"image {self.image!r} is not present locally; pull it once, deliberately: "
                               f"docker pull {self.image}")
        # The container runs as nobody: let it write the sandbox copy (a throwaway).
        for root, dirs, files in os.walk(sandbox.base):
            for name in dirs:
                (Path(root) / name).chmod(0o777)
            for name in files:
                path = Path(root) / name
                path.chmod(path.stat().st_mode | 0o666)
        sandbox.base.chmod(0o777)
        name = f"noema-probe-{token_hex(6)}"
        argv = [shutil.which(self.docker) or self.docker,
                *self.run_flags(sandbox, name, f"{CONTAINER_BASE}/repo", self._environment({})), "python", "-c", PROBE]
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=self.probe_timeout)
        except subprocess.TimeoutExpired:
            self._docker("kill", name)
            raise SandboxError("container isolation probe timed out") from None
        line = next((l for l in done.stdout.splitlines() if l.startswith("NOEMA_PROBE ")), None)
        if line is None:
            raise SandboxError("container isolation probe produced no report (exit "
                               f"{done.returncode}): {(done.stderr or done.stdout).strip()[:300]}")
        try:
            report = json.loads(line[len("NOEMA_PROBE "):])
        except ValueError:
            raise SandboxError("container isolation probe report unreadable") from None
        failures = []
        if report.get("uid") in (0, None):
            failures.append("runs as root")
        if report.get("root_writable") is not False:
            failures.append("root filesystem is writable")
        if report.get("network") is not False:
            failures.append("an outbound connection succeeded")
        if report.get("work_writable") is not True:
            failures.append("the sandbox work area is not writable")
        self.probe = report
        if failures:
            raise SandboxError("container isolation probe failed: " + "; ".join(failures))
        self.security_boundary = True

    def limits(self, timeout: float) -> dict:
        return {"backend": self.name, "security_boundary": self.security_boundary, "image": self.image,
                "timeout_s": timeout, "kills_process_tree": True, "network_blocked": True,
                "network_note": "--network none (loopback only); verified by the probe",
                "filesystem": "read-only root, /tmp tmpfs, only the sandbox folder mounted at /sandbox",
                "user": NOBODY, "capabilities": "all dropped; no-new-privileges",
                "memory_bytes": self.memory_bytes, "cpus": self.cpus, "pids": self.pids,
                "file_size_bytes": self.file_bytes, "core_dumps": "disabled", "probe": self.probe}

    def run_warnings(self, network: bool) -> list[str]:
        if network:
            raise SandboxError("the container backend never grants network access (--network none)")
        return []

    def launch(self, sandbox, command: list[str], cwd: Path, env: dict, timeout: float, run_id: str):
        if not self.security_boundary:
            raise SandboxError("container backend has not passed its isolation probe")
        name = f"noema-{run_id.lower()}-{token_hex(6)}"
        self._names = getattr(self, "_names", {})
        self._names[run_id] = name
        rel = PurePosixPath(*Path(os.path.relpath(cwd, sandbox.root)).parts)
        workdir = f"{CONTAINER_BASE}/repo" + ("" if str(rel) == "." else f"/{rel}")
        args = [self._to_container(sandbox, a) for a in command]
        if Path(command[0].replace("\\", "/")).name.lower().startswith("python"):
            args[0] = "python"   # a host interpreter path (sys.executable) means "the image's Python"
        argv = [shutil.which(self.docker) or self.docker,
                *self.run_flags(sandbox, name, workdir, self._environment(env)), *args]
        return argv, sandbox.base, None, {}

    def kill(self, proc: subprocess.Popen, run_id: str) -> None:
        name = getattr(self, "_names", {}).get(run_id)
        if name:
            self._docker("kill", name, timeout=30)   # stops the container, not just the client
        _kill_tree(proc)

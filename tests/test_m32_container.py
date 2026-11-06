"""M3.2 tests: the container sandbox backend.

* Unit: the exact hardening flags, environment and path mapping.
* Probe gating: the backend claims ``security_boundary=True`` only after an in-container probe
  shows non-root, read-only root, no network and a writable work area; anything else refuses.
* Contract (offline): the M3.1 checks through ``tests/fake_docker.py``, a stand-in for the CLI
  that runs commands on the host and does NOT isolate. It checks what NOEMA asks Docker for.
* Contract (real Docker): the same checks plus real network/root/filesystem isolation, against
  a real engine. Skipped unless Docker is reachable and the image is present
  (NOEMA_TEST_DOCKER_IMAGE, default python:3.12-slim).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from cpu_platform import container as ct
from cpu_platform.contract import Cost, EngineResult, not_measured, validate_result
from cpu_platform.sandbox import Sandbox, SandboxError
from cpu_platform.testrun import run_tests

FAKE = Path(__file__).resolve().parent / "fake_docker.py"
POSIX = os.name == "posix"
IMAGE = os.environ.get("NOEMA_TEST_DOCKER_IMAGE", ct.DEFAULT_IMAGE)


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def real_docker_ready() -> bool:
    exe = shutil.which("docker")
    if not exe:
        return False
    try:
        info = subprocess.run([exe, "info"], capture_output=True, timeout=20)
        image = subprocess.run([exe, "image", "inspect", IMAGE], capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return info.returncode == 0 and image.returncode == 0


def make_repo(root: Path):
    for rel, text in {"app/main.py": "print('main')\n", "README.md": "readme\n", ".env": "SECRET=1\n",
                      "tests/test_ok.py": "import unittest\nclass T(unittest.TestCase):\n"
                                          "    def test_ok(self):\n        self.assertTrue(True)\n"
                                          "    def test_bad(self):\n        self.assertEqual(1, 2)\n"}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")


@unittest.skipUnless(POSIX, "the fake docker wrapper is a POSIX shell script")
class FakeDockerCase(unittest.TestCase):
    probe = None

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo = self.tmp / "repo"
        make_repo(self.repo)
        self.state = self.tmp / "state"
        self.state.mkdir()
        self.log = self.tmp / "docker.log"
        self.exe = self.tmp / "docker"
        self.exe.write_text(f"#!/bin/sh\nexec {sys.executable} {FAKE} \"$@\"\n", encoding="utf-8")
        self.exe.chmod(0o755)
        env = {"FAKE_DOCKER_LOG": str(self.log), "FAKE_DOCKER_STATE": str(self.state)}
        if self.probe is not None:
            env["FAKE_DOCKER_PROBE"] = self.probe
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def backend(self, **kw):
        return ct.ContainerBackend(image="python:3.12-slim", docker=str(self.exe), **kw)

    def calls(self):
        return [json.loads(l) for l in self.log.read_text(encoding="utf-8").splitlines()] if self.log.exists() else []


class Flags(unittest.TestCase):
    def test_hardening_flags_and_mount(self):
        backend = ct.ContainerBackend(image="python:3.12-slim")
        box = mock.Mock(base=Path("/tmp/noema-sandbox-x"))
        flags = backend.run_flags(box, "noema-x1", "/sandbox/repo", {"HOME": "/sandbox/home"})
        joined = " ".join(flags)
        for required in ("--rm", "--pull never", "--network none", "--read-only", "--user 65534:65534",
                         "--cap-drop ALL", "--security-opt no-new-privileges", "--pids-limit 256",
                         "--memory 2147483648", "--memory-swap 2147483648", "--cpus 2.0", "--ulimit core=0",
                         "-w /sandbox/repo", "-e HOME=/sandbox/home", "--name noema-x1"):
            self.assertIn(required, joined)
        self.assertIn(f"--mount type=bind,source={Path('/tmp/noema-sandbox-x').resolve()},target=/sandbox", joined)
        self.assertEqual(flags[-1], "python:3.12-slim")
        self.assertEqual(sum(1 for f in flags if f in ("-v", "--mount", "--volume")), 1)   # only the sandbox folder
        self.assertNotIn("--privileged", flags)

    def test_unsafe_mount_path_refused(self):
        backend = ct.ContainerBackend()
        with self.assertRaises(SandboxError):
            backend.run_flags(mock.Mock(base=Path("/tmp/a,type=bind,source=/,target=/x")), "n", "/sandbox/repo", {})

    def test_environment_carries_no_host_values(self):
        backend = ct.ContainerBackend()
        env = backend._environment({"PATH": "/host/bin", "AWS_SECRET_ACCESS_KEY": "x", "LANG": "C.UTF-8",
                                    "HOME": "/home/me"})
        self.assertNotIn("PATH", env)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)
        self.assertEqual((env["HOME"], env["TMPDIR"], env["LANG"], env["NOEMA_SANDBOX"]),
                         ("/sandbox/home", "/sandbox/tmp", "C.UTF-8", "container"))

    def test_invalid_images(self):
        for bad in ("", "python 3", "--privileged", "x\ty"):
            with self.subTest(bad=bad), self.assertRaises(SandboxError):
                ct.ContainerBackend(image=bad)

    def test_unprobed_backend_claims_nothing(self):
        backend = ct.ContainerBackend()
        self.assertFalse(backend.security_boundary)
        with self.assertRaises(SandboxError):
            backend.launch(mock.Mock(), ["python"], Path("."), {}, 10, "X1")


class Gating(FakeDockerCase):
    def test_passing_probe_grants_the_claim(self):
        backend = self.backend()
        with Sandbox(self.repo, backend) as box:
            self.assertTrue(backend.security_boundary)
            self.assertEqual(backend.probe["uid"], 65534)
            self.assertTrue(box.run(["python", "-c", "pass"]).record.security_boundary)

    def test_each_failing_probe_check_refuses(self):
        cases = {"runs as root": {"uid": 0, "root_writable": False, "network": False, "work_writable": True},
                 "root filesystem is writable": {"uid": 65534, "root_writable": True, "network": False, "work_writable": True},
                 "outbound connection succeeded": {"uid": 65534, "root_writable": False, "network": True, "work_writable": True},
                 "work area is not writable": {"uid": 65534, "root_writable": False, "network": False, "work_writable": False},
                 "runs as root;": {"root_writable": False}}
        for expected, report in cases.items():
            with self.subTest(expected=expected), mock.patch.dict(os.environ, {"FAKE_DOCKER_PROBE": json.dumps(report)}):
                with self.assertRaisesRegex(SandboxError, "probe failed: .*" + expected.rstrip(";")):
                    Sandbox(self.repo, self.backend())

    def test_probe_without_report(self):
        with mock.patch.dict(os.environ, {"FAKE_DOCKER_PROBE": "none"}), \
                self.assertRaisesRegex(SandboxError, "no report"):
            Sandbox(self.repo, self.backend())

    def test_docker_missing_engine_down_or_image_absent(self):
        with self.assertRaisesRegex(SandboxError, "not found"):
            Sandbox(self.repo, ct.ContainerBackend(docker=str(self.tmp / "no-such-docker")))
        with mock.patch.dict(os.environ, {"FAKE_DOCKER_INFO_EXIT": "1"}), \
                self.assertRaisesRegex(SandboxError, "engine not reachable"):
            Sandbox(self.repo, self.backend())
        with mock.patch.dict(os.environ, {"FAKE_DOCKER_IMAGE_EXIT": "1"}), \
                self.assertRaisesRegex(SandboxError, "docker pull python:3.12-slim"):
            Sandbox(self.repo, self.backend())

    def test_failed_preparation_leaves_no_temp_copy(self):
        before = set(Path(tempfile.gettempdir()).glob("noema-sandbox-*"))
        with mock.patch.dict(os.environ, {"FAKE_DOCKER_IMAGE_EXIT": "1"}), self.assertRaises(SandboxError):
            Sandbox(self.repo, self.backend())
        self.assertEqual(set(Path(tempfile.gettempdir()).glob("noema-sandbox-*")), before)

    def test_network_request_is_refused(self):
        with Sandbox(self.repo, self.backend()) as box, self.assertRaisesRegex(SandboxError, "never grants network"):
            box.run(["python", "-c", "pass"], network=True)


class ContractThroughFake(FakeDockerCase):
    """The M3.1 contract checks, through the container backend's code path."""

    def setUp(self):
        super().setUp()
        self.box = Sandbox(self.repo, self.backend())
        self.addCleanup(self.box.close)

    def test_record_and_command_mapping(self):
        result = self.box.run([sys.executable, "-c", "import os; print(os.getcwd())"], workdir="app")
        r = result.record
        self.assertEqual((r.backend, r.security_boundary, r.exit_code), ("container", True, 0))
        self.assertTrue(r.stdout_tail.strip().endswith("/repo/app"))
        run = [c for c in self.calls() if c[0] == "run" and "NOEMA_PROBE" not in " ".join(c)][-1]
        tail = run[run.index("python:3.12-slim") + 1:]
        self.assertEqual(tail[0], "python")                                    # host interpreter -> image's
        self.assertEqual(run[run.index("-w") + 1], "/sandbox/repo/app")
        self.assertTrue(result.limits["network_blocked"])
        self.assertEqual(result.warnings, [])
        cost = Cost()
        validate_result(EngineResult(request_id="r", task="repo_fix", status="ok", engine_version="t", answer="a",
                                     executions=[r], cost=cost, not_run=not_measured(cost)))

    def test_host_paths_in_arguments_become_container_paths(self):
        target = self.box.tmpdir / "out.txt"
        self.box.run(["python", "-c", f"open({str(target)!r}, 'w').write('x')"])
        run = [c for c in self.calls() if c[0] == "run" and "NOEMA_PROBE" not in " ".join(c)][-1]
        self.assertIn("/sandbox/tmp/out.txt", run[-1])
        self.assertNotIn(str(self.box.base), " ".join(run[run.index("python:3.12-slim"):]))
        self.assertEqual(target.read_text(), "x")

    def test_timeout_kills_the_container(self):
        start = time.perf_counter()
        r = self.box.run(["python", "-c", "import time; time.sleep(60)"], timeout=2).record
        self.assertTrue(r.timed_out)
        self.assertIsNone(r.exit_code)
        self.assertLess(time.perf_counter() - start, 20)
        kills = [c for c in self.calls() if c[0] == "kill"]
        self.assertEqual(len(kills), 1)
        self.assertTrue(kills[0][1].startswith("noema-x1-"))
        self.assertTrue((self.state / (kills[0][1] + ".killed")).exists())

    def test_output_cap(self):
        r = self.box.run(["python", "-c", "print('A' * 300000 + 'END')"], output_tail=100).record
        self.assertTrue(r.output_truncated)
        self.assertTrue(r.stdout_tail.rstrip().endswith("END"))

    def test_original_is_never_written_and_secrets_not_copied(self):
        before = tree_digest(self.repo)
        self.box.run(["python", "-c", "open('NEW.txt','w').write('x'); open('README.md','a').write('t')"])
        self.assertEqual(tree_digest(self.repo), before)
        self.assertEqual(self.box.changes()["created"], ["NEW.txt"])
        self.assertFalse((self.box.root / ".env").exists())

    def test_test_runner_works_through_the_container_backend(self):
        run = run_tests(self.box)
        self.assertEqual(run.status, "failed")
        self.assertEqual({t.id.rsplit(".", 1)[-1]: t.outcome for t in run.tests}, {"test_ok": "passed", "test_bad": "failed"})
        self.assertEqual(run.problems, [])
        self.assertTrue(run.record.security_boundary)


class CommandLine(FakeDockerCase):
    def test_testrun_cli_with_container_backend(self):
        import contextlib, io
        from cpu_platform import testrun
        real = ct.ContainerBackend
        fake_exe = str(self.exe)
        with mock.patch.object(ct, "ContainerBackend",
                               lambda image=None: real(image=image or "python:3.12-slim", docker=fake_exe)):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = testrun.main([str(self.repo), "--backend", "container"])
        self.assertEqual(code, 1)                                             # test_bad fails
        data = json.loads(out.getvalue())
        self.assertEqual((data["record"]["backend"], data["record"]["security_boundary"]), ("container", True))
        self.assertEqual(data["status"], "failed")

    def test_dev_cli_still_requires_acknowledgement(self):
        import contextlib, io
        from cpu_platform import testrun
        with self.assertRaises(SandboxError), contextlib.redirect_stderr(io.StringIO()):
            testrun.make_backend("dev", False)


@unittest.skipUnless(real_docker_ready(), f"real Docker engine with image {IMAGE} not available (NOT RUN)")
class RealDocker(unittest.TestCase):
    """M3.2 acceptance on a machine with Docker: the M3.1 contract checks plus real isolation."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name) / "repo"
        make_repo(self.repo)
        self.box = Sandbox(self.repo, ct.ContainerBackend(image=IMAGE))
        self.addCleanup(self.box.close)

    def test_probe_passed(self):
        self.assertTrue(self.box.backend.security_boundary)
        self.assertNotEqual(self.box.backend.probe["uid"], 0)

    def test_network_root_and_filesystem_are_really_isolated(self):
        code = ("import json, os, socket\nout = {'uid': os.getuid()}\n"
                "try:\n    socket.create_connection(('8.8.8.8', 53), timeout=3); out['net'] = True\n"
                "except OSError:\n    out['net'] = False\n"
                "try:\n    open('/etc/noema', 'w'); out['etc'] = True\nexcept OSError:\n    out['etc'] = False\n"
                "out['home'] = os.path.exists('/home') and os.listdir('/home')\nprint(json.dumps(out))\n")
        r = self.box.run(["python", "-c", code]).record
        report = json.loads(r.stdout_tail)
        self.assertEqual((report["net"], report["etc"]), (False, False))
        self.assertNotEqual(report["uid"], 0)
        self.assertFalse(report["home"])                                       # host home not mounted

    def test_timeout_output_cap_and_no_writes(self):
        before = tree_digest(self.repo)
        self.assertTrue(self.box.run(["python", "-c", "import time; time.sleep(60)"], timeout=3).record.timed_out)
        self.assertTrue(self.box.run(["python", "-c", "print('x' * 300000)"], output_tail=100).record.output_truncated)
        self.box.run(["python", "-c", "open('NEW.txt','w').write('x')"])
        self.assertEqual(tree_digest(self.repo), before)

    def test_memory_limit(self):
        box = Sandbox(self.repo, ct.ContainerBackend(image=IMAGE, memory_bytes=256 * 1024 ** 2))
        self.addCleanup(box.close)
        r = box.run(["python", "-c", "x = bytearray(1024 ** 3)"]).record
        self.assertNotEqual(r.exit_code, 0)

    def test_test_runner(self):
        run = run_tests(self.box)
        self.assertEqual({t.id.rsplit(".", 1)[-1]: t.outcome for t in run.tests}, {"test_ok": "passed", "test_bad": "failed"})


if __name__ == "__main__":
    unittest.main()

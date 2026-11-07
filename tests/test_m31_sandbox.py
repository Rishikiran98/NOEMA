"""M3.1 tests: the sandbox interface and its dev backend (NOT a security boundary).

Covers the acceptance tests (timeout, output cap, no writes to the original repo) plus
adversarial cases: process trees that outlive the timeout, output floods, resource limits,
credential scrubbing, workdir escapes, shell strings, and what is (not) copied."""
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from cpu_platform import sandbox as sb
from cpu_platform.contract import EngineResult, Cost, not_measured, validate_result

PY = sys.executable
POSIX = os.name == "posix"


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() or p.is_symlink():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(os.readlink(p).encode() if p.is_symlink() else p.read_bytes())
    return h.hexdigest()


class Workspace(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo = self.tmp / "repo"
        files = {"app/main.py": "print('main')\n", "tests/test_app.py": "import unittest\n",
                 ".env": "SECRET=hunter2\n", "build/out.bin": "artifact\n", ".gitignore": "build/\n",
                 "README.md": "readme\n"}
        for rel, text in files.items():
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text(text, encoding="utf-8")
        (self.tmp / "outside.txt").write_text("outside\n", encoding="utf-8")
        self.box = sb.Sandbox(self.repo, sb.DevBackend(acknowledge_not_isolated=True))
        self.addCleanup(self.box.close)

    def run_py(self, code, **kw):
        return self.box.run([PY, "-c", code], **kw)


class Contract(Workspace):
    def test_record_states_backend_and_no_security_boundary(self):
        result = self.run_py("print('hi')")
        r = result.record
        self.assertEqual((r.backend, r.security_boundary, r.exit_code, r.timed_out), ("dev", False, 0, False))
        self.assertEqual(r.stdout_tail.strip(), "hi")
        self.assertIn("NOT a security boundary", result.warnings[0])
        self.assertFalse(result.limits["security_boundary"])
        self.assertFalse(result.limits["network_blocked"])
        # A record is valid inside an EngineResult, and could never claim isolation.
        cost = Cost()
        validate_result(EngineResult(request_id="r", task="repo_fix", status="ok", engine_version="t", answer="a",
                                     executions=[r], cost=cost, not_run=not_measured(cost)))
        self.assertFalse(sb.DevBackend.security_boundary)

    def test_dev_requires_explicit_acknowledgement(self):
        with self.assertRaisesRegex(sb.SandboxError, "not isolated"):
            sb.DevBackend()

    def test_nonzero_exit_and_stderr(self):
        r = self.run_py("import sys; print('bad', file=sys.stderr); sys.exit(3)").record
        self.assertEqual((r.exit_code, r.stderr_tail.strip()), (3, "bad"))

    def test_command_that_cannot_start(self):
        r = self.box.run(["definitely-not-a-real-program-xyz"]).record
        self.assertIsNone(r.exit_code)
        self.assertIn("could not start", r.stderr_tail)

    def test_ids_are_sequential(self):
        self.assertEqual([self.run_py("pass").record.id for _ in range(3)], ["X1", "X2", "X3"])


class Timeouts(Workspace):
    def test_timeout_kills(self):
        start = time.perf_counter()
        r = self.run_py("import time; time.sleep(60)", timeout=1).record
        self.assertTrue(r.timed_out)
        self.assertIsNone(r.exit_code)
        self.assertLess(time.perf_counter() - start, 10)

    @unittest.skipUnless(POSIX, "process-group kill is tested on POSIX")
    def test_timeout_kills_the_whole_process_tree(self):
        marker = self.box.root / "grandchild_finished.txt"
        (self.box.root / "grandchild.py").write_text(
            f"import time\ntime.sleep(3)\nopen({str(marker)!r}, 'w').write('late')\n", encoding="utf-8")
        (self.box.root / "parent.py").write_text(
            "import subprocess, sys, time\nsubprocess.Popen([sys.executable, 'grandchild.py'])\ntime.sleep(60)\n",
            encoding="utf-8")
        record = self.box.run([PY, "parent.py"], timeout=1).record
        self.assertTrue(record.timed_out, record.stderr_tail)
        time.sleep(4)
        self.assertFalse(marker.exists(), "a child process survived the timeout")

    def test_invalid_timeouts(self):
        for value in (0, -1, 3601):
            with self.assertRaises(sb.SandboxError):
                self.run_py("pass", timeout=value)


class OutputCaps(Workspace):
    def test_tail_is_kept_and_truncation_flagged(self):
        r = self.run_py("print('A' * 500000 + 'THE_END')", output_tail=200).record
        self.assertEqual(r.exit_code, 0)
        self.assertTrue(r.output_truncated)
        self.assertLessEqual(len(r.stdout_tail), 200)
        self.assertTrue(r.stdout_tail.rstrip().endswith("THE_END"))

    def test_output_flood_is_killed(self):
        result = self.run_py("import sys\nwhile True: sys.stdout.write('x' * 65536)", timeout=30,
                             output_limit=2 * 1024 * 1024)
        self.assertTrue(result.output_killed)
        self.assertIsNone(result.record.exit_code)
        self.assertTrue(result.record.output_truncated)
        self.assertFalse(result.record.timed_out)

    def test_small_output_not_truncated(self):
        self.assertFalse(self.run_py("print('ok')").record.output_truncated)

    def test_tail_never_exceeds_contract_limit(self):
        r = self.run_py("print('y' * 100000)", output_tail=10 ** 9).record
        self.assertLessEqual(len(r.stdout_tail), 16_000)


class OriginalIsNeverWritten(Workspace):
    def test_writes_modifications_and_deletions_stay_in_the_copy(self):
        before = tree_digest(self.repo)
        code = ("import os, shutil\n"
                "open('NEW.txt', 'w').write('x')\n"
                "open('README.md', 'a').write('tamper')\n"
                "os.remove('app/main.py')\n"
                "shutil.rmtree('tests')\n")
        self.assertEqual(self.run_py(code).record.exit_code, 0)
        self.assertEqual(tree_digest(self.repo), before)
        self.assertEqual(self.box.changes(), {"created": ["NEW.txt"], "modified": ["README.md"],
                                              "deleted": ["app/main.py", "tests/test_app.py"]})

    def test_copy_excludes_secrets_ignored_files_and_symlinks(self):
        try:
            os.symlink(self.tmp / "outside.txt", self.repo / "link.txt")
            linked = True
        except (OSError, NotImplementedError):
            linked = False
        with sb.Sandbox(self.repo, sb.DevBackend(acknowledge_not_isolated=True)) as box:
            self.assertEqual(sorted(box.copied), [".gitignore", "README.md", "app/main.py", "tests/test_app.py"])
            self.assertFalse((box.root / ".env").exists())
            self.assertFalse((box.root / "build").exists())
            if linked:
                self.assertFalse((box.root / "link.txt").exists())

    def test_cleanup_removes_the_copy(self):
        box = sb.Sandbox(self.repo, sb.DevBackend(acknowledge_not_isolated=True))
        root = box.root
        box.close()
        self.assertFalse(root.exists())


class Environment(Workspace):
    def test_credentials_are_dropped_and_home_is_inside(self):
        secrets = {"AWS_SECRET_ACCESS_KEY": "a", "GITHUB_TOKEN": "b", "LLAMA_API_KEY": "c",
                   "CPU_REASONING_API_KEY": "d", "OPENAI_API_KEY": "e", "NOEMA_HOME": "f"}
        with mock.patch.dict(os.environ, secrets):
            r = self.run_py("import json, os; print(json.dumps(dict(os.environ)))").record
        env = json.loads(r.stdout_tail)
        for name in secrets:
            self.assertNotIn(name, env)
        # Inside the sandbox, but beside the repository copy, not in it (M3.3: a test that
        # creates an index in TEMP must not be told it is "inside the repository").
        base, root = str(self.box.root.parent), str(self.box.root)
        for name in ("HOME", "TMPDIR", "TEMP", "TMP", "USERPROFILE"):
            self.assertTrue(env[name].startswith(base), name)
            self.assertFalse(env[name].startswith(root + os.sep), name)
        self.assertEqual(env["HTTPS_PROXY"], sb.DEAD_PROXY)
        self.assertEqual(env["NOEMA_SANDBOX"], "dev")
        self.assertIn("PATH", env)

    def test_network_flag_is_recorded_honestly(self):
        result = self.run_py("pass", network=True)
        self.assertTrue(result.record.network)
        self.assertTrue(any("does not restrict" in w for w in result.warnings))
        self.assertFalse(self.run_py("pass").record.network)


@unittest.skipUnless(POSIX, "POSIX resource limits")
class ResourceLimits(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / "a.py").write_text("pass\n", encoding="utf-8")
        self.box = sb.Sandbox(tmp.name, sb.DevBackend(acknowledge_not_isolated=True, memory_bytes=512 * 1024 ** 2,
                                                       file_bytes=1024 * 1024))
        self.addCleanup(self.box.close)

    def test_memory_limit(self):
        r = self.box.run([PY, "-c", "x = bytearray(2 * 1024 ** 3)"]).record
        self.assertNotEqual(r.exit_code, 0)
        self.assertIn("MemoryError", r.stderr_tail)

    def test_file_size_limit(self):
        r = self.box.run([PY, "-c", "open('big.bin', 'wb').write(b'0' * (4 * 1024 * 1024))"]).record
        self.assertNotEqual(r.exit_code, 0)

    def test_limits_are_reported(self):
        limits = self.box.run([PY, "-c", "pass"]).limits
        self.assertEqual((limits["memory_bytes"], limits["file_size_bytes"]), (512 * 1024 ** 2, 1024 * 1024))
        self.assertEqual(limits["core_dumps"], "disabled")


class RequestValidation(Workspace):
    def test_shell_strings_and_bad_commands_are_refused(self):
        for command in ("python -c 'print(1)'", b"ls", [], [""], ["python", None], ["py\x00thon"]):
            with self.subTest(command=command), self.assertRaises(sb.SandboxError):
                self.box.run(command)

    def test_workdir_must_stay_inside_the_copy(self):
        for workdir in ("..", "../..", str(self.tmp), "/", "app/../../", "missing"):
            with self.subTest(workdir=workdir), self.assertRaises(sb.SandboxError):
                self.box.run([PY, "-c", "pass"], workdir=workdir)
        self.assertEqual(self.box.run([PY, "-c", "import os; print(os.getcwd())"], workdir="app").workdir, "app")

    def test_not_a_directory(self):
        with self.assertRaises(sb.SandboxError):
            sb.Sandbox(self.tmp / "missing", sb.DevBackend(acknowledge_not_isolated=True))


class Cli(Workspace):
    def test_cli_requires_acknowledgement_and_reports(self):
        import contextlib
        import io
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            sb.main(["run", str(self.repo), "--", PY, "-c", "pass"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = sb.main(["run", str(self.repo), "--i-understand-dev-is-not-isolated", "--",
                            PY, "-c", "open('x.txt','w').write('1')"])
        self.assertEqual(code, 0)
        data = json.loads(out.getvalue())
        self.assertEqual((data["record"]["backend"], data["record"]["security_boundary"]), ("dev", False))
        self.assertEqual(data["changes"]["created"], ["x.txt"])
        self.assertFalse((self.repo / "x.txt").exists())


class NoemaSuiteRunsInside(unittest.TestCase):
    def test_noema_contract_tests_run_in_a_sandbox_copy(self):
        root = Path(__file__).resolve().parents[1]
        before = tree_digest(root / "cpu_platform")
        with sb.Sandbox(root, sb.DevBackend(acknowledge_not_isolated=True)) as box:
            r = box.run([PY, "-m", "unittest", "tests.test_m13_contract"], timeout=300).record
        self.assertEqual(r.exit_code, 0, r.stderr_tail)
        # "OK" or "OK (skipped=N)": optional tools (e.g. jsonschema) may be absent, as in a
        # plain python image (seen 2026-11-07 in the container run).
        self.assertRegex(r.stderr_tail.strip().splitlines()[-1], r"^OK( \(skipped=\d+\))?$")
        self.assertEqual(tree_digest(root / "cpu_platform"), before)


if __name__ == "__main__":
    unittest.main()

"""M3.3 tests: detecting and running unittest / pytest suites in the sandbox, per-test
outcomes and durations, targeted runs, and what the runner can and cannot detect.

The acceptance test runs NOEMA's own full suite inside a sandbox; it is skipped when this
suite is itself running in a sandbox (NOEMA_SANDBOX is set), to avoid recursion."""
import contextlib
import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from cpu_platform import testrun as tr
from cpu_platform.sandbox import DevBackend, Sandbox, SandboxError

try:
    import pytest  # noqa: F401  (test-only: NOEMA does not depend on it)
    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False

REPO_ROOT = Path(__file__).resolve().parents[1]

UNITTEST_SUITE = {
    "app/__init__.py": "",
    "app/calc.py": "def add(a, b):\n    return a + b\n",
    "tests/test_calc.py": textwrap.dedent('''
        import time
        import unittest
        from app.calc import add


        class Calc(unittest.TestCase):
            def test_add(self):
                time.sleep(0.05)
                self.assertEqual(add(2, 3), 5)

            def test_wrong(self):
                self.assertEqual(add(2, 2), 5)

            def test_crash(self):
                raise RuntimeError("boom")

            @unittest.skip("not today")
            def test_skipped(self):
                pass

            @unittest.expectedFailure
            def test_known_bug(self):
                self.assertEqual(add(1, 1), 3)

            @unittest.expectedFailure
            def test_fixed_bug(self):
                self.assertEqual(add(1, 1), 2)

            def test_subtests(self):
                for n in (1, 2, 3):
                    with self.subTest(n=n):
                        self.assertNotEqual(n, 2)


        class BrokenSetup(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise OSError("database unavailable")

            def test_never_runs(self):
                pass
    '''),
    "tests/test_import_error.py": "import module_that_does_not_exist\n",
}

PYTEST_SUITE = {
    "pytest.ini": "[pytest]\ntestpaths = tests\n",
    "calc.py": "def add(a, b):\n    return a + b\n",
    "tests/test_calc.py": textwrap.dedent('''
        import pytest
        from calc import add

        def test_add():
            assert add(2, 3) == 5

        def test_wrong():
            assert add(2, 2) == 5

        @pytest.mark.skip(reason="not today")
        def test_skipped():
            pass

        @pytest.mark.xfail(reason="known")
        def test_known_bug():
            assert add(1, 1) == 3

        @pytest.fixture
        def broken():
            raise OSError("fixture failed")

        def test_fixture_error(broken):
            pass
    '''),
}


def write(root: Path, files: dict):
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")


class Workspace(unittest.TestCase):
    files: dict = UNITTEST_SUITE

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.repo = Path(tmp.name) / "repo"
        write(self.repo, self.files)
        self.box = Sandbox(self.repo, DevBackend(acknowledge_not_isolated=True))
        self.addCleanup(self.box.close)

    def outcomes(self, run):
        return {t.id: t.outcome for t in run.tests}


class Unittest(Workspace):
    def test_every_outcome_is_reported_per_test(self):
        run = tr.run_tests(self.box)
        self.assertEqual((run.framework, run.status), ("unittest", "failed"))
        got = self.outcomes(run)
        expected = {
            "test_calc.Calc.test_add": "passed", "test_calc.Calc.test_wrong": "failed",
            "test_calc.Calc.test_crash": "error", "test_calc.Calc.test_skipped": "skipped",
            "test_calc.Calc.test_known_bug": "xfail", "test_calc.Calc.test_fixed_bug": "xpass",
            "test_calc.Calc.test_subtests": "failed",
        }
        for tid, outcome in expected.items():
            self.assertEqual(got.get(tid), outcome, tid)
        self.assertTrue(any("setUpClass" in t and got[t] == "error" for t in got), got)
        self.assertTrue(any("test_import_error" in t and got[t] == "error" for t in got), got)
        self.assertEqual(run.counts["total"], len(run.tests))
        self.assertEqual(run.problems, [])

    def test_durations_and_messages(self):
        run = tr.run_tests(self.box)
        tests = {t.id: t for t in run.tests}
        self.assertGreaterEqual(tests["test_calc.Calc.test_add"].duration_s, 0.04)
        self.assertIn("AssertionError", tests["test_calc.Calc.test_wrong"].message)
        self.assertIn("boom", tests["test_calc.Calc.test_crash"].message)
        self.assertIn("not today", tests["test_calc.Calc.test_skipped"].message)
        self.assertGreater(run.record.duration_s, 0)
        self.assertEqual((run.record.backend, run.record.security_boundary), ("dev", False))

    def test_targeted_run(self):
        run = tr.run_tests(self.box, select=["tests.test_calc.Calc.test_add"])
        self.assertEqual((run.status, list(self.outcomes(run).values())), ("passed", ["passed"]))
        self.assertEqual(run.selected, ["tests.test_calc.Calc.test_add"])

    def test_selectors_cannot_be_options(self):
        for bad in ("-k", "--pdb", "x\x00y"):
            with self.subTest(bad=bad), self.assertRaises(SandboxError):
                tr.run_tests(self.box, select=[bad])

    def test_all_passing_suite(self):
        (self.box.root / "tests" / "test_import_error.py").unlink()
        run = tr.run_tests(self.box, select=["tests.test_calc.Calc.test_add", "tests.test_calc.Calc.test_skipped"])
        self.assertEqual(run.status, "passed")
        self.assertEqual((run.counts["passed"], run.counts["skipped"]), (1, 1))


class EdgeCases(unittest.TestCase):
    def run_suite(self, files, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        repo = Path(tmp.name) / "repo"
        write(repo, files)
        with Sandbox(repo, DevBackend(acknowledge_not_isolated=True)) as box:
            return tr.run_tests(box, **kw)

    def test_no_tests(self):
        run = self.run_suite({"app.py": "x = 1\n"})
        self.assertEqual(run.status, "error")
        self.assertIn("no tests were collected", run.problems)

    def test_timeout(self):
        run = self.run_suite({"tests/test_slow.py": "import time, unittest\nclass S(unittest.TestCase):\n"
                                                     "    def test_slow(self):\n        time.sleep(60)\n"}, timeout=2)
        self.assertEqual(run.status, "timed_out")
        self.assertTrue(run.record.timed_out)

    def test_process_exiting_early_leaves_no_results(self):
        run = self.run_suite({"tests/test_exit.py": "import os, unittest\nclass E(unittest.TestCase):\n"
                                                     "    def test_exit(self):\n        os._exit(0)\n"})
        self.assertEqual(run.status, "error")
        self.assertTrue(any("no results file" in p for p in run.problems), run.problems)

    def test_planted_results_with_wrong_nonce_are_rejected(self):
        forge = ("import json, os, sys, unittest\nclass F(unittest.TestCase):\n    def test_forge(self):\n"
                 "        json.dump({'nonce': 'guess', 'tests': [{'id': 'x', 'outcome': 'passed'}]}, open(sys.argv[1], 'w'))\n"
                 "        os._exit(0)\n")
        run = self.run_suite({"tests/test_forge.py": forge})
        self.assertEqual(run.status, "error")
        self.assertTrue(any("nonce mismatch" in p for p in run.problems), run.problems)

    def test_in_process_forgery_is_NOT_detected_documented_limit(self):
        """Code under test runs in the reporting process, so it can read the nonce and forge
        a clean report. This test pins that limitation so the docs stay true; isolation (M3.2)
        is what would address it."""
        forge = ("import json, os, sys, unittest\nclass F(unittest.TestCase):\n    def test_forge(self):\n"
                 "        json.dump({'nonce': sys.argv[2], 'tests_run': 1, 'tests': "
                 "[{'id': 'all.good', 'outcome': 'passed', 'duration_s': 0}]}, open(sys.argv[1], 'w'))\n"
                 "        os._exit(0)\n    def test_really_broken(self):\n        self.fail('broken')\n")
        run = self.run_suite({"tests/test_forge.py": forge})
        self.assertEqual(run.status, "passed")
        self.assertIn("could forge them", run.trust_note)

    def test_naive_forgery_contradicting_exit_code_is_flagged(self):
        forge = ("import json, os, sys, unittest\nclass F(unittest.TestCase):\n    def test_forge(self):\n"
                 "        json.dump({'nonce': sys.argv[2], 'tests_run': 1, 'tests': "
                 "[{'id': 'all.good', 'outcome': 'passed', 'duration_s': 0}]}, open(sys.argv[1], 'w'))\n"
                 "        os._exit(1)\n")
        run = self.run_suite({"tests/test_forge.py": forge})
        self.assertEqual(run.status, "error")
        self.assertTrue(any("inconsistent" in p for p in run.problems), run.problems)

    def test_explicit_unknown_framework(self):
        with self.assertRaises(SandboxError):
            self.run_suite({"a.py": ""}, framework="nose")


class Detection(Workspace):
    files = PYTEST_SUITE

    def test_pytest_config_without_pytest_falls_back(self):
        plan = tr.detect(self.box, python="definitely-not-a-python-xyz")
        self.assertEqual(plan.framework, "unittest")
        self.assertIn("not importable", plan.reason)

    def test_markers(self):
        for files, marker in (({"pyproject.toml": "[tool.pytest.ini_options]\n"}, "pyproject.toml"),
                              ({"setup.cfg": "[tool:pytest]\n"}, "setup.cfg"),
                              ({"tox.ini": "[pytest]\n"}, "tox.ini"),
                              ({"tests/conftest.py": ""}, "conftest.py"),
                              ({"setup.cfg": "[metadata]\n"}, None)):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(marker=marker):
                write(Path(tmp), files)
                found = tr._pytest_configured(Path(tmp))
                self.assertEqual(found.split()[0] if found else None, marker)


@unittest.skipUnless(HAVE_PYTEST, "pytest not installed (test-only; NOEMA does not depend on it)")
class Pytest(Workspace):
    files = PYTEST_SUITE

    def test_detected_and_parsed_from_junit(self):
        run = tr.run_tests(self.box)
        self.assertEqual(run.framework, "pytest")
        got = {t.id.split("::")[-1]: t.outcome for t in run.tests}
        self.assertEqual(got, {"test_add": "passed", "test_wrong": "failed", "test_skipped": "skipped",
                               "test_known_bug": "xfail", "test_fixture_error": "error"})
        self.assertEqual(run.status, "failed")
        self.assertEqual(run.problems, [])

    def test_targeted_node_id(self):
        run = tr.run_tests(self.box, select=["tests/test_calc.py::test_add"])
        self.assertEqual((run.status, run.counts["total"]), ("passed", 1))


class JunitParsing(unittest.TestCase):
    def test_parse(self):
        xml = """<testsuites><testsuite name="s">
          <testcase classname="tests.test_a" name="test_ok" time="0.25"/>
          <testcase classname="tests.test_a" name="test_bad" time="0.1"><failure message="assert 1 == 2">trace</failure></testcase>
          <testcase classname="tests.test_a" name="test_err" time="x"><error message="boom"/></testcase>
          <testcase classname="tests.test_a" name="test_skip"><skipped message="later"/></testcase>
          <testcase classname="tests.test_a" name="test_xf"><skipped type="pytest.xfail" message="known"/></testcase>
        </testsuite></testsuites>"""
        tests = tr.parse_junit(xml)
        self.assertEqual([(t.id, t.outcome) for t in tests], [
            ("tests.test_a::test_ok", "passed"), ("tests.test_a::test_bad", "failed"),
            ("tests.test_a::test_err", "error"), ("tests.test_a::test_skip", "skipped"),
            ("tests.test_a::test_xf", "xfail")])
        self.assertEqual((tests[0].duration_s, tests[2].duration_s), (0.25, 0.0))
        self.assertIn("assert 1 == 2", tests[1].message)


class Cli(Workspace):
    def test_cli(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = tr.main([str(self.repo), "--i-understand-dev-is-not-isolated"])
        self.assertEqual(code, 1)
        data = json.loads(out.getvalue())
        self.assertEqual((data["framework"], data["status"]), ("unittest", "failed"))
        self.assertTrue(all(t["outcome"] != "passed" for t in data["tests"]))
        self.assertEqual(data["record"]["security_boundary"], False)


@unittest.skipIf(os.environ.get("NOEMA_SANDBOX"), "already inside a sandbox: avoid recursion")
class NoemaSuiteInSandbox(unittest.TestCase):
    """Acceptance (M3.3): NOEMA's own suite runs inside the sandbox with per-test results."""

    def test_full_noema_suite(self):
        with Sandbox(REPO_ROOT, DevBackend(acknowledge_not_isolated=True)) as box:
            run = tr.run_tests(box, timeout=900)
            changes = box.changes()
        failing = [(t.id, t.message[-300:]) for t in run.tests if t.outcome in ("failed", "error")]
        self.assertEqual(failing, [])
        self.assertEqual((run.framework, run.status, run.problems), ("unittest", "passed", []))
        self.assertGreater(run.counts["passed"], 400)
        self.assertEqual(changes, {"created": [], "modified": [], "deleted": []})


if __name__ == "__main__":
    unittest.main()

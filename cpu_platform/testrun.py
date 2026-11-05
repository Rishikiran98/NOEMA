"""Test runner (roadmap M3.3): detect and run a repository's ``unittest`` or ``pytest`` suite
inside a sandbox, and report pass/fail and duration per test.

    with Sandbox(repo, DevBackend(acknowledge_not_isolated=True)) as box:
        run = run_tests(box)                          # detect and run everything
        run = run_tests(box, select=["tests.test_x"])  # targeted run
        run.status, run.counts, run.tests[0]

    python -m cpu_platform.testrun PATH --i-understand-dev-is-not-isolated [--select ID ...]

How results are obtained: per-test outcomes come from a results file, not from console output
(output tails are capped). For ``unittest`` NOEMA writes a small runner script into the sandbox
that records each test's outcome and duration as JSON; for ``pytest`` its built-in
``--junitxml`` report is parsed. Each run uses a fresh nonce, so a stale or planted results file
is rejected, and the reported outcomes are cross-checked against the exit code.

What this does NOT establish: results are reported by a process that the code under test runs
in, so code that is actively malicious can forge them. Passing tests show that these tests
passed, not that the code is correct (CLAUDE.md). Tamper-resistant reporting needs isolation
(M3.2).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from secrets import token_hex
import sys
import xml.etree.ElementTree as ET

from .contract import ExecutionRecord
from .sandbox import DevBackend, Sandbox, SandboxError

OUTCOMES = ("passed", "failed", "error", "skipped", "xfail", "xpass")
MAX_MESSAGE = 2000

UNITTEST_RUNNER = r'''
import json, os, sys, time, traceback, unittest

sys.path.insert(0, os.getcwd())   # the repository, not this script's folder
out_path, nonce, start_dir, top_dir, pattern = sys.argv[1:6]
selected = sys.argv[6:]
records = {}
order = []

def rec(test, outcome, err=None, result=None, reason=None):
    tid = test.id()
    if tid not in records:
        order.append(tid)
        records[tid] = {"id": tid, "outcome": outcome, "duration_s": 0.0, "message": ""}
    entry = records[tid]
    rank = ["passed", "skipped", "xfail", "xpass", "failed", "error"]
    if rank.index(outcome) >= rank.index(entry["outcome"]):
        entry["outcome"] = outcome
    if err is not None and result is not None:
        entry["message"] = (entry["message"] + result._exc_info_to_string(err, test))[-2000:]
    elif reason:
        entry["message"] = str(reason)[-2000:]

class Result(unittest.TextTestResult):
    def startTest(self, test):
        self._noema_start = time.perf_counter()
        super().startTest(test)
    def stopTest(self, test):
        super().stopTest(test)
        if test.id() in records:
            records[test.id()]["duration_s"] = round(time.perf_counter() - self._noema_start, 6)
    def addSuccess(self, test):
        super().addSuccess(test); rec(test, "passed")
    def addFailure(self, test, err):
        super().addFailure(test, err); rec(test, "failed", err, self)
    def addError(self, test, err):
        super().addError(test, err); rec(test, "error", err, self)
    def addSkip(self, test, reason):
        super().addSkip(test, reason); rec(test, "skipped", reason=reason)
    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err); rec(test, "xfail")
    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test); rec(test, "xpass")
    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            failed = issubclass(err[0], test.failureException)
            rec(test, "failed" if failed else "error", err, self)

loader = unittest.TestLoader()
if selected:
    suite = loader.loadTestsFromNames(selected)
else:
    suite = loader.discover(start_dir, pattern=pattern, top_level_dir=top_dir if top_dir != "-" else None)
runner = unittest.TextTestRunner(stream=sys.stderr, verbosity=1, resultclass=Result)
result = runner.run(suite)
payload = {"nonce": nonce, "framework": "unittest", "tests_run": result.testsRun,
           "successful": result.wasSuccessful(), "tests": [records[t] for t in order]}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
sys.exit(0 if result.wasSuccessful() else 1)
'''


@dataclass
class TestCase:
    id: str
    outcome: str          # passed | failed | error | skipped | xfail | xpass
    duration_s: float
    message: str = ""


@dataclass
class TestPlan:
    framework: str        # unittest | pytest
    reason: str
    start_dir: str = "."


@dataclass
class TestRun:
    framework: str
    status: str           # passed | failed | error | timed_out
    record: ExecutionRecord
    tests: list[TestCase] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    selected: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    trust_note: str = ("results were reported by a process the code under test runs in; malicious code "
                       "could forge them; passing tests do not show the code is correct")

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- detection

def _mentions(path: Path, *markers: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(m in text for m in markers)


def _pytest_configured(root: Path) -> str | None:
    if (root / "pytest.ini").is_file():
        return "pytest.ini"
    if _mentions(root / "pyproject.toml", "[tool.pytest"):
        return "pyproject.toml [tool.pytest]"
    if _mentions(root / "setup.cfg", "[tool:pytest]"):
        return "setup.cfg [tool:pytest]"
    if _mentions(root / "tox.ini", "[pytest]"):
        return "tox.ini [pytest]"
    if (root / "conftest.py").is_file() or (root / "tests" / "conftest.py").is_file():
        return "conftest.py"
    return None


def detect(box: Sandbox, python: str = sys.executable) -> TestPlan:
    """Pick the framework from the repository's own configuration (in the sandbox copy)."""
    root = box.root
    marker = _pytest_configured(root)
    start = "tests" if (root / "tests").is_dir() else "."
    if marker:
        probe = box.run([python, "-c", "import pytest"], timeout=60).record
        if probe.exit_code == 0:
            return TestPlan("pytest", f"pytest configured ({marker}) and importable", start)
        return TestPlan("unittest", f"pytest configured ({marker}) but not importable here; falling back to unittest",
                        start)
    return TestPlan("unittest", "no pytest configuration found", start)


# --------------------------------------------------------------------------- running

def _counts(tests: list[TestCase]) -> dict[str, int]:
    counts = {o: 0 for o in OUTCOMES}
    for t in tests:
        counts[t.outcome] += 1
    counts["total"] = len(tests)
    return counts


def _status(record: ExecutionRecord, tests: list[TestCase], problems: list[str]) -> str:
    if record.timed_out:
        return "timed_out"
    if problems or not tests:
        return "error"
    if any(t.outcome in ("failed", "error", "xpass") for t in tests):
        return "failed"
    return "passed"


def _consistency(record: ExecutionRecord, tests: list[TestCase], framework: str) -> list[str]:
    problems = []
    bad = any(t.outcome in ("failed", "error") for t in tests)
    if record.exit_code == 0 and bad:
        problems.append("exit code 0 but failing tests were reported: results inconsistent")
    if record.exit_code not in (0, None) and tests and not bad and framework == "unittest":
        problems.append(f"exit code {record.exit_code} but no failing test was reported: results inconsistent")
    if framework == "pytest" and record.exit_code in (2, 3, 4):
        problems.append(f"pytest exit code {record.exit_code}: interrupted, internal or usage error")
    return problems


def _run_unittest(box: Sandbox, plan: TestPlan, select: list[str], timeout: float, python: str):
    nonce = token_hex(16)
    tmp = box.tmpdir
    runner = tmp / f"noema_unittest_{nonce}.py"
    out = tmp / f"noema_results_{nonce}.json"
    runner.write_text(UNITTEST_RUNNER, encoding="utf-8")
    # Like `python -m unittest discover -s tests`: the repository root is the top level only
    # when the start directory is a package; otherwise the start directory is.
    top = "." if plan.start_dir == "." or (box.root / plan.start_dir / "__init__.py").is_file() else "-"
    command = [python, str(runner), str(out), nonce, plan.start_dir, top, "test*.py", *select]
    result = box.run(command, timeout=timeout)
    problems, tests = [], []
    if not out.is_file():
        problems.append("no results file was written (crash, timeout, or the process exited early)")
    else:
        try:
            data = json.loads(out.read_text(encoding="utf-8"))
            if data.get("nonce") != nonce:
                problems.append("results file nonce mismatch: stale or planted results rejected")
            else:
                for t in data["tests"]:
                    outcome = t["outcome"] if t.get("outcome") in OUTCOMES else "error"
                    tests.append(TestCase(str(t["id"])[:500], outcome, float(t.get("duration_s", 0.0)),
                                          str(t.get("message", ""))[-MAX_MESSAGE:]))
                if data.get("tests_run", len(tests)) == 0 and not select:
                    problems.append("no tests were collected")
        except (ValueError, KeyError, TypeError) as exc:
            problems.append(f"results file unreadable: {type(exc).__name__}")
    for path in (runner, out):
        path.unlink(missing_ok=True)
    return result, tests, problems


def parse_junit(xml_text: str) -> list[TestCase]:
    tests = []
    root = ET.fromstring(xml_text)
    for case in root.iter("testcase"):
        classname, name = case.get("classname", ""), case.get("name", "")
        tid = f"{classname}::{name}" if classname else name
        outcome, message = "passed", ""
        for child in case:
            tag = child.tag
            text = (child.get("message") or "") + "\n" + (child.text or "")
            if tag == "failure":
                outcome, message = "failed", text
            elif tag == "error":
                outcome, message = "error", text
            elif tag == "skipped":
                kind = (child.get("type") or "").lower()
                outcome = "xfail" if "xfail" in kind else "skipped"
                message = text
        try:
            duration = float(case.get("time") or 0.0)
        except ValueError:
            duration = 0.0
        tests.append(TestCase(tid[:500], outcome, round(duration, 6), message.strip()[-MAX_MESSAGE:]))
    return tests


def _run_pytest(box: Sandbox, plan: TestPlan, select: list[str], timeout: float, python: str):
    nonce = token_hex(16)
    out = box.tmpdir / f"noema_junit_{nonce}.xml"
    command = [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={out}",
               "-o", f"junit_suite_name=noema-{nonce}", *select]
    result = box.run(command, timeout=timeout)
    problems, tests = [], []
    if not out.is_file():
        problems.append("no JUnit report was written (crash, timeout, or collection failed early)")
    else:
        try:
            text = out.read_text(encoding="utf-8")
            if f"noema-{nonce}" not in text:
                problems.append("JUnit report nonce mismatch: stale or planted results rejected")
            else:
                tests = parse_junit(text)
        except (ET.ParseError, OSError) as exc:
            problems.append(f"JUnit report unreadable: {type(exc).__name__}")
        out.unlink(missing_ok=True)
    if result.record.exit_code == 5:
        problems.append("pytest collected no tests")
    return result, tests, problems


def run_tests(box: Sandbox, *, select: list[str] | None = None, framework: str | None = None,
              timeout: float = 600, python: str = sys.executable) -> TestRun:
    """Detect (unless ``framework`` is given) and run the suite, or only ``select``."""
    select = [s for s in (select or []) if s]
    for s in select:
        if s.startswith("-") or "\x00" in s:
            raise SandboxError(f"test selector must be a test id, not an option: {s!r}")
    plan = TestPlan(framework, "requested", "tests" if (box.root / "tests").is_dir() else ".") if framework \
        else detect(box, python)
    if plan.framework not in ("unittest", "pytest"):
        raise SandboxError(f"unknown framework {plan.framework!r}")
    runner = _run_pytest if plan.framework == "pytest" else _run_unittest
    result, tests, problems = runner(box, plan, select, timeout, python)
    record = result.record
    problems += _consistency(record, tests, plan.framework)
    if result.output_killed:
        problems.append("killed for excessive output")
    return TestRun(plan.framework, _status(record, tests, problems), record, tests, _counts(tests), select,
                   problems, [f"framework: {plan.framework} ({plan.reason})"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a repository's tests in a sandbox copy (dev backend: "
                                                 "NOT a security boundary)")
    parser.add_argument("path")
    parser.add_argument("--i-understand-dev-is-not-isolated", action="store_true", dest="ack", required=True)
    parser.add_argument("--select", nargs="*", default=[], help="test ids (unittest dotted ids or pytest node ids)")
    parser.add_argument("--framework", choices=("unittest", "pytest"))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--details", action="store_true", help="list every test, not only non-passing ones")
    args = parser.parse_args(argv)
    try:
        with Sandbox(args.path, DevBackend(acknowledge_not_isolated=args.ack)) as box:
            run = run_tests(box, select=args.select, framework=args.framework, timeout=args.timeout)
    except SandboxError as exc:
        parser.error(str(exc))
    out = run.to_dict()
    if not args.details:
        out["tests"] = [t for t in out["tests"] if t["outcome"] != "passed"]
    out["record"] = {k: out["record"][k] for k in ("id", "backend", "security_boundary", "exit_code",
                                                     "timed_out", "duration_s")}
    print(json.dumps(out, indent=2))
    return 0 if run.status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

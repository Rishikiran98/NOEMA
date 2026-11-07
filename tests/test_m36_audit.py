"""M3.6 tests: audit reports.

Acceptance: a human can reproduce every verified claim from the report alone. The tests play
that human: they read only the saved JSON report (never the engine's objects), save the files it
names with `audit save`, run the commands it prints (making only the documented substitution of
`--backend dev --i-understand-dev-is-not-isolated` for `--backend container`), and compare the
output with the report's expectations. Refuted claims must reproduce as refuted too. Changing
the repository afterwards must make the affected reproductions fail."""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cpu_first.core import Completion
from cpu_platform import audit
from cpu_platform import fix_search as fsx
from cpu_platform.contract import (CheckedClaim, EngineResult, Evidence, from_v1_ask, request_from_dict,
                                   result_from_dict)
from cpu_platform.repo_index import RepoIndex
from cpu_platform.repo_qa import RepoQA
from cpu_platform.retrieval import Retriever
from cpu_platform.sandbox import DevBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
DEV = ["--backend", "dev", "--i-understand-dev-is-not-isolated"]

CALC = "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n"
TESTS = ("import unittest\nfrom app.calc import add, mul\n\n\nclass T(unittest.TestCase):\n"
         "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n"
         "    def test_mul(self):\n        self.assertEqual(mul(2, 3), 6)\n")


def fenced(diff):
    return f"Here is a fix.\n```diff\n{diff}```\nIt corrects the operator."


CHEAT = fenced("--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -7 +7 @@\n"
               "-        self.assertEqual(add(2, 3), 5)\n+        self.assertEqual(add(2, 3), -1)\n")
STALE = fenced("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,2 +1,2 @@\n def add(x, y):\n-    return x - y\n+    return x + y\n")
WRONG = fenced("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a * b\n")
# Miscounted header and a dropped blank line: read leniently, reported as a clean diff.
GOOD = fenced("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,3 +1,3 @@\n def add(a, b):\n-    return a - b\n"
              "+    return a + b\n \n def mul(a, b):\n")


class Scripted:
    model = "scripted"

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), 0

    def chat(self, system, prompt, **kwargs):
        self.calls += 1
        return Completion(self.answers[min(self.calls - 1, len(self.answers) - 1)], 0.5, 400, 60, None)


def make_repo(root: Path) -> Path:
    for rel, text in {"app/__init__.py": "", "app/calc.py": CALC, "tests/test_calc.py": TESTS}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


class Reader:
    """Someone with the JSON report, the repository and a NOEMA checkout, and nothing else."""

    def __init__(self, report_path: Path, repo: Path, workdir: Path):
        self.report_path, self.repo, self.workdir = report_path, repo, workdir
        self.report = json.loads(report_path.read_text(encoding="utf-8"))

    def run(self, argv: list[str]) -> subprocess.CompletedProcess:
        assert argv[:2] == ["python", "-m"], argv
        return subprocess.run([sys.executable, *argv[1:]], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)

    def save(self, name: str, spec: dict) -> Path:
        out = self.workdir / name
        done = self.run(["python", "-m", "cpu_platform.audit", "save", str(self.report_path), spec["source"],
                         "--out", str(out)])
        assert done.returncode == 0, done.stderr
        return out

    def reproduce(self, claim: dict) -> list[str]:
        """Problems found ([] = reproduced as the report says)."""
        how = claim["reproduce"]
        if not how["reproducible"]:
            return [f"not reproducible: {how['why_not']}"]
        files = {name: self.save(name, spec) for name, spec in how["files"].items()}
        problems = []
        for cmd in how["commands"]:
            argv = []
            for arg in cmd["argv"]:
                arg = arg.replace("{repo}", str(self.repo))
                for name, path in files.items():
                    arg = arg.replace(f"{{file:{name}}}", str(path))
                argv.append(arg)
            if argv[-2:] == ["--backend", "container"] or "--backend" in argv:
                at = argv.index("--backend")
                argv[at:at + 2] = DEV                     # the documented substitution
            done = self.run(argv)
            try:
                output = json.loads(done.stdout)
            except ValueError:
                return [f"{claim['id']}: output is not JSON: {done.stdout[-300:]} {done.stderr[-300:]}"]
            problems += [f"{claim['id']}: {p}" for p in audit.check_expectations(output, cmd["expect"])]
        return problems


class FixSearchReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.repo = make_repo(root / "repo")
        request = request_from_dict({"question": "make the tests pass", "task": "repo_fix",
                                     "repo": {"path": str(cls.repo)}, "allow_execution": True,
                                     "budget": {"max_model_calls": 4}})
        search = fsx.FixSearch(Scripted(CHEAT, STALE, WRONG, GOOD), lambda: DevBackend(acknowledge_not_isolated=True),
                               candidates=4)
        cls.result = search.run(request, cls.repo)
        cls.json_path, cls.md_path = audit.write(request, cls.result, root / "audits")
        cls.work = root / "reader"
        cls.work.mkdir()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_run_covers_every_claim_kind(self):
        checkers = {c.checker for c in self.result.checked_claims}
        self.assertTrue({"testrun.unittest", "patch.validate", "patch.apply"} <= checkers, checkers)
        self.assertIn("refuted", {c.result for c in self.result.checked_claims})
        self.assertEqual(self.result.status, "ok")

    def test_every_checked_claim_reproduces_from_the_report_alone(self):
        reader = Reader(self.json_path, self.repo, self.work)
        claims = reader.report["claims"]
        self.assertGreaterEqual(len(claims), 9)
        for claim in claims:
            with self.subTest(claim=claim["id"], text=claim["claim"]):
                self.assertEqual(reader.reproduce(claim), [])

    def test_repository_state_matches_digest_command(self):
        report = json.loads(self.json_path.read_text(encoding="utf-8"))
        done = subprocess.run([sys.executable, "-m", "cpu_platform.audit", "digest", str(self.repo)], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(json.loads(done.stdout)["tree_sha256"], report["repository"]["tree_sha256"])
        self.assertEqual(report["repository"]["files"], 3)

    def test_report_contents(self):
        report = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.assertEqual(report["request"]["question"], "make the tests pass")
        self.assertIn("+    return a + b\n", report["patch"])
        self.assertTrue(any(t["patch"] is None and t["not_passing"] for t in report["tests"]))   # the baseline
        self.assertTrue(all(x["backend"] == "dev" and not x["security_boundary"] for x in report["executions"]))
        items = {n["item"] for n in report["not_checked"]}
        self.assertIn("fix correctness", items)
        md = self.md_path.read_text(encoding="utf-8")
        for heading in ("## Request", "## Repository state", "## Answer", "## Checked claims", "## Patch",
                        "## Unverified claims", "## Test results", "## Commands run", "## Evidence", "## Cost",
                        "## NOT checked"):
            self.assertIn(heading, md)
        self.assertIn("python -m cpu_platform.patch try D4.diff --repo <REPO> --backend container", md)
        self.assertIn("**NO**", md)                                     # dev is not a security boundary
        self.assertIn("NOT MEASURED", md)

    def test_changing_the_repository_breaks_reproduction(self):
        changed = Path(self.tmp.name) / "changed"
        shutil.copytree(self.repo, changed)
        (changed / "app/calc.py").write_text(CALC.replace("a * b", "a ** b"), encoding="utf-8")
        reader = Reader(self.json_path, changed, self.work)
        full = [c for c in reader.report["claims"] if "full test suite passes" in c["claim"] and c["result"] == "verified"]
        self.assertEqual(len(full), 1)
        self.assertNotEqual(reader.reproduce(full[0]), [])
        digest, _ = audit.tree_digest(changed)
        self.assertNotEqual(digest, reader.report["repository"]["tree_sha256"])


AUTH = ("import hmac\n\n\ndef check_password(user, password):\n"
        "    return hmac.compare_digest(user.password_hash, hash_password(password))\n\n\n"
        "def hash_password(password):\n    return \"h:\" + password\n")


class RepoQAReport(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / "repo"
        (self.repo / "app").mkdir(parents=True)
        (self.repo / "app/auth.py").write_text(AUTH, encoding="utf-8")
        answer = ("It compares with `hmac.compare_digest(user.password_hash, hash_password(password))` "
                  "[app/auth.py:4-5]. Hashing is `bcrypt.hash(password)` [app/auth.py:8-9]. "
                  "Sessions live in [app/session.py:1-3], and more is at [app/auth.py:40-45].")
        with RepoIndex(self.repo, self.root / "i.sqlite") as index:
            index.update()
            retriever = Retriever(index)
            self.request = request_from_dict({"question": "How are passwords checked?", "task": "repo_ask",
                                              "repo": {"path": str(self.repo)}})
            self.result = RepoQA(Scripted(answer)).ask(self.request, retriever)
        self.json_path, self.md_path = audit.write(self.request, self.result, self.root / "audits")
        (self.root / "reader").mkdir()

    def test_every_citation_claim_reproduces_including_refuted_ones(self):
        reader = Reader(self.json_path, self.repo, self.root / "reader")
        claims = reader.report["claims"]
        self.assertEqual({c["checker"] for c in claims},
                         {"citation_check.file", "citation_check.lines", "citation_check.quote"})
        self.assertEqual({c["result"] for c in claims}, {"verified", "refuted"})
        for claim in claims:
            with self.subTest(claim=claim["claim"]):
                self.assertEqual(reader.reproduce(claim), [])
        md = self.md_path.read_text(encoding="utf-8")
        self.assertIn("## Citations", md)
        self.assertIn("relevance", md)

    def test_editing_a_cited_line_breaks_its_reproduction(self):
        (self.repo / "app/auth.py").write_text(AUTH.replace("compare_digest", "compare"), encoding="utf-8")
        reader = Reader(self.json_path, self.repo, self.root / "reader")
        quote = next(c for c in reader.report["claims"] if c["checker"] == "citation_check.quote" and c["result"] == "verified")
        lines = next(c for c in reader.report["claims"] if c["checker"] == "citation_check.lines"
                     and c["result"] == "verified" and "4-5" in c["claim"])
        self.assertNotEqual(reader.reproduce(quote), [])
        self.assertNotEqual(reader.reproduce(lines), [])         # the span hash changed


class Limits(unittest.TestCase):
    def test_unknown_checker_and_missing_repository_state_are_listed(self):
        result = EngineResult(request_id="r1", task="repo_ask", status="ok", engine_version="x", answer="a",
                              evidence=[Evidence(id="answer", kind="model_output", source="model", excerpt="a")],
                              checked_claims=[CheckedClaim("it is fine", "verified", "mystery_checker", "s", ["answer"])])
        request = request_from_dict({"question": "q", "task": "repo_ask", "repo": {"path": "."}})
        report = audit.build(request, result)
        self.assertFalse(report["claims"][0]["reproduce"]["reproducible"])
        items = {n["item"] for n in report["not_checked"]}
        self.assertEqual(items, {"reproduction of K1", "repository state"})
        self.assertIn("NOT POSSIBLE", audit.to_markdown(report))

    def test_v1_exact_arithmetic_gets_a_manual_recipe(self):
        result = from_v1_ask("What is 2/3 + 1/6?", {"route": "exact_arithmetic", "status": "ok", "answer": "5/6"},
                             request_id="r2", engine_version="x")
        report = audit.build(request_from_dict({"question": "What is 2/3 + 1/6?"}), result)
        how = report["claims"][0]["reproduce"]
        self.assertTrue(how["reproducible"])
        self.assertIn("Fraction", how["steps"][0])
        self.assertIsNone(report["repository"])
        self.assertNotIn("repository state", {n["item"] for n in report["not_checked"]})

    def test_fences_survive_backticks_in_content(self):
        result = EngineResult(request_id="r3", task="ask", status="ok", engine_version="x",
                              answer="````\n```\nnot closed", evidence=[])
        md = audit.to_markdown(audit.build(request_from_dict({"question": "q"}), result))
        self.assertIn("`````\n````\n```\nnot closed\n`````", md)

    def test_test_summary_fits_and_says_when_truncated(self):
        from cpu_platform.testrun import TestCase, TestRun
        from cpu_platform.contract import ExecutionRecord
        rec = ExecutionRecord("x", ["python"], "dev", False, False, 1, False, 1.0)
        tests = [TestCase(f"tests.test_mod.Case.test_{n:04d}_" + "x" * 40, "failed", 0.0, "") for n in range(300)]
        summary = json.loads(audit.test_summary(TestRun("unittest", "failed", rec, tests, {"failed": 300}), None, None))
        self.assertTrue(summary["truncated"])
        self.assertLessEqual(len(json.dumps(summary)), 4000)
        report_claim = CheckedClaim("before any patch, 300 selected test(s) fail", "verified", "testrun.unittest", "s",
                                    ["x", "x-T"])
        result = EngineResult(request_id="r4", task="repo_fix", status="incomplete", engine_version="x",
                              executions=[rec], checked_claims=[report_claim],
                              evidence=[Evidence(id="x-T", kind="command_output", source="x",
                                                 excerpt=audit.test_summary(TestRun("unittest", "failed", rec, tests,
                                                                                    {"failed": 300}), None, None),
                                                 retrieved_by="testrun.unittest")])
        report = audit.build(request_from_dict({"question": "q", "task": "repo_fix", "repo": {"path": "."},
                                                "allow_execution": True}), result)
        self.assertFalse(report["claims"][0]["reproduce"]["reproducible"])

    def test_span_refuses_paths_outside_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "r"
            root.mkdir()
            (Path(tmp) / "secret.txt").write_text("s\n")
            (root / "a.py").write_text("one\ntwo\n")
            self.assertFalse(audit.span(root, "../secret.txt", 1, 1)["exists"])
            self.assertFalse(audit.span(root, str(Path(tmp) / "secret.txt"), 1, 1)["exists"])
            ok = audit.span(root, "a.py", 2, 2, quote="two")
            self.assertEqual((ok["exists"], ok["in_range"], ok["quote_found"], ok["lines"]), (True, True, True, ["two"]))
            self.assertFalse(audit.span(root, "a.py", 2, 3)["in_range"])


class Cli(unittest.TestCase):
    def test_fix_search_writes_reports_and_render_rebuilds_them(self):
        from tests.fake_llama import FakeLlama
        fake = FakeLlama("Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", accuracy=1.0, think_capable=False)
        self.addCleanup(fake.close)
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_repo(Path(tmp) / "repo")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                fsx.main([str(repo), "-n", "1", "--backend", "dev", "--i-understand-dev-is-not-isolated",
                          "--model-url", fake.url, "--format", "json", "--audit", str(Path(tmp) / "a")])
            result = result_from_dict(json.loads(out.getvalue()))
            files = sorted(p.name for p in (Path(tmp) / "a").iterdir())
            self.assertEqual(files, [f"{result.request_id}.audit.json", f"{result.request_id}.audit.md"])
            (Path(tmp) / "result.json").write_text(out.getvalue())
            (Path(tmp) / "request.json").write_text(json.dumps({"question": "q", "task": "repo_fix",
                                                                "repo": {"path": str(repo)}, "allow_execution": True}))
            with contextlib.redirect_stdout(io.StringIO()):
                audit.main(["render", str(Path(tmp) / "result.json"), "--request", str(Path(tmp) / "request.json"),
                            "--out", str(Path(tmp) / "b")])
            again = json.loads((Path(tmp) / "b" / f"{result.request_id}.audit.json").read_text())
            self.assertEqual(again["result"]["request_id"], result.request_id)
            self.assertEqual(again["repository"]["files"], 3)


if __name__ == "__main__":
    unittest.main()

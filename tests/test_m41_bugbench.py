"""M4.1 tests: mutation operators, benchmark building (keep only caught mutants), splits by
function, freezing, the eval guard, the runner (oracle and null stand-in models, not models),
and the frozen benchmark in data/bugbench itself.

Acceptance: >= 200 mutants from NOEMA plus >= 1 other open-source Python repository, frozen
dev/eval split, generator seeds recorded."""
import ast
import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from cpu_first.core import Completion
from cpu_platform import bugbench as bb
from cpu_platform.mutate import OPERATORS, apply_site, mutation_sites
from cpu_platform.sandbox import DevBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH = REPO_ROOT / "data" / "bugbench"


def dev():
    return DevBackend(acknowledge_not_isolated=True)


def sites_by(source, operator):
    return [s for s in mutation_sites(source) if s.operator == operator]


def mutated_lines(source, site):
    return [l for l in apply_site(source, site).splitlines() if l not in source.splitlines()]


class Operators(unittest.TestCase):
    def test_flip_comparison_negates(self):
        src = "def f(a, b):\n    return a < b and a == 1 and b not in (2,) and a is not None\n"
        got = {s.description for s in sites_by(src, "flip_comparison")}
        self.assertEqual(got, {"< -> >=", "== -> !=", "not in -> in", "is not -> is"})

    def test_off_by_one_boundaries_and_integers(self):
        src = "def f(xs, n):\n    for i in range(n - 1):\n        if i <= 3:\n            xs = xs[1:]\n    return xs\n"
        got = {s.description for s in sites_by(src, "off_by_one")}
        self.assertTrue({"boundary <= -> <", "1 -> 2", "1 -> 0", "3 -> 4", "3 -> 2"} <= got, got)

    def test_swapped_operands_only_for_non_commutative(self):
        src = "def f(a, b):\n    return (a - b) + (a * b) + a / b\n"
        swaps = sites_by(src, "swapped_operands")
        self.assertEqual(sorted(s.replacement for s in swaps), ["b - a", "b / a"])

    def test_wrong_constant(self):
        src = ("def f(x):\n    if x.endswith('-'):\n        return True\n    rate = 0.25\n    big = 100\n"
               "    label = 'name'\n    return big * rate\n")
        got = {s.description for s in sites_by(src, "wrong_constant")}
        self.assertTrue({"True -> False", "0.25 -> 2.5", "100 -> 1000", "'-' -> '-_'"} <= got, got)
        self.assertNotIn("'name' -> 'name_'", got)                 # a label, not behaviour

    def test_dropped_condition_operands_and_guards(self):
        src = ("def f(x, y):\n    if x is None:\n        return 0\n    if x > 0 and y > 0:\n        return x\n"
               "    return y\n")
        got = [s for s in sites_by(src, "dropped_condition")]
        descriptions = {s.description for s in got}
        self.assertIn("delete guard `if x is None`", descriptions)
        self.assertIn("drop operand 1 of and", descriptions)
        guard = next(s for s in got if s.description.startswith("delete guard `if x is None`"))
        self.assertNotIn("x is None", apply_site(src, guard))
        compile(apply_site(src, guard), "x", "exec")

    def test_only_function_bodies_are_mutated(self):
        src = ('LIMIT = 10\n\n\n@decorate(5 < 6)\ndef f(x: int = 3 < 4) -> bool:\n'
               '    """Docstring with < and True."""\n    return x < LIMIT\n\n\nclass C:\n    flag = True\n\n'
               '    def m(self, y):\n        def inner(z):\n            return z > 1\n        return f"{y < 2}"\n')
        sites = mutation_sites(src)
        self.assertEqual({(s.function, s.line) for s in sites if s.operator == "flip_comparison"},
                         {("f", 7), ("C.m.inner", 15)})
        for s in sites:
            self.assertGreaterEqual(s.line, 7)

    def test_every_site_compiles_and_changes_the_program(self):
        for path in (REPO_ROOT / "cpu_platform" / "patch.py", BENCH / "repos/more_itertools/more_itertools/recipes.py"):
            source = path.read_text(encoding="utf-8")
            tree = ast.dump(ast.parse(source))
            for site in mutation_sites(source, str(path)):
                with self.subTest(path=path.name, site=site.description, line=site.line):
                    mutated = apply_site(source, site)
                    self.assertNotEqual(ast.dump(ast.parse(mutated)), tree)
                    self.assertEqual(mutated[:site.start], source[:site.start])         # the edit is local
                    self.assertEqual(mutated[site.start + len(site.replacement):], source[site.end:])

    def test_non_ascii_columns(self):
        src = "def f(s):\n    t = 'é' + s\n    return len(t) < 3\n"
        (site,) = [s for s in sites_by(src, "flip_comparison")]
        self.assertEqual(apply_site(src, site).splitlines()[2], "    return len(t) >= 3")

    def test_apply_site_refuses_a_changed_source(self):
        src = "def f(a):\n    return a < 1\n"
        site = sites_by(src, "flip_comparison")[0]
        with self.assertRaises(ValueError):
            apply_site(src.replace("a < 1", "a<1 "), site)


OPS = '''def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def unused(a, b):
    return a - b


def loose(x):
    return x >= 0
'''
OPS_TESTS = '''import unittest
from pkg.ops import clamp, loose


class T(unittest.TestCase):
    def test_clamp(self):
        self.assertEqual([clamp(v, 0, 10) for v in (-5, 0, 5, 10, 15)], [0, 0, 5, 10, 10])

    def test_loose(self):
        self.assertTrue(loose(5))
'''


class Building(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        repo = self.root / "small"
        for rel, text in {"pkg/__init__.py": "", "pkg/ops.py": OPS, "tests/__init__.py": "",
                          "tests/test_ops.py": OPS_TESTS}.items():
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_text(text, encoding="utf-8")
        self.spec = bb.RepoSpec("small", "vendored", ("pkg",), path="small", prefix="sm")

    def build(self, out, target=50, seed=7):
        return bb.build(noema_commit="0" * 7, seed=seed, targets={"small": target}, out_dir=out, backend_factory=dev,
                        root=self.root, repos={"small": self.spec}, log=lambda *a: None)

    def test_keeps_only_caught_mutants_of_tested_functions(self):
        manifest = self.build(self.root / "bench")
        _, mutants = bb.load(None, self.root / "bench")
        self.assertTrue(mutants)
        self.assertEqual({m["function"] for m in mutants} - {"clamp", "loose"}, set())   # `unused` has no tests
        stats = manifest["repos"]["small"]["stats"]
        self.assertGreater(stats.get("survived", 0), 0)                                    # loose's weak test
        for m in mutants:
            self.assertEqual(m["test_scope"], ["tests.test_ops"])
            self.assertTrue(m["caught_by"])
            self.assertIn(m["id"], set(manifest["dev"]) | set(manifest["eval"]))
        self.assertEqual(bb.check(self.root / "bench", self.root, run_tests_for=3, backend_factory=dev), [])

    def test_crlf_checkout(self):
        """Regression (Windows, 2026-11-12): sites computed on LF text were applied to CRLF bytes."""
        for path in (self.root / "small").rglob("*.py"):
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))  # CRLF on any OS
        bench = self.root / "bench"
        manifest = self.build(bench)
        _, mutants = bb.load(None, bench)
        self.assertTrue(mutants)
        self.assertFalse(any("\r" in m["bug"] or "\r" in m["reference_fix"] for m in mutants))
        self.assertEqual(bb.check(bench, self.root, run_tests_for=2, backend_factory=dev), [])
        work = self.root / "work"
        bb.materialize(bb.spec_of(manifest, "small"), work, root=self.root)
        bb.apply_bug(work, mutants[0])
        self.assertIn(b"\r\n", (work / mutants[0]["file"]).read_bytes())    # endings kept

    def test_deterministic_and_frozen(self):
        a = self.build(self.root / "a")
        b = self.build(self.root / "b")
        self.assertEqual((a["sha256"], a["dev"], a["eval"]), (b["sha256"], b["dev"], b["eval"]))
        with self.assertRaises(SystemExit):
            self.build(self.root / "a")                       # frozen: needs --force
        (self.root / "a" / "mutants.jsonl").write_text("{}\n")
        with self.assertRaises(SystemExit):
            bb.load(None, self.root / "a")                    # tampered file refused

    def test_splits_never_share_a_function(self):
        mutants = [{"id": f"x{n}", "repo": "r", "file": "f.py", "function": f"fn{n % 4}"} for n in range(12)]
        splits = bb.assign_splits(mutants, 1)
        side = {}
        for m in mutants:
            s = "dev" if m["id"] in splits["dev"] else "eval"
            self.assertEqual(side.setdefault(m["function"], s), s)
        self.assertEqual(Counter(side.values()), {"dev": 2, "eval": 2})

    def test_runner_with_oracle_and_null_stand_ins(self):
        bench = self.root / "bench"
        self.build(bench)
        manifest, mutants = bb.load("dev", bench)
        mutant = mutants[0]

        class Oracle:
            model = "oracle (returns the reference fix; not a model)"

            def chat(self, *args, **kwargs):
                return Completion(f"```diff\n{mutant['reference_fix']}```", 0.1, 10, 10, None)

        class Null:
            model = "null (never answers with a diff; not a model)"

            def chat(self, *args, **kwargs):
                return Completion("I am not sure.", 0.1, 10, 10, None)

        good = bb.run("dev", model=Oracle(), backend_factory=dev, candidates=1, ids=[mutant["id"]],
                      out_path=self.root / "good.jsonl", out_dir=bench, root=self.root, log=lambda *a: None)
        self.assertEqual((good["overall"]["solved"], good["exact_restore"]), (1, 1))
        bad = bb.run("dev", model=Null(), backend_factory=dev, candidates=2, out_path=self.root / "bad.jsonl",
                     out_dir=bench, root=self.root, log=lambda *a: None)
        self.assertEqual(bad["overall"]["solved"], 0)
        self.assertEqual(bad["overall"]["attempted"], len(mutants))
        rows = [json.loads(l) for l in (self.root / "bad.jsonl").read_text().splitlines()]
        self.assertEqual({r["status"] for r in rows}, {"incomplete"})

    def test_model_error_is_shown_and_stops_the_run(self):
        from cpu_first.core import ModelTransportError
        bench = self.root / "bench"
        self.build(bench)

        class Down:
            model = "down"

            def chat(self, *args, **kwargs):
                raise ModelTransportError("connection refused")

        lines = []
        summary = bb.run("dev", model=Down(), backend_factory=dev, candidates=2, out_path=self.root / "d.jsonl",
                         out_dir=bench, root=self.root, log=lines.append)
        self.assertEqual((summary["errors"], summary["overall"]["attempted"]), (1, 1))
        self.assertIn("error local_model_unavailable: ", "\n".join(lines))
        self.assertIn("connection refused", "\n".join(lines))
        self.assertTrue(lines[-1].startswith("stopping: the model server is not usable"))
        row = json.loads((self.root / "d.jsonl").read_text().splitlines()[0])
        self.assertIn("connection refused", row["error_message"])

    def test_a_timeout_does_not_stop_the_run_but_three_in_a_row_do(self):
        from cpu_first.core import ModelTransportError
        bench = self.root / "bench"
        self.build(bench)
        _, mutants = bb.load("dev", bench)

        class Slow:
            model = "slow"

            def chat(self, *args, **kwargs):
                raise ModelTransportError("timed out")

        lines = []
        bb.run("dev", model=Slow(), backend_factory=dev, candidates=1, out_path=self.root / "s.jsonl",
               out_dir=bench, root=self.root, log=lines.append)
        rows = (self.root / "s.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), min(3, len(mutants)))
        if len(mutants) >= 3:
            self.assertTrue(lines[-1].startswith("stopping:"))

    def test_eval_needs_approval_and_is_logged(self):
        bench = self.root / "bench"
        self.build(bench)

        class Null:
            model = "null"

            def chat(self, *args, **kwargs):
                return Completion("no", 0.1, 1, 1, None)

        with self.assertRaises(SystemExit):
            bb.run("eval", model=Null(), backend_factory=dev, out_path=self.root / "e.jsonl", out_dir=bench,
                   root=self.root)
        summary = bb.run("eval", model=Null(), backend_factory=dev, candidates=1, limit=1, allow_eval=True,
                         out_path=self.root / "e.jsonl", out_dir=bench, root=self.root, log=lambda *a: None)
        self.assertEqual(summary["eval_run_number"], 1)
        self.assertTrue((bench / "eval_runs.jsonl").is_file())

    def test_summary_leaves_out_mutants_that_are_not_tasks_here(self):
        rows = [{"id": "a", "repo": "r", "operator": "o", "status": "ok", "solved": True, "exact_restore": True,
                 "model_calls": 1, "model_seconds": 1.0, "total_seconds": 2.0},
                {"id": "b", "repo": "r", "operator": "o", "status": "incomplete", "solved": False, "exact_restore": None,
                 "model_calls": 4, "model_seconds": 4.0, "total_seconds": 5.0},
                {"id": "c", "repo": "r", "operator": "o", "status": "error", "solved": False, "exact_restore": None,
                 "error": "baseline_failed", "model_calls": 0, "model_seconds": 0.0, "total_seconds": 20.0},
                {"id": "d", "repo": "r", "operator": "o", "status": "incomplete", "solved": False, "exact_restore": None,
                 "not_caught_here": True, "model_calls": 0, "model_seconds": 0.0, "total_seconds": 3.0}]
        out = bb.summarize(rows)
        self.assertEqual((out["overall"]["solved"], out["overall"]["attempted"]), (1, 2))
        self.assertEqual((out["baseline_unusable"], out["not_caught_here"], out["errors_counted"]), (["c"], ["d"], 0))
        self.assertEqual(out["candidate_outcomes"], {})
        rows[1]["candidates"] = ["does_not_apply", "targeted_failed"]
        rows[2]["candidates"] = ["passed"]                                    # excluded: not counted
        self.assertEqual(bb.summarize(rows)["candidate_outcomes"], {"does_not_apply": 1, "targeted_failed": 1})
        path = self.root / "r.jsonl"
        bb.write_jsonl(path, rows)
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bb.main(["summarize", str(path)])
        self.assertEqual(json.loads(buf.getvalue())["rows"], 4)

    def test_resume_keeps_finished_mutants_and_retries_model_errors(self):
        from cpu_first.core import ModelTransportError
        bench = self.root / "bench"
        self.build(bench)
        _, mutants = bb.load("dev", bench)
        first = mutants[0]["id"]
        calls = []

        class Null:
            model = "null"

            def chat(self, *args, **kwargs):
                calls.append("null")
                return Completion("no", 0.1, 1, 1, None)

        class Down:
            model = "down"

            def chat(self, *args, **kwargs):
                calls.append("down")
                raise ModelTransportError("connection refused")

        out = self.root / "r.jsonl"
        quiet = lambda *a: None    # noqa: E731
        bb.run("dev", model=Down(), backend_factory=dev, candidates=1, ids=[first], out_path=out, out_dir=bench,
               root=self.root, log=quiet)
        self.assertEqual(bb.read_jsonl(out)[0]["error"], "local_model_unavailable")
        bb.run("dev", model=Null(), backend_factory=dev, candidates=1, ids=[first], out_path=out, out_dir=bench,
               root=self.root, log=quiet, resume=True)
        rows = bb.read_jsonl(out)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["error"])                                     # the model error was retried
        calls.clear()
        summary = bb.run("dev", model=Down(), backend_factory=dev, candidates=1, ids=[first], out_path=out,
                         out_dir=bench, root=self.root, log=quiet, resume=True)
        self.assertEqual(calls, [])                                             # finished: not run again
        self.assertEqual(bb.read_jsonl(out), rows)
        self.assertEqual(summary["overall"]["attempted"], 1)
        with self.assertRaises(SystemExit):
            bb.run("eval", model=Null(), backend_factory=dev, out_path=out, out_dir=bench, root=self.root,
                   allow_eval=True, resume=True)

    def test_an_unusable_sandbox_stops_the_run_cleanly(self):
        from cpu_platform.sandbox import SandboxError
        bench = self.root / "bench"
        self.build(bench)

        class NoDocker:
            name = "container"

            def __init__(self):
                raise SandboxError("Docker engine not reachable (is Docker Desktop / dockerd running?)")

        class Null:
            model = "null"

            def chat(self, *args, **kwargs):
                return Completion("no", 0.1, 1, 1, None)

        lines = []
        summary = bb.run("dev", model=Null(), backend_factory=NoDocker, candidates=1, out_path=self.root / "x.jsonl",
                         out_dir=bench, root=self.root, log=lines.append)
        self.assertEqual(summary["overall"]["attempted"], 0)
        self.assertTrue(any("sandbox is not usable" in l and "Docker" in l for l in lines))

    def test_failure_split(self):
        def cand(status, still=None, new=None, untargeted=False):
            return {"status": status, "still_failing": still, "new_failures": new, "untargeted": untargeted}
        base = {"repo": "r", "operator": "o", "solved": False, "exact_restore": None, "status": "incomplete",
                "model_calls": 1, "model_seconds": 1.0, "total_seconds": 1.0}
        rows = [dict(base, id="old", candidates=["full_failed", "targeted_failed", "does_not_apply"]),   # no details
                dict(base, id="new", candidates=["full_failed"] * 4 + ["targeted_failed", "passed"],
                     candidate_details=[cand("full_failed", ["t1"], [], True), cand("full_failed", [], ["t2"]),
                                        cand("full_failed", ["t1"], ["t2"]), cand("full_failed", [], []),
                                        cand("targeted_failed", ["t1"], []), cand("passed", [], [])])]
        self.assertEqual(bb.failure_split(rows), {"not_fixed": 2, "broke_others": 1, "both": 1, "no_detail": 3,
                                                  "untargeted": 1})
        self.assertIn("failed_candidates", bb.summarize(rows))

    def test_a_damaged_base_copy_is_detected_and_rebuilt(self):
        """Regression (laptop, 2026-11-15): a file vanished from the base copy in %TEMP% during a
        long run; the next mutant crashed reading it. Now the base is checked and rebuilt."""
        from cpu_platform.fix_search import FixSearch
        bench = self.root / "bench"
        self.build(bench)
        _, mutants = bb.load("eval", bench)        # the small repository's eval side has 2 mutants
        self.assertGreaterEqual(len(mutants), 2)

        class Null:
            model = "null"

            def chat(self, *args, **kwargs):
                return Completion("no", 0.1, 1, 1, None)

        class Vandal(FixSearch):
            def run(self, request, repo, **kwargs):
                result = super().run(request, repo, **kwargs)
                victim = Path(repo).parent.parent / "base" / "small-1" / "pkg" / "ops.py"
                if victim.exists():
                    victim.unlink()                     # damage the base during the first mutant
                return result

        lines = []
        summary = bb.run("eval", model=Null(), backend_factory=dev, candidates=1, allow_eval=True,
                         out_path=self.root / "v.jsonl", out_dir=bench, root=self.root, log=lines.append,
                         search_factory=lambda: Vandal(Null(), dev, candidates=1))
        rows = bb.read_jsonl(self.root / "v.jsonl")
        self.assertEqual(len(rows), len(mutants))
        self.assertIsNone(rows[0]["base_repaired"])
        self.assertEqual(rows[1]["base_repaired"], ["pkg/ops.py"])
        self.assertTrue(any("damaged" in l and "rebuilding" in l for l in lines))
        self.assertEqual(summary["overall"]["attempted"], len(mutants))
        self.assertNotIn(True, [r.get("not_caught_here") for r in rows])     # the rebuilt base still has the bug

    def test_redo_runs_chosen_mutants_again(self):
        bench = self.root / "bench"
        self.build(bench)
        _, mutants = bb.load("dev", bench)
        calls = []

        class Null:
            model = "null"

            def chat(self, *args, **kwargs):
                calls.append(1)
                return Completion("no", 0.1, 1, 1, None)

        out = self.root / "r.jsonl"
        quiet = lambda *a: None    # noqa: E731
        bb.run("dev", model=Null(), backend_factory=dev, candidates=1, out_path=out, out_dir=bench, root=self.root,
               log=quiet)
        calls.clear()
        bb.run("dev", model=Null(), backend_factory=dev, candidates=1, out_path=out, out_dir=bench, root=self.root,
               log=quiet, resume=True)
        self.assertEqual(calls, [])
        bb.run("dev", model=Null(), backend_factory=dev, candidates=1, out_path=out, out_dir=bench, root=self.root,
               log=quiet, resume=True, redo=[mutants[0]["id"]])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(bb.read_jsonl(out)), len(mutants))

    def test_waits_for_a_loading_model_server_and_retries_a_503(self):
        from cpu_first.core import ModelTransportError
        bench = self.root / "bench"
        self.build(bench)

        class Loading:
            """503 on the first call, then answers; not ready for the first two checks."""
            model = "loading"

            def __init__(self):
                self.checks = self.calls = 0

            def ready(self):
                self.checks += 1
                return self.checks > 2

            def chat(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ModelTransportError("HTTP 503 from local model server")
                return Completion("no", 0.1, 1, 1, None)

        model, lines, naps = Loading(), [], []
        import unittest.mock as mock
        with mock.patch.object(bb.time, "sleep", naps.append):
            summary = bb.run("dev", model=model, backend_factory=dev, candidates=1, out_path=self.root / "w.jsonl",
                             out_dir=bench, root=self.root, log=lines.append)
        self.assertTrue(naps)                                                   # waited before starting
        self.assertTrue(any("answered 503" in l and "retrying" in l for l in lines))
        self.assertEqual(summary["errors"], 0)
        self.assertEqual(summary["overall"]["attempted"], 1)

    def test_wait_ready_gives_up(self):
        class Never:
            def ready(self):
                return False
        t = [0.0]
        self.assertFalse(bb.wait_ready(Never(), 30, log=lambda *a: None, sleep=lambda s: t.__setitem__(0, t[0] + s),
                                       clock=lambda: t[0]))
        self.assertTrue(bb.wait_ready(object(), 30))

    def test_scratch_folder_holds_every_copy(self):
        import tempfile as tf
        bench = self.root / "bench"
        self.build(bench)
        scratch = self.root / "scratch"
        seen = []

        class Peek:
            model = "peek"

            def chat(self, *args, **kwargs):
                seen.append(tf.gettempdir())
                return Completion("no", 0.1, 1, 1, None)

        before = tf.gettempdir()
        bb.run("dev", model=Peek(), backend_factory=dev, candidates=1, out_path=self.root / "s.jsonl", out_dir=bench,
               root=self.root, log=lambda *a: None, scratch=scratch)
        self.assertEqual(seen, [str(scratch.resolve())])
        self.assertEqual(tf.gettempdir(), before)                         # restored afterwards
        self.assertTrue(scratch.is_dir())

    def test_suspended_mutants_are_left_out_of_time_figures(self):
        base = {"repo": "r", "operator": "o", "status": "ok", "solved": True, "exact_restore": True, "model_calls": 1}
        rows = [dict(base, id="a", model_seconds=100.0, total_seconds=200.0),
                dict(base, id="b", model_seconds=150.0, total_seconds=300.0),
                dict(base, id="slept", model_seconds=80000.0, total_seconds=81969.3)]
        out = bb.summarize(rows)
        self.assertEqual(out["overall"]["solved"], 3)                     # results still count
        self.assertEqual(out["time_outliers"], ["slept"])
        self.assertEqual(out["time_without_outliers"]["total_seconds"], 500.0)
        self.assertEqual(out["time_without_outliers"]["median_seconds_per_mutant"], 300.0)

    def test_wall_time_budget_is_passed_and_recorded(self):
        from cpu_platform.fix_search import FixSearch
        bench = self.root / "bench"
        self.build(bench)
        budgets = []

        class Spy(FixSearch):
            def run(self, request, repo, **kwargs):
                budgets.append(request.budget.max_wall_seconds)
                return super().run(request, repo, **kwargs)

        class Null:
            model = "null"

            def chat(self, *args, **kwargs):
                return Completion("no", 0.1, 1, 1, None)

        summary = bb.run("dev", model=Null(), backend_factory=dev, candidates=2, out_path=self.root / "b.jsonl",
                         out_dir=bench, root=self.root, log=lambda *a: None, max_wall_seconds=3600,
                         search_factory=lambda: Spy(Null(), dev, candidates=2))
        self.assertEqual(set(budgets), {3600})
        self.assertEqual(summary["budget"], {"max_model_calls": 2, "max_output_tokens": 1024, "max_wall_seconds": 3600})

    def test_wilson_interval(self):
        self.assertEqual(bb.wilson(0, 0), (0.0, 1.0))
        lo, hi = bb.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)


def _have_commit(commit):
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "cat-file", "-e", f"{commit}^{{commit}}"],
                              capture_output=True, timeout=30).returncode == 0
    except OSError:
        return False


@unittest.skipUnless((BENCH / "manifest.json").is_file(), "benchmark not built")
class FrozenBenchmark(unittest.TestCase):
    """The acceptance numbers, checked against the committed files."""

    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.mutants = bb.load(None, BENCH)

    def test_acceptance_counts(self):
        by_repo = Counter(m["repo"] for m in self.mutants)
        self.assertGreaterEqual(len(self.mutants), 200)
        self.assertGreaterEqual(by_repo["noema"], 100)
        self.assertGreaterEqual(by_repo["more_itertools"], 100)
        self.assertEqual({m["operator"] for m in self.mutants}, set(OPERATORS))
        self.assertEqual(self.manifest["generator"]["seed"], bb.DEFAULT_SEED)
        self.assertTrue(self.manifest["dev"] and self.manifest["eval"])
        self.assertFalse(set(self.manifest["dev"]) & set(self.manifest["eval"]))
        self.assertEqual(self.manifest["count"], len(self.mutants))

    def test_mutants_are_well_formed(self):
        for m in self.mutants:
            with self.subTest(id=m["id"]):
                self.assertTrue(m["caught_by"])
                self.assertTrue(m["test_scope"])
                self.assertFalse(any("/test" in m["file"] for _ in [0]))
                self.assertNotIn(m["file"], bb.spec_of(self.manifest, m["repo"]).exclude)
                self.assertIn(f"--- a/{m['file']}", m["bug"])

    def test_vendored_repository_is_pristine(self):
        from cpu_platform.audit import tree_digest
        info = self.manifest["repos"]["more_itertools"]
        digest, files = tree_digest(BENCH / "repos" / "more_itertools")
        self.assertEqual((digest, files), (info["tree_sha256"], info["files"]))

    def test_every_mutant_and_reference_fix_applies(self):
        commit = self.manifest["repos"]["noema"]["commit"]
        if not _have_commit(commit):
            self.skipTest(f"NOEMA base commit {commit} not in this clone")
        self.assertEqual(bb.check(BENCH, REPO_ROOT, run_tests_for=0, backend_factory=dev), [])

    def test_a_sample_is_still_caught_by_its_tests(self):
        sample = [m for m in self.mutants if m["repo"] == "more_itertools"][:3]
        with tempfile.TemporaryDirectory() as tmp:
            base = bb.materialize(bb.REPOS["more_itertools"], Path(tmp) / "base", root=REPO_ROOT)
            from cpu_platform.sandbox import Sandbox
            from cpu_platform.testrun import run_tests
            for m in sample:
                original = (base / m["file"]).read_bytes()
                bb.apply_bug(base, m)
                try:
                    with Sandbox(base, dev()) as box:
                        run = run_tests(box, select=m["test_scope"], timeout=120)
                finally:
                    (base / m["file"]).write_bytes(original)
                failing = sorted(t.id for t in run.tests if t.outcome in ("failed", "error"))
                self.assertEqual((run.status, failing[:20]), ("failed", m["caught_by"]))


if __name__ == "__main__":
    unittest.main()

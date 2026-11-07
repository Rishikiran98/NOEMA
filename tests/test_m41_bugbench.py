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

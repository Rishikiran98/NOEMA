"""M3.4 tests: parsing, validating and applying model-proposed unified diffs.

Acceptance: adversarial diffs (path traversal, binary, huge, touching tests to make them pass)
are rejected or flagged. Also: renames/symlinks/mode changes, interpreter start-up files,
Trojan Source characters, secrets, CI files, atomic application, line endings, offsets, and
the end-to-end "tests fail, patch, tests pass" loop in a sandbox."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from cpu_platform import patch as pt
from cpu_platform.sandbox import DevBackend, Sandbox
from cpu_platform.testrun import run_tests

CALC = "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
TEST = ("import unittest\nfrom app.calc import add, sub\n\n\nclass T(unittest.TestCase):\n"
        "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n"
        "    def test_sub(self):\n        self.assertEqual(sub(5, 3), 2)\n")
FIX = """--- a/app/calc.py
+++ b/app/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


def tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


class Repo(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo = self.tmp / "repo"
        files = {"app/__init__.py": "", "app/calc.py": CALC, "tests/test_calc.py": TEST, "README.md": "readme\n",
                 "setup.cfg": "[tool:pytest]\naddopts = -q\n", "pyproject.toml": "[project]\nname='x'\n",
                 ".env": "SECRET=1\n", ".github/workflows/ci.yml": "on: push\n"}
        for rel, text in files.items():
            (self.repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.repo / rel).write_text(text, encoding="utf-8")

    def codes(self, diff, **policy):
        check = pt.validate(pt.parse(diff), self.repo, pt.PatchPolicy(**policy))
        return check.ok, {(p.severity, p.code) for p in check.problems}

    def rejected(self, diff, code, **policy):
        ok, codes = self.codes(diff, **policy)
        self.assertFalse(ok, codes)
        self.assertIn(("reject", code), codes)


class Parsing(unittest.TestCase):
    def test_unified_diff_and_counts(self):
        p = pt.parse(FIX)
        (fp,) = p.files
        self.assertEqual((fp.path, fp.added, fp.removed, fp.new_file, fp.deleted), ("app/calc.py", 1, 1, False, False))
        self.assertEqual(p.problems, [])

    def test_extracts_from_model_answer_with_fences(self):
        answer = f"The bug is in add.\n\n```diff\n{FIX}```\n\nThat should fix it.\n```python\nprint(1)\n```\n"
        self.assertEqual([f.path for f in pt.parse(answer).files], ["app/calc.py"])

    def test_git_headers_new_and_deleted_files(self):
        diff = ("diff --git a/new.py b/new.py\nnew file mode 100644\nindex 0000000..e69de29\n--- /dev/null\n+++ b/new.py\n"
                "@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
                "diff --git a/old.py b/old.py\ndeleted file mode 100644\n--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-z = 3\n")
        new, old = pt.parse(diff).files
        self.assertEqual((new.path, new.new_file, new.added), ("new.py", True, 2))
        self.assertEqual((old.path, old.deleted, old.removed), ("old.py", True, 1))

    def test_malformed_and_empty(self):
        self.assertEqual(pt.parse("just prose, no diff").problems[0].code, "empty")
        bad = pt.parse("--- a/x.py\n+++ b/x.py\n@@ -x +1 @@\n-a\n+b\n")
        self.assertEqual(bad.problems[0].code, "malformed")
        self.assertEqual(pt.parse("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n a\n").problems[0].code, "malformed")
        # Changed on purpose (M3.5, first real-model run): a miscounted header used to be
        # rejected as malformed; it is now recounted from the body and flagged.
        miscounted = pt.parse("--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n-a\n+b\n")
        self.assertEqual([(p.severity, p.code) for p in miscounted.problems], [("flag", "recounted")])
        self.assertEqual((miscounted.files[0].hunks[0].old_len, miscounted.files[0].hunks[0].new_len), (1, 1))

    def test_undercounted_header_never_drops_changed_lines(self):
        # Before: the body was cut at the header's count, so "+    extra()" vanished silently.
        p = pt.parse("--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-a\n+b\n+    extra()\n\nThis adds a call.\n")
        (h,) = p.files[0].hunks
        self.assertEqual(h.lines, ["-a", "+b", "+    extra()"])
        self.assertIn(("flag", "recounted"), {(x.severity, x.code) for x in p.problems})

    def test_header_without_line_numbers(self):
        p = pt.parse("--- a/x.py\n+++ b/x.py\n@@ @@\n a\n-b\n+c\n")
        self.assertFalse(p.files[0].hunks[0].located)
        self.assertIn(("flag", "no_line_numbers"), {(x.severity, x.code) for x in p.problems})


class Adversarial(Repo):
    def test_path_traversal_and_absolute_paths(self):
        for path in ("../outside.py", "app/../../outside.py", "/etc/passwd", "C:/Windows/win.ini",
                     "\\\\server\\share\\x.py", "app/./calc.py"):
            with self.subTest(path=path):
                self.rejected(f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n", "path")

    def test_binary_patches(self):
        self.rejected("diff --git a/logo.png b/logo.png\nGIT binary patch\nliteral 10\nabc\n", "binary")
        self.rejected("diff --git a/a.bin b/a.bin\nBinary files a/a.bin and b/a.bin differ\n", "binary")
        self.rejected("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a\x00 + b\n",
                      "binary")

    def test_huge_patches(self):
        many = "".join(f"--- /dev/null\n+++ b/gen/f{i}.py\n@@ -0,0 +1 @@\n+x = {i}\n" for i in range(11))
        self.rejected(many, "too_many_files")
        big = "--- /dev/null\n+++ b/gen/big.py\n@@ -0,0 +1,401 @@\n" + "".join(f"+x{i} = {i}\n" for i in range(401))
        self.rejected(big, "too_large")
        wide = "--- /dev/null\n+++ b/gen/w.py\n@@ -0,0 +1 @@\n+x = '" + "y" * 250_000 + "'\n"
        self.rejected(wide, "too_large")

    def test_touching_tests_to_make_them_pass(self):
        edit = ("--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -7 +7 @@\n"
                "-        self.assertEqual(add(2, 3), 5)\n+        self.assertEqual(add(2, 3), -1)\n")
        self.rejected(edit, "touches_tests")
        for path in ("tests/test_new.py", "conftest.py", "pytest.ini", "app/test_helpers.py", "app/calc_test.py"):
            with self.subTest(path=path):
                self.rejected(f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+x = 1\n", "touches_tests")
        # setup.cfg with a pytest section is test configuration; pyproject.toml without one is not.
        self.rejected("--- a/setup.cfg\n+++ b/setup.cfg\n@@ -2 +2 @@\n-addopts = -q\n+addopts = -q -k nothing\n",
                      "touches_tests")
        ok, _ = self.codes("--- a/pyproject.toml\n+++ b/pyproject.toml\n@@ -2 +2 @@\n-name='x'\n+name='y'\n")
        self.assertTrue(ok)
        self.rejected("--- a/tests/test_calc.py\n+++ /dev/null\n@@ -1,10 +0,0 @@\n" +
                      "".join("-" + l + "\n" for l in TEST.splitlines()), "touches_tests")

    def test_test_changes_can_be_allowed_but_stay_flagged(self):
        edit = "--- /dev/null\n+++ b/tests/test_more.py\n@@ -0,0 +1 @@\n+x = 1\n"
        ok, codes = self.codes(edit, allow_test_changes=True)
        self.assertTrue(ok)
        self.assertIn(("flag", "touches_tests"), codes)

    def test_rigging_tests_from_source_code_is_flagged(self):
        diff = ("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,2 +1,3 @@\n def add(a, b):\n-    return a - b\n"
                "+    import os; os._exit(0)\n+    return a + b\n")
        ok, codes = self.codes(diff)
        self.assertIn(("flag", "test_machinery"), codes)
        self.assertTrue(ok)                                                    # flagged, not rejected

    def test_startup_hooks_ci_secrets_and_git_internals(self):
        for path in ("sitecustomize.py", "app/usercustomize.py", "evil.pth", ".github/workflows/ci.yml",
                     ".git/config", ".git/hooks/pre-commit", ".noemaignore"):
            with self.subTest(path=path):
                self.rejected(f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1 @@\n+x\n", "denied_path")
        self.rejected("--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-SECRET=1\n+SECRET=2\n", "secret_path")

    def test_renames_mode_changes_and_symlinks(self):
        self.rejected("diff --git a/app/calc.py b/app/math.py\nsimilarity index 100%\nrename from app/calc.py\n"
                      "rename to app/math.py\n", "rename")
        self.rejected("diff --git a/run.sh b/run.sh\nold mode 100644\nnew mode 100755\n", "mode_change")
        self.rejected("diff --git a/link b/link\nnew file mode 120000\n--- /dev/null\n+++ b/link\n@@ -0,0 +1 @@\n+/etc/passwd\n",
                      "mode_change")
        self.rejected("--- a/app/calc.py\n+++ b/app/other.py\n@@ -1 +1 @@\n-def add(a, b):\n+def add(a, b):\n", "rename")

    def test_trojan_source_in_added_lines(self):
        diff = ("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n"
                "+    return a + b  # \u202e } \u2066if is_admin:\u2069 \u2066\n")
        self.rejected(diff, "invisible_characters")

    def test_existence_and_duplicates(self):
        self.rejected("--- /dev/null\n+++ b/app/calc.py\n@@ -0,0 +1 @@\n+x\n", "exists")
        self.rejected("--- a/app/missing.py\n+++ b/app/missing.py\n@@ -1 +1 @@\n-a\n+b\n", "missing")
        self.rejected(FIX + FIX, "duplicate")

    def test_allow_and_deny_globs(self):
        self.rejected(FIX, "not_allowed", allow=("lib/*",))
        self.rejected(FIX, "denied_path", deny=("app/*",))
        self.assertTrue(self.codes(FIX, allow=("app/*",))[0])


class Applying(Repo):
    def setUp(self):
        super().setUp()
        self.box = Sandbox(self.repo, DevBackend(acknowledge_not_isolated=True))
        self.addCleanup(self.box.close)

    def test_fix_loop_tests_fail_then_pass_and_original_untouched(self):
        before = tree_digest(self.repo)
        self.assertEqual(run_tests(self.box).status, "failed")
        result = pt.apply(self.box, pt.parse(FIX))
        self.assertEqual((result.applied, result.files, result.problems), (True, ["app/calc.py"], []))
        self.assertEqual(result.changes, {"created": [], "modified": ["app/calc.py"], "deleted": []})
        run = run_tests(self.box)
        self.assertEqual((run.status, run.counts["passed"]), ("passed", 2))
        self.assertEqual(tree_digest(self.repo), before)

    def test_rejected_patch_writes_nothing(self):
        result = pt.apply(self.box, pt.parse("--- a/../x.py\n+++ b/../x.py\n@@ -1 +1 @@\n-a\n+b\n"))
        self.assertFalse(result.applied)
        self.assertEqual(self.box.changes(), {"created": [], "modified": [], "deleted": []})

    def test_atomic_when_a_later_file_does_not_apply(self):
        diff = FIX + "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-not the content\n+new\n"
        result = pt.apply(self.box, pt.parse(diff))
        self.assertFalse(result.applied)
        self.assertEqual(result.problems[-1].code, "does_not_apply")
        self.assertEqual(self.box.changes(), {"created": [], "modified": [], "deleted": []})

    def test_offsets_ambiguity_and_multiple_hunks(self):
        shifted = FIX.replace("@@ -1,2 +1,2 @@", "@@ -3,2 +3,2 @@")
        result = pt.apply(self.box, pt.parse(shifted))
        self.assertTrue(result.applied)
        self.assertEqual(result.offsets["app/calc.py"], [-2])
        ambiguous = "--- a/app/calc.py\n+++ b/app/calc.py\n@@ -9 +9 @@\n-    return a - b\n+    return a * b\n"
        with Sandbox(self.repo, DevBackend(acknowledge_not_isolated=True)) as box:
            r = pt.apply(box, pt.parse(ambiguous))
            self.assertFalse(r.applied)
            self.assertIn("more than one place", r.problems[-1].message)
        both = ("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
                "@@ -5,2 +5,2 @@\n def sub(a, b):\n-    return a - b\n+    return a - b  # checked\n")
        with Sandbox(self.repo, DevBackend(acknowledge_not_isolated=True)) as box:
            self.assertTrue(pt.apply(box, pt.parse(both)).applied)
            self.assertEqual((box.root / "app/calc.py").read_text(),
                             "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b  # checked\n")

    def test_new_file_deletion_crlf_and_missing_final_newline(self):
        (self.repo / "win.txt").write_bytes(b"one\r\ntwo\r\n")
        (self.repo / "noeol.txt").write_bytes(b"alpha\nbeta")
        with Sandbox(self.repo, DevBackend(acknowledge_not_isolated=True)) as box:
            diff = ("--- /dev/null\n+++ b/app/new.py\n@@ -0,0 +1,2 @@\n+X = 1\n+Y = 2\n"
                    "--- a/README.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-readme\n"
                    "--- a/win.txt\n+++ b/win.txt\n@@ -1,2 +1,2 @@\n one\n-two\n+TWO\n"
                    "--- a/noeol.txt\n+++ b/noeol.txt\n@@ -1,2 +1,2 @@\n alpha\n-beta\n\\ No newline at end of file\n+BETA\n"
                    "\\ No newline at end of file\n")
            result = pt.apply(box, pt.parse(diff))
            self.assertTrue(result.applied, result.problems)
            self.assertEqual((box.root / "app/new.py").read_text(), "X = 1\nY = 2\n")
            self.assertFalse((box.root / "README.md").exists())
            self.assertEqual((box.root / "win.txt").read_bytes(), b"one\r\nTWO\r\n")
            self.assertEqual((box.root / "noeol.txt").read_bytes(), b"alpha\nBETA")
            self.assertEqual(result.changes["created"], ["app/new.py"])

    def test_apply_only_accepts_a_sandbox(self):
        with self.assertRaises(AttributeError):
            pt.apply(str(self.repo), pt.parse(FIX))


class Lenient(Repo):
    """Diff mistakes seen from a real model (M3.5): miscounted headers, dropped or retyped context.
    Each is applied only where it matches exactly one place, and always flagged."""

    def apply(self, diff):
        box = Sandbox(self.repo, DevBackend(acknowledge_not_isolated=True))
        box.__enter__()
        self.addCleanup(box.__exit__, None, None, None)
        return box, pt.apply(box, pt.parse(diff))

    def codes_of(self, result):
        return {(p.severity, p.code) for p in result.problems}

    def test_dropped_blank_context_line(self):
        # CALC has two blank lines between the functions; the model wrote one.
        diff = ("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -2,5 +2,5 @@\n-    return a - b\n+    return a + b\n"
                " \n def sub(a, b):\n")
        box, result = self.apply(diff)
        self.assertTrue(result.applied, result.problems)
        self.assertIn(("flag", "fuzzy_context"), self.codes_of(result))
        self.assertEqual((box.root / "app/calc.py").read_text(), CALC.replace("a - b", "a + b", 1))

    def test_retyped_whitespace_inside_a_line(self):
        (self.repo / "app/rate.py").write_text("RATE = 0.02   # 20 %\n\n\ndef f():\n    return RATE\n")
        diff = "--- a/app/rate.py\n+++ b/app/rate.py\n@@ -1,3 +1,3 @@\n-RATE = 0.02 # 20 %\n+RATE = 0.20 # 20 %\n \n \n"
        box, result = self.apply(diff)
        self.assertTrue(result.applied, result.problems)
        self.assertIn("whitespace", " ".join(p.message for p in result.problems))
        self.assertEqual((box.root / "app/rate.py").read_text(), "RATE = 0.20 # 20 %\n\n\ndef f():\n    return RATE\n")

    def test_context_lines_come_from_the_file_not_the_model(self):
        (self.repo / "app/rate.py").write_text("x = 1\nRATE   =   2\ny = 3\n")
        diff = "--- a/app/rate.py\n+++ b/app/rate.py\n@@ -1,3 +1,3 @@\n x = 1\n-RATE = 2\n+RATE = 3\n y  =  3\n"
        box, result = self.apply(diff)
        self.assertTrue(result.applied, result.problems)
        self.assertEqual((box.root / "app/rate.py").read_text(), "x = 1\nRATE = 3\ny = 3\n")

    def test_removed_lines_are_never_ignored(self):
        diff = ("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a / b\n"
                "+    return a + b\n")
        box, result = self.apply(diff)
        self.assertFalse(result.applied)
        self.assertIn(("reject", "does_not_apply"), self.codes_of(result))
        self.assertEqual(box.changes(), {"created": [], "modified": [], "deleted": []})

    def test_indentation_still_exact(self):
        diff = "--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-  return a - b\n+  return a + b\n"
        _, result = self.apply(diff)
        self.assertFalse(result.applied)

    def test_lenient_placement_must_be_unique(self):
        # Both functions end in "return a - b"; with the distinguishing context line wrong,
        # trimming it would leave a hunk that matches twice: refused, not guessed.
        diff = "--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,2 +1,2 @@\n def mul(a, b):\n-    return a - b\n+    return a * b\n"
        _, result = self.apply(diff)
        self.assertFalse(result.applied)

    def test_undercounted_header_applies_whole_body(self):
        diff = ("--- a/app/calc.py\n+++ b/app/calc.py\n@@ -1,1 +1,1 @@\n def add(a, b):\n-    return a - b\n"
                "+    return a + b\n\nThe sign was wrong.\n")
        box, result = self.apply(diff)
        self.assertTrue(result.applied, result.problems)
        self.assertIn(("flag", "recounted"), self.codes_of(result))
        self.assertEqual((box.root / "app/calc.py").read_text(), CALC.replace("a - b", "a + b", 1))

    def test_no_line_numbers_placed_by_context(self):
        diff = "--- a/app/calc.py\n+++ b/app/calc.py\n@@ ... @@\n def sub(a, b):\n-    return a - b\n+    return a - b + 0\n"
        box, result = self.apply(diff)
        self.assertTrue(result.applied, result.problems)
        self.assertTrue((box.root / "app/calc.py").read_text().endswith("return a - b + 0\n"))


class Cli(Repo):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = pt.main(list(argv))
        return code, json.loads(out.getvalue())

    def test_check_and_try(self):
        good = self.tmp / "fix.diff"
        good.write_text(f"Here is the fix:\n```diff\n{FIX}```\n", encoding="utf-8")
        self.assertEqual(self.run_cli("check", str(good), "--repo", str(self.repo))[0], 0)
        code, out = self.run_cli("try", str(good), "--repo", str(self.repo), "--i-understand-dev-is-not-isolated")
        self.assertEqual((code, out["applied"], out["tests"]["status"]), (0, True, "passed"))
        self.assertIn("could forge them", out["tests"]["trust_note"])
        cheat = self.tmp / "cheat.diff"
        cheat.write_text("--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@ -7 +7 @@\n"
                         "-        self.assertEqual(add(2, 3), 5)\n+        self.assertEqual(add(2, 3), -1)\n", encoding="utf-8")
        code, out = self.run_cli("try", str(cheat), "--repo", str(self.repo), "--i-understand-dev-is-not-isolated")
        self.assertEqual((code, out["applied"]), (1, False))
        self.assertEqual(out["problems"][0]["code"], "touches_tests")
        self.assertIn("SECRET=1", (self.repo / ".env").read_text())            # original untouched throughout


if __name__ == "__main__":
    unittest.main()

"""M2.4 adversarial tests for the citation verifier: invented files, wrong lines,
paraphrased quotes, real-but-irrelevant citations, path escapes, secret and ignored files
(never read), quote-to-citation assignment, and integration with repo_qa results."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from cpu_first.core import Completion
from cpu_platform import citation_check as cc
from cpu_platform.contract import request_from_dict, result_from_dict
from cpu_platform.repo_index import RepoIndex
from cpu_platform.repo_qa import RepoQA, parse_citations
from cpu_platform.retrieval import Hit, Retriever

AUTH = '''import hmac


def check_password(user, password):
    """Compare in constant time."""
    return hmac.compare_digest(user.password_hash, hash_password(password))


def hash_password(password):
    return "h:" + password
'''
FILES = {
    "app/auth.py": AUTH,
    "app/store.py": "def load_user(name):\n    return {'name': name}\n",
    "app/empty.py": "",
    ".env": "DATABASE_PASSWORD=hunter2\n",
    "build/generated.py": "GENERATED = True\n",
    ".gitignore": "build/\n",
}


class Repo(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo = self.tmp / "repo"
        for rel, text in FILES.items():
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (self.tmp / "outside.py").write_text("OUTSIDE = 'not part of the repo'\n", encoding="utf-8")
        with RepoIndex(self.repo, self.tmp / "i.sqlite") as index:
            index.update()
            self.retriever = Retriever(index)
        self.checker = cc.checker_for(self.retriever)

    def check(self, answer, shown=None):
        return cc.check_answer(answer, parse_citations(answer, shown or {}), self.checker)

    def one(self, answer, shown=None):
        (result,) = self.check(answer, shown)
        return result


class Adversarial(Repo):
    def test_invented_file(self):
        c = self.one("It is in [app/session.py:1-5].")
        self.assertEqual((c.status, c.file_ok), ("refuted", False))
        self.assertIn("file does not exist", c.reasons)

    def test_wrong_lines(self):
        for answer, reason in (("[app/auth.py:40-45]", "outside the file (1-10)"),
                               ("[app/auth.py:8-11]", "outside the file (1-10)"),
                               ("[app/auth.py:0]", "outside the file"),
                               ("[app/empty.py:1]", "outside the file (1-0)")):
            with self.subTest(answer=answer):
                c = self.one(answer)
                self.assertEqual((c.status, c.file_ok, c.lines_ok), ("refuted", True, False))
                self.assertTrue(any(reason in r for r in c.reasons), c.reasons)

    def test_exact_and_paraphrased_quotes(self):
        ok = self.one("It calls `hmac.compare_digest(user.password_hash, hash_password(password))` [app/auth.py:4-6].")
        self.assertEqual((ok.status, [q.found for q in ok.quotes]), ("verified", [True]))
        for quote in ("hmac.compare_digest(a, b)", "hmac.compare_digest(user.password_hash,  hash_password(password))",
                      "compare_digest(password, user.password_hash)", "bcrypt.checkpw"):
            with self.subTest(quote=quote):
                c = self.one(f"It calls `{quote}` [app/auth.py:4-6].")
                self.assertEqual((c.status, [q.found for q in c.quotes]), ("refuted", [False]))
                self.assertIn("1 quoted snippet(s) not found in the cited lines", c.reasons)

    def test_quote_outside_the_cited_lines_is_refuted(self):
        c = self.one("Hashing uses `\"h:\" + password` [app/auth.py:4-6].")   # real code, but on line 10
        self.assertEqual(c.status, "refuted")

    def test_fenced_blocks(self):
        dedented = "Here:\n```python\ndef hash_password(password):\n    return \"h:\" + password   \n```\nas in [app/auth.py:9-10]."
        self.assertEqual(self.one(dedented).status, "verified")
        inner = "It returns:\n```\nreturn hmac.compare_digest(user.password_hash, hash_password(password))\n``` [app/auth.py:4-6]."
        self.assertEqual(self.one(inner).status, "verified")                 # copied from inside an indented block
        altered = "Here:\n```python\ndef hash_password(pw):\n    return \"h:\" + pw\n```\nas in [app/auth.py:9-10]."
        self.assertEqual(self.one(altered).status, "refuted")

    def test_real_but_irrelevant_citation_is_flagged_not_verified_overall(self):
        answer = "Passwords are hashed with bcrypt and a per-user salt [app/store.py:1-2]."
        c = self.one(answer)
        self.assertEqual((c.status, c.file_ok, c.lines_ok, c.quotes), ("verified", True, True, []))
        self.assertIn("relevance NOT CHECKED", c.reasons)
        self.assertEqual(cc._summary(c), "citation check: exists, no quotes, relevance NOT CHECKED")


class PathSafetyAndUnreadable(Repo):
    def setUp(self):
        super().setUp()
        self.reads = []
        original = Path.read_bytes
        test = self

        def tracking(path):
            test.reads.append(Path(path).name)
            return original(path)
        Path.read_bytes = tracking
        self.addCleanup(setattr, Path, "read_bytes", original)

    def test_escapes_are_refuted_and_never_read(self):
        for path in ("../outside.py", "app/../../outside.py", "/etc/passwd", "C:/Windows/win.ini",
                     "C:\\\\Windows\\\\win.ini", "\\\\server\\\\share\\\\x.py"):
            with self.subTest(path=path):
                c = self.checker.check(path, 1, 1)
                self.assertEqual((c.status, c.file_ok), ("refuted", False))
                self.assertTrue(c.reasons[0].startswith("invalid path"), c.reasons)
        self.assertNotIn("outside.py", self.reads)
        self.assertNotIn("passwd", self.reads)

    def test_symlink_is_refuted(self):
        try:
            os.symlink(self.tmp / "outside.py", self.repo / "app" / "link.py")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not permitted")
        c = self.checker.check("app/link.py", 1, 1)
        self.assertEqual(c.status, "refuted")
        self.assertNotIn("link.py", self.reads)

    def test_secret_and_ignored_files_are_never_opened(self):
        secret = self.one("The password is `hunter2` [.env:1].")
        self.assertEqual((secret.status, secret.quotes), ("not_checked", []))
        self.assertIn("secret_name", secret.reasons[0])
        ignored = self.one("See [build/generated.py:1].")
        self.assertEqual(ignored.status, "not_checked")
        self.assertIn("not in the index", ignored.reasons[0])
        self.assertNotIn(".env", self.reads)
        self.assertNotIn("generated.py", self.reads)

    def test_file_deleted_after_indexing(self):
        (self.repo / "app" / "store.py").unlink()
        c = self.checker.check("app/store.py", 1, 1)
        self.assertEqual((c.status, c.file_ok), ("refuted", False))


class QuoteAssignment(Repo):
    def test_two_citations_in_one_sentence(self):
        """Regression from the real laptop run: each quote belongs to the citation after it."""
        answer = ("Specifically, it defines `check_password` to compare hashes [app/auth.py:4-6], "
                  "and `hash_password` to build them [app/auth.py:9-10].")
        a, b = self.check(answer)
        self.assertEqual(([q.quote for q in a.quotes], [q.quote for q in b.quotes]),
                         (["check_password"], ["hash_password"]))
        self.assertEqual((a.status, b.status), ("verified", "verified"))

    def test_trailing_quote_and_uncited_sentence(self):
        answer = ("See [app/auth.py:9-10], which returns `\"h:\" + password`. "
                  "Elsewhere `totally_made_up()` is used.")
        (c,) = self.check(answer)
        self.assertEqual([q.quote for q in c.quotes], ['"h:" + password'])   # uncited quote is not attached

    def test_repeated_citation_collects_quotes_from_each_occurrence(self):
        answer = "`check_password` is here [app/auth.py:4-6]. It calls `hmac.compare_digest` [app/auth.py:4-6]."
        (c,) = self.check(answer)
        self.assertEqual([q.quote for q in c.quotes], ["check_password", "hmac.compare_digest"])

    def test_dots_inside_code_do_not_split_sentences(self):
        answer = "It calls `hmac.compare_digest. ` then returns [app/auth.py:4-6]."
        (c,) = self.check(answer)
        self.assertEqual(len(c.quotes), 1)

    def test_unknown_excerpt_id(self):
        (c,) = self.check("As shown [E9:3].", {})
        self.assertEqual(c.status, "refuted")


class CitationForms(unittest.TestCase):
    SHOWN = {"E3": Hit("d", "docs/x.md", 51, 88, "window", None, 1.0, [])}

    def parse(self, text):
        return [(c.path, c.start_line, c.end_line, c.unknown_excerpt) for c in parse_citations(text, self.SHOWN)]

    def test_dotfiles_and_extensionless_files_in_brackets(self):
        self.assertEqual(self.parse("[.env:1] [Makefile:3-4] [Dockerfile:1] [.github/workflows/ci.yml:2]"),
                         [(".env", 1, 1, False), ("Makefile", 3, 4, False), ("Dockerfile", 1, 1, False),
                          (".github/workflows/ci.yml", 2, 2, False)])

    def test_prose_and_excerpt_ids_are_not_misread(self):
        self.assertEqual(self.parse("At step:1 and ratio 3:2, Python 3.11 runs."), [])
        self.assertEqual(self.parse("see [E3:59]"), [("docs/x.md", 59, 59, False)])   # not also a path "E3"
        self.assertEqual(self.parse("see [E7:2]"), [("E7", 0, 0, True)])
        self.assertEqual(self.parse("see [app/a.py:0]"), [("app/a.py", 0, 0, False)])  # line 0 is a bad line, not an id


class Stub:
    model = "stub"

    def __init__(self, answer):
        self.answer = answer

    def chat(self, *args, **kwargs):
        return Completion(self.answer, 1.0, 100, 20, None)


class Integration(Repo):
    ANSWER = ("Passwords are compared with `hmac.compare_digest` [app/auth.py:4-6]. "
              "Hashing is `bcrypt.hash(password)` [app/auth.py:9-10]. It is configured in [app/settings.py:1-3]. "
              "The database password lives in [.env:1].")

    def ask(self, answer=ANSWER, **kw):
        req = request_from_dict({"question": "How are passwords compared and hashed?", "task": "repo_ask",
                                 "repo": {"path": str(self.repo)}})
        return RepoQA(Stub(answer), **kw).ask(req, self.retriever)

    def test_result_carries_specific_checked_claims(self):
        result = self.ask()
        result_from_dict(json.loads(result.to_json()))
        claims = {(k.checker, k.claim): k.result for k in result.checked_claims}
        self.assertEqual(claims[("citation_check.file", "cited file app/auth.py exists")], "verified")
        self.assertEqual(claims[("citation_check.lines", "lines 4-6 exist in app/auth.py")], "verified")
        self.assertEqual(claims[("citation_check.quote", "`hmac.compare_digest` appears in app/auth.py:4-6")], "verified")
        self.assertEqual(claims[("citation_check.quote", "`bcrypt.hash(password)` appears in app/auth.py:9-10")], "refuted")
        self.assertEqual(claims[("citation_check.file", "cited file app/settings.py exists")], "refuted")
        self.assertFalse(any(".env" in claim for _, claim in claims))         # not checked, so no claim
        for k in result.checked_claims:
            self.assertIn("says nothing", k.scope)
        reasons = {u.claim: u.reason for u in result.unverified_claims}
        self.assertIn("citation check: exists, 1/1 quotes match, relevance NOT CHECKED", reasons["citation app/auth.py:4-6"])
        self.assertIn("quote REFUTED", reasons["citation app/auth.py:9-10"])
        self.assertIn("REFUTED (file does not exist)", reasons["citation app/settings.py:1-3"])
        self.assertIn("deliberately not read", reasons["citation .env:1-1"])
        self.assertIn(result.answer, reasons)                                  # the answer stays unverified
        items = {n.item for n in result.not_run}
        self.assertIn("citation relevance", items)
        self.assertNotIn("citation verification", items)

    def test_evidence_spans_are_the_current_source(self):
        result = self.ask()
        import hashlib
        lines = AUTH.splitlines()
        span = next(e for e in result.evidence if e.retrieved_by == "citation_check" and e.start_line == 4)
        self.assertEqual(span.sha256, hashlib.sha256("\n".join(lines[3:6]).encode()).hexdigest())

    def test_checks_can_be_disabled(self):
        self.assertEqual(self.ask(check_citations=False).checked_claims, [])

    def test_cli_rechecks_a_saved_result(self):
        saved = self.tmp / "result.json"
        saved.write_text(self.ask().to_json(), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cc.main(["verify", str(saved), "--repo", str(self.repo), "--db", str(self.tmp / "v.sqlite")])
        self.assertEqual(code, 1)                                              # one refuted citation
        rows = {r["citation"]: r["status"] for r in json.loads(out.getvalue())}
        self.assertEqual(rows["app/auth.py:4-6"], "verified")
        self.assertEqual(rows["app/settings.py:1-3"], "refuted")
        saved.write_text(self.ask("Compared with `hmac.compare_digest` [app/auth.py:4-6].").to_json(), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cc.main(["verify", str(saved), "--repo", str(self.repo), "--db", str(self.tmp / "v.sqlite")]), 0)


if __name__ == "__main__":
    unittest.main()

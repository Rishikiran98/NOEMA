"""M2.2 tests: code-aware tokenizing, BM25 ranking, file-role weights, symbol definitions
and call sites, index schema migration, and the frozen retrieval question set.

No test runs the held-out "eval" split of data/retrieval: that needs the user's approval."""
import contextlib
import hashlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cpu_platform import retrieval as rt
from cpu_platform.repo_index import RepoIndex

REPO_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = REPO_ROOT / "data" / "retrieval" / "noema_questions.jsonl"
MANIFEST = REPO_ROOT / "data" / "retrieval" / "manifest.json"

FIXTURE = {
    "app/auth.py": '''"""Authentication helpers."""
import hmac
from app.store import load_user


def check_password(user, password):
    """Compare a password hash in constant time."""
    return hmac.compare_digest(user.password_hash, hash_password(password))


def hash_password(password):
    return "h:" + password


class SessionManager:
    def login(self, name, password):
        user = load_user(name)
        if check_password(user, password):
            return self.issue_token(user)

    def issue_token(self, user):
        def sign(value):
            return hash_password(value)
        return sign(user.name)
''',
    "app/store.py": '''def load_user(name):
    return {"name": name}


def save_user(user):
    pass
''',
    "docs/auth.md": "# Authentication\n\nPasswords are compared in constant time. Password password password.\n",
    "tests/test_auth.py": '''from app.auth import check_password


def test_check_password():
    assert check_password(None, "password") is False
''',
}


def build(root: Path, files=FIXTURE) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


class Workspace(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo = self.tmp / "repo"
        build(self.repo)
        self.index = RepoIndex(self.repo, self.tmp / "i.sqlite")
        self.addCleanup(self.index.close)
        self.index.update()
        self.retriever = rt.Retriever(self.index)


class Tokenizer(unittest.TestCase):
    def test_identifiers_split_and_stemmed(self):
        self.assertEqual(rt.tokenize("model_http_json"), ["model_http_json", "model", "http", "json"])
        self.assertEqual(rt.tokenize("EngineResult"), ["engineresult", "engine", "result"])
        self.assertEqual(rt.tokenize("HTTPError"), ["httperror", "http", "error"])
        self.assertEqual(rt.tokenize("How are the secrets skipped?"), ["secret", "skipped"])
        self.assertEqual(rt.tokenize("queries entries"), ["query", "entry"])
        self.assertEqual(rt.tokenize("class address"), ["address"])        # 'class' is a stopword
        self.assertEqual(rt.tokenize("HTTP 503"), ["http", "503"])
        self.assertEqual(rt.tokenize(""), [])

    def test_query_identifiers(self):
        self.assertEqual(rt.identifiers("where is check_password used?"), {"check_password"})
        self.assertEqual(rt.identifiers("what does SessionManager.issue_token return"), {"issue_token"})
        self.assertEqual(rt.identifiers("how does login work"), set())


class Ranking(Workspace):
    def test_implementation_ranks_first_among_code_and_hits_explain_themselves(self):
        hits = self.retriever.search("how is a password compared in constant time")
        code = [h for h in hits if h.path.startswith("app/")]
        self.assertEqual(code[0].symbol, "check_password")
        self.assertIn("constant", code[0].matched_terms)
        self.assertTrue(all(h.score > 0 for h in hits))
        self.assertEqual([h.score for h in hits], sorted((h.score for h in hits), reverse=True))

    def test_role_weights_scale_docs_and_tests_exactly(self):
        """A nudge, not a guarantee: a short keyword-stuffed doc can still outrank code."""
        original = dict(rt.ROLE_WEIGHTS)
        self.addCleanup(setattr, rt, "ROLE_WEIGHTS", original)
        query = "password compared in constant time"   # no identifiers: no definition bonus
        rt.ROLE_WEIGHTS = {"code": 1.0, "test": 1.0, "docs": 1.0}
        plain = {h.chunk_id: h.score for h in rt.Retriever(self.index).search(query, k=50, max_per_file=None)}
        rt.ROLE_WEIGHTS = original
        weighted = {h.chunk_id: h for h in rt.Retriever(self.index).search(query, k=50, max_per_file=None)}
        self.assertEqual(set(plain), set(weighted))
        roles_seen = set()
        for chunk_id, hit in weighted.items():
            role = rt.file_role(hit.path, "markdown" if hit.path.endswith(".md") else "python")
            roles_seen.add(role)
            self.assertAlmostEqual(hit.score, plain[chunk_id] * original[role], places=3)
        self.assertEqual(roles_seen, {"code", "docs", "test"})
        self.assertEqual(rt.file_role("tests/test_auth.py", "python"), "test")
        self.assertEqual(rt.file_role("pkg/x_test.py", "python"), "test")
        self.assertEqual(rt.file_role("docs/auth.md", "markdown"), "docs")
        self.assertEqual(rt.file_role("app/auth.py", "python"), "code")

    def test_definition_bonus_for_named_identifier(self):
        hits = self.retriever.search("what does hash_password do", k=3)
        self.assertEqual(hits[0].symbol, "hash_password")
        self.assertIn("defines:hash_password", hits[0].matched_terms)

    def test_per_file_cap_and_k(self):
        hits = self.retriever.search("password user token", k=10, max_per_file=1)
        paths = [h.path for h in hits]
        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(len(self.retriever.search("password", k=2)), 2)

    def test_no_match_returns_nothing(self):
        self.assertEqual(self.retriever.search("kubernetes helm chart"), [])
        self.assertEqual(self.retriever.search("the and of"), [])
        self.assertEqual(self.retriever.search(""), [])

    def test_hit_serialization(self):
        hit = self.retriever.search("hash password", k=1)[0]
        self.assertNotIn("text", hit.to_dict())
        self.assertIn("def hash_password", hit.to_dict(with_text=True)["text"])

    def test_empty_index(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with RepoIndex(empty, self.tmp / "e.sqlite") as index:
            index.update()
            self.assertEqual(rt.Retriever(index).search("anything"), [])


class Symbols(Workspace):
    def test_definitions(self):
        self.assertEqual([(s.name, s.path, s.line) for s in self.retriever.definitions("check_password")],
                         [("check_password", "app/auth.py", 6)])
        self.assertEqual([s.name for s in self.retriever.definitions("issue_token")], ["SessionManager.issue_token"])
        self.assertEqual([s.name for s in self.retriever.definitions("SESSIONMANAGER")], ["SessionManager"])
        self.assertEqual(self.retriever.definitions("missing"), [])

    def test_call_sites_with_callers(self):
        calls = [(r.path, r.line, r.caller, r.expr) for r in self.retriever.call_sites("check_password")]
        self.assertEqual(calls, [("app/auth.py", 18, "SessionManager.login", "check_password"),
                                 ("tests/test_auth.py", 5, "test_check_password", "check_password")])
        sites = {(r.caller, r.expr) for r in self.retriever.call_sites("hash_password")}
        self.assertEqual(sites, {("check_password", "hash_password"),
                                 ("SessionManager.issue_token.sign", "hash_password")})
        self.assertEqual([(r.caller, r.expr) for r in self.retriever.call_sites("issue_token")],
                         [("SessionManager.login", "self.issue_token")])
        self.assertEqual([r.expr for r in self.retriever.call_sites("app.auth.compare_digest")],
                         ["hmac.compare_digest"])

    def test_imports_are_refs_but_not_calls(self):
        imports = self.index.refs("load_user", "import")
        self.assertEqual([(r.path, r.expr) for r in imports], [("app/auth.py", "app.store.load_user")])
        self.assertEqual([r.caller for r in self.retriever.call_sites("load_user")], ["SessionManager.login"])
        self.assertEqual([r.expr for r in self.index.refs("hmac", "import")], ["hmac"])

    def test_refs_follow_file_changes(self):
        (self.repo / "app/auth.py").write_text("def check_password(u, p):\n    return False\n", encoding="utf-8")
        (self.repo / "tests/test_auth.py").unlink()
        self.index.update()
        self.assertEqual(self.retriever.call_sites("check_password"), [])
        self.assertEqual(self.index.refs("hash_password"), [])

    def test_unparseable_file_has_no_refs(self):
        (self.repo / "app/broken.py").write_text("def f(:\n    check_password()\n", encoding="utf-8")
        self.index.update()
        self.assertEqual([r.path for r in self.index.refs("check_password")],
                         ["app/auth.py", "tests/test_auth.py", "tests/test_auth.py"])


class SchemaMigration(unittest.TestCase):
    def test_v1_index_is_upgraded_and_fully_rebuilt(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build(tmp / "repo")
            db = tmp / "i.sqlite"
            with RepoIndex(tmp / "repo", db) as index:
                index.update()
            conn = sqlite3.connect(db)
            conn.execute("DROP TABLE refs")
            conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
            conn.commit()
            conn.close()
            with RepoIndex(tmp / "repo", db) as index:
                self.assertEqual(index.files(), [])           # forgotten, so everything is re-read
                stats = index.update()
                self.assertEqual(stats.added, len(FIXTURE))
                self.assertTrue(index.refs("check_password"))
                version = index.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
                self.assertEqual(version, "2")


class QuestionSet(unittest.TestCase):
    """The retrieval questions are frozen: written before any retrieval code existed."""

    def test_frozen_and_well_formed(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(rt.frozen_sha256(QUESTIONS), manifest["sha256"],
                         "noema_questions.jsonl changed: it is frozen (add a new versioned file instead)")
        rows = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({r["id"] for r in rows}), 40)
        self.assertEqual(sorted(r["id"] for r in rows if r["split"] == "dev"), sorted(manifest["dev"]))
        self.assertEqual(sorted(r["id"] for r in rows if r["split"] == "eval"), sorted(manifest["eval"]))
        self.assertFalse(set(manifest["dev"]) & set(manifest["eval"]))
        for row in rows:
            for gold in row["gold_files"]:
                self.assertTrue((REPO_ROOT / gold).is_file(), f"{row['id']}: gold file {gold} is missing")

    def test_windows_line_endings_do_not_unfreeze_the_file(self):
        """Regression (2026-11-07, Windows laptop): a CRLF checkout changed the raw SHA256 and
        failed the frozen check although the content was identical."""
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "noema_questions.jsonl"
            lf = QUESTIONS.read_bytes().replace(b"\r\n", b"\n")
            copy.write_bytes(lf.replace(b"\n", b"\r\n"))
            (Path(tmp) / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
            self.assertEqual(len(rt.load_questions(copy, "dev")), 20)
            # A real change is still refused, CRLF or not.
            copy.write_bytes(lf.replace(b"DNS", b"dns").replace(b"\n", b"\r\n"))
            with self.assertRaises(SystemExit):
                rt.load_questions(copy, "dev")
        self.assertEqual(hashlib.sha256(lf).hexdigest(), json.loads(MANIFEST.read_text(encoding="utf-8"))["sha256"])

    def test_tampered_question_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "noema_questions.jsonl"
            copy.write_text(QUESTIONS.read_text(encoding="utf-8").replace("DNS", "dns"), encoding="utf-8")
            (Path(tmp) / "manifest.json").write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(SystemExit):
                rt.load_questions(copy, "dev")

    def test_dev_split_quality_floor(self):
        """Regression floor on DEV only (measured 0.95 when frozen; floor leaves room for code drift)."""
        with tempfile.TemporaryDirectory() as tmp, RepoIndex(REPO_ROOT, Path(tmp) / "n.sqlite") as index:
            index.update()
            result = rt.evaluate(rt.Retriever(index), rt.load_questions(QUESTIONS, "dev"))
        self.assertEqual(result["questions"], 20)
        self.assertGreaterEqual(result["recall@5"], 0.85, result)

    def test_question_file_is_never_indexed(self):
        """Otherwise every question would match its own text (.noemaignore excludes it)."""
        with tempfile.TemporaryDirectory() as tmp, RepoIndex(REPO_ROOT, Path(tmp) / "n.sqlite") as index:
            index.update()
            self.assertFalse([f["path"] for f in index.files() if f["path"].startswith("data/retrieval/")])


class Metrics(unittest.TestCase):
    def test_recall_and_mrr(self):
        class Fixed:
            def __init__(self, ranking):
                self.ranking = ranking

            def search(self, question, k=10):
                return [rt.Hit(f"{p}:1-1", p, 1, 1, "window", None, 1.0, []) for p in self.ranking[question]][:k]

        questions = [{"id": "a", "question": "a", "gold_files": ["x.py"]},
                     {"id": "b", "question": "b", "gold_files": ["y.py", "z.py"]},
                     {"id": "c", "question": "c", "gold_files": ["w.py"]}]
        ranking = {"a": ["x.py"], "b": ["1", "2", "z.py"], "c": [str(i) for i in range(10)] + ["w.py"]}
        result = rt.evaluate(Fixed(ranking), questions)
        self.assertEqual((result["recall@1"], result["recall@5"], result["recall@10"]), (0.333, 0.667, 0.667))
        self.assertEqual(result["mrr"], round((1 + 1 / 3) / 3, 3))


class Cli(Workspace):
    def run_cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = rt.main(list(argv))
        return code, json.loads(out.getvalue()) if out.getvalue() else None

    def test_search_defs_calls(self):
        db = str(self.tmp / "cli.sqlite")
        code, hits = self.run_cli("search", str(self.repo), "constant time password", "--db", db, "-k", "2")
        self.assertEqual((code, hits[0]["symbol"]), (0, "check_password"))
        self.assertEqual(self.run_cli("defs", str(self.repo), "login", "--db", db)[1][0]["name"], "SessionManager.login")
        self.assertEqual(len(self.run_cli("calls", str(self.repo), "check_password", "--db", db)[1]), 2)

    def test_held_out_split_needs_confirmation_and_is_logged(self):
        qdir = self.tmp / "q"
        qdir.mkdir()
        rows = [{"id": "e1", "split": "eval", "question": "constant time password", "gold_files": ["app/auth.py"]}]
        (qdir / "qs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_cli("eval", str(self.repo), "--questions", str(qdir / "qs.jsonl"), "--split", "eval")
        self.assertFalse((qdir / "eval_runs.jsonl").exists())
        code, out = self.run_cli("eval", str(self.repo), "--questions", str(qdir / "qs.jsonl"),
                                 "--split", "eval", "--confirm-eval")
        self.assertEqual((code, out["recall@5"]), (0, 1.0))
        log = [json.loads(l) for l in (qdir / "eval_runs.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual((len(log), log[0]["split"], log[0]["recall@5"]), (1, "eval", 1.0))
        self.assertEqual(log[0]["settings"]["role_weights"], rt.ROLE_WEIGHTS)


if __name__ == "__main__":
    unittest.main()

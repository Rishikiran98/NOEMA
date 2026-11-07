"""M2.1 tests: gitignore matching (cross-checked against git), secret and binary skipping,
symlinks, Python and window chunking, incremental re-indexing, and index storage rules."""
import contextlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from cpu_platform import repo_index as ri
from cpu_platform.ignore import IgnoreMatcher, parse_line

GIT = shutil.which("git")
FAKE_AWS = "AKIA" + "IOSFODNN7EXAMPLE"   # AWS's documented example key, split so this file stays indexable
REPO_ROOT = Path(__file__).resolve().parents[1]

ROOT_IGNORE = """# comment
*.log
!keep.log
build/
/top_only.txt
docs/*.tmp
**/cache
a/**/z.txt
logs/**
!logs/important/
foo?.py
data[0-9].csv
[!x]y.md
\\#hash.txt
trailing.txt
dir_only/
nested/deep/
*.py[co]
"""
SUB_IGNORE = "!*.log\nlocal.txt\n/anch.txt\n"
FIXTURE_PATHS = [
    "app.log", "keep.log", "src/app.log", "src/keep.log", "build/out.o", "src/build/x", "top_only.txt",
    "src/top_only.txt", "docs/a.tmp", "docs/sub/a.tmp", "cache/x", "src/cache/y", "a/z.txt", "a/b/z.txt",
    "a/b/c/z.txt", "logs/x.txt", "logs/important/y.txt", "foo1.py", "foo12.py", "data3.csv", "datax.csv",
    "ay.md", "xy.md", "#hash.txt", "trailing.txt", "src/dir_only", "dir_only/f", "nested/deep/f", "deep/f",
    "m.pyc", "m.pyo", "m.py", "sub/q.log", "sub/local.txt", "sub/x/local.txt", "sub/anch.txt",
    "sub/x/anch.txt", "anch.txt", "sub/build/f",
]


def write(root: Path, rel: str, content="x", mode="w"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "wb":
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


class Workspace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        self.db = Path(self.tmp.name) / "index.sqlite"

    def index(self, **kw):
        idx = ri.RepoIndex(self.repo, self.db, **kw)
        self.addCleanup(idx.close)
        return idx

    def indexed_paths(self, idx):
        return {f["path"] for f in idx.files() if f["status"] == "indexed"}

    def skip_reasons(self, idx):
        return {f["path"]: f["skip_reason"] for f in idx.files() if f["status"] == "skipped"}


class IgnoreRules(unittest.TestCase):
    def check(self, pattern, path, is_dir=False, base=""):
        return parse_line(pattern, base).matches(path, is_dir)

    def test_translation_cases(self):
        self.assertTrue(self.check("*.log", "a/b/c.log"))
        self.assertFalse(self.check("/*.log", "a/c.log"))
        self.assertTrue(self.check("/*.log", "c.log"))
        self.assertTrue(self.check("a/**/z", "a/z"))
        self.assertTrue(self.check("a/**/z", "a/x/y/z"))
        self.assertTrue(self.check("**/cache", "x/y/cache", True))
        self.assertTrue(self.check("logs/**", "logs/a/b"))
        self.assertFalse(self.check("logs/**", "logs"))
        self.assertFalse(self.check("build/", "build"))           # file, dir-only pattern
        self.assertTrue(self.check("build/", "x/build", True))
        self.assertFalse(self.check("foo?.py", "foo12.py"))
        self.assertTrue(self.check("[!x]y.md", "ay.md"))
        self.assertFalse(self.check("[!x]y.md", "xy.md"))
        self.assertFalse(self.check("*.txt", "a/b.txt/c"))       # '*' never crosses '/'
        self.assertTrue(self.check("\\#hash", "#hash"))
        self.assertTrue(self.check("[unterminated", "[unterminated"))
        self.assertTrue(self.check("local.txt", "sub/local.txt", base="sub"))
        self.assertFalse(self.check("local.txt", "other/local.txt", base="sub"))
        for blank in ("", "   ", "# comment", "/", "!"):
            self.assertIsNone(parse_line(blank), blank)

    def test_last_match_wins_and_negation(self):
        m = IgnoreMatcher()
        m.rules += [parse_line("*.log"), parse_line("!keep.log")]
        self.assertTrue(m.ignored("x.log", False))
        self.assertFalse(m.ignored("keep.log", False))
        self.assertEqual(m.explain("keep.log", False).pattern, "!keep.log")
        self.assertIsNone(m.explain("x.py", False))

    @unittest.skipIf(GIT is None, "git not installed")
    def test_walker_agrees_with_git_on_tricky_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run([GIT, "init", "-q", str(root)], check=True)
            write(root, ".gitignore", ROOT_IGNORE)
            write(root, "sub/.gitignore", SUB_IGNORE)
            for rel in FIXTURE_PATHS:
                write(root, rel)
            ours = {item[0] for item in ri.iter_files(root) if item[0] != "<skip>"}
            listed = subprocess.run([GIT, "-C", str(root), "ls-files", "-co", "--exclude-standard"],
                                    capture_output=True, text=True, check=True).stdout.split()
            self.assertEqual(ours, set(listed))

    @unittest.skipIf(GIT is None or not (REPO_ROOT / ".git").exists(), "not a git checkout")
    def test_noema_itself_matches_git(self):
        ours = {item[0] for item in ri.iter_files(REPO_ROOT) if item[0] != "<skip>"}
        # NOEMA's own .noemaignore (root level only) is passed to git too, so git still
        # supplies the expected file set (M2.2 added it to keep eval questions unindexed).
        noemaignore = REPO_ROOT / ".noemaignore"
        extra = [f"--exclude-from={noemaignore}"] if noemaignore.is_file() else []
        listed = subprocess.run([GIT, "-C", str(REPO_ROOT), "ls-files", "-co", "--exclude-standard", *extra],
                                capture_output=True, text=True, check=True).stdout.split()
        if extra:
            # Exclude patterns do not hide *tracked* files from ls-files; ask git which tracked
            # files .noemaignore matches and drop them too.
            tracked_ignored = subprocess.run([GIT, "-C", str(REPO_ROOT), "ls-files", "-c", "-i", *extra],
                                             capture_output=True, text=True, check=True).stdout.split()
            listed = [p for p in listed if p not in set(tracked_ignored)]
        self.assertEqual(ours, set(listed))


class Walking(Workspace):
    def test_default_ignores_info_exclude_and_noemaignore(self):
        for rel in ("src/a.py", "node_modules/x/index.js", ".venv/lib/x.py", "pkg/__pycache__/a.pyc",
                    ".git/config", "local_only.txt", "private/notes.md", "public/readme.md"):
            write(self.repo, rel)
        write(self.repo, ".git/info/exclude", "local_only.txt\n")
        write(self.repo, ".noemaignore", "private/\n")
        idx = self.index()
        idx.update()
        self.assertEqual(self.indexed_paths(idx), {"src/a.py", "public/readme.md", ".noemaignore"})

    def test_repository_can_reinclude_a_default(self):
        write(self.repo, "venv/keep.py")
        write(self.repo, ".gitignore", "!venv/\n")
        idx = self.index()
        idx.update()
        self.assertIn("venv/keep.py", self.indexed_paths(idx))

    def test_symlinks_are_never_followed(self):
        outside = Path(self.tmp.name) / "outside"
        write(outside, "secret_design.txt", "OUTSIDE-THE-REPO-CONTENT")
        write(self.repo, "real.txt", "inside")
        try:
            os.symlink(outside / "secret_design.txt", self.repo / "link.txt")
            os.symlink(outside, self.repo / "linkdir", target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not permitted here")
        idx = self.index()
        stats = idx.update()
        self.assertEqual(self.indexed_paths(idx), {"real.txt"})
        self.assertEqual(stats.skipped.get("symlink"), 2)
        self.assertNotIn(b"OUTSIDE-THE-REPO-CONTENT", self.db.read_bytes())


class Skipping(Workspace):
    def test_secret_names(self):
        secret = [".env", ".env.local", "prod.env", "config/id_rsa", "id_ed25519.pub", "certs/server.pem",
                  "tls.key", "store.jks", "credentials", "aws/credentials.json", ".npmrc", ".pypirc",
                  ".netrc", "secrets.yaml", "infra/terraform.tfvars", "state.tfstate", "gcp-service-account.json"]
        allowed = [".env.example", ".env.sample", "credentials.py", "keys.py", "environment.py",
                   "secret_santa.py", "docs/keys.md", "config.template"]
        for rel in secret + allowed:
            write(self.repo, rel, "harmless placeholder text")
        idx = self.index()
        idx.update()
        reasons = self.skip_reasons(idx)
        for rel in secret:
            self.assertEqual(reasons.get(rel), "secret_name", rel)
        self.assertTrue(set(allowed) <= self.indexed_paths(idx))

    def test_secret_content_is_skipped_and_never_stored(self):
        # Fake tokens are assembled at run time so this test file is not itself a "secret".
        pk = "PRIVATE" + " KEY-----"
        tokens = {
            "key.txt": "-----BEGIN RSA " + pk + "\nMIIEow" + "A" * 40 + "\n-----END RSA " + pk,
            "openssh.txt": "-----BEGIN OPENSSH " + pk + "\nb3BlbnNzaC1rZXktdjEAAAAA\n",
            "deploy.py": 'AWS_KEY = "' + FAKE_AWS + '"\n',
            "ci.yml": "token: ghp_" + "a1B2c3D4e5" * 4 + "\n",
            "slack.py": 'hook = "xox' + 'b-1234567890-abcdefghij"\n',
            "llm.py": 'API_KEY = "sk-proj-' + "Z9y8X7w6V5" * 4 + '"\n',
            "pay.js": 'const k = "sk_live_' + "4eC39HqLyjWDarjtT1zdp7dc" + '";\n',
        }
        for rel, text in tokens.items():
            write(self.repo, rel, "# looks ordinary\n" + text)
        write(self.repo, "fine.py", 'AKIA = "short"  # not a key\nsk = "sk-short"\n')
        idx = self.index()
        idx.update()
        reasons = self.skip_reasons(idx)
        for rel in tokens:
            self.assertTrue(reasons.get(rel, "").startswith("secret_content:"), (rel, reasons.get(rel)))
        self.assertIn("fine.py", self.indexed_paths(idx))
        blob = self.db.read_bytes()
        for marker in (FAKE_AWS.encode(), pk.encode(), b"ghp_a1B2", b"xoxb-", b"sk-proj-", b"sk_live_"):
            self.assertNotIn(marker, blob, marker)

    def test_binary_non_utf8_large_and_bom(self):
        write(self.repo, "img.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "wb")
        write(self.repo, "latin1.txt", "caf\xe9 cr\xe8me".encode("latin-1"), "wb")
        write(self.repo, "big.txt", "y" * 2048)
        write(self.repo, "bom.py", b"\xef\xbb\xbfdef f():\n    return 1\n", "wb")
        write(self.repo, "empty.txt", "")
        idx = self.index(max_file_bytes=1024)
        idx.update()
        self.assertEqual(self.skip_reasons(idx), {"img.png": "binary", "latin1.txt": "not_utf8", "big.txt": "too_large"})
        self.assertEqual([c.symbol for c in idx.chunks("bom.py")], ["f"])
        self.assertEqual(idx.chunks("empty.txt"), [])
        self.assertIn("empty.txt", self.indexed_paths(idx))


PY_SOURCE = '''"""Module doc."""
import os

LIMIT = 3


@decorator
def top(a, b):
    def inner():
        return a
    return inner() + b


async def fetch():
    return await thing()


class Store:
    """Keeps things."""
    capacity = 10

    def __init__(self):
        self.items = []

    @property
    def size(self):
        return len(self.items)

    async def load(self):
        pass


class Empty:
    pass


if __name__ == "__main__":
    top(1, 2)
'''


class Chunking(Workspace):
    def test_python_chunks_and_symbols(self):
        write(self.repo, "m.py", PY_SOURCE)
        idx = self.index()
        idx.update()
        got = [(c.kind, c.symbol, c.start_line, c.end_line) for c in idx.chunks("m.py")]
        self.assertEqual(got, [
            ("module", None, 1, 6),
            ("function", "top", 7, 11),            # decorator included, inner() stays inside
            ("function", "fetch", 14, 15),
            ("class", "Store", 18, 21),            # header: docstring and attributes
            ("method", "Store.__init__", 22, 23),
            ("method", "Store.size", 25, 27),      # decorator included
            ("method", "Store.load", 29, 30),
            ("class", "Empty", 33, 34),
            ("module", None, 35, 38),
        ])
        lines = PY_SOURCE.splitlines()
        for chunk in idx.chunks("m.py"):
            self.assertEqual(chunk.text, "\n".join(lines[chunk.start_line - 1:chunk.end_line]))
        self.assertEqual([(s.name, s.kind) for s in idx.symbols()],
                         [("top", "function"), ("fetch", "async_function"), ("Store", "class"),
                          ("Store.__init__", "method"), ("Store.size", "method"), ("Store.load", "method"),
                          ("Empty", "class")])
        self.assertEqual([s.name for s in idx.symbols("size")], ["Store.size"])
        self.assertNotIn("inner", [s.name for s in idx.symbols()])

    def test_unparseable_python_falls_back_to_windows(self):
        write(self.repo, "broken.py", "def f(:\n    pass\n")
        idx = self.index()
        idx.update()
        (record,) = [f for f in idx.files() if f["path"] == "broken.py"]
        self.assertEqual(record["note"], "python_parse_error")
        self.assertEqual([c.kind for c in idx.chunks("broken.py")], ["window"])

    def test_windows_and_long_functions(self):
        write(self.repo, "notes.md", "\n".join(f"line {i}" for i in range(1, 131)))
        write(self.repo, "long.py", "def big():\n" + "\n".join(f"    x{i} = {i}" for i in range(250)) + "\n")
        idx = self.index()
        idx.update()
        self.assertEqual([(c.start_line, c.end_line) for c in idx.chunks("notes.md")], [(1, 60), (51, 110), (101, 130)])
        parts = idx.chunks("long.py")
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(c.symbol == "big" and c.kind == "function" for c in parts))
        self.assertEqual((parts[0].start_line, parts[-1].end_line), (1, 251))


class Incremental(Workspace):
    def test_unchanged_files_are_not_read(self):
        write(self.repo, "a.py", "def a():\n    return 1\n")
        write(self.repo, "b.md", "hello")
        idx = self.index()
        first = idx.update()
        self.assertEqual((first.added, first.unchanged), (2, 0))
        original = Path.read_bytes
        reads = []
        Path.read_bytes = lambda self: reads.append(self) or original(self)
        try:
            second = idx.update()
        finally:
            Path.read_bytes = original
        self.assertEqual((second.added, second.updated, second.unchanged, second.removed), (0, 0, 2, 0))
        self.assertEqual(reads, [])

    def test_modify_touch_delete_rename_and_newly_ignored(self):
        a = write(self.repo, "a.py", "def a():\n    return 1\n")
        write(self.repo, "b.py", "def b():\n    return 2\n")
        write(self.repo, "c.py", "def c():\n    return 3\n")
        write(self.repo, "d.txt", "plain")
        idx = self.index()
        idx.update()
        a.write_text("def a():\n    return 100\n\ndef a2():\n    return 4\n", encoding="utf-8")
        stamp = time.time() + 5
        os.utime(self.repo / "b.py", (stamp, stamp))                   # touched, same content
        (self.repo / "c.py").unlink()
        (self.repo / "d.txt").rename(self.repo / "e.txt")
        write(self.repo, ".gitignore", "e.txt\n")
        stats = idx.update()
        self.assertEqual(stats.updated, 1)
        self.assertEqual(stats.added, 1)                                 # .gitignore itself
        self.assertEqual(stats.removed, 2)                               # c.py, d.txt (e.txt ignored)
        self.assertEqual(self.indexed_paths(idx), {"a.py", "b.py", ".gitignore"})
        self.assertEqual([s.name for s in idx.symbols() if s.path == "a.py"], ["a", "a2"])
        texts = " ".join(c.text for c in idx.chunks())
        self.assertIn("return 100", texts)
        self.assertNotIn("return 1\n", texts + "\n")                     # old chunk replaced
        self.assertNotIn("return 3", texts)                              # deleted file's chunks gone
        with contextlib.closing(sqlite3.connect(self.db)) as conn:   # closed: Windows locks open files
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM symbols WHERE path='c.py'").fetchone()[0], 0)

    def test_file_that_gains_a_secret_leaves_the_searchable_index(self):
        f = write(self.repo, "settings.py", "DEBUG = True\n")
        idx = self.index()
        idx.update()
        f.write_text('DEBUG = True\nKEY = "' + FAKE_AWS + '"\n', encoding="utf-8")
        stats = idx.update()
        self.assertEqual(stats.removed, 1)
        self.assertEqual(idx.chunks("settings.py"), [])
        self.assertEqual(self.skip_reasons(idx)["settings.py"], "secret_content:aws_access_key_id")
        self.assertNotIn(FAKE_AWS.encode(), self.db.read_bytes())

    def test_skipped_file_that_becomes_clean_is_added(self):
        f = write(self.repo, "x.bin", b"\x00\x01", "wb")
        idx = self.index()
        idx.update()
        f.write_text("now text\n", encoding="utf-8")
        stats = idx.update()
        self.assertEqual(stats.added, 1)
        self.assertIn("x.bin", self.indexed_paths(idx))


class Storage(Workspace):
    def test_index_must_live_outside_the_repository(self):
        with self.assertRaises(ValueError):
            ri.RepoIndex(self.repo, self.repo / ".noema" / "i.sqlite")

    def test_one_index_per_repository(self):
        self.index().update()
        other = Path(self.tmp.name) / "other"
        other.mkdir()
        opened = []
        real_connect = ri.connect
        with mock.patch.object(ri, "connect", lambda db: opened.append(real_connect(db)) or opened[-1]):
            with self.assertRaises(ValueError):
                ri.RepoIndex(other, self.db)
        # The refused index must not keep its database open (Windows cannot delete it then).
        with self.assertRaises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")

    def test_schema_version_is_checked(self):
        idx = self.index()
        idx.conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
        idx.conn.commit()
        idx.close()
        with self.assertRaises(RuntimeError):
            ri.RepoIndex(self.repo, self.db)

    @unittest.skipIf(os.name == "nt", "POSIX permissions")
    def test_index_file_is_owner_only(self):
        self.index()
        self.assertEqual(self.db.stat().st_mode & 0o777, 0o600)

    def test_default_location_is_outside_repo_and_per_repo(self):
        os.environ["NOEMA_HOME"] = str(Path(self.tmp.name) / "home")
        self.addCleanup(os.environ.pop, "NOEMA_HOME", None)
        path = ri.default_db_path(self.repo)
        self.assertTrue(str(path).startswith(os.environ["NOEMA_HOME"]))
        self.assertNotEqual(path, ri.default_db_path(Path(self.tmp.name)))

    def test_cli_index_and_stats(self):
        write(self.repo, "a.py", "def a():\n    pass\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(ri.main(["index", str(self.repo), "--db", str(self.db)]), 0)
        self.assertEqual(json.loads(out.getvalue())["added"], 1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(ri.main(["stats", str(self.repo), "--db", str(self.db)]), 0)
        summary = json.loads(out.getvalue())
        self.assertEqual((summary["indexed_files"], summary["symbols"]), (1, 1))


class IndexesNoemaItself(unittest.TestCase):
    def test_full_and_incremental_index_of_this_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            with ri.RepoIndex(REPO_ROOT, Path(tmp) / "noema.sqlite") as idx:
                first = idx.update()
                second = idx.update()
        # Roadmap target is < 10 s on the user's laptop (NOT RUN there); this is the CI bound.
        self.assertLess(first.seconds, 10)
        self.assertGreater(first.chunks, 500)
        self.assertEqual(first.skipped, {})
        self.assertEqual((second.added, second.updated, second.removed), (0, 0, 0))
        self.assertEqual(second.unchanged, first.files_seen)


if __name__ == "__main__":
    unittest.main()

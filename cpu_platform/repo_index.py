"""Local repository indexer (roadmap M2.1). Standard library only.

    python -m cpu_platform.repo_index index PATH [--db FILE]   # build or update the index
    python -m cpu_platform.repo_index stats PATH [--db FILE]   # what is indexed and skipped

What it does:

* walks the working tree without following symlinks, honouring ``.gitignore`` files at
  every level, ``.git/info/exclude``, a ``.noemaignore`` (same syntax) and a short list of
  default excludes (VCS metadata, caches, virtualenvs, ``node_modules``);
* skips binary files, files over ``max_file_bytes``, non-UTF-8 text, and **likely
  secrets**, by name (``.env``, private keys, keystores, credential files) and by content
  (private-key blocks, cloud and VCS token formats). Only the *reason* is recorded, never
  the matched text;
* splits Python into chunks by function, class and method using ``ast`` (decorators
  included, nested functions stay with their parent), with module-level code in line
  windows; everything else is split into overlapping line windows;
* stores files, chunks (exact text, 1-based line spans, SHA256) and symbol definitions in
  a local SQLite file, by default ``~/.noema/index/<repo-hash>.sqlite`` (``NOEMA_HOME``
  overrides ``~/.noema``), never inside the repository;
* re-indexes incrementally: a file whose size and mtime are unchanged is not read; one
  whose content hash is unchanged is not re-chunked; deleted or newly ignored files are
  removed from the index.

The index holds customer source code in plain text on the local disk. It is created with
owner-only permissions where the OS supports them. Nothing leaves the machine.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import time

from .ignore import IgnoreMatcher, parse_lines

SCHEMA_VERSION = 2          # 2: adds the refs table (call sites and imports, M2.2)
DEFAULT_MAX_FILE_BYTES = 1024 * 1024
WINDOW_LINES = 60
WINDOW_OVERLAP = 10
MAX_PY_CHUNK_LINES = 200
BINARY_SNIFF_BYTES = 8192

# Applied before the repository's own ignore files, so a repository can re-include them.
DEFAULT_IGNORES = (
    ".git/", ".hg/", ".svn/", "__pycache__/", "node_modules/", ".venv/", "venv/", ".tox/",
    ".mypy_cache/", ".pytest_cache/", ".ruff_cache/", ".noema/", "*.egg-info/",
)

# Names that very likely hold secrets. Matched case-insensitively on the file name.
SECRET_NAME_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"\.env", r"\.env\..+", r".+\.env",
    r".+\.(pem|key|p12|pfx|jks|keystore|kdbx|ppk|asc|gpg|ovpn)",
    r"id_(rsa|dsa|ecdsa|ed25519)(\..+)?",
    r"\.(netrc|_netrc|npmrc|pypirc|pgpass|htpasswd|git-credentials|dockercfg)",
    r"credentials(\.(json|ya?ml|ini|toml|xml|csv|txt|conf))?", r"secrets?\.(json|ya?ml|toml|ini|txt|env)",
    r".*service[-_]?account.*\.json", r"terraform\.tfvars(\.json)?", r".+\.tfstate(\.backup)?",
))
# Template files that conventionally hold placeholders, not secrets.
SECRET_NAME_ALLOW = re.compile(r".*\.(example|sample|template|dist)$", re.I)

# Content signatures of real secrets. Matching text is never stored or reported.
SECRET_CONTENT_PATTERNS = (
    ("private_key_block", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----")),
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_secret_key", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{20,}\b")),
    ("openai_style_key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{32,}\b")),
)

LANGUAGES = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript", ".java": "java", ".kt": "kotlin",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".rb": "ruby", ".php": "php", ".swift": "swift", ".scala": "scala",
    ".sh": "shell", ".bash": "shell", ".ps1": "powershell", ".sql": "sql", ".md": "markdown",
    ".rst": "rst", ".txt": "text", ".json": "json", ".yml": "yaml", ".yaml": "yaml", ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".html": "html", ".css": "css", ".xml": "xml",
}


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    kind: str          # function | method | class | module | window
    symbol: str | None
    language: str
    text: str

    @property
    def id(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass
class Symbol:
    name: str          # qualified: Class.method
    kind: str          # function | async_function | class | method
    path: str
    line: int
    end_line: int


@dataclass
class Ref:
    """A use of a name: a call ``f(...)`` / ``obj.f(...)`` or an import ``from m import f``."""
    name: str          # the called or imported name (last component)
    kind: str          # call | import
    path: str
    line: int
    caller: str | None # enclosing function/method qualname; None at module level
    expr: str          # dotted text of the callee, e.g. "self.conn.execute"


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else f"?.{node.attr}"
    return None


def python_refs(path: str, text: str) -> list[Ref]:
    """Call sites and imports, each tagged with the enclosing definition. Empty if the
    file does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return []
    refs: list[Ref] = []

    def visit(node: ast.AST, scope: list[str]):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, scope + [child.name])
                continue
            caller = ".".join(scope) or None
            if isinstance(child, ast.Call):
                dotted = _dotted(child.func)
                if dotted:
                    refs.append(Ref(dotted.rsplit(".", 1)[-1], "call", path, child.lineno, caller, dotted[:200]))
            elif isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    refs.append(Ref(alias.name, "import", path, child.lineno, caller,
                                    f"{'.' * child.level}{child.module or ''}.{alias.name}"[:200]))
            elif isinstance(child, ast.Import):
                for alias in child.names:
                    refs.append(Ref(alias.name.rsplit(".", 1)[-1], "import", path, child.lineno, caller, alias.name[:200]))
            visit(child, scope)

    visit(tree, [])
    return refs


@dataclass
class IndexStats:
    root: str
    db: str
    files_seen: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    chunks: int = 0
    symbols: int = 0
    seconds: float = 0.0

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def to_dict(self) -> dict:
        return {**self.__dict__, "skipped": dict(sorted(self.skipped.items()))}


# --------------------------------------------------------------------------- classification

def language_of(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name in ("Makefile", "Dockerfile"):
        return name.lower()
    return LANGUAGES.get(os.path.splitext(name)[1].lower(), "text")


def secret_name_reason(name: str) -> str | None:
    if SECRET_NAME_ALLOW.fullmatch(name):
        return None
    for pattern in SECRET_NAME_PATTERNS:
        if pattern.fullmatch(name):
            return "secret_name"
    return None


def secret_content_reason(text: str) -> str | None:
    for label, pattern in SECRET_CONTENT_PATTERNS:
        if pattern.search(text):
            return f"secret_content:{label}"
    return None


def decode_text(data: bytes) -> tuple[str | None, str | None]:
    """(text, skip_reason). Binary = NUL byte in the first 8 KiB; else must be UTF-8."""
    if b"\x00" in data[:BINARY_SNIFF_BYTES]:
        return None, "binary"
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, "not_utf8"
    return text, None


# --------------------------------------------------------------------------- chunking

def _windows(path: str, lines: list[str], start: int, end: int, language: str, kind: str = "window",
             symbol: str | None = None) -> list[Chunk]:
    """Line windows over lines[start-1:end] (1-based, inclusive), skipping blank-only windows."""
    out = []
    step = WINDOW_LINES - WINDOW_OVERLAP
    first = start
    while first <= end:
        last = min(end, first + WINDOW_LINES - 1)
        text = "\n".join(lines[first - 1:last])
        if text.strip():
            out.append(Chunk(path, first, last, kind, symbol, language, text))
        if last >= end:
            break
        first += step
    return out


def chunk_text(path: str, text: str, language: str) -> list[Chunk]:
    lines = text.splitlines()
    return _windows(path, lines, 1, len(lines), language) if lines else []


def _span(node: ast.AST) -> tuple[int, int]:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return start, node.end_lineno


def chunk_python(path: str, text: str) -> tuple[list[Chunk], list[Symbol], str | None]:
    """(chunks, symbols, note). Falls back to line windows if the file does not parse."""
    lines = text.splitlines()
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return chunk_text(path, text, "python"), [], "python_parse_error"
    chunks: list[Chunk] = []
    symbols: list[Symbol] = []
    covered: list[tuple[int, int]] = []

    def emit(node, kind: str, name: str, start: int, end: int):
        if end - start + 1 > MAX_PY_CHUNK_LINES:
            chunks.extend(_windows(path, lines, start, end, "python", kind, name))
        else:
            chunks.append(Chunk(path, start, end, kind, name, "python", "\n".join(lines[start - 1:end])))
        covered.append((start, end))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _span(node)
            symbols.append(Symbol(node.name, "async_function" if isinstance(node, ast.AsyncFunctionDef)
                                  else "function", path, start, end))
            emit(node, "function", node.name, start, end)
        elif isinstance(node, ast.ClassDef):
            start, end = _span(node)
            symbols.append(Symbol(node.name, "class", path, start, end))
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if not methods:
                emit(node, "class", node.name, start, end)
                continue
            # Class header: everything in the class that is not a method (docstring, attributes).
            cursor = start
            for method in methods:
                m_start, m_end = _span(method)
                if m_start > cursor and "\n".join(lines[cursor - 1:m_start - 1]).strip():
                    emit(node, "class", node.name, cursor, m_start - 1)
                qual = f"{node.name}.{method.name}"
                symbols.append(Symbol(qual, "method", path, m_start, m_end))
                emit(method, "method", qual, m_start, m_end)
                cursor = m_end + 1
            if cursor <= end and "\n".join(lines[cursor - 1:end]).strip():
                emit(node, "class", node.name, cursor, end)
    # Module-level code outside any definition (imports, constants, __main__ blocks).
    covered.sort()
    gaps, line = [], 1
    for start, end in covered:
        if start > line:
            gaps.append((line, start - 1))
        line = max(line, end + 1)
    if line <= len(lines):
        gaps.append((line, len(lines)))
    for start, end in gaps:
        chunks.extend(_windows(path, lines, start, end, "python", "module"))
    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks, symbols, None


# --------------------------------------------------------------------------- walking

def default_db_path(root: Path) -> Path:
    home = Path(os.environ.get("NOEMA_HOME") or Path.home() / ".noema")
    digest = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    return home / "index" / f"{root.resolve().name}-{digest}.sqlite"


def iter_files(root: Path, *, extra_ignore_file: str | None = ".noemaignore", extra_ignores: tuple[str, ...] = ()):
    """Yield (rel_posix_path, os.stat_result) for files to consider, plus
    ("<skip>", rel, reason) tuples for things deliberately not followed. ``extra_ignore_file=None``
    skips ``.noemaignore`` (which only governs NOEMA's index); ``extra_ignores`` adds patterns."""
    base = IgnoreMatcher(parse_lines(DEFAULT_IGNORES + tuple(extra_ignores), "", "<default>"))
    base.add_file(root / ".git" / "info" / "exclude", "")
    stack = [("", root, base)]
    while stack:
        rel_dir, abs_dir, parent = stack.pop()
        matcher = parent.child()
        matcher.add_file(abs_dir / ".gitignore", rel_dir)
        if extra_ignore_file:
            matcher.add_file(abs_dir / extra_ignore_file, rel_dir)
        try:
            entries = sorted(os.scandir(abs_dir), key=lambda e: e.name)
        except OSError:
            yield ("<skip>", rel_dir or ".", "unreadable_dir")
            continue
        subdirs = []
        for entry in entries:
            rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                yield ("<skip>", rel, "unreadable")
                continue
            # Windows junctions are reparse points that stat() does not report as links.
            junction = getattr(entry, "is_junction", lambda: False)()
            if stat.S_ISLNK(info.st_mode) or junction:
                # Never follow: a link can point outside the repository.
                if not matcher.ignored(rel, False):
                    yield ("<skip>", rel, "symlink")
                continue
            is_dir = stat.S_ISDIR(info.st_mode)
            if matcher.ignored(rel, is_dir):
                continue
            if is_dir:
                subdirs.append((rel, Path(entry.path), matcher))
            elif stat.S_ISREG(info.st_mode):
                yield (rel, info)
            else:
                yield ("<skip>", rel, "not_regular_file")
        stack.extend(reversed(subdirs))


# --------------------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
    sha256 TEXT, language TEXT, status TEXT NOT NULL CHECK (status IN ('indexed', 'skipped')),
    skip_reason TEXT, note TEXT, chunks INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, kind TEXT NOT NULL, symbol TEXT,
    language TEXT NOT NULL, sha256 TEXT NOT NULL, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE TABLE IF NOT EXISTS symbols (
    name TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    line INTEGER NOT NULL, end_line INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS symbols_path ON symbols(path);
CREATE TABLE IF NOT EXISTS refs (
    name TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    line INTEGER NOT NULL, caller TEXT, expr TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS refs_name ON refs(name);
CREATE INDEX IF NOT EXISTS refs_path ON refs(path);
"""


def connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    new = not db.exists()
    if new:
        # Create owner-only before writing any source code into it (no-op on Windows ACLs).
        fd = os.open(db, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    version = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if version is None:
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    elif int(version[0]) == 1:
        # v1 -> v2 added refs. Forget every file so the next update re-reads and re-chunks
        # all of them (an index is a cache of the repository; nothing is lost).
        with conn:
            conn.execute("DELETE FROM files")
            conn.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),))
    elif int(version[0]) != SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(f"{db} has index schema {version[0]}, expected {SCHEMA_VERSION}; delete it to rebuild")
    return conn


class RepoIndex:
    def __init__(self, root: str | Path, db: str | Path | None = None, *,
                 max_file_bytes: int = DEFAULT_MAX_FILE_BYTES):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"not a directory: {root}")
        self.db_path = Path(db) if db else default_db_path(self.root)
        try:
            self.db_path.resolve().relative_to(self.root)
            inside = True
        except ValueError:
            inside = False
        if inside:
            raise ValueError("the index must not be stored inside the repository it indexes")
        self.max_file_bytes = max_file_bytes
        self.conn = connect(self.db_path)
        stored = self.conn.execute("SELECT value FROM meta WHERE key='root'").fetchone()
        if stored is None:
            self.conn.execute("INSERT INTO meta VALUES ('root', ?)", (str(self.root),))
            self.conn.commit()
        elif stored[0] != str(self.root):
            raise ValueError(f"{self.db_path} indexes {stored[0]}, not {self.root}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ indexing

    def update(self) -> IndexStats:
        started = time.perf_counter()
        stats = IndexStats(str(self.root), str(self.db_path))
        known = {row[0]: row[1:] for row in self.conn.execute(
            "SELECT path, size, mtime_ns, sha256, status, skip_reason FROM files")}
        seen: set[str] = set()
        with self.conn:
            for item in iter_files(self.root):
                if item[0] == "<skip>":
                    stats.skip(item[2])
                    continue
                rel, info = item
                seen.add(rel)
                stats.files_seen += 1
                previous = known.get(rel)
                if previous and previous[0] == info.st_size and previous[1] == info.st_mtime_ns:
                    stats.unchanged += 1
                    if previous[3] == "skipped":
                        stats.skip(previous[4])
                    continue
                self._index_file(rel, info, previous, stats)
            for rel in set(known) - seen:
                self.conn.execute("DELETE FROM files WHERE path = ?", (rel,))
                stats.removed += 1
            self.conn.execute("INSERT OR REPLACE INTO meta VALUES ('last_indexed_utc', ?)",
                              (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),))
        stats.chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        stats.symbols = self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
        stats.seconds = round(time.perf_counter() - started, 3)
        return stats

    def _record_skip(self, rel, info, reason, previous, stats, digest=None):
        stats.skip(reason)
        self.conn.execute("DELETE FROM files WHERE path = ?", (rel,))
        self.conn.execute("INSERT INTO files (path, size, mtime_ns, sha256, language, status, skip_reason) "
                          "VALUES (?, ?, ?, ?, ?, 'skipped', ?)",
                          (rel, info.st_size, info.st_mtime_ns, digest, language_of(rel), reason))
        if previous is not None and previous[3] == "indexed":
            stats.removed += 1   # was searchable, no longer is

    def _index_file(self, rel: str, info, previous, stats: IndexStats) -> None:
        name = rel.rsplit("/", 1)[-1]
        reason = secret_name_reason(name)
        if reason:
            return self._record_skip(rel, info, reason, previous, stats)
        if info.st_size > self.max_file_bytes:
            return self._record_skip(rel, info, "too_large", previous, stats)
        try:
            data = (self.root / rel).read_bytes()
        except OSError:
            return self._record_skip(rel, info, "unreadable", previous, stats)
        if len(data) > self.max_file_bytes:   # grew between stat and read
            return self._record_skip(rel, info, "too_large", previous, stats)
        digest = hashlib.sha256(data).hexdigest()
        if previous and previous[2] == digest and previous[3] == "indexed":
            # Touched but unchanged: refresh size/mtime, keep chunks.
            self.conn.execute("UPDATE files SET size = ?, mtime_ns = ? WHERE path = ?",
                              (info.st_size, info.st_mtime_ns, rel))
            stats.unchanged += 1
            return
        text, reason = decode_text(data)
        if reason is None:
            reason = secret_content_reason(text)
        if reason:
            return self._record_skip(rel, info, reason, previous, stats, digest)
        language = language_of(rel)
        note = None
        if language == "python":
            chunks, symbols, note = chunk_python(rel, text)
        else:
            chunks, symbols = chunk_text(rel, text, language), []
        self.conn.execute("DELETE FROM files WHERE path = ?", (rel,))
        self.conn.execute("INSERT INTO files (path, size, mtime_ns, sha256, language, status, note, chunks) "
                          "VALUES (?, ?, ?, ?, ?, 'indexed', ?, ?)",
                          (rel, info.st_size, info.st_mtime_ns, digest, language, note, len(chunks)))
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks (id, path, start_line, end_line, kind, symbol, language, sha256, text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(c.id, c.path, c.start_line, c.end_line, c.kind, c.symbol, c.language, c.sha256, c.text)
             for c in chunks])
        self.conn.executemany("INSERT INTO symbols VALUES (?, ?, ?, ?, ?)",
                              [(s.name, s.kind, s.path, s.line, s.end_line) for s in symbols])
        if language == "python" and note is None:
            self.conn.executemany("INSERT INTO refs VALUES (?, ?, ?, ?, ?, ?)",
                                  [(r.name, r.kind, r.path, r.line, r.caller, r.expr) for r in python_refs(rel, text)])
        if previous is None or previous[3] == "skipped":
            stats.added += 1
        else:
            stats.updated += 1

    # ------------------------------------------------------------------ reading

    def chunks(self, path: str | None = None) -> list[Chunk]:
        query = "SELECT path, start_line, end_line, kind, symbol, language, text FROM chunks"
        rows = (self.conn.execute(query + " WHERE path = ? ORDER BY start_line", (path,)) if path
                else self.conn.execute(query + " ORDER BY path, start_line"))
        return [Chunk(*row) for row in rows]

    def symbols(self, name: str | None = None) -> list[Symbol]:
        query = "SELECT name, kind, path, line, end_line FROM symbols"
        rows = (self.conn.execute(query + " WHERE name = ? OR name LIKE ? ORDER BY path, line",
                                  (name, "%." + name)) if name
                else self.conn.execute(query + " ORDER BY path, line"))
        return [Symbol(*row) for row in rows]

    def refs(self, name: str, kind: str | None = None) -> list[Ref]:
        query = "SELECT name, kind, path, line, caller, expr FROM refs WHERE name = ?"
        args: tuple = (name,)
        if kind:
            query += " AND kind = ?"
            args += (kind,)
        return [Ref(*row) for row in self.conn.execute(query + " ORDER BY path, line", args)]

    def files(self) -> list[dict]:
        rows = self.conn.execute("SELECT path, size, sha256, language, status, skip_reason, note, chunks "
                                 "FROM files ORDER BY path")
        keys = ("path", "size", "sha256", "language", "status", "skip_reason", "note", "chunks")
        return [dict(zip(keys, row)) for row in rows]

    def summary(self) -> dict:
        files = self.files()
        skipped: dict[str, int] = {}
        for f in files:
            if f["status"] == "skipped":
                skipped[f["skip_reason"]] = skipped.get(f["skip_reason"], 0) + 1
        languages: dict[str, int] = {}
        for f in files:
            if f["status"] == "indexed":
                languages[f["language"]] = languages.get(f["language"], 0) + 1
        meta = dict(self.conn.execute("SELECT key, value FROM meta"))
        return {"root": str(self.root), "db": str(self.db_path),
                "indexed_files": sum(f["status"] == "indexed" for f in files),
                "skipped_files": dict(sorted(skipped.items())),
                "languages": dict(sorted(languages.items())),
                "chunks": self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
                "symbols": self.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0],
                "last_indexed_utc": meta.get("last_indexed_utc")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Index a local repository for NOEMA (stays on this machine)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "stats"):
        p = sub.add_parser(name)
        p.add_argument("path")
        p.add_argument("--db", help="index file (default: ~/.noema/index/<repo>-<hash>.sqlite)")
        if name == "index":
            p.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    args = parser.parse_args(argv)
    try:
        index = RepoIndex(args.path, args.db, **({"max_file_bytes": args.max_file_bytes}
                                                 if args.command == "index" else {}))
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    with index:
        out = index.update().to_dict() if args.command == "index" else index.summary()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

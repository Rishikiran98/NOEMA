"""Repository retrieval: BM25 keyword search over index chunks plus symbol lookup
(definitions and call sites). Roadmap M2.2. Standard library only.

    python -m cpu_platform.retrieval search PATH "how are secrets skipped" [-k 10]
    python -m cpu_platform.retrieval defs   PATH verify_plan
    python -m cpu_platform.retrieval calls  PATH verify_plan
    python -m cpu_platform.retrieval eval   PATH --split dev        # retrieval quality

Search is lexical: it finds chunks that share (code-aware) words with the question. It
does not understand meaning, so a question phrased in words the code never uses can miss.
Every hit says which terms matched, so a caller can see why it was returned.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from .repo_index import Chunk, Ref, RepoIndex, Symbol

# Ranking settings. Chosen on the DEV questions only (data/retrieval, split "dev").
K1 = 1.2
B = 0.75
PATH_WEIGHT = 2          # path tokens are added this many times to a chunk's terms
SYMBOL_WEIGHT = 3        # symbol-name tokens likewise
DEFINITION_BONUS = 0.5   # x top score, for a chunk defining an identifier named in the query
MAX_PER_FILE = 3         # at most this many chunks from one file in a result list
# Score multipliers by file role: questions about how something works are answered by the
# implementation; prose docs and tests that merely mention it should not outrank it.
ROLE_WEIGHTS = {"code": 1.0, "test": 0.6, "docs": 0.6}   # dev: recall@5 0.85 -> 0.95, MRR 0.51 -> 0.82
DOC_LANGUAGES = frozenset({"markdown", "rst", "text"})


def file_role(path: str, language: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if language in DOC_LANGUAGES:
        return "docs"
    if path.startswith(("tests/", "test/")) or "/tests/" in path or name.startswith("test_") \
            or name.endswith(("_test.py", ".test.js", ".test.ts", ".spec.ts", ".spec.js")):
        return "test"
    return "code"

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset("""
a an and are as at be by can do does for from has have how if in into is it its of on or
so than that the their then there these this those to was were what when where which who
why will with would should could self none true false return def import class else elif
not pass use used using get set""".split())


def _stem(term: str) -> str:
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def tokenize(text: str) -> list[str]:
    """Lowercased, stemmed terms. Identifiers yield the whole name and its parts:
    ``model_http_json`` -> model_http_json, model, http, json; ``EngineResult`` ->
    engineresult, engine, result."""
    out = []
    for word in WORD.findall(text):
        parts = [p for chunk in word.split("_") for p in CAMEL.findall(chunk)]
        whole = word.lower().strip("_")
        candidates = [whole] + ([p.lower() for p in parts] if len(parts) > 1 else [])
        for term in candidates:
            if len(term) < 2 or term in STOPWORDS:
                continue
            out.append(_stem(term))
    return out


def identifiers(text: str) -> set[str]:
    """Words in a query that look like code identifiers (snake_case, camelCase, dotted)."""
    found = set()
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", text):
        word = word.strip(".")
        if "_" in word or "." in word or re.search(r"[a-z][A-Z]", word):
            found.add(word.rsplit(".", 1)[-1])
    return found


@dataclass
class Hit:
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    kind: str
    symbol: str | None
    score: float
    matched_terms: list[str]
    text: str = field(repr=False, default="")

    def to_dict(self, with_text: bool = False) -> dict:
        data = asdict(self)
        if not with_text:
            data.pop("text")
        return data


class Retriever:
    """In-memory BM25 over a RepoIndex. Build once per index state (fast: NOEMA's ~1,200
    chunks take well under a second)."""

    def __init__(self, index: RepoIndex):
        self.index = index
        self.root = index.root
        # path -> (status, skip_reason, sha256) at build time; the citation checker (M2.4)
        # uses it to read only files the index accepted.
        self.file_status = {f["path"]: (f["status"], f["skip_reason"], f["sha256"]) for f in index.files()}
        self.chunks: list[Chunk] = index.chunks()
        self.term_freqs: list[dict[str, int]] = []
        self.lengths: list[int] = []
        df: dict[str, int] = {}
        for chunk in self.chunks:
            terms = tokenize(chunk.text)
            terms += tokenize(chunk.path.replace("/", " ")) * PATH_WEIGHT
            if chunk.symbol:
                terms += tokenize(chunk.symbol.replace(".", " ")) * SYMBOL_WEIGHT
            tf: dict[str, int] = {}
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            self.term_freqs.append(tf)
            self.lengths.append(len(terms))
            for term in tf:
                df[term] = df.get(term, 0) + 1
        n = len(self.chunks)
        self.avg_length = (sum(self.lengths) / n) if n else 0.0
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.term_freqs):
            for term in tf:
                self.postings.setdefault(term, []).append(i)

    def search(self, query: str, k: int = 10, *, max_per_file: int | None = MAX_PER_FILE) -> list[Hit]:
        terms = list(dict.fromkeys(tokenize(query)))
        scores: dict[int, float] = {}
        matched: dict[int, list[str]] = {}
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                tf = self.term_freqs[i][term]
                norm = K1 * (1 - B + B * self.lengths[i] / (self.avg_length or 1))
                scores[i] = scores.get(i, 0.0) + idf * tf * (K1 + 1) / (tf + norm)
                matched.setdefault(i, []).append(term)
        if not scores:
            return []
        for i in scores:
            chunk = self.chunks[i]
            scores[i] *= ROLE_WEIGHTS.get(file_role(chunk.path, chunk.language), 1.0)
        named = identifiers(query)
        if named and DEFINITION_BONUS:
            top = max(scores.values())
            for i, chunk in enumerate(self.chunks):
                if chunk.symbol and chunk.symbol.rsplit(".", 1)[-1] in named:
                    scores[i] = scores.get(i, 0.0) + DEFINITION_BONUS * top
                    matched.setdefault(i, []).append(f"defines:{chunk.symbol}")
        ranked = sorted(scores, key=lambda i: (-scores[i], self.chunks[i].path, self.chunks[i].start_line))
        hits, per_file = [], {}
        for i in ranked:
            chunk = self.chunks[i]
            if max_per_file is not None and per_file.get(chunk.path, 0) >= max_per_file:
                continue
            per_file[chunk.path] = per_file.get(chunk.path, 0) + 1
            hits.append(Hit(chunk.id, chunk.path, chunk.start_line, chunk.end_line, chunk.kind, chunk.symbol,
                            round(scores[i], 4), matched[i], chunk.text))
            if len(hits) >= k:
                break
        return hits

    # ------------------------------------------------------------------ symbols

    def definitions(self, name: str) -> list[Symbol]:
        """Definitions of ``name``: exact or qualified (``Class.name``), else case-insensitive."""
        found = self.index.symbols(name)
        if not found:
            lower = name.lower()
            found = [s for s in self.index.symbols()
                     if s.name.lower() == lower or s.name.lower().endswith("." + lower)]
        return found

    def call_sites(self, name: str) -> list[Ref]:
        """Places that call ``name`` (by its last component: ``x.name(...)`` counts)."""
        return self.index.refs(name.rsplit(".", 1)[-1], "call")


# --------------------------------------------------------------------------- evaluation

DEFAULT_QUESTIONS = Path("data/retrieval/noema_questions.jsonl")
EVAL_LOG = Path("data/retrieval/eval_runs.jsonl")


def frozen_sha256(path: Path) -> str:
    """SHA256 of a frozen text file with CRLF line endings read as LF. Git for Windows checks
    files out with CRLF by default; the content (and the manifest hash, computed on LF) is
    the same. Any other change, even one character, still changes the hash."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_questions(path: Path, split: str) -> list[dict]:
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        digest = frozen_sha256(path)
        if manifest.get("sha256") != digest:
            raise SystemExit(f"{path} does not match its frozen manifest SHA256; restore the original file")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if split == "all" or r["split"] == split]


def evaluate(retriever: Retriever, questions: list[dict], k_values=(1, 5, 10)) -> dict:
    """Per question, files are ranked by their first appearance in the chunk ranking.
    hit@k (reported as recall@k) = share of questions with a gold file among the files of
    the top-k chunks. MRR uses the rank of the first gold chunk."""
    depth = max(k_values)
    per_question, hits_at = [], {k: 0 for k in k_values}
    reciprocal = 0.0
    for q in questions:
        results = retriever.search(q["question"], k=depth)
        gold = set(q["gold_files"])
        first = next((n for n, h in enumerate(results, 1) if h.path in gold), None)
        for k in k_values:
            if first is not None and first <= k:
                hits_at[k] += 1
        reciprocal += 1 / first if first else 0.0
        per_question.append({"id": q["id"], "first_gold_rank": first, "gold": sorted(gold),
                             "top5": [h.path for h in results[:5]]})
    n = len(questions) or 1
    return {"questions": len(questions),
            **{f"recall@{k}": round(hits_at[k] / n, 3) for k in k_values},
            "mrr": round(reciprocal / n, 3), "per_question": per_question}


def settings() -> dict:
    return {"k1": K1, "b": B, "path_weight": PATH_WEIGHT, "symbol_weight": SYMBOL_WEIGHT,
            "definition_bonus": DEFINITION_BONUS, "max_per_file": MAX_PER_FILE,
            "role_weights": dict(ROLE_WEIGHTS)}


def _commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search a NOEMA repository index (local only)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "defs", "calls", "eval"):
        p = sub.add_parser(name)
        p.add_argument("path")
        p.add_argument("--db", help="index file (default: ~/.noema/index/...)")
        if name in ("search", "defs", "calls"):
            p.add_argument("query")
        if name == "search":
            p.add_argument("-k", type=int, default=10)
            p.add_argument("--text", action="store_true", help="include chunk text")
        if name == "eval":
            p.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
            p.add_argument("--split", choices=("dev", "eval", "all"), default="dev")
            p.add_argument("--confirm-eval", action="store_true",
                           help="required for the held-out split; the run is logged")
            p.add_argument("--details", action="store_true", help="print per-question results")
    args = parser.parse_args(argv)
    if args.command == "eval" and args.split in ("eval", "all") and not args.confirm_eval:
        parser.error("the held-out 'eval' split is used once, with settings frozen on dev, and needs "
                     "the user's approval: add --confirm-eval (the run is logged)")
    temp = None
    db = args.db
    if args.command == "eval" and db is None:
        temp = tempfile.TemporaryDirectory()
        db = str(Path(temp.name) / "eval.sqlite")
    try:
        with RepoIndex(args.path, db) as index:
            index.update()
            retriever = Retriever(index)
            if args.command == "search":
                out = [h.to_dict(args.text) for h in retriever.search(args.query, k=args.k)]
            elif args.command == "defs":
                out = [asdict(s) for s in retriever.definitions(args.query)]
            elif args.command == "calls":
                out = [asdict(r) for r in retriever.call_sites(args.query)]
            else:
                questions = load_questions(Path(args.questions), args.split)
                out = {"split": args.split, "settings": settings(), **evaluate(retriever, questions)}
                if not args.details:
                    out["misses"] = [q for q in out["per_question"] if not q["first_gold_rank"]
                                     or q["first_gold_rank"] > 5]
                    out.pop("per_question")
                if args.split != "dev":
                    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "commit": _commit(Path(args.path)), "split": args.split,
                              "settings": settings(), "recall@5": out["recall@5"], "mrr": out["mrr"]}
                    log = Path(args.questions).parent / EVAL_LOG.name
                    with open(log, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                    out["logged_to"] = str(log)
    except ValueError as exc:
        parser.error(str(exc))
    finally:
        if temp:
            temp.cleanup()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

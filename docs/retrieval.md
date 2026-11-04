# Repository retrieval (M2.2)

Code: [`cpu_platform/retrieval.py`](../cpu_platform/retrieval.py). Standard library only; works on
the index built by [`repo_index`](repo-index.md), which it updates first.

```powershell
py -m cpu_platform.retrieval search . "how are secrets kept out of the index" -k 5
py -m cpu_platform.retrieval defs   . upstream_error      # where it is defined
py -m cpu_platform.retrieval calls  . upstream_error      # who calls it, from which function
py -m cpu_platform.retrieval eval   . --split dev         # retrieval quality on dev questions
```

## How search ranks

* **Tokens** are code-aware: `model_http_json` gives `model_http_json, model, http, json`;
  `EngineResult` gives `engineresult, engine, result`; lowercase, plural `-s`/`-ies` stripped,
  common English and Python stopwords dropped.
* **BM25** (k1 1.2, b 0.75) over chunks. A chunk's path words count twice and its symbol name
  three times.
* **Definition bonus:** if the question names an identifier (`hash_password`,
  `Store.load`), chunks defining it get +0.5 × the top score.
* **File roles:** scores of documentation (`.md/.rst/.txt`) and test files are multiplied by
  0.6, so the implementation wins over prose and tests that only mention a concept. This is
  a nudge: a short, keyword-dense doc can still rank first.
* At most 3 chunks per file in a result list.
* Every hit lists its `matched_terms` (and `defines:<symbol>` for the bonus), so the reason for
  a result is visible.

Search is **lexical**. It cannot match a question phrased in words the code never uses.

## Symbols

* `defs NAME`: definitions by exact name, as a method (`Class.NAME`), or case-insensitively.
* `calls NAME`: call sites `NAME(...)` and `x.NAME(...)` with the enclosing function
  (`ReasoningEngine.reason.model_attempt`) and the exact callee text (`self.conn.execute`).
  Calls are matched by name, not resolved by type: two different functions with the same
  name are not told apart. Imports are stored as refs too (`kind = "import"`).

Call sites come from `ast` at index time (index schema 2; a schema-1 index is upgraded by
re-reading every file on the next update).

## Evaluation

[`data/retrieval/noema_questions.jsonl`](../data/retrieval/noema_questions.jsonl): 40 questions
about NOEMA, each with gold file(s). They were **written before any retrieval code existed**,
gold files were checked by grep against the answering code, and the file is frozen by the
SHA256 in [`manifest.json`](../data/retrieval/manifest.json) (tests fail if it changes). A fixed
seed split them into 20 **dev** (for tuning) and 20 held-out **eval** questions. The eval split is
run once, with settings frozen on dev, only with the user's approval (`--confirm-eval`; every
run is appended to `data/retrieval/eval_runs.jsonl`). `.noemaignore` keeps `data/retrieval/` out
of the index so no question can match its own text.

**recall@k** = share of questions where a gold file is among the files of the top-k chunks.
**MRR** uses the rank of the first gold chunk.

| Split | Settings | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| dev (20) | plain BM25 (no weights, no bonus, no cap) | 0.35 | 0.80 | 0.85 | 0.53 |
| dev (20) | initial weights, roles 1.0 | 0.35 | 0.85 | 0.95 | 0.51 |
| dev (20) | **frozen: + docs/tests × 0.6** | **0.75** | **0.95** | **0.95** | **0.82** |
| **eval (20, held out)** | frozen, run once 2026-11-04 at `978a353` | **0.70** | **0.95** | **0.95** | **0.80** |

Dev numbers are tuned on, so they overstate quality. The questions were written by the same
author as the code changes, who knows the codebase: they are easier than real user questions.
The one dev miss (q39) asks about `tests/fake_llama.py`, which the test-file weighting pushes
down. The one held-out miss (q25, "where is the bearer token compared in constant time?") shows
the lexical limit: `server.py` does it with `hmac.compare_digest` and never says "constant time".
The held-out run was approved by the user and is logged in `data/retrieval/eval_runs.jsonl`; that
split is now spent for tuning purposes.

## Not done in M2.2

* No semantic or embedding search (would need a model or a dependency).
* Postings are rebuilt in memory per `Retriever`; fine for NOEMA-sized repositories (~1,200
  chunks in well under a second here), not measured on large ones.
* Call sites are name-based, not type-resolved; no "find callers of this exact method".
* Laptop timing: NOT RUN. A fresh held-out set is needed before any further tuning claim.

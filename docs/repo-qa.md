# Grounded repository Q&A (M2.3)

Code: [`cpu_platform/repo_qa.py`](../cpu_platform/repo_qa.py). Uses the [index](repo-index.md)
and [search](retrieval.md); returns an [`EngineResult`](engine-contract.md).

## How an answer is produced

1. The index is brought up to date (incremental) and the question is searched; the top 6
   chunks that fit a 12,000-character budget are kept.
2. The local model gets them as line-numbered `<excerpt>` blocks, marked as untrusted data,
   and is told to answer only from them and cite `[path:START-END]`. Text inside an excerpt
   cannot close or open an excerpt block (delimiters are neutralised).
3. The answer is parsed for citations. Each is classified as **inside** an excerpt the model
   was shown, **partly overlapping** one (lines off by a few), or **outside** everything it
   was shown (possibly invented).
4. If search finds nothing, the model is **not called** and the result is `incomplete`.

## What the result says, and what it does not

| In the `EngineResult` | Meaning |
|---|---|
| `evidence[]` (`file_span`) | each excerpt the model saw: path, lines, SHA256 of the exact text, matched terms |
| `unverified_claims[]` | the answer, and one entry per citation with its inside/overlap/outside status |
| `checked_claims[]` | **empty**: nothing is verified in M2.3 |
| `not_run[]` | answer correctness, citation verification (M2.4), citation relevance, unmeasured costs, dropped excerpts |
| `cost` | 1 model call, tokens, model and total seconds |

"Inside an excerpt" only means the model pointed at code it was shown. It does **not** mean
the citation is accurate or that the code supports the statement. Exact line and quote
checks arrive with M2.4. Relevance stays NOT CHECKED.

## Command line

```powershell
py -m cpu_platform.repo_qa . "How does NOEMA keep secrets out of the repository index?" --expect-model qwen3-coder-30b-a3b
```

`--format json` prints the full `EngineResult`. The exit code is 1 when the result is an error
(for example, the model server is down).

## HTTP: `POST /v2/repo/ask`

The server answers only about repositories registered at start-up. A request can name
one (or give its exact path) but can never make the server read any other directory:

```powershell
py -m cpu_platform.server --repo noema=. --expect-model qwen3-coder-30b-a3b
```
```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8070/v2/repo/ask -Method Post -ContentType 'application/json' -Body '{"question":"How are model server errors turned into HTTP status codes?","task":"repo_ask","repo":{"path":"noema"}}'
```

| Response | When |
|---|---|
| 200 + `EngineResult` | `ok` or `incomplete` |
| 400 `invalid_request` + `problems` | the body is not a valid `repo_ask` request (including `repo.revision`: working tree only) or asks a non-thinking model to think |
| 404 `repo_not_found` | repository not registered |
| 413 | body over 64 KiB (`/v1` keeps its 16 KiB limit) |
| 503 / 502 + `EngineResult` with `error` | model server down or mismatched (503) / rejected or malformed (502) |
| 503 `repos_not_configured` | server started without `--repo` |

Authentication, Host/Origin checks and the in-flight limit apply as for `/v1`.

## Real-model run (acceptance)

NOT RUN. On the laptop, with the coder model running on port 8080 (see M1.2), run three
questions and send back the text output of each:

```powershell
py -m cpu_platform.repo_qa . "How does NOEMA keep secrets out of the repository index?" --expect-model qwen3-coder-30b-a3b
```
```powershell
py -m cpu_platform.repo_qa . "What happens when the llama.cpp server returns HTTP 503?" --expect-model qwen3-coder-30b-a3b
```
```powershell
py -m cpu_platform.repo_qa . "Where is the bearer token compared, and why is that safe?" --expect-model qwen3-coder-30b-a3b
```

What gets recorded: latency, tokens, how many citations were inside, overlapping or
outside the excerpts, and your own reading of whether the answer is right. Your reading is
an observation, not a checker.

## Not done in M2.3

* No citation verification (M2.4) and no injection test suite (M2.5).
* One retrieval pass and one model call: no query rewriting, no follow-up retrieval.
* `/v2/repo/ask` is synchronous; it is not available as a background job.
* The model sees only excerpts; it cannot open other files.

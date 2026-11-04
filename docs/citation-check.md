# Citation verifier (M2.4)

Code: [`cpu_platform/citation_check.py`](../cpu_platform/citation_check.py). It runs automatically
on every [repository answer](repo-qa.md), and on a saved result:

```powershell
py -m cpu_platform.repo_qa . "How does NOEMA keep secrets out of the index?" --format json > answer.json
py -m cpu_platform.citation_check verify answer.json --repo .     # exit 1 if any citation is refuted
```

## What is checked, per citation

| Check | Refuted when | Checker |
|---|---|---|
| path | absolute (`/etc/passwd`, `C:\…`, `\\server\…`), contains `..`, or resolves outside the repository or through a symlink | `citation_check.file` |
| file | it does not exist, or no longer exists in the working tree | `citation_check.file` |
| lines | the range is not within `1..number of lines` | `citation_check.lines` |
| quotes | code quoted for this citation is not in the cited lines | `citation_check.quote` |
| relevance | **never checked** | none |

A cited file that the index **excluded** (likely secret, binary, too large) or that is **ignored**
is never opened. Checking a quote against it would let an answer probe its contents. Such
citations are labelled `not_checked` with the reason, and no checked claim is made about them.

**Which quotes belong to which citation:** within a sentence, a quote (`` `inline` `` or a fenced
block) belongs to the next citation after it; quotes after a sentence's last citation belong to
that one. Quotes in a sentence without a citation are not checked. Inline quotes must match
**exactly**. Fenced blocks may differ only in common indentation and trailing spaces, and may be
copied from inside an indented block. Quotes shorter than 2 characters are ignored.

## How it appears in the result

* `checked_claims`: one claim per fact, for example `cited file app/auth.py exists` (verified),
  `lines 4-6 exist in app/auth.py` (verified), `` `bcrypt.hash(password)` appears in app/auth.py:9-10 ``
  (refuted). Each claim's `scope` states what it does **not** show.
* `evidence`: a `file_span` with the SHA256 of the current cited lines (or of the whole file when
  the range is invalid), so the check can be reproduced.
* Each citation's unverified claim is relabelled, for example
  `citation check: exists, 1/1 quotes match, relevance NOT CHECKED`. The answer itself stays an
  unverified claim.

## What it cannot do

It confirms that cited code **exists as cited**. It does not confirm that the code supports the
statement. In the first real-model run, the wrong answer to "what happens on HTTP 503?" cited
`docs/repo-qa.md:59` and `:73`. Both verify as existing, with relevance NOT CHECKED. Catching that
kind of error needs a check of the statement itself, which this module does not attempt.

## Measured

On the real laptop answers from 2026-11-04 (checked against `ecb99ef`):

* Q1 (secrets): all 4 citations and both quotes verified. An earlier version of this checker
  tied every quote in a sentence to every citation in it and falsely refuted 2 correct
  citations. Quotes are now tied to the following citation; a regression test covers it.
* Q2 (HTTP 503): both citations verified as existing, relevance NOT CHECKED (see above).

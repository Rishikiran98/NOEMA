# v0.5 product-readiness assessment

## Implemented and regression-tested

- Local HTTP routes, bearer key comparison, key-scoped job reads/cancellation, bounded request bodies, admission limits.
- Opt-in persistent SQLite queue with WAL, idempotency, atomic transaction-based claims and an exclusive single-owner OS lock.
- Worker pool bounded to 1 to 16 workers, queue execution timing, terminal statuses, queued cancellation, restart recovery semantics.
- Existing proof and restricted semantic verification, smoke benchmarks and local browser playground.

## Explicitly not implemented

- Internet-facing TLS/identity, SSO, customer account lifecycle, per-user RBAC, enterprise audit logging, secrets rotation.
- Distributed broker, multi-host worker scheduler, model placement, GPU/CPU energy accounting, scale-out automatic failover.
- Arbitrary tool execution sandbox, customer isolation guarantees, PII retention policy or SOC 2 controls.
- Training or running a frontier-level model without GPUs. The model runner remains an independently started local CPU inference process.

## Storage and recovery

Enabling `--jobs-db` stores questions, result traces and answers as plaintext for an adjustable TTL (1 to 720 hours, default 24). The TTL is enforced lazily during admission and startup, so disk erasure may occur later than the time boundary. SQLite deletes do not guarantee physical secure erasure. Secure disk encryption and deliberate database deletion are external operational requirements. If the process dies while executing a job, the next startup marks the job `interrupted`. A previously queued job resumes. Cancellation stops *only* queued work; it cannot terminate a currently running inference call.

## Deployment guidance

Bind only to loopback. Keep the model adapter and HTTP API on the same host or in a private SSH-forwarded configuration. Do not publish the Python `ThreadingHTTPServer` to the public internet, even behind an ad hoc port-forward. Treat `CPU_REASONING_API_KEY` and `CPU_REASONING_API_KEYS_JSON` as local development credentials, not production customer authentication.

**Browser and local-process exposure (M1.5).** Loopback binding alone does not stop a web page from reaching the API: a page can re-point its own domain at 127.0.0.1 (DNS rebinding). The API therefore serves only requests whose `Host` and `Origin` name a loopback host (`127.0.0.1`, `localhost`, `::1`), and answers anything else with `403 forbidden_host` / `forbidden_origin`. A local TLS reverse proxy that forwards its public name must be listed with `--allowed-host NAME`. A request with no `Host` header (HTTP/1.0 tools) is accepted, since browsers always send one. Any unexpected server error returns `500 {"error": "internal_error", "request_id": ...}`; the traceback is logged to stderr with that request id, without request bodies, headers or query strings.

**Model server key.** The llama.cpp server is a second loopback service with the same exposure. Set `LLAMA_API_KEY` in the window that runs `cpu_platform.launch` **and** in every window that runs the API or experiments. llama-server reads that variable itself, so the key is never put on a command line, in `results/launch_<port>.json`, or in a log; NOEMA's clients send it as a bearer token. After start-up the launcher checks that an unauthenticated request is refused and records `api_key: enforced`, `NOT ENFORCED` (this build ignored the variable: do not rely on it) or `not configured` in the launch manifest. Real llama.cpp enforcement has been tested only against the offline fake server: NOT RUN on a real build. Monitor disk capacity and the queue. Model inference can be substantially slower than queue processing and can make 429 limits necessary.

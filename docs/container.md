# Container sandbox backend (M3.2)

Code: [`cpu_platform/container.py`](../cpu_platform/container.py). A [sandbox](sandbox.md) backend
that runs every command in a fresh Docker container. Unlike `dev`, it **can** be a security
boundary, and it claims to be one only after proving it, as described below.

```python
from cpu_platform.sandbox import Sandbox
from cpu_platform.container import ContainerBackend
with Sandbox("path/to/repo", ContainerBackend(image="python:3.12-slim")) as box:
    box.run(["python", "-m", "unittest"])
```

```powershell
py -m cpu_platform.testrun . --backend container --image python:3.12-slim
py -m cpu_platform.sandbox run . --backend container -- python -c "print('hi')"
```

## What every container gets

| | Flag |
|---|---|
| no network | `--network none` (loopback only); `network=True` requests are refused |
| read-only base | `--read-only`, plus `/tmp` as a 256 MiB tmpfs (`nosuid,nodev`) |
| one mount | only the sandbox folder (repository copy, HOME, TEMP) at `/sandbox` via `--mount type=bind,…` |
| non-root | `--user 65534:65534` (nobody), `--cap-drop ALL`, `--security-opt no-new-privileges` |
| limits | `--memory` = `--memory-swap` (2 GiB), `--cpus 2`, `--pids-limit 256`, `--ulimit fsize=256 MiB`, `--ulimit core=0` |
| no surprise downloads | `--pull never`; the image must already be present |
| timeout | `docker kill <container>` (killing only the CLI client would leave the container running), then the client |
| environment | only locale variables and sandbox paths; nothing from the host's environment (no PATH, no credentials) |

Host paths in command arguments (for example the test runner's results file) are rewritten to
`/sandbox/...`, and a host Python path (`sys.executable`) becomes the image's `python`.

## The isolation claim is earned

When a `Sandbox` is created, the backend checks that Docker answers and the image is present,
then runs a **probe inside a container with exactly the flags above**. It must report:
* the user is not root;
* writing to `/` fails;
* an outbound connection (to 1.1.1.1:53) fails;
* the sandbox work area is writable.

Only then does the backend set `security_boundary=True` on its records. Any failure (or Docker
missing, the engine down, the image absent) raises `SandboxError` naming what failed, and
nothing runs. The probe report is included in each result's `limits`.

What it does **not** protect against: kernel or container-runtime escapes, a Docker daemon
configured unsafely (for example its socket exposed to containers), or side channels. On Linux
the container shares the host kernel; Docker Desktop adds a VM on Windows and macOS.

## Setup

**Windows (your laptop):**
1. Install Docker Desktop and choose the WSL 2 backend (Settings → General → "Use the WSL 2 based engine").
2. Start Docker Desktop and wait until it reports "Engine running".
3. Pull the image once, deliberately:
   ```powershell
   docker pull python:3.12-slim
   ```
4. Run the acceptance tests:
   ```powershell
   py -m unittest tests.test_m32_container -v
   ```
   The `RealDocker` tests must run (not "skipped") and pass. Note: the fake-Docker tests skip on
   Windows, because their wrapper is a POSIX shell script.

The sandbox copy lives under `%TEMP%` (inside `C:\Users\…`), which Docker Desktop shares with
containers by default.

**Linux:**
1. Install Docker Engine (`docker.io` or Docker's packages) and add your user to the `docker`
   group (that group is root-equivalent on the host; prefer rootless Docker where you can).
2. Then:
   ```bash
   docker pull python:3.12-slim
   ```
   ```bash
   python -m unittest tests.test_m32_container -v
   ```

To use another image (one with your project's dependencies installed), pass `--image` /
`ContainerBackend(image=...)`, and set `NOEMA_TEST_DOCKER_IMAGE` for the tests. Pin images by
digest (`python@sha256:…`) for reproducible runs.

## Measured

* **Offline (cloud container, no Docker engine):** 18 tests pass through `tests/fake_docker.py`,
  a stand-in for the Docker CLI that runs commands on the host and does **not** isolate. They
  cover the exact flags and mount, the environment, path rewriting, every probe failure refusing,
  Docker missing / engine down / image absent, `docker kill` on timeout, the output cap, no writes
  to the original repository, and the test runner through this backend.
* **Real Docker, 2026-11-07, user's Windows laptop:** Docker Desktop 4.98 (engine 29.9.2, WSL 2
  kernel 6.18.54.1), image `python:3.12-slim` (`sha256:4840d514c583a4d3…`), code at `4b87d43`:
  **all 5 `RealDocker` tests passed** (probe; no outbound connection, `/etc` not writable,
  non-root, host home not visible; timeout, output cap, no writes to the original; 1 GiB
  allocation refused at a 256 MiB limit; test runner in the container). The 14 fake-Docker tests
  skip on Windows (POSIX-only), as expected. Docker Desktop's own HTTP proxy setting is
  irrelevant inside these containers: they have no network.
* Docker Desktop must be running first. With the engine stopped, NOEMA refuses with "Docker
  engine not reachable", the same message `docker info` shows as "failed to connect to the docker API".
* Not measured: per-run container start-up time; Linux hosts; rootless Docker.

## Not done in M3.2

* No WSL2-only backend (without Docker Desktop).
* No image building: the image must provide Python and the project's dependencies.
* One container per command (start-up cost per run, not measured); no warm pool.

"""Offline stand-in for the Docker CLI + engine (M3.2 tests). NOT a container: it does not
isolate anything. It records the exact arguments NOEMA passes, maps /sandbox paths back to
the host so commands really run, honours `docker kill`, and returns a scripted isolation-probe
report. Controlled through environment variables:

    FAKE_DOCKER_LOG         file to append one JSON argv per invocation
    FAKE_DOCKER_STATE       directory for container-name -> pid files
    FAKE_DOCKER_INFO_EXIT   exit code for `docker info` (default 0)
    FAKE_DOCKER_IMAGE_EXIT  exit code for `docker image inspect` (default 0)
    FAKE_DOCKER_PROBE       JSON probe report to print (default: an isolated container), or
                            "none" to print nothing
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

VALUE_OPTIONS = {"--name", "--pull", "--network", "--tmpfs", "--user", "--cap-drop", "--security-opt",
                 "--pids-limit", "--memory", "--memory-swap", "--cpus", "--ulimit", "--mount", "-w", "-e"}
FLAG_OPTIONS = {"--rm", "--read-only"}
ISOLATED = {"uid": 65534, "root_writable": False, "network": False, "work_writable": True}


def log(argv):
    path = os.environ.get("FAKE_DOCKER_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(argv) + "\n")


def parse_run(args):
    opts, i = [], 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] in FLAG_OPTIONS:
            opts.append((args[i], None))
            i += 1
        elif args[i] in VALUE_OPTIONS:
            opts.append((args[i], args[i + 1]))
            i += 2
        else:
            raise SystemExit(f"fake docker: unknown option {args[i]}")
    return opts, args[i], args[i + 1:]


def main():
    argv = sys.argv[1:]
    log(argv)
    command = argv[0] if argv else ""
    if command == "info":
        print("29.0-fake")
        return int(os.environ.get("FAKE_DOCKER_INFO_EXIT", "0"))
    if command == "image":
        return int(os.environ.get("FAKE_DOCKER_IMAGE_EXIT", "0"))
    if command == "kill":
        pidfile = Path(os.environ["FAKE_DOCKER_STATE"]) / (argv[1] + ".pid")
        if pidfile.is_file():
            try:
                os.killpg(int(pidfile.read_text()), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            (Path(os.environ["FAKE_DOCKER_STATE"]) / (argv[1] + ".killed")).write_text("1")
        return 0
    if command != "run":
        return 1
    opts, image, cmd = parse_run(argv[1:])
    mount = dict(part.split("=", 1) for part in next(v for k, v in opts if k == "--mount").split(","))
    assert mount["type"] == "bind" and mount["target"] == "/sandbox", mount
    base = mount["source"]
    to_host = lambda s: s.replace("/sandbox", base)
    if "NOEMA_PROBE" in " ".join(cmd):
        report = os.environ.get("FAKE_DOCKER_PROBE")
        if report != "none":
            print("NOEMA_PROBE " + (report or json.dumps(ISOLATED)))
        return 0
    workdir = to_host(next(v for k, v in opts if k == "-w"))
    env = {"PATH": os.environ.get("PATH", "")}
    for k, v in opts:
        if k == "-e":
            key, _, value = v.partition("=")
            env[key] = to_host(value)
    real = [sys.executable if a == "python" else to_host(a) for a in cmd]
    proc = subprocess.Popen(real, cwd=workdir, env=env, start_new_session=True)
    name = next(v for k, v in opts if k == "--name")
    (Path(os.environ["FAKE_DOCKER_STATE"]) / (name + ".pid")).write_text(str(proc.pid))
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Offline check (0 tokens): the harness must still save every answer it can when the launch environment is odd.

The organisers' final launch command was not published when this was written, and the one in the rules does not pass
OUT / TASKS / LOG_DIR into the container. Whatever they forward (for example a HOST path in OUT, which cannot be created
inside a `--read-only` container), a failure in a side file must never stop the run before the first task: one crash
there loses every answer, including the 0-token deterministic ones.

    python3 tools/check_startup.py <directory with the sample task dirs>

Each scenario runs run_agent.py on a fresh copy of the samples (expected/check removed from the copies), with the LLM
variables removed from the environment, and grades the answer.txt files against the originals.
Exit status 1 if any scenario does not reach 8/8 (or all samples).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_AGENT = os.path.join(HERE, "..", "run_agent.py")
UNCREATABLE = "/proc/no_such_dir_for_the_check"  # no directory can ever be created under /proc


def prepare(samples: str, dst: str) -> str:
    tasks = os.path.join(dst, "tasks")
    shutil.copytree(samples, tasks)
    for d in os.listdir(tasks):
        p = os.path.join(tasks, d, "task.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                t = json.load(f)
            t.pop("expected", None)
            t.pop("check", None)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(t, f)
    return tasks


def grade(samples: str, tasks: str) -> tuple[int, int]:
    ok = n = 0
    for d in sorted(os.listdir(samples)):
        meta = os.path.join(samples, d, "task.json")
        if not os.path.isfile(meta):
            continue
        with open(meta, encoding="utf-8") as f:
            exp = json.load(f)
        n += 1
        ans = os.path.join(tasks, d, "answer.txt")
        got = open(ans, encoding="utf-8").read().strip() if os.path.isfile(ans) else None
        if exp.get("check") == "unittest":
            # the organisers' own instruction: `python3 test_mylib.py` in the workspace (so no -I: it must import mylib)
            r = subprocess.run([sys.executable, "test_mylib.py"], cwd=os.path.join(tasks, d, "workspace"),
                               capture_output=True)
            ok += got == "DONE" and r.returncode == 0
        else:
            ok += got == exp.get("expected")
    return ok, n


def main() -> int:
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        print(__doc__)
        return 2
    samples = os.path.abspath(sys.argv[1])
    scenarios = [  # (name, extra environment, extra arguments, uses --fake-llm)
        ("OUT is a path that cannot be created (a file)", {"OUT": UNCREATABLE + "/sub/out.json"}, [], True),
        ("OUT is a path that cannot be created (a directory)", {"OUT": UNCREATABLE + "/sub"}, [], True),
        ("OUT is an existing directory", {"OUT": "@TMP/outdir"}, [], True),
        ("--out given and cannot be created", {}, ["--out", UNCREATABLE + "/x/solutions.json"], True),
        ("LOG_DIR cannot be created", {"LOG_DIR": UNCREATABLE + "/logs"}, [], True),
        ("no LLM_BASE_URL / LLM_API_KEY at all", {}, [], False),
        ("tasks dir only via the TASKS variable", {"TASKS": "@TASKS"}, ["@NOTASKSARG"], True),
        ("positional arguments instead of flags", {}, ["@POSITIONAL"], True),
    ]
    bad = 0
    for name, extra_env, extra_args, fake in scenarios:
        with tempfile.TemporaryDirectory() as tmp:
            tasks = prepare(samples, tmp)
            os.makedirs(os.path.join(tmp, "outdir"))
            env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_KEY", "LLM_BASE_URL", "OUT", "TASKS")}
            env.update({k: v.replace("@TMP", tmp).replace("@TASKS", tasks) for k, v in extra_env.items()})
            args = [sys.executable, RUN_AGENT]
            if "@NOTASKSARG" in extra_args:
                pass
            elif "@POSITIONAL" in extra_args:
                args += [tasks, os.path.join(tmp, "outdir", "s.json")]
            else:
                args += ["--tasks-dir", tasks]
            args += [a for a in extra_args if not a.startswith("@")]
            if fake:
                args.append("--fake-llm")
            r = subprocess.run(args, capture_output=True, text=True, env=env, timeout=300)
            ok, n = grade(samples, tasks)
            verdict = "ok" if (ok == n and r.returncode == 0 and "Traceback" not in r.stderr) else "BAD"
            bad += verdict != "ok"
            print(f"{verdict:3s} {ok}/{n} exit={r.returncode}  {name}")
            if verdict != "ok":
                print("    " + (r.stderr.strip().splitlines() or ["(no stderr)"])[-1][:200])
    print("RESULT:", "OK" if not bad else f"{bad} scenario(s) failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

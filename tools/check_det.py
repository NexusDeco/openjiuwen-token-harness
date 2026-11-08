#!/usr/bin/env python3
"""Validate det_solver.py without spending a single token.

    python tools/check_det.py [--seeds 20] [--samples <dir with the organisers' sample tasks>]

Ground truth comes from tools/gen_tasks.py (`--seeds` different seeds, 40 tasks each) and, optionally, from the
organisers' sample tasks (their task.json still carries `expected`). Reported per task kind: correct answers, WRONG
answers (the dangerous outcome), solvable tasks wrongly declared unsolvable, and how many fall back to the LLM.
Exit code 1 if any wrong answer or false "unsolvable" appears.
"""
import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from det_solver import _run_tests, solve_deterministic  # noqa: E402
from tasks import load_tasks  # noqa: E402


def check_generated(seeds, tally, problems):
    for seed in range(seeds):
        with tempfile.TemporaryDirectory() as td:
            out, gp = os.path.join(td, "tasks"), os.path.join(td, "grader.json")
            subprocess.run([sys.executable, os.path.join(HERE, "gen_tasks.py"), "--out", out, "--grader", gp,
                            "--seed", str(1000 + seed)], check=True, capture_output=True)
            grader = json.load(open(gp))
            for t in load_tasks(out):
                g = grader[t.id]
                r = solve_deterministic(t)
                key = (t.type, g["kind"], "solvable" if g["solvable"] else "unsolvable")
                if g["solvable"]:
                    if r.status == "nomatch":
                        verdict = "fallback"
                    elif r.status == "unsolvable":
                        verdict = "FALSE-UNSOLVABLE"
                    elif g["check"] == "unittest":
                        with open(os.path.join(t.workspace, "test_mylib.py"), "rb") as f:
                            same = hashlib.sha256(f.read()).hexdigest() == g["test_sha256"]
                        verdict = "correct" if (same and _run_tests(t.workspace)[0] and r.answer == "DONE") else "WRONG"
                    else:
                        verdict = "correct" if r.answer == g["expected"] else "WRONG"
                else:
                    verdict = {"unsolvable": "detected", "nomatch": "fallback", "answer": "answered-anyway"}[r.status]
                tally[key][verdict] += 1
                if verdict in ("WRONG", "FALSE-UNSOLVABLE"):
                    problems.append(f"seed {1000 + seed} {t.id} {g['kind']}: {verdict} (det said {r.status} {r.answer!r} {r.why})")


def check_samples(path, problems):
    print(f"\n== organisers' samples in {path}")
    with tempfile.TemporaryDirectory() as td:
        work = os.path.join(td, "tasks")
        shutil.copytree(path, work, ignore=shutil.ignore_patterns("__MACOSX", ".DS_Store"))
        for t in load_tasks(work):
            raw = json.load(open(os.path.join(t.dir, "task.json")))
            r = solve_deterministic(t)
            if r.status == "answer":
                if raw["check"] == "unittest":
                    ok = _run_tests(t.workspace)[0] and r.answer == "DONE"
                elif raw["check"] == "exact_int":
                    ok = r.answer == str(int(raw["expected"]))
                else:
                    ok = r.answer == raw["expected"]
                verdict = "correct" if ok else "WRONG"
            else:
                verdict = r.status
            print(f"   {t.id:20s} reward={t.reward:<2} -> {verdict:10s} answer={r.answer!r}  ({r.why or r.solver})")
            if verdict in ("WRONG", "unsolvable"):
                problems.append(f"sample {t.id}: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--samples")
    a = ap.parse_args()
    tally = collections.defaultdict(collections.Counter)
    problems = []
    check_generated(a.seeds, tally, problems)
    print(f"== generated tasks: {a.seeds} seeds x 40")
    print(f"{'type':14s} {'kind':22s} {'truth':10s} {'n':>5s}  results")
    for (t, k, truth), c in sorted(tally.items()):
        print(f"{t:14s} {k:22s} {truth:10s} {sum(c.values()):5d}  {dict(c)}")
    tot = collections.Counter()
    for (_t, _k, truth), c in tally.items():
        if truth == "solvable":
            tot.update(c)
    n = sum(tot.values())
    print(f"\nsolvable tasks: {n}; solved without LLM: {tot['correct']} ({100 * tot['correct'] / n:.0f}%), "
          f"fallback to LLM: {tot['fallback']}, WRONG: {tot['WRONG']}, falsely unsolvable: {tot['FALSE-UNSOLVABLE']}")
    if a.samples:
        check_samples(a.samples, problems)
    print("\nPROBLEMS:" if problems else "\nNo wrong answers and no false 'unsolvable'.")
    for p in problems[:20]:
        print("  ", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()

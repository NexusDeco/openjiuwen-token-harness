#!/usr/bin/env python3
"""Offline check (0 tokens) of det_solver.solve_math on random warehouse stories.

The ground truth is a direct simulation written here (no code shared with det_solver). A story is a start plus 1..10
operations in random order. Operations are written with the organisers' exact sentences (from math_0000 / math_0004) and
with the variants the solver claims to know; quantities are digits or number words. Checked:

  * every solvable story -> the simulated number;
  * the same story with one quantity slot blanked ("ships boxes per day ...") or made vague ("several") -> unsolvable;
  * sentences that must NOT be claimed (inconsistent "tripled ... a sister warehouse", "rounded to the nearest", a rate
    without a duration, decimals, other verbs) -> nomatch, i.e. they stay with the LLM.

    python3 tools/check_math_chains.py [--n 4000] [--seed 7] [--selftest]

`--selftest` replaces the solver by one that is wrong on purpose and requires this check to fail.
Exit status 1 on any wrong answer, wrong unsolvable claim, or claimed nonsense.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import det_solver  # noqa: E402

TAIL = ("Work only inside the current directory. When you are done, write ONLY the final answer to the file answer.txt in "
        "the current directory (for example with bash: echo 42 > answer.txt).")
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen " \
       "eighteen nineteen".split()
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    t, o = divmod(n, 10)
    return TENS[t] + (f"-{ONES[o]}" if o else "")


def q(rng: random.Random, n: int) -> str:
    """A quantity as digits, or as number words when small enough."""
    return words(n) if n < 100 and rng.random() < 0.3 else str(n)


def then(rng: random.Random, sentence: str) -> str:
    """Ship / receive sentences may or may not start with 'Then' (the samples have none; chains might)."""
    return ("Then " + sentence[0].lower() + sentence[1:]) if rng.random() < 0.3 else sentence


def make_op(rng: random.Random, v: int):
    """-> (sentence, new_value, blank) where blank() gives the same sentence with a slot removed (or None)."""
    kind = rng.choice(["split", "split", "double", "mult", "ship_day", "ship", "deliv", "recv_day", "recv"])
    if kind == "split":
        k, mode = rng.choice([2, 3, 4, 5]), rng.choice(["down", "down", "up"])
        new = v // k if mode == "down" else -(-v // k)
        mk = lambda kk: f"Then the stock is split equally into {kk} warehouses and it keeps one share (rounded {mode})."
        return mk(q(rng, k)), new, mk("")  .replace("into  warehouses", "into warehouses")
    if kind == "double":
        s = "Then the stock is doubled by merging with a sister warehouse."
        return s, v * 2, None
    if kind == "mult":
        w, f = rng.choice([("tripled", 3), ("quadrupled", 4), ("doubled", 2)])
        tail = rng.choice(["", f" by merging with {['a', 'two', 'three'][f - 2]} sister warehouse" + ("" if f == 2 else "s")])
        return f"Then the stock is {w}{tail}.", v * f, None
    if kind in ("ship_day", "ship"):
        d = rng.randint(1, 4) if kind == "ship_day" else 1
        amt = rng.randint(1, max(1, v // d)) if v else 0
        if v == 0:
            amt = 0
        total = amt * d
        if total > v:
            amt, total = 0, 0
        s = f"It ships {q(rng, amt)} boxes per day for {q(rng, d)} day{'s' if d != 1 else ''}." if kind == "ship_day" \
            else f"It ships {q(rng, amt)} boxes."
        blank = (f"It ships boxes per day for {d} days." if kind == "ship_day" else "It ships boxes.")
        return then(rng, s), v - total, blank
    if kind == "deliv":
        r, sz = rng.randint(1, 5), rng.randint(2, 90)
        s = f"It receives {q(rng, r)} deliver{'y' if r == 1 else 'ies'} of {q(rng, sz)} boxes each."
        return then(rng, s), v + r * sz, rng.choice([f"It receives deliveries of {sz} boxes each.",
                                                    f"It receives {r} deliveries of boxes each."])
    d = rng.randint(1, 4) if kind == "recv_day" else 1
    amt = rng.randint(1, 60)
    s = f"It receives {q(rng, amt)} boxes per day for {q(rng, d)} day{'s' if d != 1 else ''}." if kind == "recv_day" \
        else f"It receives {q(rng, amt)} boxes."
    blank = (f"It receives boxes per day for {d} days." if kind == "recv_day" else "It receives boxes.")
    return then(rng, s), v + amt * d, blank


def story(rng: random.Random):
    start = rng.randint(50, 3000)
    v, sents, blanks = start, [f"A warehouse starts with {q(rng, start)} boxes."], [None]
    for _ in range(rng.randint(1, 10)):
        s, v, b = make_op(rng, v)
        sents.append(s)
        blanks.append(b)
    return sents, blanks, v


def ask(sents) -> det_solver.DetResult:
    text = " ".join(sents) + " How many boxes does it have at the end? Compute it with the bash tool (python3 -c). " + TAIL
    return det_solver.solve_math(det_solver._norm(text))


# sentences that must stay with the LLM (nomatch): placed after a valid start
MUST_NOT = [
    "Then the stock is tripled by merging with a sister warehouse.",            # disagrees with the number of sisters
    "Then the stock is doubled by merging with two sister warehouses.",
    "Then the stock is split equally into 3 warehouses and it keeps one share (rounded to the nearest).",
    "It ships 36 boxes per day.",                                              # a rate without a duration
    "It ships 2.5 boxes per day for 3 days.",                                  # decimals
    "It sells 36 boxes.",                                                      # another verb
    "Then the stock is halved.",
    "It ships 36 boxes per day for 3 days. The warehouse burns down.",
    "It receives a dozen boxes.",
]


def run(n: int, seed: int) -> int:
    rng = random.Random(seed)
    bad, solved, blanked, claimed = [], 0, 0, 0
    for _ in range(n):
        sents, blanks, expect = story(rng)
        r = ask(sents)
        if r.status != "answer" or r.answer != str(expect):
            bad.append(("SOLVABLE", " ".join(sents)[:200], expect, r.status, r.answer))
        else:
            solved += 1
        cand = [i for i, b in enumerate(blanks) if b]
        if cand:
            i = rng.choice(cand)
            broken = list(sents)
            broken[i] = blanks[i]
            r2 = ask(broken)
            if r2.status != "unsolvable":
                bad.append(("BLANK-NOT-UNSOLVABLE", " ".join(broken)[:200], r2.status, r2.answer))
            else:
                blanked += 1
        if rng.random() < 0.2:  # a vague quantity in the start
            r3 = ask([f"A warehouse starts with {rng.choice(['some', 'several', 'an unknown number of'])} boxes."] + sents[1:])
            if r3.status != "unsolvable":
                bad.append(("VAGUE-START", r3.status))
    for s in MUST_NOT:
        r = ask(["A warehouse starts with 500 boxes.", s])
        if r.status != "nomatch":
            bad.append(("CLAIMED", s, r.status, r.answer))
        else:
            claimed += 1
    r = ask(["A warehouse starts with boxes.", "It ships 5 boxes."])
    if r.status != "unsolvable":
        bad.append(("EMPTY-START", r.status))
    print(f"stories: {n}; solved correctly: {solved}; blanked slots recognised as unsolvable: {blanked}; "
          f"nonsense left to the LLM: {claimed}/{len(MUST_NOT)}; problems: {len(bad)}")
    for b in bad[:8]:
        print("  ", b)
    return len(bad)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        orig = det_solver.solve_math

        def broken(text):
            r = orig(text)
            if r.status == "answer" and "tripled" in text:  # wrong on purpose
                return det_solver.DetResult("answer", str(int(r.answer) + 1), solver="math")
            return r
        det_solver.solve_math = broken
        problems = run(a.n, a.seed)
        print("SELFTEST:", "OK (the check noticed the sabotage)" if problems else "FAILED (sabotage went unnoticed)")
        return 0 if problems else 1
    problems = run(a.n, a.seed)
    print("RESULT:", "OK" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

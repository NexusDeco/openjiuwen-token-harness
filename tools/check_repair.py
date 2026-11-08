#!/usr/bin/env python3
"""Offline check of repair.py (mutation repair of mylib.py), 0 tokens.

Cases are hand-written planted bugs in helper functions that det_solver.CANON does NOT know (CANON is emptied on purpose,
so only the mutation stage can solve them). `fix=False` marks bugs that are deliberately NOT a single small edit: the
expected outcome there is "no result" (the task would go to the LLM), never a false DONE.

    python3 tools/check_repair.py [--real DIR_WITH_coding_tasks]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import det_solver  # noqa: E402

TAIL = ("Work only inside the current directory. When you are done, write ONLY the word DONE to the file answer.txt in "
        "the current directory (for example with bash: echo 42 > answer.txt).")
INSTR = det_solver._C_BODY + " " + TAIL

# (label, buggy mylib.py, test methods, fixable by one small edit per function)
CASES = [
    ("is_palindrome step", "def is_palindrome(s):\n    return s == s[::1]\n",
     "assertTrue(m.is_palindrome('abba')); self.assertFalse(m.is_palindrome('abc'))", True),
    ("factorial range", "def factorial(n):\n    r = 1\n    for i in range(1, n):\n        r *= i\n    return r\n",
     "assertEqual(m.factorial(5), 120)", True),
    ("fib returns b", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return b\n",
     "assertEqual(m.fib(10), 55); self.assertEqual(m.fib(0), 0)", True),
    ("clamp min/max", "def clamp(x, lo, hi):\n    return min(lo, min(hi, x))\n",
     "assertEqual(m.clamp(5, 0, 3), 3); self.assertEqual(m.clamp(-1, 0, 3), 0); self.assertEqual(m.clamp(2, 0, 3), 2)",
     True),
    ("count_vowels not in", "def count_vowels(s):\n    return sum(1 for c in s if c not in 'aeiou')\n",
     "assertEqual(m.count_vowels('education'), 5)", True),
    ("second_largest index", "def second_largest(xs):\n    return sorted(set(xs))[1]\n",
     "assertEqual(m.second_largest([4, 9, 1, 7]), 7)", True),
    ("is_prime bound", "def is_prime(n):\n    if n < 3:\n        return False\n    for i in range(2, n):\n"
                       "        if n % i == 0:\n            return False\n    return True\n",
     "assertTrue(m.is_prime(2)); self.assertTrue(m.is_prime(13)); self.assertFalse(m.is_prime(15)); "
     "self.assertFalse(m.is_prime(1))", True),
    ("gcd floordiv", "def gcd(a, b):\n    while b:\n        a, b = b, a // b\n    return a\n",
     "assertEqual(m.gcd(12, 18), 6)", True),
    ("sum_squares add", "def sum_squares(xs):\n    return sum(x + x for x in xs)\n",
     "assertEqual(m.sum_squares([1, 2, 3]), 14)", True),
    ("last_element idx", "def last_element(xs):\n    return xs[0]\n", "assertEqual(m.last_element([4, 5, 6]), 6)", True),
    ("abs_diff cmp", "def abs_diff(a, b):\n    return a - b if a < b else b - a\n",
     "assertEqual(m.abs_diff(9, 4), 5); self.assertEqual(m.abs_diff(4, 9), 5)", True),
    ("power augassign", "def power(base, exp):\n    r = 1\n    for _ in range(exp):\n        r += base\n    return r\n",
     "assertEqual(m.power(2, 5), 32)", True),
    ("range_sum bound", "def range_sum(a, b):\n    return sum(range(a, b))\n", "assertEqual(m.range_sum(1, 4), 10)", True),
    ("spread operands", "def spread(xs):\n    return min(xs) - max(xs)\n", "assertEqual(m.spread([3, 9, 5]), 6)", True),
    ("is_even", "def is_even(n):\n    return n % 2 == 1\n", "assertTrue(m.is_even(4)); self.assertFalse(m.is_even(7))", True),
    ("double_all", "def double_all(xs):\n    return [x + 2 for x in xs]\n", "assertEqual(m.double_all([1, 2, 3]), [2, 4, 6])",
     True),
    ("sign const", "def sign(n):\n    if n > 0:\n        return 1\n    elif n < 0:\n        return 1\n    return 0\n",
     "assertEqual(m.sign(-5), -1); self.assertEqual(m.sign(5), 1); self.assertEqual(m.sign(0), 0)", True),
    ("count_positive", "def count_positive(xs):\n    return sum(1 for x in xs if x >= 0)\n",
     "assertEqual(m.count_positive([0, 1, -2, 3]), 2)", True),
    ("mean_of_two", "def mean_of_two(a, b):\n    return (a + b) // 2\n", "assertEqual(m.mean_of_two(3, 4), 3.5)", True),
    ("middle", "def middle(xs):\n    return xs[len(xs) / 2]\n", "assertEqual(m.middle([1, 2, 3, 4, 5]), 3)", True),
    ("lower/upper", "def shout(s):\n    return s.lower() + '!'\n", "assertEqual(m.shout('hey'), 'HEY!')", True),
    ("find_last", "def last_index(xs, v):\n    return len(xs) - 1 - xs[::-1].index(v) + 1\n",
     "assertEqual(m.last_index([1, 2, 1, 3], 1), 2)", True),
    ("min as first/while", "def drop_last(xs):\n    return xs[:-2]\n", "assertEqual(m.drop_last([1, 2, 3]), [1, 2])", True),
    ("while step", "def countdown(n):\n    out = []\n    while n > 0:\n        out.append(n)\n        n += 1\n    return out\n",
     "assertEqual(m.countdown(3), [3, 2, 1])", True),
    ("class method", "class Counter:\n    def __init__(self):\n        self.n = 0\n\n    def inc(self):\n        self.n -= 1\n",
     "c = m.Counter(); c.inc(); c.inc(); self.assertEqual(c.n, 2)", True),
    ("wrong var", "def diff_of(a, b):\n    return a - a\n", "assertEqual(m.diff_of(9, 4), 5)", True),
    ("stray line", "def total_plus(xs):\n    s = 0\n    for x in xs:\n        s += x\n    s += 1\n    return s\n",
     "assertEqual(m.total_plus([1, 2, 3]), 6)", True),
    # --- not one small edit: must stay unsolved (no false DONE) ---
    ("capitalize (no swap)", "def cap_words(s):\n    return ' '.join(w.upper() for w in s.split())\n",
     "assertEqual(m.cap_words('ab cd'), 'Ab Cd')", False),
    ("two bugs in one fn", "def average2(xs):\n    return sum(xs) // len(xs) + 1\n", "assertEqual(m.average2([1, 2]), 1.5)", False),
    ("missing edge case", "def max_or_none(xs):\n    return max(xs)\n",
     "assertEqual(m.max_or_none([1, 5]), 5); self.assertIsNone(m.max_or_none([]))", False),
    ("contradictory tests", "def f(x):\n    return x\n", "assertEqual(m.f(1), 2); self.assertEqual(m.f(1), 3)", False),
]

# several functions, several bugs in one file (like the real examples): rounds of greedy repair
MULTI = [
    ("3 bugs / 5 funcs",
     "def a(xs):\n    return sum(xs) - 1\n\n\ndef b(s):\n    return s.strip().upper()\n\n\n"
     "def c(xs):\n    return [x for x in xs if x % 2]\n\n\ndef d(n):\n    return n * 2\n\n\ndef e(xs):\n    return len(xs) + 1\n",
     "assertEqual(m.a([1, 2]), 3); self.assertEqual(m.b(' hi '), 'HI'); self.assertEqual(m.c([1, 2, 3]), [1, 3]); "
     "self.assertEqual(m.d(4), 8); self.assertEqual(m.e([1, 2]), 2)"),
]


def make_dir(root: str, name: str, lib: str, body: str) -> str:
    d = os.path.join(root, name)
    os.makedirs(d)
    with open(os.path.join(d, "mylib.py"), "w") as f:
        f.write(lib)
    stmts = [s.strip() for s in body.split(";") if s.strip()]
    tests = ["import unittest", "", "import mylib as m", "", "", "class T(unittest.TestCase):", "    def test_all(self):"]
    tests += ["        self." + s if s.startswith("assert") else "        " + s for s in stmts]
    tests += ["", "", "if __name__ == '__main__':", "    unittest.main()", ""]
    with open(os.path.join(d, "test_mylib.py"), "w") as f:
        f.write("\n".join(tests))
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", help="directory with coding_* task dirs (copied; originals untouched)")
    args = ap.parse_args()
    det_solver.CANON.clear()  # CANON would solve the known names; we want to test the mutation stage alone
    bad = 0
    with tempfile.TemporaryDirectory() as root:
        rows = [(lab, lib, body, fix) for lab, lib, body, fix in CASES]
        rows += [(lab, lib, body, True) for lab, lib, body in MULTI]
        solved_expected = solved = 0
        for k, (lab, lib, body, fix) in enumerate(rows):
            d = make_dir(root, f"c{k:02d}", lib, body)
            t0 = time.time()
            r = det_solver.solve_coding(det_solver._norm(INSTR), d)
            dt = time.time() - t0
            passed = det_solver._run_tests(d)[0]
            got = r.status == "answer" and r.answer == "DONE"
            verdict = "ok" if (got == fix and (not got or passed)) else "MISMATCH"
            if got and not passed:
                verdict = "FALSE-DONE"
            bad += verdict != "ok"
            solved_expected += fix
            solved += got
            print(f"{verdict:10s} {lab:24s} expect_fix={fix!s:5s} got={got!s:5s} {dt:5.1f}s  {r.why[:90]}")
        print(f"\nsolved: {solved} (expected {solved_expected}); mismatches: {bad}")
        if args.real:
            import shutil
            for name in sorted(n for n in os.listdir(args.real) if n.startswith("coding")):
                dst = os.path.join(root, "real_" + name)
                shutil.copytree(os.path.join(args.real, name, "workspace"), dst)
                t = json.load(open(os.path.join(args.real, name, "task.json")))
                r = det_solver.solve_coding(det_solver._norm(t["instruction"]), dst)
                passed = det_solver._run_tests(dst)[0]
                bad += not (r.status == "answer" and passed)
                print(f"real {name}: {r.status} tests_pass={passed}  {r.why[:100]}")
    print("RESULT:", "OK" if bad == 0 else f"{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

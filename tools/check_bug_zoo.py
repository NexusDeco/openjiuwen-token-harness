#!/usr/bin/env python3
"""Offline benchmark (0 tokens) of the coding repair on a hand-written "zoo" of planted bugs.

The real samples (coding_0003 / coding_0007) are small files of 4-6 helper functions from one pool (reverse_words, total,
maximum, average, first_n, count_even) where one or two functions carry a one-token bug. This zoo imitates that: for every
function of a pool of ~20 helpers, several bug variants written by hand, independently of repair.py's mutation table.
Each variant is put into a file together with two correct helpers and tested with a real-style unittest file.

A case counts only if the tests pass on the correct code and fail on the buggy one (so no case is mislabelled).

    python3 tools/check_bug_zoo.py [--no-canon] [--show N] [--min-rate R]

--no-canon  empty det_solver.CANON: only the mutation stage works (what an UNKNOWN function name would get)
--min-rate  exit status 1 if the fixed share is below R (0..1)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("REPAIR_TOTAL_S", "1e9")  # many cases in one process: no global repair cap here
import det_solver  # noqa: E402

TAIL = ("Work only inside the current directory. When you are done, write ONLY the word DONE to the file answer.txt in "
        "the current directory (for example with bash: echo 42 > answer.txt).")
INSTR = det_solver._C_BODY + " " + TAIL


def F(name, src, tests, bugs, pool=False):
    return dict(name=name, src=src, tests=tests, bugs=bugs, pool=pool)


ZOO = [
    F("reverse_words", "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
      ["self.assertEqual(m.reverse_words('c a b'), 'b a c')", "self.assertEqual(m.reverse_words('hello big world'), 'world big hello')"],
      ["def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
       "def reverse_words(s):\n    return ' '.join(s.split())\n",
       "def reverse_words(s):\n    return s[::-1]\n",
       "def reverse_words(s):\n    return ' '.join(reversed(s))\n",
       "def reverse_words(s):\n    words = s.split()\n    words.reverse\n    return ' '.join(words)\n",
       "def reverse_words(s):\n    return ''.join(reversed(s.split()))\n"], pool=True),
    F("total", "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
      ["self.assertEqual(m.total([1, 2, 3]), 6)", "self.assertEqual(m.total([]), 0)", "self.assertEqual(m.total([-4, 10]), 6)"],
      ["def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
       "def total(xs):\n    s = 0\n    for x in xs:\n        s = x\n    return s\n",
       "def total(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s\n",
       "def total(xs):\n    s = 0\n    for x in xs[1:]:\n        s += x\n    return s\n",
       "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s - 1\n",
       "def total(xs):\n    s = 0\n    for x in xs:\n        s *= x\n    return s\n"], pool=True),
    F("maximum", "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
      ["self.assertEqual(m.maximum([3, 9, 2]), 9)", "self.assertEqual(m.maximum([-5, -2, -9]), -2)",
       "self.assertEqual(m.maximum([1, 5, 8]), 8)"],
      ["def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
       "def maximum(xs):\n    best = 0\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
       "def maximum(xs):\n    best = xs[0]\n    for x in xs[:-1]:\n        if x > best:\n            best = x\n    return best\n",
       "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return x\n",
       "def maximum(xs):\n    return xs[0]\n",
       "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = xs[0]\n    return best\n"], pool=True),
    F("average", "def average(xs):\n    return sum(xs) / len(xs)\n",
      ["self.assertEqual(m.average([2, 4, 6]), 4)", "self.assertEqual(m.average([1, 2]), 1.5)"],
      ["def average(xs):\n    return sum(xs) // len(xs)\n", "def average(xs):\n    return sum(xs) / (len(xs) - 1)\n",
       "def average(xs):\n    return sum(xs) / 2\n", "def average(xs):\n    return sum(xs) * len(xs)\n",
       "def average(xs):\n    return sum(xs) / len(xs) + 1\n", "def average(xs):\n    return max(xs) / len(xs)\n"], pool=True),
    F("first_n", "def first_n(xs, n):\n    return xs[:n]\n",
      ["self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])", "self.assertEqual(m.first_n([1, 2], 0), [])"],
      ["def first_n(xs, n):\n    return xs[:n - 1]\n", "def first_n(xs, n):\n    return xs[1:n]\n",
       "def first_n(xs, n):\n    return xs[:n + 1]\n", "def first_n(xs, n):\n    return xs[n:]\n",
       "def first_n(xs, n):\n    return xs[:-n]\n", "def first_n(xs, n):\n    return xs[-n:]\n"], pool=True),
    F("count_even", "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
      ["self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)", "self.assertEqual(m.count_even([]), 0)"],
      ["def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
       "def count_even(xs):\n    return sum(1 for x in xs if x % 3 == 0)\n",
       "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0) + 1\n",
       "def count_even(xs):\n    return len([x for x in xs if x % 2])\n",
       "def count_even(xs):\n    return sum(1 for x in xs if x // 2 == 0)\n",
       "def count_even(xs):\n    return sum(x for x in xs if x % 2 == 0)\n"], pool=True),
    F("minimum", "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
      ["self.assertEqual(m.minimum([3, 9, 2]), 2)", "self.assertEqual(m.minimum([4, 7, 8]), 4)", "self.assertEqual(m.minimum([5, 1, 6]), 1)"],
      ["def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
       "def minimum(xs):\n    best = 0\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
       "def minimum(xs):\n    return xs[0]\n",
       "def minimum(xs):\n    best = xs[0]\n    for x in xs[1:-1]:\n        if x < best:\n            best = x\n    return best\n"]),
    F("last_n", "def last_n(xs, n):\n    return xs[-n:] if n else []\n",
      ["self.assertEqual(m.last_n([5, 6, 7, 8], 2), [7, 8])", "self.assertEqual(m.last_n([1, 2], 0), [])"],
      ["def last_n(xs, n):\n    return xs[:n]\n", "def last_n(xs, n):\n    return xs[-n - 1:] if n else []\n",
       "def last_n(xs, n):\n    return xs[-n + 1:] if n else []\n", "def last_n(xs, n):\n    return xs[len(xs) + n:] if n else []\n"]),
    F("count_odd", "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
      ["self.assertEqual(m.count_odd([1, 2, 3, 5, 6]), 3)", "self.assertEqual(m.count_odd([]), 0)"],
      ["def count_odd(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n", "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 != 1)\n",
       "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 == 1) - 1\n", "def count_odd(xs):\n    return len(xs) - 1\n"]),
    F("product", "def product(xs):\n    p = 1\n    for x in xs:\n        p *= x\n    return p\n",
      ["self.assertEqual(m.product([2, 3, 4]), 24)", "self.assertEqual(m.product([]), 1)"],
      ["def product(xs):\n    p = 0\n    for x in xs:\n        p *= x\n    return p\n",
       "def product(xs):\n    p = 1\n    for x in xs:\n        p += x\n    return p\n",
       "def product(xs):\n    p = 1\n    for x in xs[1:]:\n        p *= x\n    return p\n",
       "def product(xs):\n    p = 1\n    for x in xs:\n        p = x\n    return p\n"]),
    F("is_palindrome", "def is_palindrome(s):\n    return s == s[::-1]\n",
      ["self.assertTrue(m.is_palindrome('abba'))", "self.assertFalse(m.is_palindrome('abc'))", "self.assertTrue(m.is_palindrome(''))"],
      ["def is_palindrome(s):\n    return s != s[::-1]\n", "def is_palindrome(s):\n    return s == s[::1]\n",
       "def is_palindrome(s):\n    return s == s[1:][::-1]\n", "def is_palindrome(s):\n    return s[0] == s[-1]\n"]),
    F("factorial", "def factorial(n):\n    r = 1\n    for i in range(1, n + 1):\n        r *= i\n    return r\n",
      ["self.assertEqual(m.factorial(5), 120)", "self.assertEqual(m.factorial(0), 1)"],
      ["def factorial(n):\n    r = 1\n    for i in range(1, n):\n        r *= i\n    return r\n",
       "def factorial(n):\n    r = 0\n    for i in range(1, n + 1):\n        r *= i\n    return r\n",
       "def factorial(n):\n    r = 1\n    for i in range(1, n + 1):\n        r += i\n    return r\n",
       "def factorial(n):\n    r = 1\n    for i in range(n + 1):\n        r *= i\n    return r\n"]),
    F("fibonacci", "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
      ["self.assertEqual(m.fibonacci(10), 55)", "self.assertEqual(m.fibonacci(0), 0)", "self.assertEqual(m.fibonacci(1), 1)"],
      ["def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return b\n",
       "def fibonacci(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
       "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return a\n",
       "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a - b\n    return a\n"]),
    F("second_largest", "def second_largest(xs):\n    return sorted(set(xs))[-2]\n",
      ["self.assertEqual(m.second_largest([4, 9, 1, 7]), 7)", "self.assertEqual(m.second_largest([5, 9, 9]), 5)"],
      ["def second_largest(xs):\n    return sorted(set(xs))[1]\n", "def second_largest(xs):\n    return sorted(xs)[-2]\n",
       "def second_largest(xs):\n    return sorted(set(xs))[-1]\n",
       "def second_largest(xs):\n    return sorted(set(xs), reverse=True)[-2]\n"]),
    F("count_vowels", "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n",
      ["self.assertEqual(m.count_vowels('Education'), 5)", "self.assertEqual(m.count_vowels('rhythm'), 0)"],
      ["def count_vowels(s):\n    return sum(1 for c in s.lower() if c not in 'aeiou')\n",
       "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
       "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeio')\n",
       "def count_vowels(s):\n    return len(s.lower())\n"]),
    F("clamp", "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
      ["self.assertEqual(m.clamp(5, 0, 3), 3)", "self.assertEqual(m.clamp(-1, 0, 3), 0)", "self.assertEqual(m.clamp(2, 0, 3), 2)"],
      ["def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n", "def clamp(x, lo, hi):\n    return max(lo, min(x, lo))\n",
       "def clamp(x, lo, hi):\n    return min(hi, max(x, hi))\n", "def clamp(x, lo, hi):\n    return max(hi, min(x, lo))\n"]),
    F("gcd", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
      ["self.assertEqual(m.gcd(12, 18), 6)", "self.assertEqual(m.gcd(7, 5), 1)"],
      ["def gcd(a, b):\n    while b:\n        a, b = b, a // b\n    return a\n", "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return b\n",
       "def gcd(a, b):\n    while a:\n        a, b = b, a % b\n    return a\n", "def gcd(a, b):\n    while b:\n        a, b = a, a % b\n    return a\n"]),
    F("is_prime", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n",
      ["self.assertTrue(m.is_prime(2))", "self.assertTrue(m.is_prime(13))", "self.assertFalse(m.is_prime(15))", "self.assertFalse(m.is_prime(1))"],
      ["def is_prime(n):\n    if n < 3:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n",
       "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i != 0:\n            return False\n    return True\n",
       "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(1, n):\n        if n % i == 0:\n            return False\n    return True\n",
       "def is_prime(n):\n    if n < 2:\n        return True\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n"]),
    F("capitalize_words", "def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())\n",
      ["self.assertEqual(m.capitalize_words('ab cd'), 'Ab Cd')", "self.assertEqual(m.capitalize_words('hello'), 'Hello')"],
      ["def capitalize_words(s):\n    return ' '.join(w.upper() for w in s.split())\n", "def capitalize_words(s):\n    return ' '.join(w for w in s.split())\n",
       "def capitalize_words(s):\n    return ''.join(w.capitalize() for w in s.split())\n", "def capitalize_words(s):\n    return s.capitalize()\n"]),
    F("unique", "def unique(xs):\n    seen = []\n    for x in xs:\n        if x not in seen:\n            seen.append(x)\n    return seen\n",
      ["self.assertEqual(m.unique([3, 1, 3, 2, 1]), [3, 1, 2])", "self.assertEqual(m.unique([]), [])"],
      ["def unique(xs):\n    seen = []\n    for x in xs:\n        if x in seen:\n            seen.append(x)\n    return seen\n",
       "def unique(xs):\n    return list(set(xs))\n",
       "def unique(xs):\n    seen = []\n    for x in xs:\n        if x not in seen:\n            seen.append(x)\n    return xs\n",
       "def unique(xs):\n    seen = []\n    for x in xs[1:]:\n        if x not in seen:\n            seen.append(x)\n    return seen\n"]),
]


def build_file(rng, target, bug_src, others):
    """mylib source and test source: the buggy function plus two correct helpers (like the real files)."""
    picks = [target] + rng.sample([o for o in others if o["name"] != target["name"]], 2)
    rng.shuffle(picks)
    lib, tests = [], []
    for f in picks:
        lib.append(bug_src if f is target else f["src"])
        tests.append((f["name"], f["tests"]))
    body = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n"
    for name, ts in tests:
        body += f"    def test_{name}(self):\n" + "".join(f"        {t}\n" for t in ts) + "\n"
    body += "\nif __name__ == '__main__':\n    unittest.main()\n"
    return "\n\n".join(lib), body


def write(root, name, lib, tests):
    d = os.path.join(root, name)
    os.makedirs(d)
    open(os.path.join(d, "mylib.py"), "w").write(lib)
    open(os.path.join(d, "test_mylib.py"), "w").write(tests)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-canon", action="store_true")
    ap.add_argument("--show", type=int, default=40)
    ap.add_argument("--min-rate", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    if a.no_canon:
        det_solver.CANON.clear()
    rng = random.Random(a.seed)
    fixed = total = 0
    unfixed, slow, skipped = [], [], []
    with tempfile.TemporaryDirectory() as root:
        k = 0
        for f in ZOO:
            for bug in f["bugs"]:
                lib, tests = build_file(rng, f, bug, ZOO)
                d_ok = write(root, f"ok{k}", lib.replace(bug, f["src"]), tests)
                d = write(root, f"c{k}", lib, tests)
                k += 1
                good = det_solver._run_tests(d_ok)[0]
                bad = not det_solver._run_tests(d)[0]
                if not (good and bad):  # a mistake in the zoo itself, not a solver result
                    skipped.append((f["name"], bug.strip().splitlines()[-1].strip()[:60], "tests-pass-on-correct" if good else "correct-fails"))
                    continue
                t0 = time.time()
                r = det_solver.solve_coding(det_solver._norm(INSTR), d)
                dt = time.time() - t0
                total += 1
                ok = r.status == "answer" and det_solver._run_tests(d)[0]
                fixed += ok
                if dt > 5:
                    slow.append((f["name"], round(dt, 1)))
                if not ok:
                    unfixed.append((f["name"], [ln.strip() for ln in bug.splitlines()[1:]]))
    rate = fixed / total if total else 0
    print(f"mode: {'mutation only (CANON empty)' if a.no_canon else 'full flow (CANON + mutation)'}; cases: {total}; "
          f"fixed: {fixed} ({rate:.0%}); not fixed (would go to the LLM): {len(unfixed)}; zoo mistakes skipped: {len(skipped)}")
    for n, b in unfixed[:a.show]:
        print(f"   NOT FIXED {n:18s} {' | '.join(b)[:110]}")
    for s in skipped:
        print("   SKIPPED (zoo mistake):", s)
    if slow:
        print("   slow cases (>5 s):", slow)
    return 1 if rate < a.min_rate else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Edge-case regression tests for the no-LLM detectors (zero tokens, standard library only).

    python tools/check_edge_cases.py

The cost of a false "unsolvable" is a lost task (several points), the cost of a refusal is some tokens, so these cases
pin down both directions: what MUST be detected and, more importantly, what must NOT be (stateful functions,
decorated or conditional tests, records in an unfamiliar format, free-text mentions, vague-looking but valid numbers,
reworded boilerplate that is still the familiar sentence vs. one that names another file or adds a condition).
Exit code 1 if any case fails.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from det_solver import solve_data, solve_math, solve_search  # noqa: E402
from prechecks import contradictory_tests, failing_assertions, pure_functions  # noqa: E402
from solver import answer_ready, normalize_answer  # noqa: E402
from tasks import Task, missing_files  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"   [{detail}]" if detail and not ok else ""))


# ---------------------------------------------------------------- tests that cannot pass (prechecks)
print("== contradictory tests / purity")
H = "import unittest\nimport mylib as m\nfrom mylib import total as tot\n\nclass T(unittest.TestCase):\n"
PURE = "def total(xs):\n    return sum(xs)\n\ndef bump(n):\n    return n + 1\n"
STATEFUL = "_c = 0\n\ndef bump(n):\n    global _c\n    _c += n\n    return _c\n\ndef total(xs):\n    return sum(xs)\n"
RANDOM = "import random\n\ndef roll(n):\n    return random.randint(1, n)\n"
MUTDEF = "def push(x, acc=[]):\n    acc.append(x)\n    return len(acc)\n"
CALLSIMP = "_s = 0\n\ndef inc():\n    global _s\n    _s += 1\n    return _s\n\ndef wrapper(n):\n    return inc() + n\n"
SAME = "    def test_a(self):\n        self.assertEqual(m.{f}(1), 2)\n        self.assertEqual(m.{f}(1), 3)\n"
CROSS = "    def test_a(self):\n        self.assertEqual(m.{f}(1), 2)\n    def test_b(self):\n        self.assertEqual(m.{f}(1), 3)\n"
for name, tsrc, lsrc, want in [
    ("same method, pure function -> contradiction", H + SAME.format(f="bump"), PURE, True),
    ("same method, stateful (global counter) -> not flagged", H + SAME.format(f="bump"), STATEFUL, False),
    ("same method, mylib.py unknown -> not flagged", H + SAME.format(f="bump"), None, False),
    ("cross method, pure -> contradiction", H + CROSS.format(f="bump"), PURE, True),
    ("cross method, stateful -> not flagged", H + CROSS.format(f="bump"), STATEFUL, False),
    ("cross method, mylib.py unknown -> flagged", H + CROSS.format(f="bump"), None, True),
    ("random-based function -> not flagged", H + SAME.format(f="roll"), RANDOM, False),
    ("mutable default argument -> not flagged", H + SAME.format(f="push"), MUTDEF, False),
    ("wrapper around a stateful function -> not flagged", H + SAME.format(f="wrapper"), CALLSIMP, False),
    ("same call, same expected twice -> fine", H + "    def test_a(self):\n        self.assertEqual(m.bump(1), 2)\n        self.assertEqual(m.bump(1), 2)\n", PURE, False),
    ("different arguments -> fine", H + "    def test_a(self):\n        self.assertEqual(m.bump(1), 2)\n        self.assertEqual(m.bump(2), 3)\n", PURE, False),
    ("zero-argument calls are never compared", H + "    def test_a(self):\n        self.assertEqual(m.bump(), 1)\n        self.assertEqual(m.bump(), 2)\n", PURE, False),
]:
    check(name, bool(contradictory_tests(tsrc, lsrc)) == want)
check("pure_functions: pure / stateful / wrapper of stateful",
      pure_functions(PURE) == {"bump", "total"} and pure_functions(STATEFUL) == {"total"} and pure_functions(CALLSIMP) == set())
print("== always-failing assertions")
for name, body, want in [
    ("assertEqual(1, 2)", "self.assertEqual(1, 2)", True), ("assertEqual(2, 2) is fine", "self.assertEqual(2, 2)", False),
    ("assertTrue(False)", "self.assertTrue(False)", True), ("assertTrue(True) is fine", "self.assertTrue(True)", False),
    ("assertFalse(True)", "self.assertFalse(True)", True), ("self.fail()", "self.fail('x')", True),
    ("inside assertRaises -> ignored", "with self.assertRaises(ValueError):\n            self.assertEqual(1, 2)", False),
    ("inside if -> ignored", "if False:\n            self.fail()", False),
    ("a call compared with a literal -> not decidable", "self.assertEqual(m.total([1]), 2)", False),
]:
    check(name, bool(failing_assertions(H + "    def test_a(self):\n        " + body + "\n")) == want)
for deco, name in (("@unittest.expectedFailure", "expectedFailure"), ("@unittest.skip('x')", "skip")):
    check(f"@{name} -> ignored", not failing_assertions(H + f"    {deco}\n    def test_a(self):\n        self.assertEqual(1, 2)\n"))

# ---------------------------------------------------------------- math
print("== math quantities and boilerplate")
SFX = ("Work only inside the current directory. When you are done, write ONLY the final answer to the file "
       "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
Q = "How many boxes does it have at the end?"


def math(story, tail=SFX, hint="Compute it with the bash tool (python3 -c)."):
    return solve_math(" ".join(x for x in (story, Q, hint, tail) if x))


A = "A warehouse starts with 100 boxes."
for name, story, status, ans in [
    ("number words", A + " It ships twenty boxes per day for three days.", "answer", "40"),
    ("hyphenated number words", A + " It ships twenty-five boxes per day for two days.", "answer", "50"),
    ("split by a word", A + " Then the stock is split equally into four warehouses and it keeps one share (rounded down).", "answer", "25"),
    ("one hundred", A + " It receives one hundred deliveries of 2 boxes each.", "answer", "300"),
    ("'several' is vague", A + " It ships several boxes per day for 3 days.", "unsolvable", None),
    ("'some days' is vague", A + " It ships 5 boxes per day for some days.", "unsolvable", None),
    ("an unknown number of deliveries", A + " It receives an unknown number of deliveries of 9 boxes each.", "unsolvable", None),
    ("'a dozen' is not vague and not parsed -> LLM", A + " It ships a dozen boxes per day for 3 days.", "nomatch", None),
    ("an unknown extra sentence -> LLM", A + " It ships 5 boxes per day for 3 days. The warehouse burns down.", "nomatch", None),
    ("split by zero -> LLM", A + " Then the stock is split equally into 0 warehouses and it keeps one share (rounded down).", "nomatch", None),
]:
    r = math(story)
    check(name, r.status == status and (ans is None or r.answer == ans), f"{r.status} {r.answer} {r.why}")
T = "Work only inside the current directory. When you are done, write ONLY the final answer to the file answer.txt"
S5 = A + " It ships 5 boxes per day for 3 days."
for name, kw, status in [
    ("tail 'next to task.json' + reworded hint", dict(tail=T + " next to task.json.", hint="Compute it with python3."), "answer"),
    ("tail without the bash example, no hint", dict(tail=T + " in the current directory.", hint=""), "answer"),
    ("a path in the tail -> LLM", dict(tail=T + " in /tmp/out.", hint=""), "nomatch"),
    ("another file name -> LLM", dict(tail="Work only inside the current directory. When you are done, write ONLY the final answer to the file result.txt.", hint=""), "nomatch"),
    ("an extra condition instead of a hint -> LLM", dict(tail=T + " next to task.json.", hint="Assume 3 boxes are lost."), "nomatch"),
    ("a number inside a hint -> LLM", dict(tail=T + " next to task.json.", hint="Compute it in 5 steps."), "nomatch"),
    ("a coding-style 'word DONE' tail on a math task -> LLM", dict(tail="Work only inside the current directory. When you are done, write ONLY the word DONE to the file answer.txt.", hint=""), "nomatch"),
]:
    check(name, math(S5, **kw).status == status)

# ---------------------------------------------------------------- search / data
print("== search records")
TS = ("The directory kb/ contains a small knowledge base of markdown files. {q} Use the file tools or bash (ls, grep, cat) to find it. " + SFX)


def kb(files):
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "kb"))
    for n, c in files.items():
        with open(os.path.join(d, "kb", n), "w") as f:
            f.write(c)
    return d


def P(name, born=1980, emp="Acme", mentor="Bob Roe", drop=None):
    lines = [f"Born: {born}", f"Employer: {emp}", f"Mentor: {mentor}"]
    return f"# {name}\n\n" + "\n".join(ln for ln in lines if not (drop and ln.startswith(drop))) + "\n"


def C(name, hq="Oslo", f=1990, drop=None):
    lines = [f"Headquarters: {hq}", f"Founded: {f}"]
    return f"# {name}\n\n" + "\n".join(ln for ln in lines if not (drop and ln.startswith(drop))) + "\n"


base = {"person_alice_smith.md": P("Alice Smith"), "person_bob_roe.md": P("Bob Roe", 1970, "Acme", "Alice Smith"),
        "company_acme.md": C("Acme")}
QB, QH = "In which year was Alice Smith born?", "In which city is the headquarters of Alice Smith's employer?"
for name, files, q, status, ans in [
    ("complete record", base, QB, "answer", "1980"),
    ("record lacks Born -> unsolvable", {**base, "person_alice_smith.md": P("Alice Smith", drop="Born")}, QB, "unsolvable", None),
    ("lacks Born but a free-text note mentions the person -> LLM", {**base, "person_alice_smith.md": P("Alice Smith", drop="Born"), "notes.md": "# Notes\n\nAlice Smith was born in 1981.\n"}, QB, "nomatch", None),
    ("company lacks Headquarters -> unsolvable", {**base, "company_acme.md": C("Acme", drop="Headquarters")}, QH, "unsolvable", None),
    ("company lacks Founded, HQ asked -> still answers", {**base, "company_acme.md": C("Acme", drop="Founded")}, QH, "answer", "Oslo"),
    ("two records with the same name -> LLM", {**base, "person_alice_smith2.md": P("Alice Smith", 1999)}, QB, "nomatch", None),
    ("person known only as somebody's mentor -> LLM", {k: v for k, v in base.items() if k != "person_alice_smith.md"}, QB, "nomatch", None),
    ("name nowhere in the kb -> unsolvable", base, "In which year was Zoltan Quimby born?", "unsolvable", None),
    ("name only in a free-text note -> LLM", {**base, "notes.md": "# Notes\n\nZoltan Quimby was born in 1981.\n"}, "In which year was Zoltan Quimby born?", "nomatch", None),
    ("mentor's name comes from the record", base, "Who is the mentor of Bob Roe?", "answer", "Alice Smith"),
    ("record under the right title in an unfamiliar schema -> LLM", {"person_alice_smith.md": "# Alice Smith\n\nShe is a person.\n"}, QB, "nomatch", None),
    ("company title exists in another format -> LLM", {**base, "company_acme.md": "# Acme\n\nA company in Oslo.\n"}, QH, "nomatch", None),
    ("company file missing, only Employer: lines mention it -> unsolvable", {k: v for k, v in base.items() if k != "company_acme.md"}, QH, "unsolvable", None),
]:
    r = solve_search(TS.format(q=q), kb(files))
    check(name, r.status == status and (ans is None or r.answer == ans), f"{r.status} {r.answer} {r.why}")

print("== data records")
TD = ("The file sales.csv has columns {cols} ({n} rows). {q} Use bash with python3 to compute it. " + SFX)
H6 = "order_id,region,product,month,units,price"
ROWS = ["1,west,widget,7,10,5", "2,east,widget,8,20,6", "3,west,gizmo,7,5,9"]
QU = "How many units of product 'widget' were sold in months 7 through 8 inclusive?"


def data(name, header, rows, cols_in_text, q, status, ans=None):
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "sales.csv"), "w") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")
    r = solve_data(TD.format(cols=cols_in_text, n=len(rows), q=q), d)
    check(name, r.status == status and (ans is None or r.answer == ans), f"{r.status} {r.answer} {r.why}")


data("standard question", H6, ROWS, H6, QU, "answer", "30")
data("question needs units, file lacks it -> unsolvable", "order_id,region,product,month,price", ["1,west,widget,7,5"], "order_id,region,product,month,price", QU, "unsolvable")
data("revenue needs price, file lacks it -> unsolvable", "order_id,region,product,month,units", ["1,west,widget,7,5"], "order_id,region,product,month,units", "What is the total revenue (units * price) of product 'widget' in region 'west'?", "unsolvable")
data("instruction header differs from the file -> LLM", H6, ROWS, "order_id,region,product,month,units", QU, "nomatch")
data("total discount (not a column) -> unsolvable", H6, ROWS, H6, "What is the total discount given on orders from region 'west'?", "unsolvable")
data("total cost (ambiguous, may be units*price) -> LLM", H6, ROWS, H6, "What is the total cost of orders from region 'west'?", "nomatch")
data("a product named like a concept, quoted -> answer", H6, ["1,west,tax,7,10,5"], H6, "How many units of product 'tax' were sold in months 7 through 8 inclusive?", "answer", "10")
data("discount IS a column and the question is unknown -> LLM", H6 + ",discount", ["1,west,widget,7,10,5,2"], H6 + ",discount", "What is the total discount given on orders from region 'west'?", "nomatch")

# ---------------------------------------------------------------- small helpers
print("== helpers")
mk = lambda instr, ws: Task("t", "/x", "math", instr, 1.0, ws)  # noqa: E731
d = tempfile.mkdtemp()
open(os.path.join(d, "sales.csv"), "w").close()
os.makedirs(os.path.join(d, "kb"))
check("missing_files ignores answer.txt, task.json and workspace/", missing_files(mk("Write the answer to answer.txt next to task.json. Work inside workspace/.", d)) == [])
check("missing_files still flags a really missing file", missing_files(mk("Read missing.csv and sales.csv please.", d)) == ["missing.csv"])
check("missing_files accepts present file and kb/", missing_files(mk("The file sales.csv and the directory kb/ are here.", d)) == [])
check("normalize_answer", [normalize_answer(x) for x in ("680.0", " 42 \n", "-3.00", "3.5", "Osaka", "1,234.0", None)] == ["680", "42", "-3", "3.5", "Osaka", "1,234.0", None])
task = tempfile.mkdtemp()
ws = os.path.join(task, "workspace")
os.makedirs(ws)
check("answer_ready: nothing written", not answer_ready(ws, "", task))
with open(os.path.join(task, "answer.txt"), "w") as f:
    f.write("42\n")
check("answer_ready: answer next to task.json counts", answer_ready(ws, "", task))
check("answer_ready: ... but not when that directory is not passed", not answer_ready(ws, ""))
with open(os.path.join(ws, "answer.txt"), "w") as f:
    f.write("42\n")
check("answer_ready: a command that failed ([exit 1]) does not count", not answer_ready(ws, "Traceback\n[exit 1]", task))
check("answer_ready: a timeout does not count", not answer_ready(ws, "[timeout after 30s]", task))

failed = RESULTS.count(False)
print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} edge cases pass" + ("" if not failed else f"  ({failed} FAILED)"))
sys.exit(1 if failed else 0)

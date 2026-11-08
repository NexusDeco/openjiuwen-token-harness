#!/usr/bin/env python3
"""Synthetic task generator that mimics the organisers' SAMPLE formats (math / search / data_analysis / coding).

    python tools/gen_tasks.py --out <tasks_dir> --grader <grader.json> [--seed 20261107]

* `<tasks_dir>/<task_id>/task.json` holds only {task_id, task_type, instruction, reward} (no answers), plus an
  optional `workspace/` exactly like the samples.
* `<grader.json>` holds what the agent must never see: expected answer / check type, whether the task is solvable,
  and for coding tasks the sha256 of the original test file (so edited tests can be detected).

Solvable tasks follow the two templates seen per type in the samples (a few extra "probe" variants that are NOT in
the samples are marked `variant: "probe"`). The unsolvable ones are GUESSES about what unsolvable tasks look like:
person absent from kb, employer file missing, nonexistent CSV column, contradictory tests, under-determined math.
`--extended` appends more cases (used by tools/check_det.py, the default 40 tasks and their seed are unchanged):
numbers written as words, instructions with reworded boilerplate (`variant: "boilerplate"`), and more unsolvable kinds:
a vague quantity in the story, a CSV without the column the question needs, a kb record lacking the asked field,
an always-failing test.
Standard library only; every coding task is self-checked (tests fail before the fix and pass after it).
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile

SUFFIX = ("Work only inside the current directory. When you are done, write ONLY the final answer to the file "
          "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
SUFFIX_DONE = ("Work only inside the current directory. When you are done, write ONLY the word DONE to the file "
               "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
SEARCH_TAIL = "Use the file tools or bash (ls, grep, cat) to find it."

FIRST = ["alice", "boris", "chen", "dmitri", "farid", "grace", "hiro", "irina", "jonas", "kira", "maya", "nikolai",
         "olga", "pavel", "quinn", "rosa", "sergei", "tanya"]
LAST = ["garcia", "haddad", "ivanova", "kowalski", "muller", "novak", "petrov", "sato", "smith", "wang"]
COMPANIES = ["Fjord Data", "Dune Robotics", "Arktika Labs", "Cobalt Works", "Iris Optics", "Bluefin Systems",
             "Ember Bio", "Juniper Foods", "Helix Motors", "Granite AI"]
CITIES = ["Quito", "Osaka", "Graz", "Lyon", "Oslo", "Porto", "Tartu", "Bern", "Split", "Gdansk", "Cork", "Turku"]
ABSENT_FIRST = ["zoltan", "ulrich", "viktor", "wanda", "xenia"]
ABSENT_LAST = ["quimby", "yamada", "zielinski", "voss", "abbott"]
PRODUCTS = ["widget", "sprocket", "gizmo", "gadget"]
REGIONS = ["west", "north", "east", "south"]


def title(name):
    return " ".join(p.capitalize() for p in name.split("_"))


# ---------------------------------------------------------------- math
def math_chain(rng):
    n = rng.randint(500, 3000)
    text, val = f"A warehouse starts with {n} boxes. ", n
    for i in range(rng.randint(3, 6)):
        if i == 0 or rng.random() < 0.55:
            k = rng.choice([2, 3, 4, 5])
            text += f"Then the stock is split equally into {k} warehouses and it keeps one share (rounded down). "
            val //= k
        else:
            text += "Then the stock is doubled by merging with a sister warehouse. "
            val *= 2
    text += "How many boxes does it have at the end? Compute it with the bash tool (python3 -c). " + SUFFIX
    return text, str(val)


def math_simple(rng):
    n, s, d, r, q = rng.randint(800, 2500), rng.randint(10, 60), rng.randint(2, 6), rng.randint(1, 4), rng.randint(40, 120)
    text = (f"A warehouse starts with {n} boxes. It ships {s} boxes per day for {d} days. It receives {r} deliveries of "
            f"{q} boxes each. How many boxes does it have at the end? Compute it with the bash tool (python3 -c). "
            + SUFFIX)
    return text, str(n - s * d + r * q)


_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 10: "ten", 12: "twelve", 15: "fifteen", 20: "twenty"}


def math_words(rng):
    """Same story, quantities spelled as words (a solvable variant the samples do not show)."""
    n = rng.randint(500, 3000)
    text, val = f"A warehouse starts with {n} boxes. ", n
    for i in range(rng.randint(3, 5)):
        r = rng.random()
        if i == 0 or r < 0.35:
            k = rng.choice([2, 3, 4, 5])
            text += f"Then the stock is split equally into {_WORDS[k]} warehouses and it keeps one share (rounded down). "
            val //= k
        elif r < 0.55:
            text += "Then the stock is doubled by merging with a sister warehouse. "
            val *= 2
        elif r < 0.8:
            s, d = rng.choice([10, 12, 15, 20]), rng.choice([2, 3, 4, 5, 6])
            text += f"It ships {_WORDS[s]} boxes per day for {_WORDS[d]} days. "
            val -= s * d
        else:
            kd, q = rng.choice([2, 3, 4]), rng.randint(40, 120)
            text += f"It receives {_WORDS[kd]} deliveries of {q} boxes each. "
            val += kd * q
    text += "How many boxes does it have at the end? Compute it with the bash tool (python3 -c). " + SUFFIX
    return text, str(val)


def math_vague(rng):
    """A known sentence whose quantity is not a number: the answer cannot be computed."""
    n, d, r, q = rng.randint(800, 2500), rng.randint(2, 6), rng.randint(1, 4), rng.randint(40, 120)
    vague = rng.choice(["some", "several", "a few", "an unknown number of", "an unspecified number of"])
    if rng.random() < 0.5:
        ships, recv = f"It ships {vague} boxes per day for {d} days.", f"It receives {r} deliveries of {q} boxes each."
    else:
        ships, recv = f"It ships {rng.randint(10, 60)} boxes per day for {d} days.", f"It receives {vague} deliveries of {q} boxes each."
    return (f"A warehouse starts with {n} boxes. {ships} {recv} How many boxes does it have at the end? Compute it with "
            f"the bash tool (python3 -c). " + SUFFIX)


def reword_boiler(instr):
    """Cosmetic rewording of the boilerplate around the question (tail location, hint sentences)."""
    for old, new in ((SUFFIX, "Work only inside the current directory. When you are done, write ONLY the final answer to the "
                              "file answer.txt next to task.json."),
                     (SUFFIX_DONE, "Work only inside the current directory. When you are done, write ONLY the word DONE to "
                                   "the file answer.txt next to task.json.")):
        instr = instr.replace(old, new)
    for old, new in (("Compute it with the bash tool (python3 -c).", "Calculate it with python3."),
                     ("Use bash with python3 to compute it.", "Use python3 to calculate it."),
                     (SEARCH_TAIL, "Use grep or cat to find it.")):
        instr = instr.replace(old, new)
    return instr


def math_underdetermined(rng):
    n, d, r, q = rng.randint(800, 2500), rng.randint(2, 6), rng.randint(1, 4), rng.randint(40, 120)
    return (f"A warehouse starts with {n} boxes. It ships an unknown number of boxes per day for {d} days. It receives "
            f"{r} deliveries of {q} boxes each. How many boxes does it have at the end? Compute it with the bash tool "
            f"(python3 -c). " + SUFFIX)


# ---------------------------------------------------------------- search
def make_kb(rng):
    names = set()
    while len(names) < rng.randint(26, 28):
        names.add(f"{rng.choice(FIRST)}_{rng.choice(LAST)}")
    names = sorted(names)
    comp = {c: {"hq": rng.choice(CITIES), "founded": rng.randint(1950, 2015)} for c in COMPANIES}
    persons = {}
    for nm in names:
        persons[nm] = {"born": rng.randint(1950, 2005), "employer": rng.choice(COMPANIES),
                       "mentor": rng.choice([x for x in names if x != nm])}
    return persons, comp


def write_kb(ws, persons, comp, skip_company=None, drop_field=None):
    """drop_field = ("person", key, "Born") or ("company", name, "Headquarters"): leave that one line out."""
    kb = os.path.join(ws, "kb")
    os.makedirs(kb)
    drop = drop_field or (None, None, None)
    for nm, p in persons.items():
        lines = [f"Born: {p['born']}", f"Employer: {p['employer']}", f"Mentor: {title(p['mentor'])}"]
        if drop[0] == "person" and drop[1] == nm:
            lines = [ln for ln in lines if not ln.startswith(drop[2] + ":")]
        with open(os.path.join(kb, f"person_{nm}.md"), "w") as f:
            f.write(f"# {title(nm)}\n\n" + "\n".join(lines) + "\n")
    for c, v in comp.items():
        if c == skip_company:
            continue
        lines = [f"Headquarters: {v['hq']}", f"Founded: {v['founded']}"]
        if drop[0] == "company" and drop[1] == c:
            lines = [ln for ln in lines if not ln.startswith(drop[2] + ":")]
        with open(os.path.join(kb, f"company_{c.lower().replace(' ', '_')}.md"), "w") as f:
            f.write(f"# {c}\n\n" + "\n".join(lines) + "\n")


def search_task(rng, kind):
    persons, comp = make_kb(rng)
    nm = rng.choice(sorted(persons))
    p = persons[nm]
    q, exp, skip, drop_field = None, None, None, None
    if kind == "birth":
        q, exp = f"In which year was {title(nm)} born?", str(p["born"])
    elif kind == "hq":
        q, exp = f"In which city is the headquarters of {title(nm)}'s employer?", comp[p["employer"]]["hq"]
    elif kind == "mentor_born":  # probe: not in the samples
        mn = p["mentor"]  # (stored as the person's key)
        q, exp = f"In which year was {title(nm)}'s mentor born?", str(persons[mn]["born"])
    elif kind == "mentor_hq":  # probe: three hops
        mn = p["mentor"]
        q, exp = (f"In which city is the headquarters of the employer of {title(nm)}'s mentor?",
                  comp[persons[mn]["employer"]]["hq"])
    elif kind == "absent":
        while True:
            fake = f"{rng.choice(ABSENT_FIRST)}_{rng.choice(ABSENT_LAST)}"
            if fake not in persons:
                break
        q, exp = f"In which year was {title(fake)} born?", None
    elif kind == "employer_missing":
        q, exp, skip = f"In which city is the headquarters of {title(nm)}'s employer?", None, p["employer"]
    elif kind == "field_missing":  # the person's record exists but has no Born line
        q, exp, drop_field = f"In which year was {title(nm)} born?", None, ("person", nm, "Born")
    elif kind == "company_field_missing":  # the employer's record exists but has no Headquarters line
        q, exp, drop_field = (f"In which city is the headquarters of {title(nm)}'s employer?", None,
                              ("company", p["employer"], "Headquarters"))
    text = f"The directory kb/ contains a small knowledge base of markdown files. {q} {SEARCH_TAIL} {SUFFIX}"
    return text, exp, (persons, comp, skip, drop_field)


# ---------------------------------------------------------------- data analysis
def write_sales(ws, rng, n, drop=None):
    """sales.csv; `drop` names one column to leave out (header and rows)."""
    cols = ["order_id", "region", "product", "month", "units", "price"]
    rows = [",".join(c for c in cols if c != drop)]
    full = []
    for i in range(1, n + 1):
        rec = dict(zip(cols, [i, rng.choice(REGIONS), rng.choice(PRODUCTS), rng.randint(1, 12), rng.randint(1, 45), rng.randint(10, 99)]))
        full.append([str(rec[c]) for c in cols])
        rows.append(",".join(str(rec[c]) for c in cols if c != drop))
    with open(os.path.join(ws, "sales.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    return full


def data_task(rng, ws, kind):
    n = rng.choice([20, 60, 150, 300])
    if kind == "missing_units_column":  # the file really lacks the column the question is about
        write_sales(ws, rng, n, drop="units")
        prod, a = rng.choice(PRODUCTS), rng.randint(1, 11)
        b = min(12, a + rng.randint(1, 3))
        return (f"The file sales.csv has columns order_id,region,product,month,price ({n} rows). How many units of product "
                f"'{prod}' were sold in months {a} through {b} inclusive? Use bash with python3 to compute it. {SUFFIX}"), None
    rows = write_sales(ws, rng, n)
    head = f"The file sales.csv has columns order_id,region,product,month,units,price ({n} rows). "
    if kind == "units":
        prod, a = rng.choice(PRODUCTS), rng.randint(1, 11)
        b = min(12, a + rng.randint(1, 3))  # always two or more distinct months, like the samples
        exp = sum(int(r[4]) for r in rows if r[2] == prod and a <= int(r[3]) <= b)
        q = f"How many units of product '{prod}' were sold in months {a} through {b} inclusive?"
        return f"{head}{q} Use bash with python3 to compute it. {SUFFIX}", str(exp)
    if kind == "revenue":  # probe: not in the samples
        prod, reg = rng.choice(PRODUCTS), rng.choice(REGIONS)
        exp = sum(int(r[4]) * int(r[5]) for r in rows if r[2] == prod and r[1] == reg)
        q = f"What is the total revenue (units * price) of product '{prod}' in region '{reg}'?"
        return f"{head}{q} Use bash with python3 to compute it. {SUFFIX}", str(exp)
    col = rng.choice(["discount", "tax", "cost", "profit"])  # missing_column
    reg = rng.choice(REGIONS)
    return (f"{head}What is the total {col} given on orders from region '{reg}'? Use bash with python3 to compute it. "
            f"{SUFFIX}"), None


# ---------------------------------------------------------------- coding
def _words(rng):
    return " ".join(rng.sample(["a", "b", "c", "d", "e", "f"], rng.choice([3, 4])))


BANK = {
    "reverse_words": dict(
        correct="def reverse_words(s):\n    return ' '.join(reversed(s.split()))",
        bugs=["def reverse_words(s):\n    return ' '.join(sorted(s.split()))",
              "def reverse_words(s):\n    return ' '.join(s.split())"],
        args=lambda rng: (_words(rng),)),
    "total": dict(
        correct="def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s",
        bugs=["def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s",
              "def total(xs):\n    s = 1\n    for x in xs:\n        s += x\n    return s"],
        args=lambda rng: ([rng.randint(1, 9) for _ in range(rng.choice([3, 4]))],)),
    "maximum": dict(
        correct="def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best",
        bugs=["def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best",
              "def maximum(xs):\n    return xs[-1]"],
        args=lambda rng: (rng.sample(range(1, 20), 3),)),
    "average": dict(
        correct="def average(xs):\n    return sum(xs) / len(xs)",
        bugs=["def average(xs):\n    return sum(xs) / (len(xs) + 1)"],
        args=lambda rng: ((lambda b, d: [b - d, b, b + d])(rng.randint(3, 20), rng.randint(1, 3)),)),
    "first_n": dict(
        correct="def first_n(xs, n):\n    return xs[:n]",
        bugs=["def first_n(xs, n):\n    return xs[:n + 1]", "def first_n(xs, n):\n    return xs[1:n]"],
        args=lambda rng: (rng.sample(range(1, 30), 4), rng.choice([2, 3]))),
    "count_even": dict(
        correct="def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)",
        bugs=["def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)"],
        args=lambda rng: (rng.sample(range(1, 30), rng.choice([5, 6])),)),
}


def _run(src, name, args):
    ns = {}
    exec(src, ns)
    return ns[name](*args)


def coding_task(rng, ws, kind):
    names = rng.sample(sorted(BANK), rng.randint(4, 6))
    n_bugs = rng.randint(1, min(3, len(names)))
    buggy = set(rng.sample(names, n_bugs))
    srcs, fixed, tests = {}, {}, []
    for nm in names:
        spec = BANK[nm]
        fixed[nm] = spec["correct"]
        bug = rng.choice(spec["bugs"]) if nm in buggy else None
        srcs[nm] = bug or spec["correct"]
        for _ in range(200):
            args = spec["args"](rng)
            exp = _run(spec["correct"], nm, args)
            if bug is None or _run(bug, nm, args) != exp:
                break
        else:
            raise RuntimeError(f"could not find a distinguishing test for {nm}")
        call = f"m.{nm}({', '.join(repr(a) for a in args)})"
        tests.append((nm, call, exp))
    order = list(names)
    rng.shuffle(order)
    t_lines = ["import unittest", "", "import mylib as m", "", "", "class T(unittest.TestCase):"]
    contradiction = None
    for nm, call, exp in tests:
        t_lines += [f"    def test_{nm}(self):", f"        self.assertEqual({call}, {exp!r})", ""]
    if kind == "contradictory_cross":      # caught by the planner's AST precheck
        nm, call, exp = tests[0]
        alt = exp + 1 if isinstance(exp, (int, float)) else (exp + "x" if isinstance(exp, str) else exp + [0])
        t_lines += [f"    def test_{nm}_alt(self):", f"        self.assertEqual({call}, {alt!r})", ""]
        contradiction = "cross-method"
    elif kind == "always_fail":            # fails whatever mylib.py says
        t_lines += ["    def test_impossible(self):", "        self.assertEqual(1, 2)", ""]
        contradiction = "always-fail"
    elif kind == "contradictory_same":     # same test method (caught only when mylib.py shows the function is pure)
        nm, call, exp = tests[0]
        alt = exp + 1 if isinstance(exp, (int, float)) else (exp + "x" if isinstance(exp, str) else exp + [0])
        i = t_lines.index(f"        self.assertEqual({call}, {exp!r})")
        t_lines.insert(i + 1, f"        self.assertEqual({call}, {alt!r})")
        contradiction = "same-method"
    t_lines += ["", "if __name__ == '__main__':", "    unittest.main()", ""]
    test_src = "\n".join(t_lines).replace("\n\n\n\n", "\n\n\n")
    with open(os.path.join(ws, "mylib.py"), "w") as f:
        f.write("\n\n\n".join(srcs[n] for n in order) + "\n")
    with open(os.path.join(ws, "test_mylib.py"), "w") as f:
        f.write(test_src)
    # self-check: tests must fail as written, and pass once every function is the correct one
    if kind == "solvable":
        assert not _tests_pass(ws), "generated coding task already passes"
        with tempfile.TemporaryDirectory() as td:
            shutil.copy(os.path.join(ws, "test_mylib.py"), td)
            with open(os.path.join(td, "mylib.py"), "w") as f:
                f.write("\n\n\n".join(fixed[n] for n in order) + "\n")
            assert _tests_pass(td), "reference solution does not pass its own tests"
    digest = hashlib.sha256(test_src.encode()).hexdigest()
    instr = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not "
             "edit the tests) until all tests pass. " + SUFFIX_DONE)
    return instr, digest, {"bugs": sorted(buggy), "functions": sorted(names), "contradiction": contradiction}


def _tests_pass(d):
    return subprocess.run([sys.executable, "test_mylib.py"], cwd=d, capture_output=True,
                          env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}).returncode == 0


# ---------------------------------------------------------------- assembly
PLAN = [  # (type, kind, solvable, variant)
    *[("math", "chain", True, "sample")] * 4, *[("math", "simple", True, "sample")] * 4,
    *[("search", "birth", True, "sample")] * 3, *[("search", "hq", True, "sample")] * 3,
    ("search", "mentor_born", True, "probe"), ("search", "mentor_hq", True, "probe"),
    *[("data_analysis", "units", True, "sample")] * 6, *[("data_analysis", "revenue", True, "probe")] * 2,
    *[("coding", "solvable", True, "sample")] * 8,
    *[("math", "underdetermined", False, "guess")] * 2,
    ("search", "absent", False, "guess"), ("search", "employer_missing", False, "guess"),
    *[("data_analysis", "missing_column", False, "guess")] * 2,
    ("coding", "contradictory_cross", False, "guess"), ("coding", "contradictory_same", False, "guess"),
]


EXTRA = [  # appended by --extended
    *[("math", "words", True, "extended")] * 3,
    *[("math", "chain", True, "boilerplate")] * 2, *[("math", "simple", True, "boilerplate")] * 2,
    *[("data_analysis", "units", True, "boilerplate")] * 2,
    *[("search", "hq", True, "boilerplate")] * 2, *[("search", "birth", True, "boilerplate")] * 1,
    *[("coding", "solvable", True, "boilerplate")] * 2,
    *[("math", "vague", False, "extended")] * 3,
    *[("data_analysis", "missing_units_column", False, "extended")] * 3,
    *[("search", "field_missing", False, "extended")] * 2, *[("search", "company_field_missing", False, "extended")] * 2,
    *[("coding", "always_fail", False, "extended")] * 2,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--grader", required=True)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--first-index", type=int, default=100, help="numeric suffix of the first task id")
    ap.add_argument("--extended", action="store_true", help="append the EXTRA kinds (words, boilerplate, more unsolvable)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    plan = list(PLAN) + (EXTRA if args.extended else [])
    rng.shuffle(plan)
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out)
    grader = {}
    for i, (ttype, kind, solvable, variant) in enumerate(plan):
        tid = f"{ttype}_{args.first_index + i:04d}"
        d = os.path.join(args.out, tid)
        ws = os.path.join(d, "workspace")
        os.makedirs(d)
        meta = {"kind": kind, "solvable": solvable, "variant": variant, "reward": rng.randint(1, 5)}
        if ttype == "math":
            instr, exp = (math_chain(rng) if kind == "chain" else math_simple(rng) if kind == "simple" else
                          math_words(rng) if kind == "words" else
                          (math_vague(rng), None) if kind == "vague" else (math_underdetermined(rng), None))
            meta.update(expected=exp, check="exact_int")
        elif ttype == "search":
            instr, exp, (persons, comp, skip, drop_field) = search_task(rng, kind)
            os.makedirs(ws)
            write_kb(ws, persons, comp, skip, drop_field)
            meta.update(expected=exp, check="exact_str")
        elif ttype == "data_analysis":
            os.makedirs(ws)
            instr, exp = data_task(rng, ws, kind)
            meta.update(expected=exp, check="exact_int")
        else:
            os.makedirs(ws)
            instr, digest, info = coding_task(rng, ws, kind)
            meta.update(expected="tests_pass", check="unittest", test_sha256=digest, **info)
        if variant == "boilerplate":
            instr = reword_boiler(instr)
        with open(os.path.join(d, "task.json"), "w") as f:
            json.dump({"task_id": tid, "task_type": ttype, "instruction": instr, "reward": meta["reward"]}, f,
                      ensure_ascii=False, indent=1)
        grader[tid] = meta
    with open(args.grader, "w") as f:
        json.dump(grader, f, ensure_ascii=False, indent=1)
    solv = sum(1 for m in grader.values() if m["solvable"])
    print(f"generated {len(grader)} tasks in {args.out}: {solv} solvable, {len(grader) - solv} unsolvable; grader -> {args.grader}")


if __name__ == "__main__":
    main()

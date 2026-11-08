# coding: utf-8
"""Deterministic (no-LLM) solvers for the task templates seen in the organisers' samples. Cost: 0 tokens.

The organisers allow it ("можно решить все, если получится"). The design rule is: a solver answers only when the
WHOLE instruction matches a known template word for word (anchored regexes, the standard boilerplate suffix included)
and the data it relies on is consistent (header, row count, unambiguous lookups). Anything else returns `nomatch`
and the task goes to the LLM engine exactly as before. A wrong parse is therefore much less likely than a
refusal; refusals only cost tokens.

Results: `answer` (write answer.txt), `unsolvable` (the template matched but the data provably cannot answer it:
do NOT create answer.txt), `nomatch` (hand the task to the LLM).

  math          warehouse stories: start / ships X per day for D days / receives R deliveries of Q / split into K
                keeping one share (floor) / doubled; a sentence with an "unknown number" makes the task unsolvable
  data_analysis sales.csv: units of a product in a month range (optionally in a region), revenue of product in region
  search        kb/ markdown base: born / employer's headquarters / mentor's birth year / employer of the mentor /
                mentor's name / employer's founding year; a missing entity makes the task unsolvable
  coding        mylib.py + test_mylib.py: replaces known functions (a small library of canonical implementations)
                one by one while the number of failing tests drops; only if ALL tests pass it answers DONE
"""
from __future__ import annotations

import ast
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from tasks import Task

SUFFIX = ("Work only inside the current directory. When you are done, write ONLY the final answer to the file "
          "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
SUFFIX_DONE = ("Work only inside the current directory. When you are done, write ONLY the word DONE to the file "
               "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")


@dataclass
class DetResult:
    status: str                    # "answer" | "unsolvable" | "nomatch"
    answer: Optional[str] = None
    why: str = ""
    solver: str = ""


def _nomatch(why: str = "") -> DetResult:
    return DetResult("nomatch", why=why)


def _norm(s: str) -> str:
    return " ".join(s.split())


# ------------------------------------------------------------------------------------------------ math
_M_START = re.compile(r"A warehouse starts with (\d+) boxes")
_M_SPLIT = re.compile(r"Then the stock is split equally into (\d+) warehouses and it keeps one share \(rounded down\)")
_M_DOUBLE = re.compile(r"Then the stock is doubled by merging with a sister warehouse")
_M_SHIP = re.compile(r"It ships (\d+) boxes per day for (\d+) days?")
_M_RECV = re.compile(r"It receives (\d+) deliver(?:y|ies) of (\d+) boxes each")
_M_SHIP_UNKNOWN = re.compile(r"It ships an unknown (?:number|amount) of boxes per day for (\d+) days?")
_M_BODY = re.compile(r"(?P<pre>.+\.) How many boxes does it have at the end\? Compute it with the bash tool "
                     r"\(python3 -c\)\.")


def solve_math(text: str) -> DetResult:
    if not text.endswith(SUFFIX):
        return _nomatch("math: unknown suffix")
    m = _M_BODY.fullmatch(text[: -len(SUFFIX)].strip())
    if not m:
        return _nomatch("math: body does not match")
    val: Optional[int] = None
    unknown = False
    for s in re.split(r"\.\s+", m.group("pre").rstrip().rstrip(".")):
        s = s.strip()
        if val is None:
            st = _M_START.fullmatch(s)
            if not st:
                return _nomatch("math: first sentence is not the start")
            val = int(st.group(1))
        elif (mm := _M_SPLIT.fullmatch(s)):
            k = int(mm.group(1))
            if k == 0:
                return _nomatch("math: split by zero")
            val //= k
        elif _M_DOUBLE.fullmatch(s):
            val *= 2
        elif (mm := _M_SHIP.fullmatch(s)):
            val -= int(mm.group(1)) * int(mm.group(2))
        elif (mm := _M_RECV.fullmatch(s)):
            val += int(mm.group(1)) * int(mm.group(2))
        elif _M_SHIP_UNKNOWN.fullmatch(s):
            unknown = True
        else:
            return _nomatch(f"math: unknown sentence {s[:40]!r}")
    if unknown:
        return DetResult("unsolvable", why="math: an unknown quantity", solver="math")
    return DetResult("answer", str(val), solver="math")


# ------------------------------------------------------------------------------------------------ data analysis
_D_BODY = re.compile(r"The file sales\.csv has columns (?P<cols>[a-z_,]+) \((?P<n>\d+) rows\)\. (?P<q>.+?) "
                     r"Use bash with python3 to compute it\.")
_D_UNITS = re.compile(r"How many units of product '(\w+)' were sold in months (\d+) through (\d+) inclusive\?")
_D_UNITS_REGION = re.compile(r"How many units of product '(\w+)' were sold in region '(\w+)' in months (\d+) through "
                             r"(\d+) inclusive\?")
_D_REVENUE = re.compile(r"What is the total revenue \(units \* price\) of product '(\w+)' in region '(\w+)'\?")


def solve_data(text: str, workspace: Optional[str]) -> DetResult:
    if workspace is None or not text.endswith(SUFFIX):
        return _nomatch("data: no workspace or unknown suffix")
    m = _D_BODY.fullmatch(text[: -len(SUFFIX)].strip())
    if not m:
        return _nomatch("data: body does not match")
    cols = m.group("cols").split(",")
    try:
        with open(os.path.join(workspace, "sales.csv"), newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != cols:
                return _nomatch("data: header differs from the instruction")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        return _nomatch("data: cannot read sales.csv")
    if len(rows) != int(m.group("n")):
        return _nomatch("data: row count differs from the instruction")
    q = m.group("q")
    try:
        if (mm := _D_UNITS.fullmatch(q)):
            p, a, b = mm.group(1), int(mm.group(2)), int(mm.group(3))
            total = sum(int(r["units"]) for r in rows if r["product"] == p and a <= int(r["month"]) <= b)
        elif (mm := _D_UNITS_REGION.fullmatch(q)):
            p, reg, a, b = mm.group(1), mm.group(2), int(mm.group(3)), int(mm.group(4))
            total = sum(int(r["units"]) for r in rows
                        if r["product"] == p and r["region"] == reg and a <= int(r["month"]) <= b)
        elif (mm := _D_REVENUE.fullmatch(q)):
            p, reg = mm.group(1), mm.group(2)
            total = sum(int(r["units"]) * int(r["price"]) for r in rows if r["product"] == p and r["region"] == reg)
        else:
            return _nomatch("data: question is not a known template")
    except (KeyError, ValueError, TypeError):
        return _nomatch("data: a column is missing or not numeric")
    return DetResult("answer", str(total), solver="data")


# ------------------------------------------------------------------------------------------------ search / kb
_S_HEAD = "The directory kb/ contains a small knowledge base of markdown files. "
_S_TAIL = " Use the file tools or bash (ls, grep, cat) to find it. " + SUFFIX
_NAME = r"[A-Z][\w'’-]*(?: [A-Z][\w'’-]*)+"
# (question regex, kind). Every regex has exactly one group: the person's name.
_S_TEMPLATES = [
    (re.compile(rf"In which year was ({_NAME}) born\?"), "born"),
    (re.compile(rf"In which city is the headquarters of ({_NAME})'s employer\?"), "employer_hq"),
    (re.compile(rf"In which year was ({_NAME})'s mentor born\?"), "mentor_born"),
    (re.compile(rf"In which city is the headquarters of the employer of ({_NAME})'s mentor\?"), "mentor_employer_hq"),
    (re.compile(rf"Who is the mentor of ({_NAME})\?"), "mentor"),
    (re.compile(rf"In which year was ({_NAME})'s employer founded\?"), "employer_founded"),
]


def _load_kb(kb_dir: str):
    """title.lower() -> list of (filename, fields dict, text). Titles are the first `# ` line of each file."""
    ents: Dict[str, list] = {}
    for fn in sorted(os.listdir(kb_dir)):
        p = os.path.join(kb_dir, fn)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), None)
        if title:
            fields = {k.strip().lower(): v.strip() for k, v in re.findall(r"^([A-Za-z ]+):[ \t]*(.+?)[ \t]*$", text, re.M)}
            ents.setdefault(title.lower(), []).append((fn, fields, text))
    return ents


class _Absent(Exception):
    """The entity is provably not in the knowledge base."""


class _Unsure(Exception):
    """Ambiguous or incomplete data: let the LLM look."""


def _entity(ents, name: str, is_person: bool) -> Dict[str, str]:
    """Fields of the person (is_person) or company called `name`.

    Absence is claimed only when no file OF THE SAME KIND mentions the name at all (a person file mentioning it as a
    mentor, or a company file mentioning it, makes us unsure and the task goes to the LLM). References from the other
    kind do not count: every person file names its employer, which says nothing about the company's data."""
    def is_kind(fields):
        return ("born" in fields or "employer" in fields) if is_person else ("headquarters" in fields)

    found = [e for e in ents.get(name.lower(), []) if is_kind(e[1])]
    if len(found) > 1:
        raise _Unsure(f"{name} is ambiguous")
    if found:
        return found[0][1]
    same_kind_texts = [t for lst in ents.values() for _fn, f, t in lst if is_kind(f)]
    if any(name.lower() in t.lower() for t in same_kind_texts):
        raise _Unsure(f"{name} appears in the kb but not as a title")
    raise _Absent(name)


def solve_search(text: str, workspace: Optional[str]) -> DetResult:
    if workspace is None or not text.startswith(_S_HEAD) or not text.endswith(_S_TAIL):
        return _nomatch("search: unknown frame")
    q = text[len(_S_HEAD): -len(_S_TAIL)].strip()
    for rx, kind in _S_TEMPLATES:
        mm = rx.fullmatch(q)
        if mm:
            name = mm.group(1)
            break
    else:
        return _nomatch("search: question is not a known template")
    kb = os.path.join(workspace, "kb")
    if not os.path.isdir(kb):
        return _nomatch("search: no kb/ directory")
    ents = _load_kb(kb)
    try:
        def person(n):
            return _entity(ents, n, True)

        def company(n):
            return _entity(ents, n, False)

        def need(fields, key):
            if key not in fields or not fields[key]:
                raise _Unsure(f"field {key} is missing")
            return fields[key]

        p = person(name)
        if kind == "born":
            ans = need(p, "born")
        elif kind == "employer_hq":
            ans = need(company(need(p, "employer")), "headquarters")
        elif kind == "mentor":
            ans = need(p, "mentor")
        elif kind == "employer_founded":
            ans = need(company(need(p, "employer")), "founded")
        else:
            mentor = person(need(p, "mentor"))
            if kind == "mentor_born":
                ans = need(mentor, "born")
            else:  # mentor_employer_hq
                ans = need(company(need(mentor, "employer")), "headquarters")
    except _Absent as e:
        return DetResult("unsolvable", why=f"search: {e} is not in the kb", solver="search")
    except _Unsure as e:
        return _nomatch(f"search: {e}")
    return DetResult("answer", ans.strip(), solver="search")


# ------------------------------------------------------------------------------------------------ coding
_C_TEXT = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not "
           "edit the tests) until all tests pass. " + SUFFIX_DONE)

# Canonical correct implementations of helper functions that appear in such tasks (and close siblings).
CANON: Dict[str, str] = {
    "reverse_words": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))",
    "total": "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s",
    "maximum": "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best",
    "minimum": "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best",
    "average": "def average(xs):\n    return sum(xs) / len(xs)",
    "first_n": "def first_n(xs, n):\n    return xs[:n]",
    "last_n": "def last_n(xs, n):\n    return xs[len(xs) - n:] if n > 0 else []",
    "count_even": "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)",
    "count_odd": "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 == 1)",
    "product": "def product(xs):\n    p = 1\n    for x in xs:\n        p *= x\n    return p",
}


def _run_tests(ws: str, timeout: float = 60.0):
    """(all_passed, number_of_failing_tests). Unknown outcomes count as a huge number of failures."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run([sys.executable, "test_mylib.py"], cwd=ws, env=env, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return False, 10 ** 6
    if p.returncode == 0:
        return True, 0
    m = re.search(r"FAILED \((.*?)\)", p.stderr)
    return False, (sum(int(x) for x in re.findall(r"=(\d+)", m.group(1))) if m else 10 ** 6)


def _write(path: str, src: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)


def solve_coding(text: str, workspace: Optional[str]) -> DetResult:
    if workspace is None or text != _C_TEXT:
        return _nomatch("coding: instruction is not the known template")
    lib, tests = os.path.join(workspace, "mylib.py"), os.path.join(workspace, "test_mylib.py")
    if not (os.path.isfile(lib) and os.path.isfile(tests)):
        return _nomatch("coding: mylib.py / test_mylib.py missing")
    with open(lib, encoding="utf-8") as f:
        orig = f.read()
    ok0, fail0 = _run_tests(workspace)
    if ok0:
        return _nomatch("coding: no test fails, unexpected for this template")
    try:
        ast.parse(orig)
    except SyntaxError:
        return _nomatch("coding: mylib.py does not parse")
    cur, fcur = orig, fail0
    try:
        for name in [n.name for n in ast.parse(orig).body if isinstance(n, ast.FunctionDef) and n.name in CANON]:
            node = next((n for n in ast.parse(cur).body if isinstance(n, ast.FunctionDef) and n.name == name), None)
            canon = ast.parse(CANON[name]).body[0]
            if node is None or node.decorator_list or ast.dump(node) == ast.dump(canon) \
                    or [a.arg for a in node.args.args] != [a.arg for a in canon.args.args]:
                continue  # missing, decorated, already canonical, or a different signature: leave it alone
            lines = cur.splitlines(keepends=True)
            new = "".join(lines[: node.lineno - 1]) + CANON[name] + "\n" + "".join(lines[node.end_lineno:])
            _write(lib, new)
            ok, nfail = _run_tests(workspace)
            if ok or nfail < fcur:
                cur, fcur = new, nfail
                if ok:
                    return DetResult("answer", "DONE", why="replaced known functions", solver="coding")
            else:
                _write(lib, cur)
    except Exception:
        _write(lib, orig)
        raise
    if fcur >= fail0:
        _write(lib, orig)  # nothing helped: leave the file exactly as it was for the LLM
    return _nomatch(f"coding: still {fcur} failing test(s) after the known fixes")


# ------------------------------------------------------------------------------------------------ dispatcher
def solve_deterministic(task: Task) -> DetResult:
    text = _norm(task.instruction)
    solvers: Dict[str, Callable[[], DetResult]] = {
        "math": lambda: solve_math(text),
        "data_analysis": lambda: solve_data(text, task.workspace),
        "search": lambda: solve_search(text, task.workspace),
        "coding": lambda: solve_coding(text, task.workspace),
    }
    order = [task.type] if task.type in solvers else list(solvers)
    last = _nomatch("no solver for this task type")
    for t in order:
        r = solvers[t]()
        if r.status != "nomatch":
            return r
        last = r
    return last

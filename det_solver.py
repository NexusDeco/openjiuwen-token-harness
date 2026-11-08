# coding: utf-8
"""Deterministic (no-LLM) solvers for the task templates seen in the organisers' samples. Cost: 0 tokens.

The organisers allow it ("можно решить все, если получится"). The design rule is: a solver answers only when the
QUESTION matches a known template word for word (anchored regexes) and the data it relies on is consistent (header,
row count, unambiguous lookups). Anything else returns `nomatch` and the task goes to the LLM engine exactly as
before. A wrong parse is therefore much less likely than a refusal; refusals only cost tokens.

Only the boilerplate around the question is read tolerantly: the trailing "Work only inside ... write ONLY the final
answer to the file answer.txt <where>" may word its location differently (the organisers said the file goes next to
task.json, so that sentence may change), and imperative hint sentences without numbers ("Compute it ...", "Use bash
...") may be reworded. Any other file name, a path, or a number in a hint still means `nomatch`.

Results: `answer` (write answer.txt), `unsolvable` (the template matched but the data provably cannot answer it:
do NOT create answer.txt), `nomatch` (hand the task to the LLM).

  math          warehouse stories: start / ships X per day for D days / receives R deliveries of Q / split into K
                keeping one share (floor) / doubled. Numbers may be written as words. A vague quantity ("an unknown
                number of", "several", "some") in one of these sentences makes the task unsolvable
  data_analysis sales.csv: units of a product in a month range (optionally in a region), revenue of product in region;
                unsolvable when the question needs a column the file does not have, or asks about a concept (discount,
                tax, profit ...) that is not a column
  search        kb/ markdown base: born / employer's headquarters / mentor's birth year / employer of the mentor /
                mentor's name / employer's founding year; unsolvable when the person or company is missing from the
                kb, or its record exists but lacks the needed field
  coding        mylib.py + test_mylib.py: unsolvable when the tests cannot pass (see prechecks.tests_flags); otherwise
                replaces known functions (a small library of canonical implementations) one by one while the number of
                failing tests drops, and answers DONE only if ALL tests pass
"""
from __future__ import annotations

import ast
import csv
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import repair
from prechecks import tests_flags
from tasks import Task


@dataclass
class DetResult:
    status: str                    # "answer" | "unsolvable" | "nomatch"
    answer: Optional[str] = None
    why: str = ""
    solver: str = ""


class _NoMatch(Exception):
    pass


def _nomatch(why: str = "") -> DetResult:
    return DetResult("nomatch", why=why)


def _norm(s: str) -> str:
    return " ".join(s.split())


# ------------------------------------------------------------------------------------------------ boilerplate
_TAIL_RX = re.compile(r"Work only inside the current directory\. When you are done, write ONLY the "
                      r"(?P<what>final answer|word DONE) to the file answer\.txt(?P<where>[^(]*?)"
                      r"(?: \(for example with bash: echo 42 > answer\.txt\))?\.?")
_HINT_START = re.compile(r"(?:Compute|Calculate|Use|Find|Run|Solve)\b")


def _before_tail(text: str, what: str) -> Optional[str]:
    """The part of the instruction before the standard trailing boilerplate, or None if that boilerplate is not the
    familiar 'write the answer to answer.txt' kind (a different file name or a path is rejected)."""
    i = text.find("Work only inside the current directory")
    if i < 0:
        return None
    m = _TAIL_RX.fullmatch(text[i:].strip())
    if not m or m.group("what") != what:
        return None
    where = m.group("where")
    if "/" in where or any(f != "task.json" for f in re.findall(r"[\w-]+\.(?:txt|json|csv|md|py)\b", where)):
        return None
    return text[:i].strip()


def _question_part(text: str) -> Optional[str]:
    """Instruction without the tail and without trailing imperative hint sentences; ends with the question's '?'."""
    core = _before_tail(text, "final answer")
    if core is None:
        return None
    q = core.rfind("?")
    if q < 0:
        return None
    hints = core[q + 1:].strip()
    for s in (re.split(r"(?<=[.?])\s+", hints) if hints else []):
        if not _HINT_START.match(s) or re.search(r"\d", s.replace("python3", "python")):
            return None
    return core[: q + 1]


# ------------------------------------------------------------------------------------------------ math
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_VAGUE = re.compile(r"(?:some|several|many|few|a few|numerous|multiple|various|unknown|unspecified|"
                    r"(?:an? )?(?:unknown|unspecified|certain|undisclosed|unstated) (?:number|amount)(?: of)?|"
                    r"a number of|a (?:number|couple) of)", re.I)


def _words_to_int(s: str) -> Optional[int]:
    toks = s.lower().replace("-", " ").split()
    if not toks:
        return None
    cur = 0
    for w in toks:
        if w in _UNITS:
            cur += _UNITS[w]
        elif w in _TENS:
            cur += _TENS[w]
        elif w == "hundred":
            cur = max(cur, 1) * 100
        else:
            return None
    return cur


def _qty(s: str) -> Union[int, str]:
    """A number (digits or words), or "vague" for an unspecified quantity; anything else is not our template."""
    s = s.strip()
    if s.isdigit():
        return int(s)
    n = _words_to_int(s)
    if n is not None:
        return n
    if _VAGUE.fullmatch(s):
        return "vague"
    raise _NoMatch(f"unrecognised quantity {s[:30]!r}")


_Q = r"[A-Za-z0-9][A-Za-z0-9 -]*?"
_M_START = re.compile(rf"A warehouse starts with (?P<n>{_Q}) boxes")
_M_BODY = re.compile(r"(?P<pre>.+\.) How many boxes does it have at the end\?")
# The story is a sequence of operations on one stock, any order and length. A sentence is matched in canonical form: a
# leading "Then " is dropped and the first letter lower-cased, so "Then it ships ..." and "It ships ..." are one form.
_M_SPLIT = re.compile(rf"the stock is split equally into (?P<k>{_Q}) warehouses and it keeps one share "
                      r"\(rounded (?P<mode>down|up)\)")
_M_MULT = re.compile(r"the stock is (?P<w>doubled|tripled|quadrupled)"
                     r"(?: by merging with (?P<c>a|one|two|three) sister warehouses?)?")
_M_SHIP = re.compile(rf"it ships (?P<a>{_Q}) boxes(?: per day for (?P<b>{_Q}) days?)?")
_M_RECV = re.compile(rf"it receives (?P<a>{_Q}) boxes(?: per day for (?P<b>{_Q}) days?)?")
_M_DELIV = re.compile(rf"it receives (?P<a>{_Q}) deliver(?:y|ies) of (?P<b>{_Q}) boxes each")
_MULT = {"doubled": 2, "tripled": 3, "quadrupled": 4}
_MULT_TAIL = {"a": 1, "one": 1, "two": 2, "three": 3}
# A known sentence with an EMPTY quantity slot ("It ships boxes per day for 3 days"). A solvable task cannot contain one,
# so it is unsolvable, with no guessing.
_M_HOLES = [
    re.compile(rf"it (?:ships|receives) boxes(?: per day for (?:{_Q}) days?)?"),
    re.compile(rf"it (?:ships|receives) (?:{_Q}) boxes per day for days?"),
    re.compile(rf"it receives deliver(?:y|ies) of (?:{_Q}) boxes each"),
    re.compile(rf"it receives (?:{_Q}) deliver(?:y|ies) of boxes each"),
    re.compile(r"the stock is split equally into warehouses and it keeps one share \(rounded (?:down|up)\)"),
]
_M_START_HOLE = re.compile(r"A warehouse starts with boxes")


def solve_math(text: str) -> DetResult:
    core = _question_part(text)
    m = _M_BODY.fullmatch(core) if core else None
    if not m:
        return _nomatch("math: body does not match")
    val: Optional[int] = None
    unknown = False
    try:
        for i, s in enumerate(re.split(r"\.\s+", m.group("pre").rstrip().rstrip("."))):
            s = s.strip()
            if i == 0:
                if _M_START_HOLE.fullmatch(s):
                    unknown, val = True, 0
                    continue
                st = _M_START.fullmatch(s)
                if not st:
                    return _nomatch("math: first sentence is not the start")
                q = _qty(st.group("n"))
                unknown, val = (True, 0) if q == "vague" else (unknown, q)
                continue
            c = re.sub(r"^Then,?\s+", "", s)
            c = c[:1].lower() + c[1:]
            if (mm := _M_SPLIT.fullmatch(c)):
                k = _qty(mm.group("k"))
                if k == "vague":
                    unknown = True
                elif k == 0:
                    return _nomatch("math: split by zero")
                elif not unknown:
                    val = val // k if mm.group("mode") == "down" else -(-val // k)
            elif (mm := _M_MULT.fullmatch(c)):
                f = _MULT[mm.group("w")]
                if mm.group("c") and _MULT_TAIL[mm.group("c")] != f - 1:
                    return _nomatch(f"math: {mm.group('w')} does not agree with the sister warehouses")
                if not unknown:
                    val *= f
            elif (mm := _M_DELIV.fullmatch(c)) or (mm := _M_SHIP.fullmatch(c)) or (mm := _M_RECV.fullmatch(c)):
                a = _qty(mm.group("a"))
                b = _qty(mm.group("b")) if mm.group("b") is not None else 1
                if "vague" in (a, b):
                    unknown = True
                elif not unknown:
                    val += (-1 if c.startswith("it ships") else 1) * a * b
            elif any(h.fullmatch(c) for h in _M_HOLES):
                unknown = True
            else:
                return _nomatch(f"math: unknown sentence {s[:40]!r}")
    except _NoMatch as e:
        return _nomatch(f"math: {e}")
    if unknown:
        return DetResult("unsolvable", why="math: an unspecified quantity", solver="math")
    return DetResult("answer", str(val), solver="math")


# ------------------------------------------------------------------------------------------------ data analysis
_D_BODY = re.compile(r"The file sales\.csv has columns (?P<cols>[a-z_,]+)(?: \((?P<n>\d+) rows\))?\. (?P<q>.+\?)")
# Two metrics (units sold; revenue = units * price, defined in the question itself) over optional filters, always written
# in this order: product, region, months ("in months A through B inclusive" or "in month M"). The three forms seen or
# guessed so far (product+months, product+region+months, revenue of product in region) are special cases.
_D_FILT = (r"(?: in region '(?P<reg>\w+)')?"
           r"(?: in (?:months (?P<a>\d+) through (?P<b>\d+) inclusive|month (?P<m>\d+)))?")
_D_UNITS_Q = re.compile(r"How many units(?: of product '(?P<p>\w+)')? were sold" + _D_FILT + r"\?")
_D_REVENUE_Q = re.compile(r"What is the total revenue \(units \* price\)(?: of product '(?P<p>\w+)')?" + _D_FILT + r"\?")
# Concepts a sales table with these columns cannot answer. Deliberately a short list of unambiguous words: a question
# that mentions one while the file has no such column cannot be solved exactly. ("cost" is left out: it may well
# mean units * price.)
_ABSENT_CONCEPTS = {"discount", "discounts", "tax", "taxes", "profit", "profits", "margin", "margins", "commission",
                    "commissions", "shipping", "freight", "refund", "refunds", "rating", "ratings", "bonus", "tip",
                    "tips", "fee", "fees", "customer", "customers", "salary", "salaries", "age", "email", "address"}


def solve_data(text: str, workspace: Optional[str]) -> DetResult:
    core = _question_part(text) if workspace is not None else None
    m = _D_BODY.fullmatch(core) if core else None
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
    if m.group("n") is not None and len(rows) != int(m.group("n")):
        return _nomatch("data: row count differs from the instruction")
    q = m.group("q")

    def total(needed, fn):
        absent = [c for c in needed if c not in cols]
        if absent:
            return DetResult("unsolvable", why=f"data: column {absent[0]!r} is not in sales.csv", solver="data")
        try:
            return DetResult("answer", str(sum(fn(r) for r in rows)), solver="data")
        except (KeyError, ValueError, TypeError):
            return _nomatch("data: a column is not numeric")

    for rx, revenue in ((_D_UNITS_Q, False), (_D_REVENUE_Q, True)):
        mm = rx.fullmatch(q)
        if mm:
            p, reg, a, b, mo = (mm.group(k) for k in ("p", "reg", "a", "b", "m"))
            lo, hi = (int(a), int(b)) if a is not None else ((int(mo), int(mo)) if mo is not None else (None, None))
            needed = ["units"] + (["price"] if revenue else []) + (["product"] if p else []) \
                + (["region"] if reg else []) + (["month"] if lo is not None else [])

            def value(r, p=p, reg=reg, lo=lo, hi=hi, revenue=revenue):
                if (p and r["product"] != p) or (reg and r["region"] != reg) \
                        or (lo is not None and not lo <= int(r["month"]) <= hi):
                    return 0
                return int(r["units"]) * (int(r["price"]) if revenue else 1)
            return total(needed, value)
    words = set(re.findall(r"[a-z]+", re.sub(r"'[^']*'", " ", q.lower())))  # quoted values are data, not concepts
    gone = sorted((words & _ABSENT_CONCEPTS) - set(cols))
    if gone:
        return DetResult("unsolvable", why=f"data: sales.csv has no {gone[0]!r} column", solver="data")
    return _nomatch("data: question is not a known template")


# ------------------------------------------------------------------------------------------------ search / kb
_S_HEAD = re.compile(r"The directory kb/ [^.?]*knowledge base[^.?]*\. (?P<q>.+\?)")
_NAME = r"[A-Z][\w'’-]*(?: [A-Z][\w'’-]*)+"
# The kb is a graph: person --mentor--> person, person --employer--> company. A question names a start person, follows
# `hops` mentor links (written either "X's mentor" or "the mentor of X", any depth, mixed freely) and asks for one fact.
_POSS = r"['\u2019]s"
_MAX_HOPS = 12


def _person_expr(s: str):
    """'Anna Lee' -> ('Anna Lee', 0); "Anna Lee's mentor's mentor" / 'the mentor of Anna Lee' -> hops counted; else None."""
    s, hops = s.strip(), 0
    while True:
        m = re.fullmatch(r"the mentor of (.+)", s) or re.fullmatch(rf"(.+){_POSS} mentor", s)
        if not m:
            break
        s, hops = m.group(1).strip(), hops + 1
        if hops > _MAX_HOPS:
            return None
    return (s, hops) if re.fullmatch(_NAME, s) else None


def _company_expr(s: str):
    """The employer of a person expression: "X's employer" / 'the employer of X' -> that person's (name, hops)."""
    s = s.strip()
    m = (re.fullmatch(r"the employer of (.+)", s) or re.fullmatch(rf"(.+){_POSS} employer", s)
         or re.fullmatch(r"the company where (.+) works", s) or re.fullmatch(r"the company that employs (.+)", s)
         or re.fullmatch(r"the company (.+) works for", s))
    return _person_expr(m.group(1)) if m else None


# (kind, question frame with one group, parser of that group)
_S_FRAMES = [
    ("born", re.compile(r"In which year was (.+) born\?"), _person_expr),
    ("born", re.compile(r"In what year was (.+) born\?"), _person_expr),
    ("born", re.compile(r"What year was (.+) born\?"), _person_expr),
    ("born", re.compile(r"What is the birth year of (.+)\?"), _person_expr),
    ("hq", re.compile(r"In which city is the headquarters of (.+)\?"), _company_expr),
    ("hq", re.compile(r"Where is the headquarters of (.+)\?"), _company_expr),
    ("hq", re.compile(r"In which city is (.+) headquartered\?"), _company_expr),
    ("founded", re.compile(r"In which year was (.+) founded\?"), _company_expr),
    ("founded", re.compile(r"In what year was (.+) founded\?"), _company_expr),
    ("founded", re.compile(r"What is the founding year of (.+)\?"), _company_expr),
    ("mentor", re.compile(r"Who is the mentor of (.+)\?"), _person_expr),
    ("mentor", re.compile(rf"Who is (.+){_POSS} mentor\?"), _person_expr),
    ("employer", re.compile(r"Which company employs (.+)\?"), _person_expr),
    ("employer", re.compile(r"Who is the employer of (.+)\?"), _person_expr),
]
_PERSON_KEYS = {"born", "employer", "mentor"}
_COMPANY_KEYS = {"headquarters", "founded"}


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
    """The entity (or its needed field) is provably not in the knowledge base."""


class _Unsure(Exception):
    """Ambiguous or incomplete data: let the LLM look."""


def _entity(ents, name: str, is_person: bool) -> Dict[str, str]:
    """Fields of the person (is_person) or company called `name`.

    Absence is claimed only when no file OF THE SAME KIND mentions the name at all (a person file mentioning it as a
    mentor, or a company file mentioning it, makes us unsure and the task goes to the LLM). References from the other
    kind do not count: every person file names its employer, which says nothing about the company's data."""
    def is_kind(fields):  # a record of the person / company schema is recognised by ANY of its known fields
        return any(k in fields for k in (_PERSON_KEYS if is_person else _COMPANY_KEYS))

    titled = ents.get(name.lower(), [])
    found = [e for e in titled if is_kind(e[1])]
    if len(found) > 1:
        raise _Unsure(f"{name} is ambiguous")
    if found:
        return found[0][1]
    if titled:  # a file with this very title exists but not in the schema we know: the answer may be in its text
        raise _Unsure(f"{name} has a record in an unfamiliar format")
    same_kind_texts = [t for lst in ents.values() for _fn, f, t in lst if is_kind(f)]
    if any(name.lower() in t.lower() for t in same_kind_texts) or _free_text_mentions(ents, name):
        raise _Unsure(f"{name} appears in the kb but not as a title")
    raise _Absent(f"{name} is not in the kb")


def _free_text_mentions(ents, name: str) -> bool:
    """Is the name mentioned anywhere except as a title or in a `Mentor:` / `Employer:` reference line?"""
    low = name.lower()
    for lst in ents.values():
        for _fn, _f, text in lst:
            for line in text.splitlines():
                if low in line.lower() and not line.startswith("# ") \
                        and not re.match(r"\s*(mentor|employer)\s*:", line, re.I):
                    return True
    return False


def solve_search(text: str, workspace: Optional[str]) -> DetResult:
    core = _question_part(text) if workspace is not None else None
    hm = _S_HEAD.fullmatch(core) if core else None
    if not hm:
        return _nomatch("search: unknown frame")
    q = hm.group("q")
    for kind, rx, parse in _S_FRAMES:
        mm = rx.fullmatch(q)
        expr = parse(mm.group(1)) if mm else None
        if expr:
            name, hops = expr
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

        def need(fields, key, who, is_person):
            if fields.get(key):
                return fields[key]
            known = _PERSON_KEYS if is_person else _COMPANY_KEYS
            if any(k in fields for k in known if k != key):
                # A record of this schema that simply lacks the field. If the name is mentioned in free text somewhere
                # the information may live there: leave it to the LLM; otherwise it is not in the base.
                if _free_text_mentions(ents, who):
                    raise _Unsure(f"{who}: no {key!r} field, but mentioned in free text")
                raise _Absent(f"{who} has no {key!r} recorded in the kb")
            raise _Unsure(f"field {key} is missing")

        p, cur = person(name), name
        for _ in range(hops):  # follow the mentor links
            cur = need(p, "mentor", cur, True)
            p = person(cur)
        if kind == "born":
            ans = need(p, "born", cur, True)
        elif kind == "mentor":
            ans = need(p, "mentor", cur, True)
        elif kind == "employer":
            ans = need(p, "employer", cur, True)
        else:  # hq / founded: through the person's employer
            emp = need(p, "employer", cur, True)
            ans = need(company(emp), "headquarters" if kind == "hq" else "founded", emp, False)
    except _Absent as e:
        return DetResult("unsolvable", why=f"search: {e}", solver="search")
    except _Unsure as e:
        return _nomatch(f"search: {e}")
    return DetResult("answer", ans.strip(), solver="search")


# ------------------------------------------------------------------------------------------------ coding
_C_BODY = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not "
           "edit the tests) until all tests pass.")

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


# Tests run in a child with hard memory / CPU limits: a buggy (or mutated) function that loops forever and appends to a
# list would otherwise eat all RAM of the container before anything stops it. (No preexec_fn: unsafe with threads.)
_TEST_WRAP = (
    "import resource, runpy, sys\n"
    "mb, cpu = int(sys.argv[1]), int(sys.argv[2])\n"
    "try:\n"
    "    resource.setrlimit(resource.RLIMIT_AS, (mb << 20, mb << 20))\n"
    "    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))\n"
    "except (ValueError, OSError):\n"
    "    pass\n"
    "sys.argv = ['test_mylib.py']\n"
    "runpy.run_path('test_mylib.py', run_name='__main__')\n"
)


def _test_cmd(timeout: float) -> List[str]:
    mb = os.environ.get("TEST_MEM_MB", "1024")
    return [sys.executable, "-B", "-c", _TEST_WRAP, mb, str(int(timeout) + 1)]


def _run_tests(ws: str, timeout: float = 60.0):
    """(all_passed, number_of_failing_tests). Unknown outcomes count as a huge number of failures."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(_test_cmd(timeout), cwd=ws, env=env, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return False, 10 ** 6
    if p.returncode == 0:
        return True, 0
    m = re.search(r"FAILED \((.*?)\)", p.stderr)
    return False, (sum(int(x) for x in re.findall(r"=(\d+)", m.group(1))) if m else 10 ** 6)


def _run_tests_score(ws: str, timeout: float = 60.0):
    """(all_passed, score) with score = (failing tests, -sum of the test-file line of each first failure): smaller is
    better, and a failure that moves further down a test (an earlier assertion now passes) counts as progress."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(_test_cmd(timeout), cwd=ws, env=env, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return False, (10 ** 6, 0)
    if p.returncode == 0:
        return True, (0, 0)
    m = re.search(r"FAILED \((.*?)\)", p.stderr)
    nfail = sum(int(x) for x in re.findall(r"=(\d+)", m.group(1))) if m else 10 ** 6
    depth = 0
    for block in re.split(r"^={20,}$", p.stderr, flags=re.M)[1:]:
        lines = re.findall(r'File "[^"]*test_mylib\.py", line (\d+)', block)
        if lines:
            depth += int(lines[-1])
    return False, (nfail, -depth)


def _failing_funcs(ws: str, timeout: float = 60.0):
    """Names of the mylib.py functions the failing tests point at: frames inside mylib.py, identifiers in the failing line
    of the test file, and `test_<name>` method names. Used only to decide which functions to try first."""
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(_test_cmd(timeout), cwd=ws, env=env, capture_output=True, text=True, errors="replace",
                           timeout=timeout)
        with open(os.path.join(ws, "mylib.py"), encoding="utf-8") as f:
            defined = {n.name for n in ast.walk(ast.parse(f.read())) if isinstance(n, ast.FunctionDef)}
    except (subprocess.TimeoutExpired, OSError, SyntaxError):
        return set()
    if p.returncode == 0:
        return set()
    names = set()
    for block in re.split(r"^={20,}$", p.stderr, flags=re.M)[1:]:
        names.update(re.findall(r'File "(?:[^"]*/)?mylib\.py", line \d+, in (\w+)', block))
        for fn, code in re.findall(r'File "[^"]*test_mylib\.py", line \d+, in (\w+)\n\s*(.+)', block):
            names.update(re.findall(r"[A-Za-z_]\w*", code))
            if fn.startswith("test_"):
                names.add(fn[5:])
    return names & defined


_REPAIR_LOCK = threading.Lock()
_REPAIR_SPENT = [0.0]  # seconds spent in mutation repair by this process, all tasks together


def _write(path: str, src: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)


def solve_coding(text: str, workspace: Optional[str]) -> DetResult:
    if workspace is None or _before_tail(text, "word DONE") != _C_BODY:
        return _nomatch("coding: instruction is not the known template")
    lib, tests = os.path.join(workspace, "mylib.py"), os.path.join(workspace, "test_mylib.py")
    if not (os.path.isfile(lib) and os.path.isfile(tests)):
        return _nomatch("coding: mylib.py / test_mylib.py missing")
    flags = tests_flags(workspace)
    if flags:
        return DetResult("unsolvable", why=f"coding: {flags[0]}", solver="coding")
    with open(lib, encoding="utf-8") as f:
        orig = f.read()
    t_base = time.monotonic()
    ok0, fail0 = _run_tests(workspace)
    t_base = time.monotonic() - t_base
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
    # Stage 2: functions CANON does not know. Single-edit mutations verified against the real tests, in a copy of the
    # workspace (repair.py); the result counts only if the whole test file passes.
    if os.environ.get("NO_REPAIR") != "1":
        try:
            # unit tests of helpers are fast; a slow baseline is a runaway loop (memory cap, timeout), not a measure of speed
            base = min(t_base, 0.5)
            # the det pass is sequential and shares the run's time limit with the LLM stage: cap the total
            with _REPAIR_LOCK:
                left = float(os.environ.get("REPAIR_TOTAL_S", "600")) - _REPAIR_SPENT[0]
            found = None
            if left >= 3:
                t_rep = time.monotonic()
                try:
                    found = repair.search(cur, workspace, _run_tests_score, _run_tests_score(workspace)[1],
                                          baseline_s=base, budget_s=min(30.0, left), focus=_failing_funcs)
                finally:
                    with _REPAIR_LOCK:
                        _REPAIR_SPENT[0] += time.monotonic() - t_rep
        except Exception:
            found = None
        if found is not None:
            fixed, steps = found
            _write(lib, fixed)
            if _run_tests(workspace)[0]:
                return DetResult("answer", "DONE", why="mutation repair: " + "; ".join(steps), solver="coding")
            _write(lib, cur)  # the copy and the real directory disagree: do not trust it
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

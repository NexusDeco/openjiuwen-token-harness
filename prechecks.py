# coding: utf-8
"""Cheap, LLM-free feasibility checks (cost: 0 tokens). Each returned reason lowers p(success) 10x in the planner.

Written for the SAMPLE formats and deliberately conservative: a task is flagged only when it provably cannot be
solved as stated.
  * search : a `kb/` directory of markdown files, each starting with a `# Title` line, person files carrying
             `Employer: <Company>`; flagged when the question names entities that are absent from the kb.
  * coding : test files that contradict each other (the same pure module-level call asserted equal to two
             different literals in different test methods).
Unsolvable tasks of other shapes are NOT caught here (the solver's IMPOSSIBLE channel and step caps are the
second line of defence).
"""
from __future__ import annotations

import ast
import glob
import os
import re
from typing import Any, Dict, List, Set, Tuple

from tasks import Task

# sequences of capitalised words: "Irina Smith", "Zoltan Quimby" (a trailing 's is not part of the name)
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b")
MAX_FILES, MAX_BYTES = 2000, 200_000


# ---------------------------------------------------------------- search / kb
def _read_kb(kb_dir: str) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    for i, fn in enumerate(sorted(os.listdir(kb_dir))):
        if i >= MAX_FILES:
            break
        p = os.path.join(kb_dir, fn)
        if os.path.isfile(p):
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    docs[fn] = f.read(MAX_BYTES)
            except OSError:
                pass
    return docs


def _title(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.lstrip("# ").strip().lower()
    return ""


def kb_precheck(task: Task) -> List[str]:
    if task.type != "search" or not task.workspace:
        return []
    kb_dir = os.path.join(task.workspace, "kb")
    if not os.path.isdir(kb_dir):
        return []
    docs = _read_kb(kb_dir)
    names = NAME_RE.findall(task.instruction)
    if not docs or not names:
        return []
    corpus = "\n".join(docs.values()).lower()
    if not any(n.lower() in corpus for n in names):
        # Every entity the question names is absent from the knowledge base.
        return [f"entity-not-in-kb:{names[0]}"]
    flags: List[str] = []
    if re.search(r"\bemployer\b", task.instruction, re.I):
        titles = {_title(t) for t in docs.values()}
        for n in names:
            for text in docs.values():
                if _title(text) == n.lower():
                    m = re.search(r"^Employer:\s*(.+?)\s*$", text, re.M)
                    if m and m.group(1).strip().lower() not in titles:
                        flags.append(f"employer-not-in-kb:{m.group(1).strip()}")
    return flags


# ---------------------------------------------------------------- coding / contradictory tests
def _literal(node: ast.AST) -> Tuple[bool, Any]:
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _is_pure_call(call: ast.AST, modules: Set[str], funcs: Set[str]) -> bool:
    """`mod.func(<literals>)` or `func(<literals>)` for a name imported at module level, with >= 1 argument.

    Zero-argument calls and method calls on objects built inside a test (`stack.pop()`, `counter.next()`) can
    legitimately return different values on different calls, so they are never considered.
    """
    if not isinstance(call, ast.Call):
        return False
    f = call.func
    named = (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in modules) or \
            (isinstance(f, ast.Name) and f.id in funcs)
    if not named or not (call.args or call.keywords):
        return False
    return all(_literal(a)[0] for a in call.args) and all(_literal(k.value)[0] for k in call.keywords)


def _imports(tree: ast.Module) -> Tuple[Set[str], Set[str]]:
    modules: Set[str] = set()
    funcs: Set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            modules.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            funcs.update(a.asname or a.name for a in n.names if a.name != "*")
    return modules, funcs


def contradictory_tests(source: str) -> List[str]:
    """Reasons found in one test file's source (empty if consistent or unparsable)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    modules, funcs = _imports(tree)
    seen: Dict[str, List[Tuple[str, Any]]] = {}  # ast.dump(call) -> [(test method id, expected literal)]
    text: Dict[str, str] = {}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if not (isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name.startswith("test")):
                continue
            method = f"{cls.name}.{fn.name}"
            for node in ast.walk(fn):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("assertEqual", "assertEquals") and len(node.args) == 2):
                    continue
                for call, lit in ((node.args[0], node.args[1]), (node.args[1], node.args[0])):
                    ok, val = _literal(lit)
                    if ok and _is_pure_call(call, modules, funcs):
                        key = ast.dump(call)
                        seen.setdefault(key, []).append((method, val))
                        text[key] = ast.unparse(call)
                        break
    out: List[str] = []
    for key, items in seen.items():
        for i, (m1, v1) in enumerate(items):
            for m2, v2 in items[i + 1:]:
                if m1 != m2 and v1 != v2:
                    out.append(f"contradictory-tests:{text[key]}=={v1!r}|{v2!r}")
    return sorted(set(out))


def tests_precheck(task: Task) -> List[str]:
    if task.type != "coding" or not task.workspace:
        return []
    flags: List[str] = []
    for path in sorted(glob.glob(os.path.join(task.workspace, "test_*.py")))[:5]:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                flags.extend(contradictory_tests(f.read(MAX_BYTES)))
        except OSError:
            pass
    return flags[:1]  # one reason is enough (each flag multiplies p by 0.1)


def precheck(task: Task) -> List[str]:
    return kb_precheck(task) + tests_precheck(task)

# coding: utf-8
"""Cheap, LLM-free feasibility checks (cost: 0 tokens). Each returned reason lowers p(success) 10x in the planner.

Written for the SAMPLE formats and deliberately conservative: a task is flagged only when it provably cannot be
solved as stated.
  * search : a `kb/` directory of markdown files, each starting with a `# Title` line, person files carrying
             `Employer: <Company>`; flagged when the question names entities that are absent from the kb.
  * coding : tests that cannot pass: the same PURE module-level call asserted equal to two different literals
             (in one test method or in different ones), or an assertion that fails whatever mylib.py does
             (assertEqual of two different literals, assertTrue(False), self.fail()).
Unsolvable tasks of other shapes are NOT caught here (the solver's IMPOSSIBLE channel and step caps are the
second line of defence).

Purity matters: a module function that keeps state (a global counter, random, time, a mutable default) may honestly
return different values for the same arguments, so such calls are never treated as contradictory. Purity is decided
from mylib.py itself (no global/nonlocal, no reference to module-level variables or impure modules, no mutable
defaults, only calls to other pure functions).
"""
from __future__ import annotations

import ast
import glob
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from tasks import Task

# sequences of capitalised words: "Irina Smith", "Zoltan Quimby" (a trailing 's is not part of the name)
NAME_RE = re.compile(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b")
MAX_FILES, MAX_BYTES = 2000, 200_000
IMPURE_MODULES = {"random", "time", "datetime", "os", "sys", "uuid", "secrets", "tempfile", "socket", "subprocess"}


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


# ---------------------------------------------------------------- coding / tests that cannot pass
def _literal(node: ast.AST) -> Tuple[bool, Any]:
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def pure_functions(lib_source: str) -> Optional[Set[str]]:
    """Names of top-level functions in mylib.py that are safe to treat as pure; None if the file does not parse."""
    try:
        tree = ast.parse(lib_source)
    except (SyntaxError, ValueError, RecursionError):
        return None
    module_vars: Set[str] = set()
    funcs: Dict[str, ast.FunctionDef] = {}
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                module_vars.update(x.id for x in ast.walk(t) if isinstance(x, ast.Name))
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                nm = (a.asname or a.name).split(".")[0]
                if nm in IMPURE_MODULES:
                    module_vars.add(nm)
        elif isinstance(n, ast.FunctionDef):
            funcs[n.name] = n
        elif isinstance(n, ast.ClassDef):
            module_vars.add(n.name)  # classes may carry state
    pure = set(funcs)
    for name, fn in funcs.items():
        bad = bool(fn.decorator_list)
        defaults = list(fn.args.defaults) + [d for d in fn.args.kw_defaults if d is not None]
        bad = bad or any(isinstance(d, (ast.List, ast.Dict, ast.Set)) for d in defaults)
        for x in ast.walk(fn):
            if isinstance(x, (ast.Global, ast.Nonlocal, ast.Yield, ast.YieldFrom, ast.Await)):
                bad = True
            if isinstance(x, ast.Name) and x.id in module_vars:
                bad = True
        if bad:
            pure.discard(name)
    changed = True
    while changed:  # a function calling an impure function is impure too
        changed = False
        for name in list(pure):
            called = {x.id for x in ast.walk(funcs[name]) if isinstance(x, ast.Name) and x.id in funcs and x.id != name}
            if called - pure:
                pure.discard(name)
                changed = True
    return pure


def _imports(tree: ast.Module) -> Tuple[Set[str], Set[str]]:
    modules: Set[str] = set()
    funcs: Set[str] = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            modules.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            funcs.update(a.asname or a.name for a in n.names if a.name != "*")
    return modules, funcs


def _is_pure_call(call: ast.AST, modules: Set[str], funcs: Set[str], pure: Optional[Set[str]]) -> bool:
    """`mod.func(<literals>)` or `func(<literals>)` for a name imported at module level, with >= 1 argument.

    Zero-argument calls and method calls on objects built inside a test (`stack.pop()`, `counter.next()`) can
    legitimately return different values on different calls, so they are never considered. When mylib.py is known,
    the callee must also be one of its pure functions."""
    if not isinstance(call, ast.Call):
        return False
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in modules:
        callee = f.attr
    elif isinstance(f, ast.Name) and f.id in funcs:
        callee = f.id
    else:
        return False
    if pure is not None and callee not in pure:
        return False
    if not (call.args or call.keywords):
        return False
    return all(_literal(a)[0] for a in call.args) and all(_literal(k.value)[0] for k in call.keywords)


def contradictory_tests(source: str, lib_source: Optional[str] = None) -> List[str]:
    """Reasons found in one test file (empty if consistent or unparsable).

    Without mylib.py only calls in DIFFERENT test methods are compared (as before). With mylib.py and a function proven
    pure, two different expectations for the same call are contradictory even inside one method."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    pure = pure_functions(lib_source) if lib_source is not None else None
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
                    if ok and _is_pure_call(call, modules, funcs, pure):
                        key = ast.dump(call)
                        seen.setdefault(key, []).append((method, val))
                        text[key] = ast.unparse(call)
                        break
    out: List[str] = []
    same_method_ok = pure is not None
    for key, items in seen.items():
        for i, (m1, v1) in enumerate(items):
            for m2, v2 in items[i + 1:]:
                if (m1 != m2 or same_method_ok) and v1 != v2:
                    out.append(f"contradictory-tests:{text[key]}=={v1!r}|{v2!r}")
    return sorted(set(out))


def failing_assertions(source: str) -> List[str]:
    """Unconditional assertions that fail whatever mylib.py contains (top-level statements of test methods only,
    undecorated methods only, so skips, expected failures and assertRaises blocks are never misread)."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return []
    out: List[str] = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("test") and not fn.decorator_list):
                continue
            for st in fn.body:
                if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Call)
                        and isinstance(st.value.func, ast.Attribute)):
                    continue
                c, attr = st.value, st.value.func.attr
                if attr in ("assertEqual", "assertEquals") and len(c.args) == 2:
                    (ok1, v1), (ok2, v2) = _literal(c.args[0]), _literal(c.args[1])
                    if ok1 and ok2 and v1 != v2:
                        out.append(f"always-failing-assertion:{ast.unparse(c)}")
                elif attr == "assertTrue" and c.args:
                    ok, v = _literal(c.args[0])
                    if ok and not v:
                        out.append(f"always-failing-assertion:{ast.unparse(c)}")
                elif attr == "assertFalse" and c.args:
                    ok, v = _literal(c.args[0])
                    if ok and v:
                        out.append(f"always-failing-assertion:{ast.unparse(c)}")
                elif attr == "fail" and isinstance(c.func.value, ast.Name) and c.func.value.id == "self":
                    out.append("always-failing-assertion:self.fail()")
    return out


def tests_flags(workspace: Optional[str]) -> List[str]:
    """Reasons why the tests in a coding workspace cannot all pass (empty list if none is provable)."""
    if not workspace:
        return []
    lib_src: Optional[str] = None
    try:
        with open(os.path.join(workspace, "mylib.py"), encoding="utf-8", errors="replace") as f:
            lib_src = f.read(MAX_BYTES)
    except OSError:
        pass
    flags: List[str] = []
    for path in sorted(glob.glob(os.path.join(workspace, "test_*.py")))[:5]:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                src = f.read(MAX_BYTES)
        except OSError:
            continue
        flags.extend(failing_assertions(src))
        flags.extend(contradictory_tests(src, lib_src))
    return flags[:1]  # one reason is enough (each flag multiplies p by 0.1)


def tests_precheck(task: Task) -> List[str]:
    return tests_flags(task.workspace) if task.type == "coding" else []


def precheck(task: Task) -> List[str]:
    return kb_precheck(task) + tests_precheck(task)

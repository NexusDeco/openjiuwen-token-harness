"""Test-verified program repair by single AST mutations (no LLM, 0 tokens).

The coding tasks ("some tests in test_mylib.py fail, fix mylib.py") plant small bugs in otherwise ordinary helper
functions: a flipped comparison, `-=` instead of `+=`, `sorted` instead of `reversed`, `% 2 == 1` instead of `== 0`.
`det_solver.CANON` fixes the functions it knows by name; this module fixes ANY function whose bug is one small edit:

    * every function (and method) of mylib.py is enumerated site by site (comparison / arithmetic / boolean operators,
      well-known call swaps, integer constants +-1, slice bounds and indices, `not`, a wrong variable, a deleted
      statement ...);
    * each mutant is run against the real tests inside a throw-away COPY of the workspace (the original is untouched);
    * a mutant is kept only if the number of failing tests strictly decreases; the best one per round is applied and the
      search continues until every test passes, nothing helps, or the time budget is spent.

Nothing is ever reported as fixed unless the whole test file passes, so the worst case is "no result" (the task then
goes to the LLM exactly as before). Standard library only.
"""
from __future__ import annotations

import ast
import copy
import os
import queue
import shutil
import tempfile
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, List, Optional, Tuple

# (workspace, timeout) -> (all_passed, score); a smaller score is better (e.g. (failing tests, -depth of the failures))
Run = Callable[[str, float], Tuple[bool, tuple]]

_CMP = {
    ast.Lt: [ast.Gt, ast.LtE], ast.Gt: [ast.Lt, ast.GtE], ast.LtE: [ast.GtE, ast.Lt], ast.GtE: [ast.LtE, ast.Gt],
    ast.Eq: [ast.NotEq], ast.NotEq: [ast.Eq], ast.Is: [ast.IsNot], ast.IsNot: [ast.Is],
    ast.In: [ast.NotIn], ast.NotIn: [ast.In],
}
_BIN = {
    ast.Add: [ast.Sub, ast.Mult], ast.Sub: [ast.Add], ast.Mult: [ast.Add, ast.FloorDiv, ast.Div],
    ast.Div: [ast.FloorDiv, ast.Mult], ast.FloorDiv: [ast.Div, ast.Mod, ast.Mult], ast.Mod: [ast.FloorDiv],
    ast.Pow: [ast.Mult], ast.BitAnd: [ast.BitOr], ast.BitOr: [ast.BitAnd],
}
_CALL_SWAP = {
    "min": ["max"], "max": ["min"], "any": ["all"], "all": ["any"], "sorted": ["reversed"], "reversed": ["sorted"],
    "sum": ["len", "max", "min"], "len": ["sum"], "floor": ["ceil"], "ceil": ["floor"], "round": ["int"],
    "int": ["round", "float"], "float": ["int"], "ord": ["chr"], "chr": ["ord"],
}
_ATTR_SWAP = {
    "upper": ["lower"], "lower": ["upper"], "lstrip": ["rstrip", "strip"], "rstrip": ["lstrip", "strip"],
    "startswith": ["endswith"], "endswith": ["startswith"], "append": ["insert"], "pop": ["popleft"],
    "find": ["rfind"], "rfind": ["find"], "title": ["capitalize"], "capitalize": ["title"], "isdigit": ["isalpha"],
    "isalpha": ["isdigit"], "islower": ["isupper"], "isupper": ["islower"], "sort": ["reverse"],
}
_UNWRAP = {"abs", "sorted", "reversed", "list", "set", "int", "float", "round", "str", "tuple", "sum", "len"}

# priority: smaller is more natural; used only to break ties between mutants that fix equally many tests
P_OP, P_BOOL, P_CONST, P_INDEX, P_NOT, P_NAME, P_DELETE = 1, 2, 3, 3, 4, 5, 6

Mutation = Tuple[int, str, Callable[[ast.AST], Optional[ast.AST]]]  # (priority, description, apply)


def _num(v: int) -> ast.Constant:
    return ast.Constant(value=v)


def _plus(node: ast.expr, d: int) -> ast.expr:
    return ast.BinOp(left=node, op=ast.Add() if d > 0 else ast.Sub(), right=_num(abs(d)))


def _swap_operands(x: ast.BinOp) -> None:
    x.left, x.right = x.right, x.left


def _swap_branches(x: ast.If) -> None:
    x.body, x.orelse = x.orelse, x.body


def _node_mutations(n: ast.AST, names: List[str]) -> Iterator[Mutation]:
    if isinstance(n, ast.Compare):
        for j, op in enumerate(n.ops):
            for alt in _CMP.get(type(op), []):
                yield P_OP, f"cmp {type(op).__name__}->{alt.__name__}", \
                    (lambda x, j=j, alt=alt: x.ops.__setitem__(j, alt()))
    elif isinstance(n, (ast.BinOp, ast.AugAssign)):
        for alt in _BIN.get(type(n.op), []):
            yield P_OP, f"op {type(n.op).__name__}->{alt.__name__}", (lambda x, alt=alt: setattr(x, "op", alt()))
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod)):
            yield P_OP, "swap operands", _swap_operands
    elif isinstance(n, ast.BoolOp):
        alt = ast.Or if isinstance(n.op, ast.And) else ast.And
        yield P_BOOL, "and<->or", (lambda x, alt=alt: setattr(x, "op", alt()))
    elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.Not, ast.USub)):
        yield P_BOOL, "drop unary", (lambda x: x.operand)
    elif isinstance(n, ast.Constant):
        v = n.value
        if isinstance(v, bool):
            yield P_BOOL, "flip bool", (lambda x: setattr(x, "value", not x.value))
        elif isinstance(v, int):
            for d in (1, -1):
                yield P_CONST, f"const {d:+d}", (lambda x, d=d: setattr(x, "value", x.value + d))
            if v:
                yield P_CONST, "const negate", (lambda x: setattr(x, "value", -x.value))
        elif isinstance(v, str) and v in ("", " "):
            yield P_CONST, "str ''<->' '", (lambda x: setattr(x, "value", " " if x.value == "" else ""))
    elif isinstance(n, ast.Call):
        f = n.func
        if isinstance(f, ast.Name):
            for alt in _CALL_SWAP.get(f.id, []):
                yield P_OP, f"call {f.id}->{alt}", (lambda x, alt=alt: setattr(x.func, "id", alt))
            if f.id in _UNWRAP and len(n.args) == 1 and not n.keywords:
                yield P_BOOL, f"unwrap {f.id}()", (lambda x: x.args[0])
            if f.id == "sorted" and not any(k.arg == "reverse" for k in n.keywords):
                yield P_BOOL, "sorted(reverse=True)", \
                    (lambda x: x.keywords.append(ast.keyword(arg="reverse", value=ast.Constant(value=True))))
            if f.id == "range":
                if len(n.args) == 1:
                    yield P_INDEX, "range(n)->range(1, n+1)", \
                        (lambda x: setattr(x, "args", [_num(1), _plus(x.args[0], 1)]))
                for k in range(len(n.args)):
                    for d in (1, -1):
                        yield P_INDEX, f"range arg{k}{d:+d}", \
                            (lambda x, k=k, d=d: x.args.__setitem__(k, _plus(x.args[k], d)))
        elif isinstance(f, ast.Attribute):
            for alt in _ATTR_SWAP.get(f.attr, []):
                yield P_OP, f"method {f.attr}->{alt}", (lambda x, alt=alt: setattr(x.func, "attr", alt))
    elif isinstance(n, ast.Slice):
        for fld in ("lower", "upper"):
            cur = getattr(n, fld)
            if cur is not None:
                yield P_INDEX, f"slice drop {fld}", (lambda x, fld=fld: setattr(x, fld, None))
                for d in (1, -1):
                    yield P_INDEX, f"slice {fld}{d:+d}", \
                        (lambda x, fld=fld, d=d: setattr(x, fld, _plus(getattr(x, fld), d)))
    elif isinstance(n, ast.Subscript) and not isinstance(n.slice, (ast.Slice, ast.Tuple, ast.Constant)):
        for d in (1, -1):
            yield P_INDEX, f"index{d:+d}", (lambda x, d=d: setattr(x, "slice", _plus(x.slice, d)))
    elif isinstance(n, (ast.If, ast.While, ast.IfExp)):
        if not (isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not)):
            yield P_NOT, "negate condition", \
                (lambda x: setattr(x, "test", ast.UnaryOp(op=ast.Not(), operand=x.test)))
        if isinstance(n, ast.If) and n.orelse:
            yield P_NOT, "swap if/else", _swap_branches
    elif isinstance(n, (ast.Break, ast.Continue)):
        alt = ast.Continue if isinstance(n, ast.Break) else ast.Break
        yield P_BOOL, "break<->continue", (lambda x, alt=alt: alt())
    elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names:
        for alt in names:
            if alt != n.id:
                yield P_NAME, f"name {n.id}->{alt}", (lambda x, alt=alt: setattr(x, "id", alt))
    # statement deletion (a stray or misplaced line)
    if isinstance(n, (ast.FunctionDef, ast.For, ast.While, ast.If)):
        for fld in ("body", "orelse"):
            body = getattr(n, fld, None) or []
            for k, st in enumerate(body):
                if isinstance(st, (ast.Assign, ast.AugAssign, ast.Expr, ast.If)) and not (
                        isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)):
                    yield P_DELETE, f"delete stmt line {st.lineno}", \
                        (lambda x, fld=fld, k=k: _delete(x, fld, k))


def _delete(node: ast.AST, fld: str, k: int) -> None:
    body = getattr(node, fld)
    del body[k]
    if not body and fld == "body":
        body.append(ast.Pass())


class _Replace(ast.NodeTransformer):
    def __init__(self, target: ast.AST, new: ast.AST):
        self.target, self.new = target, new

    def visit(self, node):
        if node is self.target:
            return self.new
        return self.generic_visit(node)


def _local_names(fn: ast.AST) -> List[str]:
    out: List[str] = []
    args = getattr(fn, "args", None)
    if args is not None:
        out += [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id not in out:
            out.append(n.id)
    return [x for x in out if x not in ("self", "cls", "_")]


def _enumerate(fn: ast.AST) -> List[Tuple[int, int, str, int]]:
    """[(priority, order, description, walk_index)] for one function."""
    names = _local_names(fn)
    out = []
    for i, n in enumerate(ast.walk(fn)):
        for prio, desc, _ in _node_mutations(n, names):
            out.append((prio, len(out), desc, i))
    return out


def _build(fn: ast.AST, walk_index: int, desc: str) -> Optional[ast.AST]:
    """Deep copy of `fn` with exactly the mutation (walk_index, desc) applied; None if it cannot be applied."""
    c = copy.deepcopy(fn)
    nodes = list(ast.walk(c))
    target = nodes[walk_index]
    for _, d, apply in _node_mutations(target, _local_names(c)):
        if d == desc:
            try:
                rep = apply(target)
            except Exception:
                return None
            if rep is not None:  # replacement of the node itself
                if target is c:
                    return rep
                c = _Replace(target, rep).visit(c)
            return ast.fix_missing_locations(c)
    return None


def _functions(tree: ast.Module) -> List[ast.FunctionDef]:
    """Top-level functions and methods of top-level classes (undecorated: their line span is exact)."""
    out: List[ast.FunctionDef] = []
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            out.append(n)
        elif isinstance(n, ast.ClassDef):
            out += [m for m in n.body if isinstance(m, ast.FunctionDef)]
    return [f for f in out if not f.decorator_list]


def _splice(src: str, fn: ast.FunctionDef, new_fn: ast.AST) -> str:
    lines = src.splitlines(keepends=True)
    text = textwrap.indent(ast.unparse(new_fn), " " * fn.col_offset) + "\n"
    return "".join(lines[: fn.lineno - 1]) + text + "".join(lines[fn.end_lineno:])


def _candidates(src: str) -> List[Tuple[int, int, str, str]]:
    """[(priority, order, description, mutated_source)] for every single mutation of every function."""
    tree = ast.parse(src)
    out: List[Tuple[int, int, str, str]] = []
    for fn in _functions(tree):
        for prio, order, desc, idx in _enumerate(fn):
            m = _build(fn, idx, desc)
            if m is None:
                continue
            try:
                new_src = _splice(src, fn, m)
                ast.parse(new_src)
            except Exception:
                continue
            if new_src != src:
                out.append((prio, len(out), f"{fn.name}: {desc}", new_src))
    seen, uniq = set(), []
    for c in out:  # identical sources from different mutations: keep the first
        if c[3] not in seen:
            seen.add(c[3])
            uniq.append(c)
    return uniq


def search(src: str, workspace: str, run: Run, score0: tuple, baseline_s: float = 0.2, budget_s: float = 60.0,
           max_rounds: int = 8, workers: Optional[int] = None) -> Optional[Tuple[str, List[str]]]:
    """Greedy repair. Returns (fixed_source, [applied mutation descriptions]) or None.

    `run(ws, timeout)` runs the tests in directory `ws`; `workspace` is only read (copied), never written.
    """
    deadline = time.monotonic() + budget_s
    # mutants that loop forever are the expensive ones: they can only be cut off by the timeout
    timeout = min(3.0, max(1.0, 5 * baseline_s))
    workers = workers or max(1, min(6, os.cpu_count() or 2))
    root = tempfile.mkdtemp(prefix="repair-")
    try:
        pool_dirs: "queue.Queue[str]" = queue.Queue()
        for k in range(workers):
            d = os.path.join(root, f"w{k}")
            shutil.copytree(workspace, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "answer.txt"))
            pool_dirs.put(d)

        def evaluate(source: str) -> Tuple[bool, tuple]:
            d = pool_dirs.get()
            try:
                with open(os.path.join(d, "mylib.py"), "w", encoding="utf-8") as f:
                    f.write(source)
                return run(d, timeout)
            finally:
                pool_dirs.put(d)

        cur, scur, applied = src, score0, []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in range(max_rounds):
                if time.monotonic() > deadline:
                    return None
                try:
                    cands = _candidates(cur)
                except Exception:
                    return None
                if not cands:
                    return None
                cands.sort(key=lambda c: c[:2])  # most natural mutations are evaluated first
                results: List[Tuple[int, tuple, int, str, str, bool]] = []
                for start in range(0, len(cands), workers * 4):
                    if time.monotonic() > deadline:
                        return None
                    chunk = cands[start:start + workers * 4]
                    for (prio, order, desc, source), (ok, score) in zip(chunk, ex.map(evaluate, [c[3] for c in chunk])):
                        if ok or score < scur:
                            results.append((0 if ok else 1, score, prio * 10 ** 6 + order, desc, source, ok))
                    if any(r[5] for r in results):
                        break  # a mutant passes everything: no need to look at the rest
                if not results:
                    return None
                results.sort(key=lambda r: r[:3])  # all-pass first, then best score, then most natural
                _, score, _, desc, source, ok = results[0]
                cur, scur = source, score
                applied.append(desc)
                if ok:
                    return cur, applied
        return None
    finally:
        shutil.rmtree(root, ignore_errors=True)

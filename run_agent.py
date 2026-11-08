# coding: utf-8
"""Entry point:  python run_agent.py --tasks-dir /data/tasks

Env: TASKS (tasks dir), OUT (answers file or dir), LLM_BASE_URL, LLM_API_KEY (never printed),
     TOKEN_BUDGET (output-token equivalents, unknown by default), PER_TASK_CAP (default 12000),
     MAX_PARALLEL (default 3), TIME_LIMIT_S (default 3300), TIERS (comma-separated model list),
     LOG_DIR (default $TMPDIR/ojw_logs).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from budget import Ledger
from llm import FakeLLM, LLM
from planner import make_plan
from results import ResultStore, resolve_out
from solver import ToolLoopSolver
from tasks import load_tasks


def env_float(name: str, default):
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=float, default=env_float("TOKEN_BUDGET", None))
    ap.add_argument("--per-task-cap", type=float, default=env_float("PER_TASK_CAP", 12000.0))
    ap.add_argument("--parallel", type=int, default=int(env_float("MAX_PARALLEL", 3)))
    ap.add_argument("--time-limit", type=float, default=env_float("TIME_LIMIT_S", 3300.0))
    ap.add_argument("--log-dir", default=os.environ.get("LOG_DIR") or os.path.join(tempfile.gettempdir(), "ojw_logs"))
    ap.add_argument("--resume", action="store_true", help="skip tasks already present in the OUT file")
    ap.add_argument("--fake-llm", action="store_true", help="offline mode: no network, no real tokens")
    ap.add_argument("--engine", choices=["auto", "openjiuwen", "loop"], default=os.environ.get("ENGINE", "auto"),
                    help="openjiuwen = ReActAgent (imports openjiuwen, ~7 s); loop = dependency-free chat loop; "
                         "auto (default) = openjiuwen, falling back to loop if it cannot be loaded or fails inside")
    ap.add_argument("--no-precheck", action="store_true", default=bool(os.environ.get("NO_PRECHECK")),
                    help="disable the LLM-free feasibility pre-checks (for measuring the solver on its own)")
    args = ap.parse_args()
    if not args.tasks_dir or not os.path.isdir(args.tasks_dir):
        print("error: --tasks-dir (or TASKS) must point to the tasks directory", file=sys.stderr)
        return 2

    started = time.time()
    deadline = started + args.time_limit
    os.makedirs(args.log_dir, exist_ok=True)
    ledger = Ledger(total_budget=args.budget, per_task_cap=args.per_task_cap,
                    log_path=os.path.join(args.log_dir, "calls.jsonl"))
    base = key = None
    engine = args.engine
    if args.fake_llm:
        if engine == "openjiuwen":
            print("error: --fake-llm cannot drive the openjiuwen engine (use tools/mock_llm_server.py instead)",
                  file=sys.stderr)
            return 2
        engine = "loop"
        llm = FakeLLM(ledger)
    else:
        base, key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
        if not base or not key:
            print("error: LLM_BASE_URL and LLM_API_KEY must be set", file=sys.stderr)
            return 2
        llm = LLM(base, key, ledger)
    tiers = [t.strip() for t in os.environ.get("TIERS", "").split(",") if t.strip()] or None
    solver_kw = dict(tiers=tiers, log_dir=os.path.join(args.log_dir, "traj"))
    loop_solver = ToolLoopSolver(llm, ledger, **solver_kw)
    if engine in ("auto", "openjiuwen"):
        try:
            from ojw_solver import OjwSolver  # imports openjiuwen (slow, ~7 s) - only when this engine is selected
            solver = OjwSolver(base, key, ledger, fallback=loop_solver if engine == "auto" else None, **solver_kw)
        except Exception as e:
            if engine == "openjiuwen":
                raise
            print(f"[run] WARNING: the openJiuwen engine could not be loaded ({type(e).__name__}: {str(e)[:120]}); "
                  f"using the dependency-free chat loop instead", flush=True)
            solver = loop_solver
    else:
        solver = loop_solver
    print(f"[run] engine: {'openjiuwen (ReActAgent)' if solver is not loop_solver else 'chat loop (no openjiuwen)'}",
          flush=True)

    store = ResultStore(resolve_out(args.out, os.environ.get("OUT")))
    done = store.load_existing() if args.resume else {}
    tasks = load_tasks(args.tasks_dir)
    plan = make_plan([t for t in tasks if t.id not in done], use_prechecks=not args.no_precheck)
    print(f"[run] {len(tasks)} tasks, {len(plan.order)} planned, {len(plan.skipped)} skipped as unlikely, "
          f"{len(done)} already done; budget={args.budget}, parallel={args.parallel}", flush=True)
    for tid, why in plan.skipped.items():
        store.note(tid, status="skipped", reason=why)

    def work(task):
        while True:
            if time.time() > deadline:
                store.note(task.id, status="skipped", reason="deadline")
                return
            verdict = ledger.try_reserve(task.id, plan.est_cost[task.id])
            if verdict == "ok":
                break
            if verdict == "no":
                store.note(task.id, status="skipped", reason="budget")
                return
            time.sleep(0.5)  # 'wait': blocked only by reservations of tasks in flight, which will be released
        scratch = None
        try:
            cwd = task.workspace
            if cwd is None:  # e.g. math: no files, give the agent an empty scratch dir (it must not see task.json)
                cwd = scratch = tempfile.mkdtemp(prefix=task.id + "_")
            res = solver.solve(task, cwd)
            if res.answer is not None:
                store.record(task, res.answer)
            store.note(task.id, status="answered" if res.answer is not None else "no-answer", reward=task.reward,
                       verified=res.verified, spent=round(res.spent, 1), model=res.model, steps=res.steps,
                       reason=res.reason)
            print(f"[{task.id}] {'ok' if res.answer is not None else 'NO ANSWER'} verified={res.verified} "
                  f"spent={res.spent:.0f} model={res.model} steps={res.steps} ({res.reason})", flush=True)
        except Exception as e:  # one bad task must never kill the run
            store.note(task.id, status="error", reason=f"{type(e).__name__}: {str(e)[:120]}")
            print(f"[{task.id}] ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
        finally:
            ledger.release(task.id)
            if scratch:
                shutil.rmtree(scratch, ignore_errors=True)

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        list(pool.map(work, plan.order))

    answered = sum(1 for v in store.report.values() if v.get("status") == "answered")
    store.write_report({"total_spent": round(ledger.spent(), 1), "llm_calls": ledger.calls,
                        "elapsed_s": round(time.time() - started, 1), "answered": answered,
                        "budget": args.budget})
    print(f"[run] done: answered={answered}/{len(tasks)} spent={ledger.spent():.0f} calls={ledger.calls} "
          f"elapsed={time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# coding: utf-8
"""Entry point:  python run_agent.py --tasks-dir /data/tasks

Env: TASKS (tasks dir), OUT (answers file or dir), LLM_BASE_URL, LLM_API_KEY (never printed),
     BUDGET (organisers pass it, e.g. BUDGET=100000; output-token equivalents; TOKEN_BUDGET is the old alias; when
     neither is set we assume the announced 100000), PER_TASK_CAP (default 12000),
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
from det_solver import solve_deterministic
from llm import FakeLLM, LLM
from planner import Plan, make_plan
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
    # The organisers' final launch command was not published yet (due 9 Nov): accept the obvious spellings of the two
    # arguments the rules mention (tasks directory, solutions file) instead of dying on an unknown flag.
    ap.add_argument("--tasks-dir", "--tasks_dir", "--tasks", default=os.environ.get("TASKS"))
    ap.add_argument("--out", "--output", "--out-file", "--output-file", "--solutions", "--solutions-file",
                    "--results", "--result-file", "-o", dest="out", default=None)
    # The organisers announced ~100K tokens, passed in the BUDGET environment variable (kick-off meeting, 7 Nov).
    ap.add_argument("--budget", type=float, default=env_float("BUDGET", env_float("TOKEN_BUDGET", 100000.0)))
    ap.add_argument("--per-task-cap", type=float, default=env_float("PER_TASK_CAP", 12000.0))
    ap.add_argument("--parallel", type=int, default=int(env_float("MAX_PARALLEL", 3)))
    ap.add_argument("--time-limit", type=float, default=env_float("TIME_LIMIT_S", 3300.0))
    ap.add_argument("--log-dir", default=os.environ.get("LOG_DIR") or os.path.join(tempfile.gettempdir(), "ojw_logs"))
    ap.add_argument("--no-auto-stop", action="store_true", default=os.environ.get("AUTO_STOP") == "0",
                    help="always let the model send its own final message (baseline for measuring the auto-stop saving)")
    ap.add_argument("--resume", action="store_true", help="skip tasks already present in the OUT file")
    ap.add_argument("--fake-llm", action="store_true", help="offline mode: no network, no real tokens")
    ap.add_argument("--engine", choices=["auto", "openjiuwen", "loop"], default=os.environ.get("ENGINE", "auto"),
                    help="openjiuwen = ReActAgent (imports openjiuwen, ~7 s); loop = dependency-free chat loop; "
                         "auto (default) = openjiuwen, falling back to loop if it cannot be loaded or fails inside")
    ap.add_argument("--no-deterministic", action="store_true", default=os.environ.get("DETERMINISTIC") == "0",
                    help="skip the no-LLM template solvers (det_solver.py); every task then goes to the LLM engine")
    ap.add_argument("--no-precheck", action="store_true", default=bool(os.environ.get("NO_PRECHECK")),
                    help="disable the LLM-free feasibility pre-checks (for measuring the solver on its own)")
    args, unknown = ap.parse_known_args()
    if unknown:  # never die on an argument we do not know: a lost run costs far more than an ignored flag
        print(f"[run] WARNING: ignoring unknown arguments: {unknown}", file=sys.stderr, flush=True)
        positional = [u for u in unknown if not u.startswith("-")]
        if not args.tasks_dir and positional:       # `run_agent.py <tasks_dir> [<solutions_file>]`
            args.tasks_dir = positional.pop(0)
        if not args.out and positional:
            args.out = positional.pop(0)
    if not args.tasks_dir and os.path.isdir("/data/tasks"):  # the mount point of the organisers' launch command
        args.tasks_dir = "/data/tasks"
    if not args.tasks_dir or not os.path.isdir(args.tasks_dir):
        print("error: --tasks-dir (or TASKS) must point to the tasks directory", file=sys.stderr)
        return 2

    started = time.time()
    deadline = started + args.time_limit
    # Nothing below this line may stop the run before the first task: a log directory that cannot be created is only
    # a lost log (every writer of these logs already tolerates OSError), never a lost answer.
    try:
        os.makedirs(args.log_dir, exist_ok=True)
    except OSError as e:
        print(f"[run] WARNING: cannot create the log directory {args.log_dir!r} ({type(e).__name__}); logging into "
              f"{tempfile.gettempdir()}", file=sys.stderr, flush=True)
        args.log_dir = tempfile.gettempdir()
    ledger = Ledger(total_budget=args.budget, per_task_cap=args.per_task_cap,
                    log_path=os.path.join(args.log_dir, "calls.jsonl"))
    base = key = None
    engine = args.engine
    llm = None
    if args.fake_llm:
        if engine == "openjiuwen":
            print("error: --fake-llm cannot drive the openjiuwen engine (use tools/mock_llm_server.py instead)",
                  file=sys.stderr)
            return 2
        engine = "loop"
        llm = FakeLLM(ledger)
    else:
        base, key = os.environ.get("LLM_BASE_URL"), os.environ.get("LLM_API_KEY")
        if base and key:
            llm = LLM(base, key, ledger)
        else:  # the template solvers need no model: still solve and save everything they can (0 tokens)
            print("[run] WARNING: LLM_BASE_URL and/or LLM_API_KEY are not set: only the deterministic solvers will "
                  "run, tasks that need the LLM stay unanswered", file=sys.stderr, flush=True)
    solver = loop_solver = None
    if llm is not None:
        tiers = [t.strip() for t in os.environ.get("TIERS", "").split(",") if t.strip()] or None
        solver_kw = dict(tiers=tiers, log_dir=os.path.join(args.log_dir, "traj"), auto_stop=not args.no_auto_stop)
        loop_solver = ToolLoopSolver(llm, ledger, **solver_kw)
        if engine in ("auto", "openjiuwen"):
            try:
                from ojw_solver import OjwSolver  # imports openjiuwen (slow, ~7 s) - only when this engine is selected
                solver = OjwSolver(base, key, ledger, fallback=loop_solver if engine == "auto" else None, **solver_kw)
            except Exception as e:
                if engine == "openjiuwen":
                    raise
                print(f"[run] WARNING: the openJiuwen engine could not be loaded ({type(e).__name__}: "
                      f"{str(e)[:120]}); using the dependency-free chat loop instead", flush=True)
                solver = loop_solver
        else:
            solver = loop_solver
    if solver is None:
        print("[run] engine: none (no LLM credentials)", flush=True)
    else:
        print(f"[run] engine: {'openjiuwen (ReActAgent)' if solver is not loop_solver else 'chat loop (no openjiuwen)'}; "
              f"auto-stop {'off' if args.no_auto_stop else 'on'}", flush=True)

    store = ResultStore(resolve_out(args.out, os.environ.get("OUT")))
    done = store.load_existing() if args.resume else {}
    tasks = load_tasks(args.tasks_dir)

    # The answers are only read from <task>/answer.txt. Under `--cap-drop ALL` root has no CAP_DAC_OVERRIDE, so if the
    # task directories belong to someone else and are not world-writable, nothing can be saved: say so loudly at once.
    unwritable = [t.id for t in tasks
                  if not os.access(t.dir, os.W_OK) or (t.workspace and not os.access(t.workspace, os.W_OK))]
    if unwritable:
        print(f"[run] WARNING: {len(unwritable)} of {len(tasks)} task directories are NOT writable by this process "
              f"(uid {os.getuid()}), e.g. {unwritable[:3]}: answer.txt cannot be saved there. The tasks directory must "
              f"be writable for the container user.", file=sys.stderr, flush=True)

    # Pass 1 - no LLM, 0 tokens: strict-template solvers (det_solver.py). Whatever they do not recognise word for word
    # simply stays in the list and goes through the planner and the LLM engine below.
    handled = set()
    if not args.no_deterministic:
        n_ans = n_uns = 0
        for t in tasks:
            if t.id in done:
                continue
            try:
                r = solve_deterministic(t)
            except Exception as e:  # a solver bug must never lose a task: the LLM engine takes it
                print(f"[det] {t.id}: solver error {type(e).__name__}: {str(e)[:80]}; falling back to the LLM",
                      flush=True)
                continue
            if r.status == "answer":
                saved = store.record(t, r.answer)
                store.note(t.id, status="answered" if saved else "write-failed", reward=t.reward, verified=None,
                           spent=0.0, model="deterministic", steps=0, reason=f"deterministic {r.solver}")
                handled.add(t.id)
                n_ans += 1 if saved else 0
            elif r.status == "unsolvable":
                store.clear(t)  # organisers: do not create answer.txt for an unsolvable task
                store.note(t.id, status="skipped", reason=f"deterministic: {r.why}")
                handled.add(t.id)
                n_uns += 1
        print(f"[det] solved without the LLM: {n_ans}, recognised as unsolvable: {n_uns}, "
              f"left for the LLM: {len(tasks) - len(done) - len(handled)}", flush=True)

    rest = [t for t in tasks if t.id not in done and t.id not in handled]
    try:
        plan = make_plan(rest, use_prechecks=not args.no_precheck)
    except Exception as e:  # a planner bug must not cost the tasks the LLM could still solve: plain reward order
        print(f"[run] WARNING: planner failed ({type(e).__name__}: {str(e)[:80]}); using reward order", flush=True)
        plan = Plan(order=sorted(rest, key=lambda t: -t.reward), est_cost={t.id: 600.0 for t in rest},
                    p_success={t.id: 0.5 for t in rest})
    if solver is None:  # no credentials: nothing can be sent to a model
        for t in plan.order:
            store.note(t.id, status="skipped", reason="no LLM credentials")
        plan.order = []
    print(f"[run] {len(tasks)} tasks, {len(handled)} settled deterministically, {len(plan.order)} planned for the LLM, "
          f"{len(plan.skipped)} skipped as unlikely, {len(done)} already done; budget={args.budget}, "
          f"parallel={args.parallel}", flush=True)
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
            saved = False
            if res.answer is not None:
                saved = store.record(task, res.answer)
            else:
                store.clear(task)  # organisers: for an unsolvable task do NOT create answer.txt
            status = ("answered" if saved else "write-failed") if res.answer is not None else "no-answer"
            store.note(task.id, status=status, reward=task.reward, verified=res.verified, spent=round(res.spent, 1),
                       model=res.model, steps=res.steps, reason=res.reason)
            print(f"[{task.id}] {'ok' if saved else ('WRITE FAILED' if res.answer is not None else 'NO ANSWER')} "
                  f"verified={res.verified} spent={res.spent:.0f} model={res.model} steps={res.steps} ({res.reason})",
                  flush=True)
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

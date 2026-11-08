# coding: utf-8
"""Per-task solver: a minimal bash-tool loop with a tiered model policy and free local verification.

This is a dependency-free baseline. An openJiuwen-based solver can be dropped in later: it only has to
provide `solve(task, cwd) -> SolveResult` and charge the same Ledger.

Policy (from measurements on sample + synthetic unsolvable tasks):
  * the model may give up explicitly with IMPOSSIBLE (no answer written, no escalation);
  * a stronger tier is tried ONLY when a local test says the attempt failed - never because steps ran out
    (a stronger model cannot find data that is not there);
  * step caps per task type, set from the steps solvable tasks actually needed (+ margin).
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from budget import Ledger
from llm import LLMError
from tasks import Task

# Cheapest first. Reasoning models are NOT in the default list: they burn output tokens (see probe results).
DEFAULT_TIERS = ["qwen3.6-35b-a3b-noreason", "qwen3.8-27b-noreason"]

# Solvable samples needed 3 (math), 3 (data_analysis), 4-5 (search), 7 (coding) calls.
STEP_CAPS = {"math": 5, "data_analysis": 5, "search": 7, "coding": 9}

SYSTEM = ("You solve one task inside the current working directory using the bash tool. Keep commands short and "
          "their output small. Follow the task instructions exactly, including writing the final answer to "
          "answer.txt. When finished, reply with the single word DONE. If the task cannot be solved because "
          "required data or files are missing, or its requirements contradict each other, do not guess and do not "
          "work around it: reply with exactly IMPOSSIBLE and nothing else.")
BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Run a bash command in the task directory.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}

MIN_ATTEMPT_BUDGET = 300.0   # do not start/continue an attempt with less headroom than this
MAX_TOOL_OUTPUT = 1500       # chars of tool output returned to the model


@dataclass
class SolveResult:
    answer: Optional[str]
    verified: Optional[bool]   # True/False if a local test could judge it, None if there is no verifier
    spent: float
    model: str
    steps: int
    reason: str


def run_bash(cmd: str, cwd: str, timeout: float) -> str:
    # The child gets a minimal environment: it must never see LLM_API_KEY / LLM_BASE_URL.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "LANG": "C.UTF-8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(["bash", "-c", cmd], cwd=cwd, env=env, capture_output=True, text=True,
                           errors="replace", timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            out += f"\n[exit {p.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[timeout after {timeout:.0f}s]"
    if len(out) > MAX_TOOL_OUTPUT:
        half = MAX_TOOL_OUTPUT // 2
        out = out[:half] + "\n...[truncated]...\n" + out[-half:]
    return out or "(no output)"


def verify(cwd: str, timeout: float = 60.0) -> Optional[bool]:
    """Free (no tokens) check: if the workspace has test_*.py, run them. None if there is nothing to run.

    NOTE: passing visible tests is not proof of a correct fix - a model can special-case the tests (seen on a
    task with contradictory tests). Hidden tests will judge the real result.
    """
    tests = sorted(glob.glob(os.path.join(cwd, "test_*.py")))
    if not tests:
        return None
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        p = subprocess.run(["python3", os.path.basename(tests[0])], cwd=cwd, env=env, capture_output=True,
                           timeout=timeout)
        return p.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _answer_path(cwd: str) -> str:
    return os.path.join(cwd, "answer.txt")


def _read_answer(cwd: str) -> Optional[str]:
    try:
        with open(_answer_path(cwd), encoding="utf-8", errors="replace") as f:
            s = f.read().strip()
        return s or None
    except OSError:
        return None


def _remove_answer(cwd: str) -> None:
    try:
        os.remove(_answer_path(cwd))
    except OSError:
        pass


def _is_impossible(s: Optional[str]) -> bool:
    return bool(s) and s.strip().upper().startswith("IMPOSSIBLE")


_FLOAT_INT = re.compile(r"^(-?\d+)\.0+$")


def normalize_answer(a: Optional[str]) -> Optional[str]:
    """`680.0` -> `680`. The model often sums with float() and writes `680.0`; an integer-exact checker (the samples
    use `exact_int`) would reject that although the value is right. Anything else is only stripped."""
    if a is None:
        return None
    s = a.strip()
    m = _FLOAT_INT.match(s)
    return m.group(1) if m else s


def answer_ready(cwd: str, last_tool_output: str = "") -> bool:
    """True when the attempt can end WITHOUT another model call whose only job is to say DONE.

    Every task used to finish with a separate call (the whole history is sent again) in which the model just writes
    DONE, and data tasks added a verifying `cat answer.txt`. Ready means: a non-empty answer.txt exists (a stale one
    is deleted at the start of every attempt), the last command did not fail or time out, and the local tests, if the
    workspace has any, pass. If a test fails the conversation simply continues, so correctness does not change.
    """
    if _read_answer(cwd) is None:
        return False
    tail = (last_tool_output or "")[-80:]
    if re.search(r"\[exit \d+\]\s*$", tail) or "[timeout after" in tail:
        return False
    return verify(cwd) is not False


class BaseSolver:
    """Policy shared by every engine (IMPOSSIBLE, escalation only on a failed local test, step caps, budget).

    An engine only has to implement `_attempt(task, cwd, model, messages) -> (steps, reason, final_text)`: run one
    conversation on `model` inside `cwd`, leave the transcript in `messages` and charge every call to `self.ledger`.
    """

    def __init__(self, ledger: Ledger, tiers: Optional[List[str]] = None, max_steps: int = 8,
                 call_max_tokens: int = 1024, bash_timeout: float = 30.0, log_dir: Optional[str] = None,
                 step_caps: Optional[Dict[str, int]] = None, auto_stop: bool = True):
        self.ledger = ledger
        self.auto_stop = auto_stop  # end an attempt as soon as answer_ready() is true (see there)
        self.tiers = tiers or list(DEFAULT_TIERS)
        self.max_steps, self.call_max_tokens, self.bash_timeout = max_steps, call_max_tokens, bash_timeout
        self.step_caps = STEP_CAPS if step_caps is None else step_caps
        self.log_dir = log_dir

    def _attempt(self, task: Task, cwd: str, model: str, messages: List[Dict[str, Any]]):
        raise NotImplementedError

    def solve(self, task: Task, cwd: str) -> SolveResult:
        attempts: List[Dict[str, Any]] = []
        answer: Optional[str] = None
        verified: Optional[bool] = None
        model, steps, reason = self.tiers[0], 0, "no attempt"
        try:
            for i, model in enumerate(self.tiers):
                room = self.ledger.headroom(task.id)
                if room is not None and room < MIN_ATTEMPT_BUDGET:
                    reason = "budget"
                    break
                _remove_answer(cwd)  # a stale answer from a failed tier must not count
                messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM},
                                                  {"role": "user", "content": task.instruction}]
                steps, reason, final_text = self._attempt(task, cwd, model, messages)
                attempts.append({"model": model, "reason": reason, "messages": messages})
                answer, verified = _read_answer(cwd), verify(cwd)
                if _is_impossible(answer) or _is_impossible(final_text):
                    # Explicit give-up: write nothing, never escalate (a stronger model cannot invent missing data).
                    _remove_answer(cwd)
                    answer, reason = None, "impossible"
                    break
                if answer is None and verified:
                    answer = "DONE"
                if answer is None and final_text and final_text.strip().upper() != "DONE" and len(final_text) <= 200:
                    answer, reason = final_text.strip(), "answer-from-final-text"
                if verified is False and i < len(self.tiers) - 1:
                    continue  # the one reason to escalate: a local test says this attempt failed
                break
        except LLMError as e:
            reason = f"llm-error: {str(e)[:120]}"
        finally:
            self._dump(task, attempts)
        return SolveResult(normalize_answer(answer), verified, self.ledger.spent(task.id), model, steps, reason)

    def _dump(self, task: Task, attempts: List[Dict[str, Any]]) -> None:
        if not self.log_dir:
            return
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, f"{task.id}.json"), "w", encoding="utf-8") as f:
                json.dump({"task": task.id, "type": task.type, "reward": task.reward, "attempts": attempts}, f,
                          ensure_ascii=False, indent=1)
        except OSError:
            pass


class ToolLoopSolver(BaseSolver):
    """Dependency-free engine: a hand-written OpenAI tool-calling loop over `llm.LLM`."""

    def __init__(self, llm, ledger: Ledger, **kwargs):
        super().__init__(ledger, **kwargs)
        self.llm = llm

    def _attempt(self, task: Task, cwd: str, model: str, messages: List[Dict[str, Any]]):
        final_text = ""
        cap = self.step_caps.get(task.type, self.max_steps)
        for step in range(1, cap + 1):
            room = self.ledger.headroom(task.id)
            if room is not None and room < MIN_ATTEMPT_BUDGET:
                return step - 1, "budget", final_text
            max_tokens = self.call_max_tokens if room is None else int(max(64, min(self.call_max_tokens, room)))
            r = self.llm.chat(task.id, model, messages, tools=[BASH_TOOL], max_tokens=max_tokens)
            msg = r["message"]
            # Only send back what the API needs (drops reasoning_content etc. to keep input small).
            keep = {"role": "assistant", "content": msg.get("content") or ""}
            calls = msg.get("tool_calls") or []
            if calls:
                keep["tool_calls"] = calls
            messages.append(keep)
            final_text = msg.get("content") or ""
            if not calls:
                return step, "finished", final_text
            out = ""
            for c in calls:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                    out = run_bash(str(args["command"]), cwd, self.bash_timeout)
                except (KeyError, TypeError, ValueError):
                    out = "error: tool arguments must be JSON like {\"command\": \"...\"}"
                messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": out})
            if self.auto_stop and answer_ready(cwd, out):
                messages.append({"role": "assistant", "content": "DONE"})  # not a model call: nothing is charged
                return step, "finished", "DONE"
        return cap, "max-steps", final_text

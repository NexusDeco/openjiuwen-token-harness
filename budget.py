# coding: utf-8
"""Thread-safe token ledger.

Budget unit = output tokens; 10 input tokens == 1 output token (competition rules).

`completion_tokens` reported by the proxy ALREADY includes reasoning tokens (checked on the
real proxy for qwen3.8-27b and qwen3.6-35b-a3b), so `reasoning_tokens` must never be added on top.
"""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from typing import Dict, Optional

IN_PER_OUT = 10.0


def cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Budget cost of one LLM call in output-token equivalents."""
    return completion_tokens + prompt_tokens / IN_PER_OUT


class Ledger:
    """Global + per-task accounting with reservations so parallel workers cannot overspend together.

    total_budget  : None = unknown/unlimited (only per-task caps apply)
    reserve_frac  : share of the budget try_reserve() will never hand out (safety margin)
    per_task_cap  : hard cap on what a single task may spend (None = no cap)
    """

    def __init__(self, total_budget: Optional[float] = None, per_task_cap: Optional[float] = None,
                 reserve_frac: float = 0.10, log_path: Optional[str] = None):
        self.total_budget = total_budget
        self.per_task_cap = per_task_cap
        self.reserve_frac = reserve_frac
        self._log_path = log_path
        self._lock = threading.Lock()
        self._spent = 0.0
        self._task_spent: Dict[str, float] = defaultdict(float)
        self._reserved: Dict[str, float] = {}
        self.calls = 0

    # ---- accounting -------------------------------------------------------------------
    def charge(self, task_id: str, model: str, prompt_tokens: int, completion_tokens: int, note: str = "") -> float:
        c = cost(prompt_tokens, completion_tokens)
        with self._lock:
            self._spent += c
            self._task_spent[task_id] += c
            self.calls += 1
            if self._log_path:
                rec = {"ts": round(time.time(), 3), "task": task_id, "model": model, "prompt": prompt_tokens,
                       "completion": completion_tokens, "cost": round(c, 2), "note": note}
                try:
                    with open(self._log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except OSError:
                    pass
        return c

    def spent(self, task_id: Optional[str] = None) -> float:
        with self._lock:
            return self._spent if task_id is None else self._task_spent[task_id]

    def remaining(self) -> Optional[float]:
        if self.total_budget is None:
            return None
        with self._lock:
            return self.total_budget - self._spent

    # ---- reservations -----------------------------------------------------------------
    def _outstanding(self, exclude: Optional[str] = None) -> float:
        # what other tasks may still spend according to their reservation (caller holds the lock)
        return sum(max(0.0, r - self._task_spent[t]) for t, r in self._reserved.items() if t != exclude)

    def try_reserve(self, task_id: str, est: float) -> str:
        """'ok' (reserved) | 'wait' (would fit once tasks in flight release their reservations) | 'no' (never fits)."""
        with self._lock:
            if self.total_budget is not None:
                usable = self.total_budget * (1.0 - self.reserve_frac) - self._spent
                if est > usable:
                    return "no"
                if est > usable - self._outstanding():
                    return "wait"
            self._reserved[task_id] = est
            return "ok"

    def release(self, task_id: str) -> None:
        with self._lock:
            self._reserved.pop(task_id, None)

    def headroom(self, task_id: str) -> Optional[float]:
        """How much this task may still spend right now (None = unlimited)."""
        with self._lock:
            limits = []
            if self.per_task_cap is not None:
                limits.append(self.per_task_cap - self._task_spent[task_id])
            if self.total_budget is not None:
                limits.append(self.total_budget - self._spent - self._outstanding(exclude=task_id))
            return min(limits) if limits else None

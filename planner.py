# coding: utf-8
"""Task selection and ordering: maximise  reward * p(success) / expected_tokens.

!! PRIORS ARE ESTIMATES from very few samples. Re-calibrate on real runs (collect tokens + success per task type,
   then replace this table with a regression/boosting model over tasks.features()).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from prechecks import precheck
from tasks import Task, missing_files

# task_type -> (p_success, expected cost in output-token equivalents)
# Cost = mean measured per type with qwen3.6-35b-a3b-noreason (8 organiser samples, then 32 synthetic solvable tasks
# in the same formats, openjiuwen engine): math 353 (n=8), data_analysis 309 (n=6), search 458 (n=6; 3-hop probes
# 622), coding 1033 (n=8).   p_success: 28/28 solved in the sample formats, so these stay deliberately cautious.
PRIORS: Dict[str, Tuple[float, float]] = {
    "math": (0.90, 350.0),
    "search": (0.90, 460.0),
    "data_analysis": (0.85, 310.0),
    "coding": (0.80, 1050.0),
}
DEFAULT_PRIOR = (0.50, 1500.0)
MIN_P = 0.15  # below this a task is not attempted at all
FLAG_FACTOR = 0.1  # every feasibility flag multiplies p(success) by this


@dataclass
class Plan:
    order: List[Task] = field(default_factory=list)
    est_cost: Dict[str, float] = field(default_factory=dict)
    p_success: Dict[str, float] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)  # task_id -> reason


def estimate(task: Task, use_prechecks: bool = True) -> Tuple[float, float, List[str]]:
    p, est = PRIORS.get(task.type, DEFAULT_PRIOR)
    flags: List[str] = []
    miss = missing_files(task)
    if miss:  # instruction refers to files that are not there: probably unsolvable
        flags.append(f"missing:{','.join(miss)}")
    if use_prechecks:
        flags.extend(precheck(task))
    for _ in flags:
        p *= FLAG_FACTOR
    # TODO: scale `est` with tasks.features(task) (csv_rows, n_files, instr_chars) once data is collected.
    return p, est, flags


def make_plan(tasks: List[Task], use_prechecks: bool = True) -> Plan:
    plan = Plan()
    scored = []
    for t in tasks:
        p, est, flags = estimate(t, use_prechecks)
        plan.est_cost[t.id], plan.p_success[t.id] = est, p
        if p < MIN_P:
            plan.skipped[t.id] = "unlikely solvable (" + "; ".join(flags or ["low prior"]) + ")"
            continue
        scored.append((t.reward * p / est, t))
    scored.sort(key=lambda s: -s[0])
    plan.order = [t for _, t in scored]
    return plan

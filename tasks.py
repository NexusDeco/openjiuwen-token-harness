# coding: utf-8
"""Loading task directories and cheap (no-LLM) features.

The model only ever sees `instruction`. `expected` / `check` fields (present in the sample task.json files)
are never loaded into the Task object, so they cannot leak into prompts.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

FILE_RE = re.compile(r"(?<![\w/.-])([A-Za-z_][\w.-]*\.(?:csv|tsv|py|md|json|txt|xlsx|yaml|yml))(?![\w])")
DIR_RE = re.compile(r"(?<![\w/.-])([A-Za-z_][\w-]*)/(?=[\s,.;:)]|$)")


@dataclass
class Task:
    id: str
    dir: str
    type: str
    instruction: str
    reward: float
    workspace: Optional[str]  # <task>/workspace if it exists; the agent works there (instructions say "current directory")


def load_tasks(tasks_dir: str) -> List[Task]:
    tasks: List[Task] = []
    for name in sorted(os.listdir(tasks_dir)):
        d = os.path.join(tasks_dir, name)
        meta_path = os.path.join(d, "task.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[tasks] skip {name}: unreadable task.json ({type(e).__name__})", file=sys.stderr)
            continue
        if not isinstance(meta, dict):
            print(f"[tasks] skip {name}: task.json is not a JSON object", file=sys.stderr)
            continue
        tid = str(meta.get("task_id") or name)
        ttype = str(meta.get("task_type") or tid.rsplit("_", 1)[0])
        try:
            reward = float(meta.get("reward", 1))
        except (TypeError, ValueError):
            reward = 1.0
        ws = os.path.join(d, "workspace")
        tasks.append(Task(id=tid, dir=d, type=ttype, instruction=str(meta.get("instruction", "")),
                          reward=reward, workspace=ws if os.path.isdir(ws) else None))
    return tasks


def missing_files(task: Task) -> List[str]:
    """Files/dirs the instruction mentions that do not exist in the workspace.

    A *signal*, not proof of an unsolvable task (regexes can misfire), so the planner only lowers p(success).
    """
    names = {m for m in FILE_RE.findall(task.instruction)} | {m + "/" for m in DIR_RE.findall(task.instruction)}
    # Files the boilerplate talks about but that never live in the workspace: the answer file we write, task.json that
    # sits next to it ("write the answer next to task.json"), and the workspace directory itself.
    names -= {"answer.txt", "task.json", "workspace/"}
    base = task.workspace
    return sorted(n for n in names if base is None or not os.path.exists(os.path.join(base, n)))


def features(task: Task) -> Dict[str, float]:
    """Cheap numeric features for the (future) cost estimator."""
    n_files = total_bytes = csv_rows = 0
    if task.workspace:
        for root, _dirs, files in os.walk(task.workspace):
            for fn in files:
                n_files += 1
                p = os.path.join(root, fn)
                try:
                    total_bytes += os.path.getsize(p)
                    if fn.endswith(".csv") and csv_rows == 0:
                        with open(p, "rb") as f:
                            csv_rows = sum(1 for _ in f) - 1
                except OSError:
                    pass
    return {"instr_chars": len(task.instruction), "n_files": n_files, "total_bytes": total_bytes,
            "csv_rows": max(csv_rows, 0), "reward": task.reward}

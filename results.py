# coding: utf-8
"""Persisting answers. A task counts only if its answer is on disk, so every answer is written immediately:
<task_dir>/answer.txt  and  the OUT file (JSON {task_id: answer}), the latter atomically (tmp + rename).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from typing import Any, Dict

from tasks import Task


def resolve_out(arg_out: str | None, env_out: str | None) -> str:
    """The rules call OUT 'the output file' but also run `mkdir -p "$OUT"`; accept both a file and a directory.

    OUT may be a HOST path that does not exist inside the container (`-e OUT` forwards only the value) and the root
    file system is read-only there, so creating its directory can fail. That must never stop the run: the answers the
    organisers read are the answer.txt files in the task directories, this file is only a copy. Fall back to the
    temporary directory (and say so) instead of dying before the first task."""
    p = arg_out or env_out or os.path.join(tempfile.gettempdir(), "solutions.json")
    if p.endswith("/") or os.path.isdir(p) or not os.path.splitext(p)[1]:
        p = os.path.join(p, "solutions.json")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    except OSError as e:
        fallback = os.path.join(tempfile.gettempdir(), "solutions.json")
        print(f"[results] WARNING: cannot create the directory for the solutions file {p!r} ({type(e).__name__}: {e}); "
              f"keeping that copy in {fallback}. answer.txt in every task directory is not affected.",
              file=sys.stderr, flush=True)
        p = fallback
    return p


class ResultStore:
    def __init__(self, out_path: str):
        self.out_path = out_path
        self._lock = threading.Lock()
        self.answers: Dict[str, str] = {}
        self.report: Dict[str, Any] = {}

    def load_existing(self) -> Dict[str, str]:
        try:
            with open(self.out_path, encoding="utf-8") as f:
                data = json.load(f)
            self.answers.update({str(k): str(v) for k, v in data.items()})
        except (OSError, ValueError, AttributeError):
            pass
        return dict(self.answers)

    def record(self, task: Task, answer: str) -> bool:
        """Save the answer. Returns True only if answer.txt really landed next to task.json (the file the organisers
        read); a failure is logged and must not be reported as a solved task."""
        written = True
        with self._lock:
            self.answers[task.id] = answer
            for d in {task.dir, task.workspace}:
                if d:
                    try:
                        with open(os.path.join(d, "answer.txt"), "w", encoding="utf-8") as f:
                            f.write(answer.strip() + "\n")
                    except OSError as e:
                        if d == task.dir:
                            written = False
                        print(f"[results] cannot write answer.txt in {d}: {e}", file=sys.stderr)
            self._flush(self.out_path, self.answers)
        return written

    def clear(self, task: Task) -> None:
        """No answer for this task: make sure no answer.txt is left behind (the model may have written a
        placeholder before giving up). The organisers asked us not to create the file for unsolvable tasks."""
        with self._lock:
            self.answers.pop(task.id, None)
            for d in {task.dir, task.workspace}:
                if d:
                    try:
                        os.remove(os.path.join(d, "answer.txt"))
                    except FileNotFoundError:
                        pass
                    except OSError as e:
                        print(f"[results] cannot remove answer.txt in {d}: {e}", file=sys.stderr)

    def note(self, task_id: str, **info: Any) -> None:
        with self._lock:
            self.report[task_id] = info

    def write_report(self, extra: Dict[str, Any]) -> None:
        with self._lock:
            path = os.path.splitext(self.out_path)[0] + ".report.json"
            self._flush(path, {**extra, "tasks": self.report})

    @staticmethod
    def _flush(path: str, data: Any) -> None:
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except OSError as e:
            print(f"[results] cannot write {path}: {e}", file=sys.stderr)

# coding: utf-8
"""Thin OpenAI-compatible client for the organisers' proxy (stdlib only) + an offline fake for tests.

Every call is charged to the ledger from `usage` (prompt_tokens / completion_tokens). The API key is read
by the caller from the environment and is never logged or put into exception messages.
"""
from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from budget import Ledger

# Only these four models are allowed by the rules; anything else is refused client-side.
ALLOWED_MODELS = (
    "qwen3.8-27b",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-noreason",
)
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Deliberately pessimistic (3 chars/token) estimate, used when usage is unavailable."""
    chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    return chars // 3 + 8


class LLM:
    def __init__(self, base_url: str, api_key: str, ledger: Ledger, timeout: float = 180.0, max_retries: int = 4):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self.ledger = ledger
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, task_id: str, model: str, messages: List[Dict[str, Any]], tools: Optional[list] = None,
             max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        if model not in ALLOWED_MODELS:
            raise ValueError(f"model {model!r} is not allowed")
        payload: Dict[str, Any] = {"model": model, "messages": messages, "max_tokens": int(max_tokens),
                                   "temperature": temperature}
        if tools:
            payload["tools"] = tools
        data = json.dumps(payload).encode()
        last = "no attempt made"
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self._url, data=data, method="POST", headers={
                "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = json.loads(r.read().decode("utf-8", "replace"))
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    # The proxy echoes a masked copy of the key in this body: never keep it (stdout/report/logs).
                    last = f"http {e.code}: authentication rejected"
                else:
                    last = f"http {e.code}: " + e.read().decode("utf-8", "replace")[:300].replace(self._key, "***")
                if e.code not in RETRY_STATUS:
                    raise LLMError(last) from None  # 400/401/403/404: retrying will not help
                self._sleep(attempt, e.headers.get("Retry-After"))
                continue
            except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
                last = f"network: {type(e).__name__}"
                # The request may have reached the model: charge a pessimistic prompt-only estimate.
                self.ledger.charge(task_id, model, estimate_tokens(messages), 0, note="lost-request-estimate")
                self._sleep(attempt, None)
                continue
            if "choices" not in body:
                raise LLMError(f"unexpected response: {json.dumps(body, ensure_ascii=False)[:300]}")
            choice = body["choices"][0]
            msg = choice.get("message") or {}
            usage = body.get("usage") or {}
            if "prompt_tokens" in usage and "completion_tokens" in usage:
                pt, ct, note = int(usage["prompt_tokens"]), int(usage["completion_tokens"]), ""
            else:  # no usage in the answer: fall back to a pessimistic estimate
                pt = estimate_tokens(messages)
                ct = len(json.dumps(msg, ensure_ascii=False)) // 3 + 1
                note = "usage-missing-estimate"
            self.ledger.charge(task_id, model, pt, ct, note=note)
            return {"message": msg, "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": pt, "completion_tokens": ct}
        raise LLMError(f"giving up after {self.max_retries + 1} attempts ({last})")

    @staticmethod
    def _sleep(attempt: int, retry_after: Optional[str]) -> None:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        time.sleep(min(30.0, max(delay, 2.0 ** attempt) + random.random()))


class FakeLLM:
    """Offline stand-in: writes `0` into answer.txt via a bash tool call, then says DONE. Charges fake usage."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def chat(self, task_id, model, messages, tools=None, max_tokens=512, temperature=0.0):
        pt = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages) // 4 + 20
        if not any(m.get("role") == "tool" for m in messages):
            msg = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": "echo 0 > answer.txt"})}}]}
            ct = 12
        else:
            msg, ct = {"role": "assistant", "content": "DONE"}, 2
        time.sleep(0.05)
        self.ledger.charge(task_id, model, pt, ct, note="fake")
        return {"message": msg, "finish_reason": "stop", "prompt_tokens": pt, "completion_tokens": ct}

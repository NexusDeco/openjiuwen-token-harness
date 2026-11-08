# coding: utf-8
"""openJiuwen engine: a ReActAgent driven by our policy (solver.BaseSolver) and charged to our token ledger.

Why a bare ReActAgent and not create_deep_agent (measured with tools/mock_llm_server.py + the real proxy, first call
of the same task, prompt tokens): ReActAgent + our system prompt 452 (identical to the hand-written loop), DeepAgent
568 by default / 434 without its rails; DeepAgent also fires a hidden image-capability probe request at start-up
(enable_read_image_multimodal=None) and ships retry logic only inside a rail. ReActAgent sends exactly what we give it.

openJiuwen facts this module relies on (all checked against 0.1.19):
  * ReActAgent has NO transient-error retry (max_retries in ModelClientConfig is not passed to the SDK), so
    LedgerModel retries 429 / 5xx / timeouts itself;
  * model errors surface as FrameworkError whose __cause__ is the openai SDK exception;
  * temperature / max_tokens are not set by default -> LedgerModel sets them on every call;
  * by default the library also logs to ./logs/ (relative to the CWD): with `docker run --read-only` that makes the
    agent fail at start-up (FrameworkError 183001), hence the logging prelude below.
"""
from __future__ import annotations

# --- Must run before ANY other openjiuwen import: its loggers initialise lazily from this dict. -----------------
import os
import tempfile

from openjiuwen.core.common.logging.default.constant import DEFAULT_INNER_LOG_CONFIG as _LOG_CFG

# CRITICAL, not WARNING: at ERROR level the library prints whole prompts and, for a 401, the masked key the proxy
# echoes back, and it logs every transient failure that LedgerModel retries on purpose.
_LOG_CFG.update(level="CRITICAL", log_path=os.path.join(tempfile.gettempdir(), "ojw_logs") + "/",
                output=["console"], interface_output=["console"], performance_output=["console"])
# -------------------------------------------------------------------------------------------------------------

import asyncio
import random
import threading
import uuid
from typing import Any, Dict, List, Optional

from openjiuwen.core.foundation.llm import Model
from openjiuwen.core.foundation.llm.schema.config import ModelClientConfig, ModelRequestConfig
from openjiuwen.core.foundation.llm.schema.message import AssistantMessage
from openjiuwen.core.foundation.tool import ToolCard
from openjiuwen.core.foundation.tool.function.function import LocalFunction
from openjiuwen.core.single_agent.agents.react_agent import ReActAgent, ReActAgentConfig
from openjiuwen.core.single_agent.schema.agent_card import AgentCard

from budget import Ledger
from llm import LLMError, estimate_tokens
from solver import BASH_TOOL, MIN_ATTEMPT_BUDGET, SYSTEM, BaseSolver, answer_ready, run_bash
from tasks import Task

RETRYABLE = {"RateLimitError", "InternalServerError", "APITimeoutError", "APIConnectionError"}
LOST_REQUEST = {"APITimeoutError", "APIConnectionError"}  # the request may have reached the model
AUTH_ERRORS = {"AuthenticationError", "PermissionDeniedError"}


class OutOfBudget(Exception):
    pass


def _last_tool_text(messages) -> Optional[str]:
    """Text of the newest message if it is a tool result (i.e. a command has just finished), else None."""
    if not messages:
        return None
    m = messages[-1]
    role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
    if role != "tool":
        return None
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
    return c if isinstance(c, str) else str(c)


def _plain(m: Any) -> Dict[str, Any]:
    if isinstance(m, dict):
        return m
    try:
        return m.model_dump()
    except Exception:  # pragma: no cover - defensive
        return {"role": getattr(m, "role", "?"), "content": str(getattr(m, "content", ""))}


class LedgerModel(Model):
    """openJiuwen Model that charges every call to the Ledger, enforces per-call/task caps and retries transient
    failures. One instance per attempt, so `task_id` is fixed."""

    def __init__(self, client_cfg: ModelClientConfig, req_cfg: ModelRequestConfig, *, ledger: Ledger, task_id: str,
                 api_key: str, call_max_tokens: int, cwd: str, task_dir: Optional[str] = None, auto_stop: bool = True,
                 max_attempts: int = 5):
        super().__init__(client_cfg, req_cfg)
        self._lg_ledger, self._lg_task, self._lg_key = ledger, task_id, api_key
        self._lg_call_max, self._lg_attempts = call_max_tokens, max_attempts
        self._lg_cwd, self._lg_task_dir, self._lg_auto_stop = cwd, task_dir, auto_stop
        self._lg_name = req_cfg.model_name
        self.calls = 0
        self.auto_stopped = False
        self.out_of_budget = False
        self.last_messages: Optional[List[Any]] = None
        self.last_reply: Any = None

    async def invoke(self, messages, *, tools=None, **kwargs):  # type: ignore[override]
        if self._lg_auto_stop:
            # A command has just finished and the answer is on disk: skip the model call whose only job would be
            # to write DONE (see solver.answer_ready). Run in a thread: it may execute the task's tests.
            tool_text = _last_tool_text(messages)
            if tool_text is not None and await asyncio.to_thread(
                    answer_ready, self._lg_cwd, tool_text, self._lg_task_dir):
                reply = AssistantMessage(content="DONE")
                self.last_messages, self.last_reply, self.auto_stopped = messages, reply, True
                return reply
        last = "no attempt made"
        for attempt in range(self._lg_attempts):
            room = self._lg_ledger.headroom(self._lg_task)
            if room is not None and room < MIN_ATTEMPT_BUDGET:
                self.out_of_budget = True
                raise OutOfBudget()
            kwargs["max_tokens"] = self._lg_call_max if room is None else int(max(64, min(self._lg_call_max, room)))
            kwargs.setdefault("temperature", 0.0)
            try:
                reply = await super().invoke(messages, tools=tools, **kwargs)
            except OutOfBudget:
                raise
            except Exception as e:  # FrameworkError wrapping the SDK error
                cause = e.__cause__ or e.__context__
                name = type(cause).__name__ if cause is not None else type(e).__name__
                if name in AUTH_ERRORS:
                    # The proxy echoes a masked key in 401 bodies: never keep the text.
                    raise LLMError(f"{name}: authentication rejected") from None
                last = f"{name}: {str(e).replace(self._lg_key, '***')[:160]}"
                if name not in RETRYABLE:
                    raise LLMError(last) from None
                if name in LOST_REQUEST:
                    self._lg_ledger.charge(self._lg_task, self._lg_name, estimate_tokens([_plain(m) for m in messages]),
                                           0, note="lost-request-estimate")
                headers = getattr(getattr(cause, "response", None), "headers", None)
                try:
                    delay = float(headers.get("retry-after")) if headers and headers.get("retry-after") else 0.0
                except (TypeError, ValueError):
                    delay = 0.0
                await asyncio.sleep(min(30.0, max(delay, 2.0 ** attempt) + random.random()))
                continue
            self._charge(messages, reply)
            self.calls += 1
            self.last_messages, self.last_reply = messages, reply
            return reply
        raise LLMError(f"giving up after {self._lg_attempts} attempts ({last})")

    async def stream(self, *args, **kwargs):  # type: ignore[override]
        raise RuntimeError("streaming is disabled: it would bypass the token ledger")

    def _charge(self, messages, reply) -> None:
        um = getattr(reply, "usage_metadata", None)
        pt = int(getattr(um, "input_tokens", 0) or 0)
        ct = int(getattr(um, "output_tokens", 0) or 0)
        note = ""
        if pt == 0 and ct == 0:  # provider returned no usage: fall back to a pessimistic estimate
            pt = estimate_tokens([_plain(m) for m in messages])
            ct = len(str(getattr(reply, "content", "") or "")) // 3 + 1
            note = "usage-missing-estimate"
        self._lg_ledger.charge(self._lg_task, self._lg_name, pt, ct, note=note)


class _LoopThread:
    """One asyncio loop in a background thread: openJiuwen keeps a shared long-lived HTTP client that must live on a
    single loop, while run_agent.py drives tasks from a thread pool."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="ojw-loop", daemon=True).start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


class OjwSolver(BaseSolver):
    """`fallback`: an engine used when the framework itself fails (anything except an LLM/auth error, which would
    fail the same way there). Losing the whole qualifying run to a framework bug would cost far more than a
    differently-implemented chat loop, so the fallback is on by default in run_agent.py (`--engine auto`)."""

    def __init__(self, base_url: str, api_key: str, ledger: Ledger, fallback: Optional[BaseSolver] = None, **kwargs):
        super().__init__(ledger, **kwargs)
        self._base, self._key = base_url.rstrip("/"), api_key
        self._fallback = fallback
        self._warned = False
        self._lt: Optional[_LoopThread] = None
        self._lt_lock = threading.Lock()

    def _loop_thread(self) -> _LoopThread:
        with self._lt_lock:
            if self._lt is None:
                self._lt = _LoopThread()
            return self._lt

    def _attempt(self, task: Task, cwd: str, model: str, messages: List[Dict[str, Any]]):
        try:
            return self._loop_thread().run(self._attempt_async(task, cwd, model, messages))
        except LLMError:
            raise
        except Exception as e:  # a framework failure, not a model/proxy one
            what = f"{type(e).__name__}: {str(e).replace(self._key, '***')[:160]}"
            if self._fallback is None:
                raise LLMError(f"openjiuwen engine failed: {what}") from None
            if not self._warned:
                self._warned = True
                print(f"[ojw] WARNING: openJiuwen engine failed ({what}); using the fallback chat loop for this and "
                      f"later failures", flush=True)
            messages[:] = [m for m in messages if m.get("role") in ("system", "user")][:2]
            return self._fallback._attempt(task, cwd, model, messages)

    async def _attempt_async(self, task: Task, cwd: str, model: str, messages: List[Dict[str, Any]]):
        cap = self.step_caps.get(task.type, self.max_steps)
        llm = LedgerModel(
            ModelClientConfig(client_provider="OpenAI", api_key=self._key, api_base=self._base, timeout=120.0),
            ModelRequestConfig(model_name=model),
            ledger=self.ledger, task_id=task.id, api_key=self._key, call_max_tokens=self.call_max_tokens,
            cwd=cwd, task_dir=task.dir, auto_stop=self.auto_stop)
        # Unique ids: the framework keeps tools and sessions in process-wide registries keyed by these ids.
        aid = f"{task.id}-{uuid.uuid4().hex[:8]}"
        agent = ReActAgent(AgentCard(id=aid, name="solver", description="Solves one task with bash."))
        cfg = ReActAgentConfig().configure_model_client("OpenAI", self._key, self._base, model)
        cfg.configure_max_iterations(cap)  # one iteration == one model call, same meaning as the step cap
        cfg.configure_prompt_template([{"role": "system", "content": SYSTEM}])
        agent.configure(cfg)
        agent.set_llm(llm)

        async def bash(command: str) -> str:
            return await asyncio.to_thread(run_bash, command, cwd, self.bash_timeout)

        fn = BASH_TOOL["function"]
        card = ToolCard(id="bash", name=fn["name"], description=fn["description"], input_params=fn["parameters"])
        agent.ability_manager.add_ability(card, LocalFunction(card, bash))

        result: Any = None
        try:
            result = await agent.invoke({"query": task.instruction, "conversation_id": aid})
        except Exception:
            if not llm.out_of_budget:
                raise
        finally:
            if llm.last_messages is not None:  # leave the transcript in `messages` for the trajectory dump
                messages[:] = [_plain(m) for m in llm.last_messages] + [_plain(llm.last_reply)]

        last_text = str(getattr(llm.last_reply, "content", "") or "")
        if llm.out_of_budget:
            return llm.calls, "budget", last_text
        out = str(result.get("output", "")) if isinstance(result, dict) else str(result)
        if isinstance(result, dict) and result.get("result_type") == "error":
            if "max iterations" in out.lower():
                return llm.calls, "max-steps", last_text
            raise LLMError(f"agent error: {out[:150]}")
        return llm.calls, "finished", out

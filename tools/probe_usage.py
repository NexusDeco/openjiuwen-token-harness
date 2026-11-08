"""Step 2 probe: how the proxy reports token usage. Tiny calls only. Never prints the API key."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ["LLM_BASE_URL"].rstrip("/")
KEY = os.environ["LLM_API_KEY"]
OUT = sys.argv[1]  # where to dump the raw results (no secrets in there)

MODELS = [
    "qwen3.6-35b-a3b-noreason",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.8-27b",
]
PROMPT = "What is 17*3? Answer with just the number."
MAX_TOKENS = 400  # hard cap so a chatty reasoning model cannot burn the budget
HDR_HINTS = ("limit", "balance", "remaining", "token", "cost", "usage", "budget", "credit", "quota", "spent")


def call(path, payload=None, stream=False):
    req = urllib.request.Request(
        BASE + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=120)
        status, headers = r.status, dict(r.headers)
        body = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status, headers, body = e.code, dict(e.headers), e.read().decode("utf-8", "replace")
    return status, headers, body, time.time() - t0


def interesting_headers(h):
    return {k: v for k, v in h.items() if any(s in k.lower() for s in HDR_HINTS)}


results = {"chat": [], "stream": None, "probes": []}
total_in = total_out = 0

print(f"== non-streaming chat, max_tokens={MAX_TOKENS} ==")
for m in MODELS:
    payload = {"model": m, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": MAX_TOKENS, "temperature": 0}
    status, headers, body, dt = call("/chat/completions", payload)
    try:
        d = json.loads(body)
    except Exception:
        print(f"[{m}] http={status} non-JSON: {body[:200]}")
        continue
    results["chat"].append({"model": m, "status": status, "response": d, "header_names": sorted(headers), "interesting_headers": interesting_headers(headers)})
    if "choices" not in d:
        print(f"[{m}] http={status} error: {json.dumps(d, ensure_ascii=False)[:300]}")
        continue
    ch = d["choices"][0]
    msg = ch.get("message", {})
    usage = d.get("usage", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    reasoning_keys = [k for k in msg if "reason" in k.lower() or "think" in k.lower()]
    total_in += usage.get("prompt_tokens", 0)
    total_out += usage.get("completion_tokens", 0)
    print(f"[{m}] http={status} {dt:.1f}s finish={ch.get('finish_reason')}")
    print(f"   usage = {json.dumps(usage, ensure_ascii=False)}")
    print(f"   content({len(content)} chars) = {content[:80]!r}")
    print(f"   reasoning fields in message: {reasoning_keys or 'none'}; reasoning text length = {len(reasoning)} chars")
    print(f"   message keys: {sorted(msg)}")
    print(f"   response top-level keys: {sorted(d)}")
    ih = interesting_headers(headers)
    print(f"   budget-like headers: {ih or 'none'}")

print()
print("== streaming (usage chunk?) on qwen3.6-35b-a3b-noreason ==")
payload = {"model": "qwen3.6-35b-a3b-noreason", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 50,
           "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
status, headers, body, dt = call("/chat/completions", payload)
usage_chunks = []
n_chunks = 0
for line in body.splitlines():
    if line.startswith("data:") and line.strip() != "data: [DONE]":
        try:
            c = json.loads(line[5:].strip())
        except Exception:
            continue
        n_chunks += 1
        if c.get("usage"):
            usage_chunks.append(c["usage"])
results["stream"] = {"status": status, "chunks": n_chunks, "usage_chunks": usage_chunks}
print(f"   http={status} chunks={n_chunks} usage chunks={json.dumps(usage_chunks, ensure_ascii=False)}")
for u in usage_chunks[-1:]:
    total_in += u.get("prompt_tokens", 0)
    total_out += u.get("completion_tokens", 0)

print()
print("== GET probes for a balance/usage endpoint (no tokens spent) ==")
for p in ["/usage", "/balance", "/credits", "/key", "/me", "/dashboard/billing/usage"]:
    status, headers, body, dt = call(p)
    snippet = body.strip().replace("\n", " ")[:160]
    results["probes"].append({"path": p, "status": status, "body": body[:2000]})
    print(f"   GET {p}: http={status} body={snippet!r}")

print()
print(f"TOTAL spent in this probe: input={total_in} output={total_out} -> budget-equivalent output tokens = {total_out + total_in / 10:.1f}")
json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1)

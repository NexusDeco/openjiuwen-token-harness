"""Tiny OpenAI-compatible mock of the proxy for OFFLINE tests (no network, no real tokens).

    python tools/mock_llm_server.py [--mode answer|impossible] [--fail-first K] [--fail-status 429] [--require-key KEY]

Prints `PORT <n>` once it listens on 127.0.0.1, then serves until killed. Point the harness at it:
    LLM_BASE_URL=http://127.0.0.1:<n>/v1 LLM_API_KEY=mock python run_agent.py --tasks-dir ...

* mode `answer`     : first call of a conversation -> tool call `echo 0 > answer.txt`; after a tool result -> `DONE`
* mode `answer-parent`: like `answer` but writes ../answer.txt (the answer file next to task.json)
* mode `impossible` : always replies `IMPOSSIBLE`
* --fail-first K    : the first K requests fail with --fail-status (429 carries `Retry-After: 1`), then it works;
                      used to check retries and that failed requests are not charged
* --require-key KEY : any other bearer token gets a 401 (body echoes a masked key, like the real proxy)
Usage numbers are rough (JSON chars / 4). Standard library only.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["answer", "answer-parent", "impossible"], default="answer")
ap.add_argument("--port", type=int, default=0)
ap.add_argument("--fail-first", type=int, default=0)
ap.add_argument("--fail-status", type=int, default=429)
ap.add_argument("--require-key", default=None)
ARGS = ap.parse_args()
LOCK = threading.Lock()
STATE = {"seen": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj, headers=None):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        if ARGS.require_key and self.headers.get("Authorization") != f"Bearer {ARGS.require_key}":
            tok = (self.headers.get("Authorization") or "")[7:]
            return self._send(401, {"error": {"message": f"Authentication Error. Received API Key = {tok[:3]}...{tok[-3:]}",
                                              "type": "invalid_api_key", "code": 401}})
        with LOCK:
            STATE["seen"] += 1
            fail = STATE["seen"] <= ARGS.fail_first
        if fail:
            hdr = {"Retry-After": "1"} if ARGS.fail_status == 429 else {}
            return self._send(ARGS.fail_status, {"error": {"message": "mock failure", "type": "mock", "code": ARGS.fail_status}}, hdr)
        msgs = body.get("messages", [])
        if ARGS.mode == "impossible":
            msg, fin = {"role": "assistant", "content": "IMPOSSIBLE"}, "stop"
        elif body.get("tools") and not any(m.get("role") == "tool" for m in msgs):
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "bash", "arguments": json.dumps({"command": "echo 0 > " + ("../answer.txt" if ARGS.mode == "answer-parent" else "answer.txt")})}}]}
            fin = "tool_calls"
        else:
            msg, fin = {"role": "assistant", "content": "DONE"}, "stop"
        pt = len(json.dumps(body)) // 4 + 1
        self._send(200, {"id": "mock", "object": "chat.completion", "created": 0, "model": body.get("model"),
                         "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                         "usage": {"prompt_tokens": pt, "completion_tokens": 3, "total_tokens": pt + 3}})


srv = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
print(f"PORT {srv.server_address[1]}", flush=True)
srv.serve_forever()

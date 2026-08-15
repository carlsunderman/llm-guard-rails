"""Minimal OpenAI-compatible mock LLM for end-to-end proxy testing.

- Prompts containing "REPEAT: <text>" get <text> echoed back (to exercise
  the output guardrail block path).
- Everything else gets a benign canned response.
"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        last = ""
        for m in reversed(body.get("messages", [])):
            c = m.get("content")
            if isinstance(c, str) and c:
                last = c
                break
        if "demo record" in last:
            # Benign prompt that makes the "model" emit PII (exercises the
            # output-block path without tripping the input injection scanner).
            content = "The SSN on file is 123-45-6789."
        elif "REPEAT:" in last:
            content = last.split("REPEAT:", 1)[1].strip()
        else:
            content = "Hello from mock upstream."
        resp = {
            "id": "mock-1",
            "object": "chat.completion",
            "model": body.get("model", "mock"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 9310), Handler).serve_forever()

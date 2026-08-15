"""
Guardrail proxy for LLM API calls.

Sits between agents/tools and upstream LLM providers.
For each request:
1. Calls guardrail service /check-input on the input text (system, user, and tool messages).
2. If allowed, forwards to upstream LLM provider (OpenAI-compatible).
3. Calls guardrail service /check-output on the model response.
4. Returns the upstream response to the client, or blocks with an error if
   a check fails on a block-action issue. Redact-action issues (low-confidence
   PII) are masked in place instead: input messages are redacted before the
   upstream call, assistant content is redacted before the response (and SSE
   replay) reaches the client. See docs/proxy-design.md.

Key behaviors:
- Fail-closed by default if guardrail service is unreachable.
- Structured JSON-line audit logging to stdout.
"""

import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

app = FastAPI(title="LLM Guardrail Proxy", version="0.1.0")

# Configuration from environment
GUARDRAILS_URL = os.getenv("GUARDRAILS_URL", "http://guardrails:8090")
UPSTREAM_API_BASE = os.getenv("UPSTREAM_API_BASE", "https://api.openai.com/v1")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "gpt-4o")

FAIL_CLOSED = os.getenv("FAIL_CLOSED", "true").lower() in ("true", "1", "yes")
GUARDRAILS_TIMEOUT = float(os.getenv("GUARDRAILS_TIMEOUT_SECONDS", "3"))
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "120"))


def infer_upstream_provider(base_url: str) -> str:
    host = urlparse(base_url).netloc.lower()
    if "openai" in host:
        return "openai"
    if "anthropic" in host:
        return "anthropic"
    return host or "unknown"


UPSTREAM_PROVIDER = infer_upstream_provider(UPSTREAM_API_BASE)

if not UPSTREAM_API_KEY:
    raise SystemExit(
        "UPSTREAM_API_KEY is not set; refusing to start. "
        "Provide the upstream LLM API key before starting the proxy."
    )

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("guardrail-proxy")


class ChatCompletionRequest(BaseModel):
    """Validation model — only the fields the proxy depends on are declared.

    The raw request body is what gets forwarded upstream, so arbitrary
    OpenAI parameters (tools, tool_choice, stream, response_format, top_p,
    ...) pass through untouched instead of being dropped by pydantic.
    """

    model: Optional[str] = None
    # OpenAI allows message content as a string, a list of typed parts, or
    # null (assistant messages carrying tool_calls commonly use null).
    messages: List[Dict[str, Any]]
    agent_id: Optional[str] = None
    user_id: Optional[str] = None


class AuditLog(BaseModel):
    timestamp: str
    request_id: str
    user_id: Optional[str]
    agent_id: Optional[str]
    upstream_provider: str
    input_decision: str
    input_issues: List[Dict[str, Any]]
    output_decision: str
    output_issues: List[Dict[str, Any]]
    overall_decision: str
    latency_ms: int


def write_audit_log(
    *,
    request_id: str,
    agent_id: str,
    user_id: Optional[str],
    input_decision: str,
    input_issues: List[Dict[str, Any]],
    output_decision: str,
    output_issues: List[Dict[str, Any]],
    overall_decision: str,
    start_time: float,
) -> None:
    """Write a single JSON-line audit log to stdout."""
    log = AuditLog(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        request_id=request_id,
        user_id=user_id,
        agent_id=agent_id,
        upstream_provider=UPSTREAM_PROVIDER,
        input_decision=input_decision,
        input_issues=input_issues,
        output_decision=output_decision,
        output_issues=output_issues,
        overall_decision=overall_decision,
        latency_ms=int((time.time() - start_time) * 1000),
    )
    print(json.dumps(log.model_dump()), flush=True)


async def call_guardrails_check(
    text: str,
    endpoint: str,
    agent_id: Optional[str],
    user_id: Optional[str],
    segments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Call the guardrail service and return the response.

    `text` is the newline-joined scan text (sent always, for back-compat);
    `segments` is the per-message/per-part list the service redacts
    independently, with `redacted` returned parallel to it.
    """
    payload = {"text": text}
    if segments:
        payload["segments"] = segments
    if agent_id:
        payload["agent_id"] = agent_id
    if user_id:
        payload["user_id"] = user_id

    try:
        async with httpx.AsyncClient(timeout=GUARDRAILS_TIMEOUT) as client:
            resp = await client.post(f"{GUARDRAILS_URL}{endpoint}", json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        logger.exception("Guardrail check failed (endpoint=%s)", endpoint)
        if FAIL_CLOSED:
            raise HTTPException(
                status_code=503,
                detail="Guardrail service unavailable (fail-closed mode); request blocked.",
            )
        # Fail-open: assume ok if guardrails are down
        return {"ok": True, "issues": []}


def message_text(content: Any) -> str:
    """Flatten an OpenAI message content (string, list of parts, or null) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


# Roles whose text is input-scanned (see docs/proxy-design.md, "Input scan
# scope"). Assistant turns are covered by the output check on the final
# response; other roles are not scanned.
SCANNED_INPUT_ROLES = ("system", "user", "tool")


def build_input_segments(messages: List[Dict[str, Any]]) -> tuple:
    """Build the per-message segments to input-scan.

    Returns (segments, targets): segments are the non-empty texts of
    system/user/tool messages only (assistant turns are covered by the
    output check on the final response), targets the parallel message
    indices so redacted segments can be substituted back in place.
    """
    segments: List[str] = []
    targets: List[int] = []
    for idx, m in enumerate(messages):
        if m.get("role") not in SCANNED_INPUT_ROLES:
            continue
        text = message_text(m.get("content"))
        if text.strip():
            segments.append(text)
            targets.append(idx)
    return segments, targets


def _is_all_text_parts(content: Any) -> bool:
    return (
        isinstance(content, list)
        and len(content) > 0
        and all(isinstance(p, dict) and p.get("type") == "text" for p in content)
    )


def substitute_input_redaction(
    messages: List[Dict[str, Any]],
    targets: List[int],
    segments: List[str],
    redacted: List[str],
    notes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy `messages` with redacted segment text substituted in place.

    Substitution rules: string content is replaced; an all-text-part content
    list is replaced by a single text part carrying the redacted text; a
    list containing non-text parts (images, etc.) is left untouched and
    recorded in `notes` for the audit log.
    """
    new_messages = [dict(m) for m in messages]
    for msg_idx, original, red in zip(targets, segments, redacted):
        if red == original:
            continue
        content = new_messages[msg_idx].get("content")
        if isinstance(content, str):
            new_messages[msg_idx]["content"] = red
        elif _is_all_text_parts(content):
            new_messages[msg_idx]["content"] = [{"type": "text", "text": red}]
        else:
            notes.append(
                {
                    "scanner": "pii",
                    "severity": "low",
                    "message": "redact skipped: message content has non-text parts",
                    "action": "allow",
                }
            )
    return new_messages


def build_output_segments(response_data: Dict[str, Any]) -> tuple:
    """Build the per-part segments to output-scan from the upstream response.

    Returns (segments, has_content): the assistant content string (if any)
    first, then each tool-call arguments string. Tool-call arguments are
    included because that is where exfiltration payloads live when the
    agent's tools write to external systems (DBs, APIs, files).
    """
    choices = response_data.get("choices") or []
    if not choices:
        return [], False
    message = choices[0].get("message") or {}
    segments: List[str] = []
    has_content = False
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        segments.append(content)
        has_content = True
    for tool_call in message.get("tool_calls") or []:
        function = (tool_call or {}).get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        if isinstance(arguments, str) and arguments.strip():
            segments.append(arguments)
    return segments, has_content


def substitute_output_redaction(
    response_data: Dict[str, Any], segments: List[str], redacted: List[str]
) -> bool:
    """Redact the assistant content in place from a `redacted` segment.

    Returns True when the content was actually changed. Tool-call arguments
    are deliberately NOT redacted (v1): they are machine-consumed (DSNs,
    JSON envelopes) and masking them would silently break tool execution.
    Credential-shaped hits in arguments still block (see guardrail policy).
    """
    if not redacted or not segments or segments[0] == redacted[0]:
        return False
    message = response_data["choices"][0]["message"]
    if not isinstance(message.get("content"), str):
        return False
    message["content"] = redacted[0]
    return True


def build_sse_events(response_data: Dict[str, Any], model: str) -> List[str]:
    """Synthesize an OpenAI-compatible SSE stream from a buffered response.

    The proxy always pulls from upstream non-streaming so the output check
    runs before any model bytes reach the client. Streaming clients then
    receive the fully checked response replayed as standard
    chat.completion.chunk events (content, tool calls, finish, [DONE]).
    Trade-off: no incremental tokens until the full response is generated.
    """
    choices = response_data.get("choices") or [{}]
    message = choices[0].get("message") or {}
    base_id = response_data.get("id") or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(response_data.get("created") or time.time())
    model_name = response_data.get("model") or model

    def chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> str:
        payload = {
            "id": base_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    events: List[str] = [chunk({"role": "assistant", "content": ""})]
    content = message.get("content")
    if isinstance(content, str) and content:
        events.append(chunk({"content": content}))
    for i, tool_call in enumerate(message.get("tool_calls") or []):
        function = tool_call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})
        events.append(
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": tool_call.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": arguments,
                            },
                        }
                    ]
                }
            )
        )
    finish = choices[0].get("finish_reason") or "stop"
    events.append(chunk({}, finish))
    events.append("data: [DONE]\n\n")
    return events


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with guardrail enforcement."""
    try:
        raw = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")

    try:
        req = ChatCompletionRequest(**raw)
    except ValidationError:
        raise HTTPException(status_code=422, detail="Invalid request: 'messages' must be a list")

    request_id = str(uuid.uuid4())
    start_time = time.time()

    model = req.model or UPSTREAM_MODEL
    agent_id = req.agent_id or "unknown"
    user_id = req.user_id

    # Step 1: Check input via guardrails
    input_segments, input_targets = build_input_segments(req.messages)
    try:
        input_result = await call_guardrails_check(
            text="\n".join(input_segments),
            endpoint="/check-input",
            agent_id=agent_id,
            user_id=user_id,
            segments=input_segments,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected guardrail input check failure")
        if FAIL_CLOSED:
            raise HTTPException(status_code=503, detail="Guardrail check failed (fail-closed); request blocked.")
        input_result = {"ok": True, "issues": []}

    input_decision = "allow" if input_result.get("ok") else "block"
    input_issues = input_result.get("issues", [])

    if not input_result.get("ok"):
        # Block the request
        write_audit_log(
            request_id=request_id,
            agent_id=agent_id,
            user_id=user_id,
            input_decision=input_decision,
            input_issues=input_issues,
            output_decision="skip",
            output_issues=[],
            overall_decision="block",
            start_time=start_time,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Request blocked by guardrails. Issues: {input_issues}",
        )

    # Redact-action issues: mask low-confidence PII in the input before it
    # reaches the model (the model then operates on [REDACTED] placeholders).
    upstream_messages = req.messages
    redacted_in = input_result.get("redacted")
    if redacted_in:
        redact_notes: List[Dict[str, Any]] = []
        upstream_messages = substitute_input_redaction(
            req.messages, input_targets, input_segments, redacted_in, redact_notes
        )
        if redact_notes:
            input_issues = input_issues + redact_notes
        if any(r != s for r, s in zip(redacted_in, input_segments)):
            input_decision = "redact"

    # Step 2: Forward to upstream LLM provider.
    # Forward the raw body verbatim (minus proxy-only fields) so no OpenAI
    # parameters (tools, tool_choice, stream, response_format, ...) are lost.
    upstream_body = {k: v for k, v in raw.items() if k not in ("agent_id", "user_id")}
    upstream_body["model"] = model
    upstream_body["messages"] = upstream_messages
    # Always buffer upstream non-streaming: the output check must see the full
    # response before anything reaches the client. Streaming clients get the
    # checked response replayed as SSE (see build_sse_events).
    upstream_body["stream"] = False

    try:
        async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
            upstream_resp = await client.post(
                f"{UPSTREAM_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {UPSTREAM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=upstream_body,
            )
            upstream_resp.raise_for_status()
            response_data = upstream_resp.json()
    except httpx.HTTPStatusError as e:
        # Surface the upstream status code (429, 500, ...); the response
        # body stays on stderr only.
        logger.exception("Upstream LLM call failed (model=%s)", model)
        write_audit_log(
            request_id=request_id,
            agent_id=agent_id,
            user_id=user_id,
            input_decision=input_decision,
            input_issues=input_issues,
            output_decision="error",
            output_issues=[],
            overall_decision="error",
            start_time=start_time,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM error (upstream status {e.response.status_code})",
        )
    except Exception:
        logger.exception("Upstream LLM call failed (model=%s)", model)
        write_audit_log(
            request_id=request_id,
            agent_id=agent_id,
            user_id=user_id,
            input_decision=input_decision,
            input_issues=input_issues,
            output_decision="error",
            output_issues=[],
            overall_decision="error",
            start_time=start_time,
        )
        raise HTTPException(status_code=502, detail="Upstream LLM error")

    # Step 3: Check output via guardrails
    output_segments, output_has_content = build_output_segments(response_data)
    try:
        output_result = await call_guardrails_check(
            text="\n".join(output_segments),
            endpoint="/check-output",
            agent_id=agent_id,
            user_id=user_id,
            segments=output_segments,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected guardrail output check failure")
        if FAIL_CLOSED:
            raise HTTPException(status_code=503, detail="Guardrail check failed on output (fail-closed); response blocked.")
        output_result = {"ok": True, "issues": []}

    output_decision = "allow" if output_result.get("ok") else "block"
    output_issues = output_result.get("issues", [])

    if not output_result.get("ok"):
        # Block the response
        write_audit_log(
            request_id=request_id,
            agent_id=agent_id,
            user_id=user_id,
            input_decision=input_decision,
            input_issues=input_issues,
            output_decision=output_decision,
            output_issues=output_issues,
            overall_decision="block",
            start_time=start_time,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Response blocked by guardrails. Issues: {output_issues}",
        )

    # Redact-action issues: mask low-confidence PII in the assistant content
    # before it reaches the client. build_sse_events reads response_data, so
    # the redacted content propagates to streaming replays automatically.
    redacted_out = output_result.get("redacted")
    if output_has_content and redacted_out and substitute_output_redaction(
        response_data, output_segments, redacted_out
    ):
        output_decision = "redact"

    # Step 4: Return allowed response. Streaming clients receive the checked
    # response replayed as an SSE stream; the block above still fires before
    # any bytes reach them. Audit first so both paths are logged.
    write_audit_log(
        request_id=request_id,
        agent_id=agent_id,
        user_id=user_id,
        input_decision=input_decision,
        input_issues=input_issues,
        output_decision=output_decision,
        output_issues=output_issues,
        overall_decision="allow",
        start_time=start_time,
    )
    if raw.get("stream"):
        return StreamingResponse(
            iter(build_sse_events(response_data, model)),
            media_type="text/event-stream",
        )

    return response_data


@app.get("/health")
def health():
    return {"status": "ok"}

"""
Guardrail proxy for LLM API calls.

Sits between agents/tools and upstream LLM providers.
For each request:
1. Calls guardrail service /check-input on the input text (system, user, and tool messages).
2. If allowed, forwards to upstream LLM provider (OpenAI-compatible).
3. Calls guardrail service /check-output on the model response.
4. Returns final response (allow/block/redact) to the client.

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
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

app = FastAPI(title="LLM Guardrail Proxy", version="0.1.0")

# Configuration from environment
GUARDRAILS_URL = os.getenv("GUARDRAILS_URL", "http://guardrails:8090")
UPSTREAM_API_BASE = os.getenv("UPSTREAM_API_BASE", "https://api.openai.com/v1")
UPSTREAM_API_KEY = os.getenv("UPSTREAM_API_KEY", "")
UPSTREAM_MODEL = os.getenv("UPSTREAM_MODEL", "gpt-4o")

FAIL_CLOSED = os.getenv("FAIL_CLOSED", "true").lower() in ("true", "1", "yes")
GUARDRAILS_TIMEOUT = float(os.getenv("GUARDRAILS_TIMEOUT_SECONDS", "3"))

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


class ChatMessage(BaseModel):
    role: str
    # OpenAI allows a string, a list of typed content parts, or null
    # (assistant messages carrying tool_calls commonly use null).
    content: Optional[Union[str, List[Dict[str, Any]]]] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
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


def write_audit_log(log: AuditLog):
    """Write a single JSON-line audit log to stdout."""
    print(json.dumps(log.model_dump()), flush=True)


async def call_guardrails_check(text: str, endpoint: str, agent_id: Optional[str], user_id: Optional[str]) -> Dict[str, Any]:
    """Call the guardrail service and return the response."""
    payload = {"text": text}
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


def extract_input_text(messages: List[ChatMessage]) -> str:
    """Extract the text to input-scan from the message list.

    Scans system, user, and tool messages; assistant turns are covered by
    the output check on the final response.
    """
    parts = []
    for m in messages:
        if m.role == "assistant":
            continue
        text = message_text(m.content)
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def extract_model_response(response_data: Dict[str, Any]) -> str:
    """Extract the model's response text from the upstream API response."""
    choices = response_data.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content", "")


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint with guardrail enforcement."""
    request_id = str(uuid.uuid4())
    start_time = time.time()

    model = req.model or UPSTREAM_MODEL
    agent_id = req.agent_id or "unknown"
    user_id = req.user_id

    # Step 1: Check input via guardrails
    input_text = extract_input_text(req.messages)
    try:
        input_result = await call_guardrails_check(
            text=input_text,
            endpoint="/check-input",
            agent_id=agent_id,
            user_id=user_id,
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
        latency_ms = int((time.time() - start_time) * 1000)
        write_audit_log(
            AuditLog(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                request_id=request_id,
                user_id=user_id,
                agent_id=agent_id,
                upstream_provider="openai",
                input_decision=input_decision,
                input_issues=input_issues,
                output_decision="skip",
                output_issues=[],
                overall_decision="block",
                latency_ms=latency_ms,
            )
        )
        raise HTTPException(
            status_code=400,
            detail=f"Request blocked by guardrails. Issues: {input_issues}",
        )

    # Step 2: Forward to upstream LLM provider
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            upstream_resp = await client.post(
                f"{UPSTREAM_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {UPSTREAM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": m.role, "content": m.content} for m in req.messages],
                    "temperature": req.temperature,
                    **({"max_tokens": req.max_tokens} if req.max_tokens else {}),
                },
            )
            upstream_resp.raise_for_status()
            response_data = upstream_resp.json()
    except Exception:
        logger.exception("Upstream LLM call failed (model=%s)", model)
        latency_ms = int((time.time() - start_time) * 1000)
        write_audit_log(
            AuditLog(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                request_id=request_id,
                user_id=user_id,
                agent_id=agent_id,
                upstream_provider="openai",
                input_decision=input_decision,
                input_issues=input_issues,
                output_decision="error",
                output_issues=[],
                overall_decision="error",
                latency_ms=latency_ms,
            )
        )
        raise HTTPException(status_code=502, detail="Upstream LLM error")

    # Step 3: Check output via guardrails
    model_response = extract_model_response(response_data)
    try:
        output_result = await call_guardrails_check(
            text=model_response,
            endpoint="/check-output",
            agent_id=agent_id,
            user_id=user_id,
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
        latency_ms = int((time.time() - start_time) * 1000)
        write_audit_log(
            AuditLog(
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                request_id=request_id,
                user_id=user_id,
                agent_id=agent_id,
                upstream_provider="openai",
                input_decision=input_decision,
                input_issues=input_issues,
                output_decision=output_decision,
                output_issues=output_issues,
                overall_decision="block",
                latency_ms=latency_ms,
            )
        )
        raise HTTPException(
            status_code=400,
            detail=f"Response blocked by guardrails. Issues: {output_issues}",
        )

    # Step 4: Return allowed response
    latency_ms = int((time.time() - start_time) * 1000)
    write_audit_log(
        AuditLog(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            request_id=request_id,
            user_id=user_id,
            agent_id=agent_id,
            upstream_provider="openai",
            input_decision=input_decision,
            input_issues=input_issues,
            output_decision=output_decision,
            output_issues=output_issues,
            overall_decision="allow",
            latency_ms=latency_ms,
        )
    )

    return response_data


@app.get("/health")
def health():
    return {"status": "ok"}

"""
Guardrail service for coding agents.

Uses LLM Guard (v0.3.16) to scan prompts and responses for:
- Prompt injection attempts (input)
- Toxicity in prompts (input)
- Sensitive patterns (secrets, SSN, credit cards) in outputs via regex (output)

Run via Docker Compose; exposes HTTP endpoints on port 8090.
"""

import os
from typing import List, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# LLM Guard imports (loaded once at startup in container)
from llm_guard.input_scanners import PromptInjection, Toxicity as InputToxicity
from llm_guard.output_scanners import Regex as OutputRegex

app = FastAPI(title="Agent Guardrails", version="0.1.0")


# Input scanners
prompt_injection_scanner = PromptInjection(threshold=0.85)
toxicity_scanner = InputToxicity(threshold=0.6)

# Output scanner: regex-based detection of high-confidence secrets and strong PII.
# Deliberately excludes low-confidence patterns (emails, IPs, phone numbers):
# coding agents routinely emit 127.0.0.1, example.com addresses, and
# phone-shaped test fixtures, which would false-positive as hard blocks.
blocking_pii_patterns = [
    # US SSN (xxx-xx-xxxx)
    r"\b\d{3}-\d{2}-\d{4}\b",
    # Credit-card-like: 13-16 digits in groups of 4 with required separators
    # (avoids matching long contiguous numeric ids like transaction numbers)
    r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]?\d{1,4}\b",
    # Generic API key / token patterns (e.g. sk-..., pk_..., ghp_...)
    r"\b(?:sk|pk|rk)[- _][a-zA-Z0-9]{20,}\b",
    r"\bghp_[a-zA-Z0-9]{36,}\b",
    r"\bgp_[a-zA-Z0-9]{22,}\b",
]

# Low-confidence PII patterns: NOT blocking. Reserved for the future
# redact-only mode (see docs/proxy-design.md) so they can redact instead of
# failing the request.
redact_candidate_patterns = [
    # Email
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    # IPv4 addresses
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    # US phone numbers (various formats)
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
]

sensitive_pattern_scanner = OutputRegex(patterns=blocking_pii_patterns, is_blocked=True, redact=False)


class CheckRequest(BaseModel):
    text: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None


class Issue(BaseModel):
    scanner: str
    severity: Literal["low", "medium", "high", "critical"]
    message: str


class CheckResponse(BaseModel):
    ok: bool
    issues: List[Issue] = []


def run_input_scanners(text: str) -> List[Issue]:
    """Run input scanners using llm-guard 0.3.x API: scan(prompt) -> (text, is_valid, score)."""
    issues: List[Issue] = []

    # Prompt injection
    _, is_valid, score = prompt_injection_scanner.scan(text)
    if not is_valid:
        severity = "critical" if score > 0.9 else "high"
        issues.append(
            Issue(
                scanner="prompt_injection",
                severity=severity,
                message=f"Possible prompt injection detected (score={score:.2f})",
            )
        )

    # Toxicity in input
    _, is_valid, score = toxicity_scanner.scan(text)
    if not is_valid:
        severity = "high" if score > 0.9 else "medium"
        issues.append(
            Issue(
                scanner="toxicity",
                severity=severity,
                message=f"Toxic or abusive prompt detected (score={score:.2f})",
            )
        )

    return issues


def run_output_scanners(prompt_text: str, output_text: str) -> List[Issue]:
    """Run output scanners using llm-guard 0.3.x API: scan(prompt, output) -> (text, is_valid, score)."""
    issues: List[Issue] = []

    # Regex-based sensitive/PII-like pattern detection in output
    _, is_valid, score = sensitive_pattern_scanner.scan(prompt_text, output_text)
    if not is_valid:
        issues.append(
            Issue(
                scanner="sensitive_patterns",
                severity="high",
                message=f"Sensitive/PII-like patterns detected in output (score={score:.2f})",
            )
        )

    return issues


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check-input", response_model=CheckResponse)
def check_input(req: CheckRequest):
    """Scan an incoming prompt/context before the agent processes it."""
    if not req.text.strip():
        return CheckResponse(ok=True)

    issues = run_input_scanners(req.text)
    return CheckResponse(ok=len(issues) == 0, issues=issues)


@app.post("/check-output", response_model=CheckResponse)
def check_output(req: CheckRequest):
    """Scan an agent's response before it is returned/executed.

    For output scanners that need context, we currently only have the output text.
    In a fuller integration, the caller would also send the original prompt.
    Here we pass empty string as prompt since our regex scanner only looks at output.
    """
    if not req.text.strip():
        return CheckResponse(ok=True)

    issues = run_output_scanners("", req.text)
    return CheckResponse(ok=len(issues) == 0, issues=issues)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host=host, port=port)

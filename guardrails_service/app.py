"""
Guardrail service for coding agents.

Uses LLM Guard (v0.3.16) to scan prompts and responses for:
- Prompt injection attempts (input)
- Toxicity in prompts (input)
- Sensitive patterns (secrets, SSN, credit cards) in outputs via regex (output)
- Credential leaks (API keys, connection strings with embedded passwords,
  private keys, password assignments, canary tokens) in inputs and outputs

Run via Docker Compose; exposes HTTP endpoints on port 8090.
"""

import os
import re
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

# Credential-leak scanner (regex, no ML). Runs on BOTH inputs and outputs:
# input hits stop credentials from ever entering model context, output hits
# stop exfiltration (including tool-call arguments, which the proxy scans).
# Covers AWS, GitHub, JWTs, PEM private keys, generic connection strings
# with embedded user:pass (mongodb, postgres, mssql, hana/SAP, redis, ...),
# and named secret assignments (password=..., secret: ...).
CREDENTIAL_PATTERNS: List[tuple] = [
    ("aws_access_key_id", r"\bAKIA[0-9A-Z]{16}\b"),
    ("aws_temp_access_key_id", r"\bASIA[0-9A-Z]{16}\b"),
    ("github_token", r"\bgh[opsur]_[A-Za-z0-9]{36,}\b"),
    ("github_fine_grained_token", r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    # Databricks personal access token (dapi + 44 alphanumerics)
    ("databricks_pat", r"\bdapi[a-zA-Z0-9]{44}\b"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ("pem_private_key", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"),
    # scheme://user:pass@ — any URL with embedded basic-auth credentials
    ("connection_string_credentials", r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^/\s:@]+:[^/\s@]+@"),
    # password= / secret: style assignments; value captured for placeholder check
    (
        "named_secret_assignment",
        r"(?i)\b(?:password|passwd|pwd|passphrase|secret|client_secret|access_key"
        r"|secret_key|api_key|apikey|auth_token|access_token)\s*[:=]\s*['\"]?"
        r"([A-Za-z0-9!@#$%^&*_,./+~-]{8,})",
    ),
]

# Values that are clearly not real secrets (templates, type annotations,
# prose). Checked against the captured assignment value only.
PLACEHOLDER_MARKERS = ("${", "{{", "<", ">", "%s", "{0}")
PLACEHOLDER_VALUES = {
    "none", "null", "nil", "undefined", "true", "false", "string", "str",
    "text", "value", "values", "parameter", "parameters", "param",
    "placeholder", "example", "changeme", "change_me", "your", "xxx",
    "xxxx", "dummy", "test", "password", "passwd", "secret", "token",
    "todo", "fixme", "redacted",
}


class Issue(BaseModel):
    scanner: str
    severity: Literal["low", "medium", "high", "critical"]
    message: str


def _is_placeholder(value: str) -> bool:
    v = value.strip("'\"").lower()
    if any(marker in v for marker in PLACEHOLDER_MARKERS):
        return True
    return v in PLACEHOLDER_VALUES


def _load_canary_tokens() -> List[str]:
    """Exact-match canary tokens from CANARY_TOKENS (comma-separated env).

    Change canaries by editing CANARY_TOKENS (e.g. in .env) and restarting
    the service.
    """
    raw = os.getenv("CANARY_TOKENS", "")
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def run_credential_scanners(text: str) -> List[Issue]:
    """Regex-based credential leak detection. Never echoes matched values."""
    issues: List[Issue] = []
    seen: set = set()
    for name, pattern in CREDENTIAL_PATTERNS:
        for match in re.finditer(pattern, text):
            if name == "named_secret_assignment" and _is_placeholder(match.group(1)):
                continue
            if name in seen:
                continue
            seen.add(name)
            issues.append(
                Issue(
                    scanner="credentials",
                    severity="high",
                    message=f"Possible credential leak ({name})",
                )
            )
    for token in _load_canary_tokens():
        if token in text:
            issues.append(
                Issue(
                    scanner="canary_token",
                    severity="critical",
                    message="Canary token detected",
                )
            )
            break
    return issues


class CheckRequest(BaseModel):
    text: str
    agent_id: Optional[str] = None
    user_id: Optional[str] = None


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

    issues = run_input_scanners(req.text) + run_credential_scanners(req.text)
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

    issues = run_output_scanners("", req.text) + run_credential_scanners(req.text)
    return CheckResponse(ok=len(issues) == 0, issues=issues)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host=host, port=port)

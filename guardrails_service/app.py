"""
Guardrail service for coding agents.

Uses LLM Guard (v0.3.16) to scan prompts and responses for:
- Prompt injection attempts (input)
- Sensitive patterns (secrets, SSN, credit cards) in outputs via regex (output)
- Credential leaks (API keys, connection strings with embedded passwords,
  private keys, password assignments, canary tokens) in inputs and outputs
- Low-confidence PII (emails, IPv4, phone numbers) in inputs and outputs —
  redacted to [REDACTED] in the `redacted` response field, never blocking
- Optional Needle 3 local triage classifier (audit-only, action "allow"):
  labels segments benign/instruction_override/exfiltration; requires the
  cactus-needle package + baked weights, self-disables otherwise

Each scanner maps to an enforcement action via SCANNER_POLICY:
block (fail the request), redact (mask + pass), or allow (log + pass).

Run via Docker Compose; exposes HTTP endpoints on port 8090.
"""

import concurrent.futures
import os
import re
import threading
import unicodedata
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

# LLM Guard imports (loaded once at startup in container)
from llm_guard.input_scanners import InvisibleText, PromptInjection
from llm_guard.output_scanners import Regex as OutputRegex

# Needle 3 (cactus-needle): local triage classifier. Optional at import so
# the image still builds/runs without it; the scanner self-disables when the
# engine or weights are unavailable (audit-only, action "allow", D-13).
try:
    import needle as _needle

    class _TriageVerdict(BaseModel):
        verdict: Literal[
            "benign", "instruction_override", "exfiltration", "unclear"
        ]

    def _needle_extract(text: str) -> Any:
        return _needle.extract(text, _TriageVerdict)

    _NEEDLE_AVAILABLE = True
except Exception:  # pragma: no cover - depends on image build args
    _NEEDLE_AVAILABLE = False

    def _needle_extract(text: str) -> Any:
        return None

app = FastAPI(title="Agent Guardrails", version="0.1.0")


# Input scanners
prompt_injection_scanner = PromptInjection(threshold=0.85)
# Invisible-text sanitizer: strips Unicode format/private-use/unassigned
# characters (BOM, zero-width, bidi controls) that carry invisible-character
# injection payloads. scan() returns the cleaned text, so this is a redact
# scanner (see SCANNER_POLICY); inputs only, since it's an input-side
# attack vector.
invisible_text_scanner = InvisibleText()
# Toxicity scanning (llm_guard InputToxicity) is intentionally NOT enabled:
# it adds a DeBERTa model download + per-request inference for a signal
# that false-positives on benign-but-aggressive agent prompts, and this
# deployment's threats are secrets/injection/PII, not abuse. Re-enable via
# InputToxicity(threshold=0.6) + a "toxicity" SCANNER_POLICY entry if needed.

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

# Low-confidence PII patterns: never blocking. Redacted to [REDACTED] by
# pii_redaction_scanner below (see SCANNER_POLICY). Coding agents routinely
# emit 127.0.0.1, example.com addresses, and phone-shaped test fixtures,
# which would false-positive as hard blocks.
redact_candidate_patterns = [
    # Email
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    # IPv4 addresses
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    # US phone numbers (various formats)
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
]

sensitive_pattern_scanner = OutputRegex(patterns=blocking_pii_patterns, is_blocked=True, redact=False)

# Redact-mode scanners for low-confidence PII. is_blocked=True is required
# for the LLM Guard regex scanner to perform the substitution; with
# redact=True scan() returns (redacted_text, False, 1.0) on a match, where
# redacted_text has hits replaced with [REDACTED]. One scanner per pattern
# is needed because the 0.3.16 scanner returns after the FIRST matching
# pattern, so a multi-pattern list would only ever redact one kind. They are
# chained in run_pii_redaction. The "pii" policy action is "redact", so
# matches never fail the request (see SCANNER_POLICY).
pii_redaction_scanners = [
    OutputRegex(patterns=[pattern], is_blocked=True, redact=True)
    for pattern in redact_candidate_patterns
]

# Per-scanner enforcement action (the policy knob; externalizing this to a
# YAML policy file is still pending, see docs/enhancements.md).
#   block  -> an issue fails the request (400 at the proxy)
#   redact -> matches are masked in the `redacted` response field; pass
#   allow  -> logged to the audit trail only; pass
# Change a value here and restart the service to flip behavior.
ScannerAction = Literal["block", "redact", "allow"]
SCANNER_POLICY: Dict[str, ScannerAction] = {
    "prompt_injection": "block",
    "credentials": "block",
    "canary_token": "block",
    "sensitive_patterns": "block",
    "pii": "redact",
    "invisible_text": "redact",
    "needle_triage": "allow",
}

# Credential-leak scanner (regex, no ML). Runs on BOTH inputs and outputs:
# input hits stop credentials from ever entering model context, output hits
# stop exfiltration (including tool-call arguments, which the proxy scans).
# Covers AWS, GitHub, JWTs, PEM private keys, generic connection strings
# with embedded user:pass (mongodb, postgres, mssql, hana/SAP, redis, ...),
# and named secret assignments (password=..., secret: ...).
CREDENTIAL_PATTERNS: List[Tuple[str, str]] = [
    ("aws_access_key_id", r"\bAKIA[0-9A-Z]{16}\b"),
    ("aws_temp_access_key_id", r"\bASIA[0-9A-Z]{16}\b"),
    ("github_token", r"\bgh[opsur]_[A-Za-z0-9]{36,}\b"),
    ("github_fine_grained_token", r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    # Databricks personal access token (dapi + 44 alphanumerics)
    ("databricks_pat", r"\bdapi[a-zA-Z0-9]{44}\b"),
    # OpenAI project API key
    ("openai_project_key", r"\bsk-proj-[A-Za-z0-9_-]{20,}"),
    # GCP API key (AIza + 35 base64url chars)
    ("gcp_api_key", r"\bAIza[0-9A-Za-z_-]{35}"),
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


# Lock serializing Needle engine access across scan threads.
_needle_lock = threading.Lock()


class Issue(BaseModel):
    scanner: str
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    # Enforcement action applied to this issue, from SCANNER_POLICY.
    action: ScannerAction = "block"
    # Index into the request's segments list when the issue is attributable
    # to a single segment; None for scanners that run on the joined text.
    segment: Optional[int] = None


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
    """Regex-based credential leak detection. Never echoes matched values.

    Scans an NFKC-normalized copy: NFKC folds compatibility lookalikes
    (fullwidth letters/digits, ligatures, ...) to canonical ASCII, so
    patterns cannot be evaded by homoglyph-style substitution. Detection
    only — the normalized copy is never returned, audited, or forwarded.
    The PII redaction scanners intentionally keep the original text: their
    redacted output is forwarded upstream and must not be rewritten.
    """
    text = unicodedata.normalize("NFKC", text)
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
                    action=SCANNER_POLICY["credentials"],
                )
            )
    for token in _load_canary_tokens():
        # The canary is matched against the NFKC copy, so fold the token
        # too (e.g. a fullwidth-pasted CANARY_TOKENS env value).
        if unicodedata.normalize("NFKC", token) in text:
            issues.append(
                Issue(
                    scanner="canary_token",
                    severity="critical",
                    message="Canary token detected",
                    action=SCANNER_POLICY["canary_token"],
                )
            )
            break
    return issues


class CheckRequest(BaseModel):
    # Either `text` (single blob, back-compat) or `segments` (per-message /
    # per-part list; scanners run on the newline-joined text, redaction runs
    # per segment so the caller can substitute each one back in place).
    text: str = ""
    segments: Optional[List[str]] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None


class CheckResponse(BaseModel):
    # ok == no issue with a "block" policy action. Redact-action issues do
    # not affect ok; their masked text comes back in `redacted`.
    ok: bool
    issues: List[Issue] = []
    # Parallel to the effective segments (single element for text-only
    # requests). None when nothing was redacted.
    redacted: Optional[List[str]] = None


def _effective_segments(req: CheckRequest) -> List[str]:
    return req.segments if req.segments is not None else [req.text]


def run_pii_redaction(segments: List[str]) -> Tuple[Optional[List[str]], List[Issue]]:
    """Run the redact-mode PII scanner per segment.

    The redact patterns never span newlines, so per-segment redaction is
    equivalent to redacting the joined text. Returns
    (redacted_segments_or_None, issues): the list is parallel to `segments`
    and None when no segment changed.
    """
    redacted: List[str] = []
    hits = 0
    for segment in segments:
        # Chain the per-pattern scanners; each either leaves the text
        # unchanged or replaces its hits with [REDACTED].
        text = segment
        for scanner in pii_redaction_scanners:
            text, _, _ = scanner.scan("", text)
        if text != segment:
            hits += 1
        redacted.append(text)
    issues: List[Issue] = []
    if hits:
        issues.append(
            Issue(
                scanner="pii",
                severity="low",
                message=f"Low-confidence PII (email/IP/phone) found in {hits} segment(s); redacted",
                action=SCANNER_POLICY["pii"],
            )
        )
    return (redacted if hits else None), issues


def run_invisible_text_sanitization(segments: List[str]) -> Tuple[Optional[List[str]], List[Issue]]:
    """Strip invisible characters per segment (input path only).

    Like run_pii_redaction, returns (redacted_segments_or_None, issues):
    the list is parallel to `segments` and None when no segment changed.
    Stripping is lossless for the model (tokenizers drop these characters
    anyway) and unmasks invisible-character injection payloads, making the
    hidden instructions visible to the user and the injection scanner.
    """
    redacted: List[str] = []
    hits = 0
    for segment in segments:
        text, is_valid, _ = invisible_text_scanner.scan(segment)
        if not is_valid:
            hits += 1
        redacted.append(text)
    issues: List[Issue] = []
    if hits:
        issues.append(
            Issue(
                scanner="invisible_text",
                severity="medium",
                message=f"Invisible characters found in {hits} segment(s); removed",
                action=SCANNER_POLICY["invisible_text"],
            )
        )
    return (redacted if hits else None), issues


def run_input_scanners(segments: List[str]) -> List[Issue]:
    """Run input scanners using llm-guard 0.3.x API: scan(prompt) -> (text, is_valid, score).

    The injection classifier runs per segment (message), not on the joined
    text: a legitimate-looking system prompt in context suppresses its score
    for a malicious user message (see test_check_input_injection_after_system_prompt).
    Segments are scanned concurrently and EVERY offending segment is
    reported with its caller-side index (Issue.segment), so the caller can
    attribute each hit to a specific message.
    """
    indexed: List[Tuple[int, str]] = [(i, s) for i, s in enumerate(segments) if s.strip()]
    if not indexed:
        return []

    lock = threading.Lock()
    issues: List[Issue] = []

    def scan_segment(item: Tuple[int, str]) -> None:
        idx, segment = item
        # Prompt injection
        _, is_valid, score = prompt_injection_scanner.scan(segment)
        if is_valid:
            return
        with lock:
            severity = "critical" if score > 0.9 else "high"
            issues.append(
                Issue(
                    scanner="prompt_injection",
                    severity=severity,
                    message=f"Possible prompt injection detected (score={score:.2f})",
                    action=SCANNER_POLICY["prompt_injection"],
                    segment=idx,
                )
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(indexed))) as pool:
        futures = [pool.submit(scan_segment, it) for it in indexed]
        for f in futures:
            f.result()

    return issues


def run_needle_triage(segments: List[str]) -> List[Issue]:
    """Needle 3 local triage classifier, one call per segment (D-13).

    Audit-only signal: SCANNER_POLICY["needle_triage"] is "allow", so a
    suspicious verdict never fails the request — it exists to give the
    operators an independent, calibrated signal alongside the DeBERTa
    injection scanner (see active issue I-02) and to gate future promotion.
    Serialized under a module lock: the Needle engine instance is not
    documented thread-safe (same caveat as the shared PromptInjection
    scanner). Any engine failure yields no issue (silent skip) — an
    audit-only signal must never break the request path.
    """
    if os.getenv("NEEDLE_ENABLED", "true").lower() not in ("true", "1", "yes"):
        return []
    if not _NEEDLE_AVAILABLE or not segments:
        return []
    issues: List[Issue] = []
    with _needle_lock:
        for idx, segment in enumerate(segments):
            try:
                verdict = _needle_extract(segment)
            except Exception:
                continue
            if verdict is None:
                continue
            v = getattr(verdict, "verdict", None)
            if v in ("instruction_override", "exfiltration"):
                issues.append(
                    Issue(
                        scanner="needle_triage",
                        severity="medium",
                        message=f"Needle triage: {v}",
                        action=SCANNER_POLICY["needle_triage"],
                        segment=idx,
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
                action=SCANNER_POLICY["sensitive_patterns"],
            )
        )

    return issues


@app.get("/health")
def health():
    return {"status": "ok"}


def _decide(issues: List[Issue]) -> bool:
    """ok == no issue whose policy action is "block"."""
    return not any(issue.action == "block" for issue in issues)


@app.post("/check-input", response_model=CheckResponse)
def check_input(req: CheckRequest):
    """Scan an incoming prompt/context before the agent processes it."""
    segments = _effective_segments(req)
    text = "\n".join(segments)
    if not text.strip():
        return CheckResponse(ok=True)

    issues: List[Issue] = []
    # Invisible-text sanitization first: strips zero-width/bidi characters so
    # evasion-split payloads are reassembled before any classifier sees them.
    stripped, invisible_issues = run_invisible_text_sanitization(segments)
    issues = issues + invisible_issues
    # Injection classifier + credential regexes run on the stripped text
    # (pre-PII-redaction): the DeBERTa classifier scores the literal
    # "[REDACTED]" marker at 1.00, so it must judge the user's actual text.
    scan_segments = stripped or segments
    issues = (
        issues
        + run_input_scanners(scan_segments)
        + run_credential_scanners("\n".join(scan_segments))
        + run_needle_triage(scan_segments)
    )
    # PII redaction last: its output is the text forwarded upstream. When no
    # PII was masked, the stripped text is the forwarded copy.
    redacted, pii_issues = run_pii_redaction(scan_segments)
    issues = issues + pii_issues
    return CheckResponse(
        ok=_decide(issues),
        issues=issues,
        redacted=redacted if redacted is not None else stripped,
    )


@app.post("/check-output", response_model=CheckResponse)
def check_output(req: CheckRequest):
    """Scan an agent's response before it is returned/executed.

    For output scanners that need context, we currently only have the output text.
    In a fuller integration, the caller would also send the original prompt.
    Here we pass empty string as prompt since our regex scanner only looks at output.
    """
    segments = _effective_segments(req)
    text = "\n".join(segments)
    if not text.strip():
        return CheckResponse(ok=True)

    stripped, invisible_issues = run_invisible_text_sanitization(segments)
    scan_text = "\n".join(stripped or segments)
    issues = (
        run_output_scanners("", scan_text)
        + run_credential_scanners(scan_text)
        + invisible_issues
    )
    # PII redaction last: its output is the text forwarded to the client.
    redacted, pii_issues = run_pii_redaction(stripped or segments)
    issues = issues + pii_issues
    return CheckResponse(
        ok=_decide(issues),
        issues=issues,
        redacted=redacted if redacted is not None else stripped,
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8090"))
    uvicorn.run(app, host=host, port=port)

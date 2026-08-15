# LLM Guardrail Proxy — Design

Purpose:
- Provide a single enforcement layer that sits between all coding agents/tools (Pi, Claude Code, Copilot, Codex, custom UIs) and upstream LLM providers.
- Ensure every prompt and response passes through the guardrail service before it reaches the model or the user.
- Centralize policy, logging, and secrets management for org-wide protection.

## Architecture

Components:
- Guardrail Service (`guardrails_service/`)
  - Runs LLM Guard scanners (prompt injection, toxicity, sensitive patterns) plus a regex credential-leak scanner (inputs and outputs) and exact-match canary tokens (`CANARY_TOKENS` env).
  - Exposes:
    - POST /check-input { text, agent_id?, user_id? } -> { ok, issues[] }
    - POST /check-output { text, agent_id?, user_id? } -> { ok, issues[] }

- Guardrail Proxy (`proxy/`)
  - HTTP server that accepts standard LLM-style requests (OpenAI-compatible chat completions for now).
  - Accepts OpenAI message content as a string, a list of typed parts, or null; content is forwarded to upstream verbatim.
  - Arbitrary OpenAI parameters (tools, tool_choice, stream, response_format, top_p, ...) are forwarded to upstream verbatim; only the proxy-only `agent_id`/`user_id` fields are stripped before forwarding.
  - For each request:
    1. Extracts the input text (system, user, and tool messages) and metadata.
    2. Calls guardrail service /check-input.
    3. If allowed, forwards request to the upstream LLM provider (OpenAI-compatible).
    4. Calls guardrail service /check-output on the model response, including tool-call arguments (where exfiltration payloads to external systems appear).
    5. Returns the upstream response to the client, or blocks with an error if either check fails.
  - Logs every decision as structured JSON for auditing.

Clients (current):
- Any tool that can be pointed at a custom OpenAI-compatible base URL:
  - Custom apps, scripts, and OpenAI-compatible SDKs
  - Tools with configurable API base (e.g., some Pi/OMP model provider configs)

Not yet directly compatible (speak Anthropic-native or proprietary formats):
- Claude Code (ANTHROPIC_BASE_URL) — needs an Anthropic-format route on the proxy, or
  use the hook integration (`hooks/`) for per-turn checks instead.
Note: until the proxy supports the Anthropic wire format, Claude Code should use the
hook-based integration (`hooks/`) rather than the proxy.

## Key Design Decisions

### 1. Fail-open vs fail-closed

Decision: fail-closed by default, configurable.

Behavior:
- If the guardrail service is unreachable or times out (default 3s):
  - Input check failure → block the LLM call; return an error to the client.
  - Output check failure → block the response; return an error to the client.
- Rationale:
  - This is a security layer for the organization; silently losing protection is worse than brief unavailability.
  - A short blip blocks calls temporarily but preserves the security guarantee.

Configuration:
- FAIL_CLOSED=true|false (default true)
- GUARDRAILS_TIMEOUT_SECONDS=3
- UPSTREAM_TIMEOUT_SECONDS=120 (upstream LLM call timeout)

Future consideration:
- Per-policy severity rules (e.g., fail-closed for critical scanners, fail-open for lower-severity checks).

### 2. Audit logging

Decision: structured JSON-line audit log on every request.

Format (one JSON object per line to stdout):
{
  "timestamp": "2026-08-14T12:00:00Z",
  "request_id": "79e52825-9a0a-43f6-815f-ab913fe47704",
  "user_id": "user-or-agent-id-if-known",
  "agent_id": "claude-code|pi|copilot-proxy|unknown",
  "upstream_provider": "openai",
  "input_decision": "allow|block",
  "input_issues": [{"scanner": "prompt_injection", "severity": "critical"}],
  "output_decision": "allow|block|skip|error",
  "output_issues": [{"scanner": "sensitive_patterns", "severity": "high"}],
  "overall_decision": "allow|block|error",
  "latency_ms": 1234
}

Decision enums (as implemented):
- input_decision: allow | block
- output_decision: allow | block | skip (skipped when the input was already blocked) | error (upstream call failed)
- overall_decision: allow | block | error

Properties:
- Captures who called, what was flagged, and the final decision.
- Enables security reviews, incident response, and compliance reporting.
- Stdout-based so Docker/K8s log collectors can ingest it without extra config.
- stdout is a pure JSON-line stream: uvicorn access logs are disabled (`--no-access-log`) and guardrail/upstream error details are logged to stderr, so collectors can parse every stdout line as JSON.

Future consideration:
- Add a redact mode that masks sensitive patterns in outputs instead of blocking,
  adding a "redact" decision value.

### 3. Secrets handling for upstream LLM keys

Decision (POC): environment variables injected at deployment time.

Behavior:
- The proxy reads a single upstream key from the UPSTREAM_API_KEY env var (OpenAI-compatible upstream for now).
- Keys are provided at container start and kept in memory only.
- Keys are never written to logs or config files in the image.
- If UPSTREAM_API_KEY is unset, the proxy exits at startup with a clear error instead of failing per request.

Requirements:
- Keys must be managed by your deployment pipeline (Docker secrets, K8s Secrets, CI/CD vault), not stored in the repo.
- .env examples are for local development only and must be gitignored.

Future consideration:
- Integrate with Vault / AWS SSM / GCP Secret Manager for dynamic fetching and rotation.

### 4. Scope of support (initial)

Decision: start with OpenAI-compatible chat completions endpoint.

Rationale:
- Many tools and wrappers already speak this format, so custom clients need no adapter.
- Anthropic-native (Claude Code, Copilot-style clients) and other provider-specific routes can be added later; until then those tools use the hook integration.

### 5. Input scan scope

Decision: scan system, user, and tool messages; do not input-scan assistant turns.

Rationale:
- Indirect prompt injection in agent workloads arrives via tool results and fetched context (tool and system messages), not only user text.
- Assistant turns are covered by the output check on the final response; input-scanning them would double-scan model output and add false-positive surface.

### 6. Scanner behaviour (notes)

Investigated 2026-08-15 (review doc, item 7). No code change.

- The `PromptInjection` scanner runs `protectai/deberta-v3-base-prompt-injection-v2`.
- Its score is **bimodal and saturated**: ~1.0 for any text it flags, ~0.0-0.05 otherwise. There is no usable probability band, so threshold tuning cannot trade precision for recall (and vice versa).
- It treats **verbatim-repetition / "echo this string" requests as injection** (e.g. `REPEAT: ...`, `Please echo this exact string: ...`). This is deliberate in the model, not a bug: asking a model to replay sensitive context verbatim is a known exfiltration / indirect-injection payload. Do not paper over it with a length-based skip or a pattern exception — that re-opens a real attack surface.
- A lone SSN-shaped reference in normal prose (`Reference number 123-45-6789`) passes at ~0.01; it is the *repetition request* combined with the sensitive data that trips it.
- Future mitigation options if this becomes operationally noisy: per-policy severity rules (see above) or substituting a different injection scanner. Recorded in `docs/review-2026-08-14.md`.

### 7. Credential-leak scanner (added 2026-08-15, incident-driven)

Incident context: a model was asked to chunk a credentials file and the chunks were written to an external-facing database. The exfiltration vector was the **tool-call arguments** in the model response, which the output check originally did not inspect.

Changes:
- The proxy output check now scans `choices[].message.tool_calls[].function.arguments` in addition to `content` (see `extract_model_response`).
- The guardrail service runs `run_credential_scanners` on **both** `/check-input` and `/check-output` (input hits stop credentials from ever entering model context; output hits stop exfiltration). Coverage:
  - AWS access keys (`AKIA...`, `ASIA...`), GitHub tokens (`gh[opsur]_...`, `github_pat_...`), JWTs (`eyJ...`), PEM private-key blocks.
  - Any URL with embedded basic-auth credentials (`scheme://user:pass@...`), which covers MongoDB, SQL Server (`mssql://`), Postgres, SAP HANA (`hana://`), Azure, Redis, etc.
  - Named secret assignments (`password=...`, `secret: ...`, `api_key=...`, `access_token=...`, ...) with a placeholder filter so env indirection (`${DB_PASSWORD}`), angle-bracket placeholders, type annotations, and prose don't block.
  - Exact-match **canary tokens** from `CANARY_TOKENS` (comma-separated env var, severity `critical`). Recommended: seed canaries into the systems most likely to leak and verify detection end-to-end.

Known limits (honest scope):
- Unstructured plaintext credential dumps (bare `user123  hunter22` lines with no key/URL structure) are **not** detectable by regex; that needs a classifier or a known-value list (e.g. sync real usernames and match against them).
- Protection covers the LLM API path only. Tool executions that never traverse the model (direct file writes, direct HTTP) are out of scope; extend the Claude Code hook or add egress-side detection for those.
- Streaming responses (`"stream": true`) are not output-checked: the proxy's response parsing fails and the request errors (fail-closed, nothing leaks), but there is no streamed-inspection path yet.
- Matched values are never echoed into issues or logs.

## Deployment

POC (local):
- docker-compose.yml with two services:
  - guardrails
  - proxy
- Clients point at http://localhost:8000 (proxy) instead of upstream providers.

Org-wide (future):
- Kubernetes Deployment + Service for both guardrails and proxy.
- HPA for scalability.
- Centralized secrets management.
- Optional auth (API key or mTLS) so only authorized agents/tools can use the proxy.

## Status (as of 2026-08-14)

- Implemented: OpenAI-compatible /v1/chat/completions route, fail-closed mode, JSON-line audit log on a pure-JSON stdout (access logs disabled), two-service docker-compose stack.
- Input scanning covers system, user, and tool messages (assistant turns are covered by the output check); message content accepts string, list of parts, or null per the OpenAI schema.
- Credit-card output pattern requires grouped 4-digit separators so long contiguous numeric ids (transaction numbers, build ids) no longer trigger false-positive blocks.
- Proxy fails fast at startup if UPSTREAM_API_KEY is unset; upstream/guardrail error details are logged to stderr, never returned to clients.
- Verified end-to-end (2026-08-15, against a local OpenAI-compatible mock upstream): benign prompt returned the upstream response (200); model output containing an SSN blocked post-upstream (400, sensitive_patterns); injection prompt blocked pre-upstream (400, prompt_injection). Pure-JSON audit lines written for all three paths with the derived `upstream_provider`.
- Credential-leak protection added 2026-08-15 (incident-driven): output check covers tool-call arguments; regex credential scanner + `CANARY_TOKENS` exact-match canaries run on inputs and outputs (see section 7).
- Not yet implemented: redaction, streaming-response inspection, Anthropic-format route, tenant/user policy scoping, auth on the proxy, Kubernetes manifests.

## Known issues (from 2026-08-14 review, see `docs/review-2026-08-14.md`)

- **[P0, resolved 31a8186]** The proxy now forwards the raw request body verbatim (minus `agent_id`/`user_id`), so `tools`, `tool_choice`, `stream`, `response_format`, etc. reach upstream untouched. Covered by `proxy/test_app.py`.
- **[P1, resolved]** The output block list is now high-confidence only (SSN, grouped credit cards, API-key/token shapes). Email/phone/IPv4 moved to `redact_candidate_patterns` (non-blocking, reserved for the future redact-only mode).
- **[P1, resolved]** `proxy/test_app.py` covers field forwarding, model default, null-content/tool_calls verbatim forwarding, input block, output block, fail-closed 503, fail-open passthrough, and allow-path passthrough (mocked upstream + guardrails).
- **[P3]** `extract_input_text` scans every non-assistant role (whitelist to system/user/tool per design); `upstream_provider` audit field is hardcoded to "openai"; upstream timeout hardcoded at 120 s.

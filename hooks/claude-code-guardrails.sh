#!/usr/bin/env bash
# Example hook for Claude Code (or similar agent) to call the guardrail service.
#
# Usage:
#   - Pre-turn (prompt check):
#       echo "$PROMPT_TEXT" | ./claude-code-guardrails.sh check-input
#   - Post-turn (response check):
#       echo "$RESPONSE_TEXT" | ./claude-code-guardrails.sh check-output
#
# Exit codes:
#   0 - ok
#   1 - issues detected (details printed to stdout)
#   2 - service error (unreachable, or invalid/non-JSON response)

set -euo pipefail

GUARDRAILS_URL="${GUARDRAILS_URL:-http://localhost:8090}"
CURL_TIMEOUT="${GUARDRAILS_TIMEOUT_SECONDS:-10}"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <check-input|check-output>" >&2
  exit 2
fi

MODE="$1"
TEXT=$(cat)

ENDPOINT=""
case "$MODE" in
  check-input)
    ENDPOINT="${GUARDRAILS_URL}/check-input"
    ;;
  check-output)
    ENDPOINT="${GUARDRAILS_URL}/check-output"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac

# Build the JSON body with jq so backslashes, quotes, and newlines in code
# text cannot produce a malformed payload.
PAYLOAD=$(jq -n --arg text "$TEXT" '{text: $text}')

if ! RESPONSE=$(curl -s --max-time "$CURL_TIMEOUT" -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD"); then
  echo "Guardrail service unreachable at $ENDPOINT" >&2
  exit 2
fi

OK=$(echo "$RESPONSE" | jq -r '.ok' 2>/dev/null) || OK="null"

case "$OK" in
  true)
    exit 0
    ;;
  false)
    echo "Guardrail issues detected:"
    echo "$RESPONSE" | jq -r '.issues[] | "[\(.severity)] \(.scanner): \(.message)"'
    exit 1
    ;;
  *)
    echo "Unexpected guardrail response: $RESPONSE" >&2
    exit 2
    ;;
esac

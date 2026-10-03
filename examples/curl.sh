#!/usr/bin/env bash
# Support-ticket triage with plain curl. Reads CLEF_URL (default http://127.0.0.1:8910) and CLEF_API_KEY.
set -euo pipefail

URL="${CLEF_URL:-http://127.0.0.1:8910}"
AUTH=()
if [[ -n "${CLEF_API_KEY:-}" ]]; then AUTH=(-H "X-API-Key: ${CLEF_API_KEY}"); fi

post() { curl -sS --fail-with-body "${AUTH[@]}" -H 'Content-Type: application/json' -X "$1" "${URL}$2" -d "$3"; echo; }

echo "== single label"
post POST /v1/classify '{"input": "Checkout is down, orders blocked", "labels": ["billing", "technical", "account"]}'

echo "== multi label (every label >= threshold)"
post POST /v1/classify '{"input": "I was charged twice and the app crashes on login",
  "labels": {"billing": "Payments, invoices, refunds", "technical": "Bugs and outages", "account": "Login, profile, permissions"},
  "multi_label": true, "threshold": 0.5}'

echo "== score"
post POST /v1/score '{"input": "Checkout is down, orders blocked", "levels": ["low", "medium", "high"],
  "instructions": "How urgent is this ticket?"}'

echo "== saved classifier: create, use, delete"
post PUT /v1/classifiers/support-triage '{"kind": "classify", "labels": ["billing", "technical", "account"],
  "description": "Support ticket routing"}'
post POST /v1/classifiers/support-triage '{"input": "Checkout is down, orders blocked"}'
post POST /v1/classifiers/support-triage/batch '{"inputs": ["Refund my invoice", "Cannot reset my password"]}'
post DELETE /v1/classifiers/support-triage '{}'

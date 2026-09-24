#!/bin/sh
# The same three calls as client.py, in curl. `j3v serve` speaks plain HTTP/1.1 and needs no client library.
#
#   ./client.sh [base-url]        default: http://127.0.0.1:8000
set -eu
J3V="${1:-http://127.0.0.1:8000}"

# The server checks the bearer token before it routes, so every call below sends it when it is set.
echo "== is it up, and which artifact is loaded?"
curl -fsS ${J3V_API_KEY:+-H "Authorization: Bearer $J3V_API_KEY"} "$J3V/healthz"; echo

echo
echo "== what was this artifact compiled to answer, and at which thresholds?"
# The thresholds come from the schema, so a client never hardcodes them.
curl -fsS ${J3V_API_KEY:+-H "Authorization: Bearer $J3V_API_KEY"} "$J3V/v1/schema" | head -c 400; echo

echo
echo "== ask the whole schema"
curl -fsS -X POST "$J3V/v1/systemone" \
  -H 'Content-Type: application/json' \
  ${J3V_API_KEY:+-H "Authorization: Bearer $J3V_API_KEY"} \
  -d '{"state": {"message": "I was charged twice, please refund me"}}'
echo

echo
echo "== ask one question (the definition must match what was compiled, or you get a 422)"
curl -fsS -X POST "$J3V/v1/systemone" \
  -H 'Content-Type: application/json' \
  ${J3V_API_KEY:+-H "Authorization: Bearer $J3V_API_KEY"} \
  -d '{"state": {"message": "where is my order"},
       "questions": {"refund_requested": {"type": "noul",
                                          "instructions": "Does the user ask to get their money back?"}}}'
echo

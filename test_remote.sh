#!/usr/bin/env bash
# Test the deployed API on the Hetzner box via an SSH tunnel.
#   ./test_remote.sh [image] [threshold]
# Examples:
#   ./test_remote.sh ids/id-connecticut.jpeg
#   ./test_remote.sh some.jpg 0.4
#
# API key: read from $MVSS_API_KEY, else pulled from DEPLOY_NOTES.local.md.
set -euo pipefail
cd "$(dirname "$0")"

IMG="${1:-ids/id-connecticut.jpeg}"
THRESH="${2:-0.5}"
SSH_HOST="${SSH_HOST:-id-tamper}"
LOCAL_PORT="${LOCAL_PORT:-18000}"

[ -f "$IMG" ] || { echo "no such file: $IMG" >&2; exit 1; }

KEY="${MVSS_API_KEY:-$(grep -m1 'MVSS_API_KEY' DEPLOY_NOTES.local.md 2>/dev/null | sed 's/.*= *//')}"
[ -n "$KEY" ] || { echo "no API key (set MVSS_API_KEY or DEPLOY_NOTES.local.md)" >&2; exit 1; }

# open tunnel, ensure it's torn down on exit
ssh -f -N -L "${LOCAL_PORT}:localhost:8000" "$SSH_HOST"
trap 'pkill -f "${LOCAL_PORT}:localhost:8000" 2>/dev/null || true' EXIT
sleep 2

echo "-> $IMG (threshold=$THRESH)"
curl -s -H "X-API-Key: $KEY" \
     -F "file=@${IMG}" \
     "http://localhost:${LOCAL_PORT}/detect?threshold=${THRESH}" \
  | { command -v jq >/dev/null && jq || cat; }

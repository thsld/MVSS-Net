#!/usr/bin/env bash
# Launch the tamper-detection API locally for testing.
#   ./run_local.sh
# Reads .env.local if present (for MVSS_API_KEY etc.), then serves on :8000
# with auto-reload. Requires the uv-managed .venv and the checkpoints in
# ckpt/mvssnet_model/.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env.local ] && source .env.local
: "${MVSS_API_KEY:=dev-local-secret}"
export MVSS_API_KEY

echo "API:  http://localhost:8000   (docs: /docs)"
echo "Key:  X-API-Key: ${MVSS_API_KEY}"
exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000 --reload

#!/bin/sh
# Container entrypoint: start the Cloudflare Tunnel connector (if a token is
# provided) so id-check.rntg.in points here, then serve the API.
set -e

if [ -n "$CF_TUNNEL_TOKEN" ]; then
    echo "starting cloudflared tunnel connector ..."
    cloudflared tunnel --no-autoupdate run --token "$CF_TUNNEL_TOKEN" &
fi

exec uvicorn app:app --host 0.0.0.0 --port 8000

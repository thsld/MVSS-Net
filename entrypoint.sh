#!/bin/sh
# Container entrypoint: start the Cloudflare Tunnel connector (if a token is
# provided) so id-check.rntg.in points here, then serve the API.
set -e

if [ -n "$CF_TUNNEL_TOKEN" ]; then
    echo "starting cloudflared tunnel connector ..."
    # --protocol http2: use TCP (not QUIC/UDP) — many cloud/container networks
    #   (e.g. RunPod) block the UDP that cloudflared's default QUIC needs.
    cloudflared tunnel --no-autoupdate --protocol http2 run --token "$CF_TUNNEL_TOKEN" &
fi

exec uvicorn app:app --host 0.0.0.0 --port 8000

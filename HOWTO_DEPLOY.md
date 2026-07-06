# Deploying the tamper-detection API to Hetzner

Build-on-box deployment of the MVSS-Net tamper-detection API to a **Hetzner
CCX13** (dedicated vCPU — no fair-use throttling). No container registry; the
image is built directly on the server. CPU-only, no GPU.

- **App:** `app.py` — FastAPI, loads both checkpoints once, answers `POST /detect`.
- **Weights:** two 588 MB checkpoints, moved to the box **once** (they never change).
- **TLS:** Caddy in front, automatic HTTPS via Let's Encrypt.

---

## 0. Prerequisites (on your laptop)

```bash
brew install hcloud
```
- A Hetzner Cloud **API token** (Console → project → Security → API Tokens).
- Your SSH public key: `~/.ssh/id_ed25519.pub`.
- A **domain/subdomain** you can point at the box (needed for TLS), e.g.
  `id-tamper.rentengine.io`.
- The two checkpoints present locally at `ckpt/mvssnet_model/mvssnet_{casia,defacto}.pt`.

> **Before anything else:** the service files must be pushed to the fork, or the
> `git clone` below pulls the old upstream code with no `app.py`/`Dockerfile`.
> See "Publishing the code" at the bottom.

---

## 1. Provision the server (one-time)

```bash
hcloud context create rentengine          # paste API token
hcloud ssh-key create --name laptop --public-key-from-file ~/.ssh/id_ed25519.pub

hcloud server create \
  --name id-tamper \
  --type ccx13 \            # 2 dedicated vCPU / 8 GB — no throttling
  --location ash \          # Ashburn VA (us-east); use hil for Hillsboro OR
  --image docker-ce \       # Docker preinstalled
  --ssh-key laptop

SERVER=root@$(hcloud server ip id-tamper)
echo "$SERVER"
```

Lock down the firewall to just SSH + web:

```bash
hcloud firewall create --name id-tamper-fw
hcloud firewall add-rule id-tamper-fw --direction in --protocol tcp --port 22  --source-ips 0.0.0.0/0 --source-ips ::/0
hcloud firewall add-rule id-tamper-fw --direction in --protocol tcp --port 80  --source-ips 0.0.0.0/0 --source-ips ::/0
hcloud firewall add-rule id-tamper-fw --direction in --protocol tcp --port 443 --source-ips 0.0.0.0/0 --source-ips ::/0
hcloud firewall apply-to-resource id-tamper-fw --type server --server id-tamper
```

Point your DNS `A` record (`id-tamper.rentengine.io`) at the server IP before
requesting TLS.

---

## 2. Get code + weights onto the box

Code and weights travel separately — the weights are gitignored, so they are
**not** in the repo.

```bash
# a) code — clone the fork (requires a deploy key on the box; see bottom)
ssh $SERVER 'git clone git@github.com:thsld/MVSS-Net.git app 2>/dev/null || (cd app && git pull)'

# b) weights — copy the 2 checkpoints to the path the Dockerfile expects (~1.2 GB, ONCE)
ssh $SERVER 'mkdir -p app/ckpt/mvssnet_model'
scp ckpt/mvssnet_model/mvssnet_casia.pt   $SERVER:app/ckpt/mvssnet_model/
scp ckpt/mvssnet_model/mvssnet_defacto.pt $SERVER:app/ckpt/mvssnet_model/
```

> No GitHub deploy key? Replace step (a) with a direct sync from your laptop:
> ```bash
> rsync -av --exclude .git --exclude .venv --exclude ids --exclude 'out*' \
>   ./ $SERVER:app/
> ```

---

## 3. Build + run

```bash
# build natively on amd64 (fast — no emulation)
ssh $SERVER 'cd app && docker build -t id-tamper:latest .'

# a private docker network so Caddy can reach the app by name
ssh $SERVER 'docker network create web 2>/dev/null || true'

# run the API (not published to the host — only reachable inside the network)
ssh $SERVER 'docker run -d --name id-tamper --restart unless-stopped \
  --network web \
  -e MVSS_API_KEY="CHANGE-ME-strong-secret" \
  -e OMP_NUM_THREADS=2 \           # match CCX13 vCPU count
  --memory 7g \                    # headroom on the 8 GB box
  id-tamper:latest'
```

### Caddy for automatic HTTPS

```bash
ssh $SERVER 'cat > /root/Caddyfile <<EOF
id-tamper.rentengine.io {
    reverse_proxy id-tamper:8000
}
EOF'

ssh $SERVER 'docker run -d --name caddy --restart unless-stopped \
  --network web \
  -p 80:80 -p 443:443 \
  -v /root/Caddyfile:/etc/caddy/Caddyfile \
  -v caddy_data:/data \
  caddy:2'
```

Caddy fetches a Let's Encrypt cert automatically once DNS resolves.

---

## 4. Verify

```bash
# health (open, no key)
curl -s https://id-tamper.rentengine.io/health

# detect — requires the API key
curl -s -H "X-API-Key: CHANGE-ME-strong-secret" \
  -F "file=@/path/to/id.jpg" \
  https://id-tamper.rentengine.io/detect | jq
```

Expected:
```json
{"manipulated": true, "score": 0.687, "threshold": 0.5,
 "models": {"casia": 0.687, "defacto": 0.072}}
```

---

## 5. Redeploying (the rare case)

Weights don't change, so you **never re-copy them**. Docker's layer cache reuses
the Torch install layer, so only changed code re-copies — rebuilds take seconds.

```bash
ssh $SERVER 'cd app && git pull && docker build -t id-tamper:latest . \
  && docker rm -f id-tamper \
  && docker run -d --name id-tamper --restart unless-stopped \
     --network web \
     -e MVSS_API_KEY="CHANGE-ME-strong-secret" \
     -e OMP_NUM_THREADS=2 --memory 7g \
     id-tamper:latest'
```

---

## 6. Monitoring / confirming you're not throttled

```bash
ssh $SERVER 'docker stats --no-stream'   # live CPU% / MEM per container
ssh $SERVER 'vmstat 1 5'                 # watch the "st" (steal) column
```
On CCX13 (dedicated vCPU) `st` should stay ~0 even at full load. Sustained high
`st` would indicate hypervisor contention — not expected on a dedicated plan.

---

## Environment variables (`app.py`)

| Var | Default | Purpose |
|-----|---------|---------|
| `MVSS_API_KEY` | *(unset)* | Required in `X-API-Key`. **Unset = open — never in prod.** |
| `MVSS_MAX_UPLOAD_MB` | `15` | Reject larger uploads (413). |
| `MVSS_THRESHOLD` | `0.5` | Default decision threshold. |
| `MVSS_DEVICE` | `auto` | `cpu` on the server (no GPU). |
| `OMP_NUM_THREADS` | *(all cores)* | Pin Torch threads to the vCPU count. |

---

## Security notes

- The API processes **government IDs (PII)**. It holds nothing on disk —
  each request is decoded in memory and discarded after the JSON response.
- Always set a strong `MVSS_API_KEY`; the app binds only to the private docker
  network, so Caddy (HTTPS) is the only public entry point.
- Keep the box patched: `ssh $SERVER 'apt-get update && apt-get upgrade -y'`.

---

## Publishing the code (do this first)

The `git clone` step needs the service files on the fork:

```bash
git push origin <branch>     # push the committed service files
# then merge to master (or clone the branch on the box)
```

For step 2a's `git clone` over SSH, add a **deploy key** to the fork:
```bash
ssh $SERVER 'ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 && cat ~/.ssh/id_ed25519.pub'
# add that public key to github.com/thsld/MVSS-Net → Settings → Deploy keys
```
Or skip GitHub entirely and use the `rsync` alternative in step 2.

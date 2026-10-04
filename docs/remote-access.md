# Remote access

By default clef binds `127.0.0.1`: only the machine it runs on can call it. This page covers calling it from other
machines on your local network, and what to add if you expose it further.

## LAN setup

1. Create a keys file (one `name:key` per line, `#` comments allowed). Generate strong keys:

   ```bash
   mkdir -p ~/.config/clef
   {
     echo "# name:key"
     echo "ci-bot:$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
     echo "laptop:$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
   } > ~/.config/clef/keys
   chmod 600 ~/.config/clef/keys
   ```

2. Start the server bound to all interfaces:

   ```bash
   CLEF_HOST=0.0.0.0 CLEF_API_KEYS=$HOME/.config/clef/keys CLEF_RATE_LIMIT=120 clef serve --detach
   ```

   `CLEF_API_KEYS` also accepts the list inline: `ci-bot:KEY1,laptop:KEY2`. `CLEF_API_KEY=KEY` is a single key named
   `default`. Binding a non-loopback host with no key logs a loud warning at startup; do not ignore it.

3. Allow the port through the firewall for your LAN only (below).

4. Call it from another machine:

   ```bash
   curl http://192.168.1.50:8910/v1/classify -H "X-API-Key: $CLEF_KEY" -H 'Content-Type: application/json' \
     -d '{"input": "Checkout is down", "labels": ["billing", "technical"]}'
   ```

   `Authorization: Bearer <key>` works too. The Python and JS clients take the key as `api_key=` / `apiKey`.

All `/v1/*` routes, and the Hugging Face routes (`/hf/models/*`), then require a key (401 otherwise); the OpenAI SDKs
send it as `Authorization: Bearer` from their `api_key` setting. Open routes: `/`, static files, `/health`, `/livez`, `/docs`,
`/openapi.json`. The console is served without a key, but its API calls need one: enter it in the console settings. The
live event stream accepts `?key=` because browsers cannot set headers on `EventSource`; this puts the key in the URL,
so on untrusted networks use TLS.

## Keys and logging

- Each key has a name. The name (never the key) is stored on every request in `/v1/log` and counted in `/v1/stats`
  (`by_key`), so you can see who is calling.
- Keys are compared in constant time. They are never logged.
- Rotate by editing the keys file and restarting `clef serve` (keys are read at startup).

## Rate limit

`CLEF_RATE_LIMIT=N` allows N inference requests per minute per key (per client IP when auth is off), sliding 60 s
window. Counted routes: `/v1/systemone`, `/v1/batch`, `/v1/classify`, `/v1/classify/batch`, `/v1/score`,
`POST /v1/classifiers/{name}[/batch]`, `/v1/chat/completions`, `/hf/models/*`, `/v1/evaluate` and `POST /v1/jobs`. Over the limit: `429` with `Retry-After: <seconds>`. `0` (default) disables it.
A batch call counts as one request, so also keep `CLEF_MAX_BATCH` reasonable.

## Jobs, webhooks and keys

Async jobs belong to the key that submitted them: other keys get `404` for them and do not see them in the job list.
Webhooks are off until you set `CLEF_WEBHOOK_ALLOW`; list only the receivers you trust (a private address needs its
IP or CIDR listed explicitly) and see [Jobs & webhooks](jobs.md#webhooks-are-off-by-default-ssrf) and the
[security policy](../SECURITY.md) before enabling them on an exposed server. `jobs.db` in the state directory holds
your submitted inputs and the webhook secrets in clear text, so protect that directory like the keys file.

## CORS (browser callers)

Off by default. To let a web page on another origin call the API:

```bash
CLEF_CORS_ORIGINS=https://app.example.com,http://localhost:5173 clef serve
```

`*` is accepted, but do not combine it with an exposed server. Allowed methods GET/POST/PUT/DELETE/OPTIONS, headers
`*`, credentials off. Never put an API key in public front-end code: anyone who loads the page can read it. Call clef
from your own backend instead.

## Firewall

Open TCP 8910 (or your `CLEF_PORT`) to your LAN subnet only, not to the world. Replace `192.168.1.0/24`.

**Ubuntu (ufw)**

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8910 proto tcp
```

**macOS**: System Settings > Network > Firewall; allow incoming connections for the Python/`clef` process when prompted.
For a subnet rule use `pf`.

**Windows (PowerShell, as administrator)**

```powershell
New-NetFirewallRule -DisplayName "clef" -Direction Inbound -Protocol TCP -LocalPort 8910 -RemoteAddress 192.168.1.0/24 -Action Allow
```

**Windows + WSL2**: the server listens inside WSL, so Windows must forward the port. With the default NAT networking
use a port proxy (the WSL address changes on restart), or set `networkingMode=mirrored` in `%USERPROFILE%\.wslconfig`
(Windows 11 22H2+) and then only the firewall rule is needed:

```powershell
$wsl = (wsl -d Ubuntu hostname -I).Trim().Split(' ')[0]
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8910 connectaddress=$wsl connectport=8910
```

**Docker**: `-p 8910:8910` already publishes on all interfaces; bind to a LAN address with
`-p 192.168.1.50:8910:8910` if you want to limit it, and still use keys.

## Beyond the LAN: TLS reverse proxy (Caddy)

Do not expose the plain HTTP port to the internet. Put a reverse proxy with TLS in front, keep clef bound to
`127.0.0.1`, and keep API keys on. `Caddyfile` (certificates are obtained automatically for a public hostname):

```caddyfile
clef.example.com {
    encode zstd gzip

    # large image/video payloads: match CLEF_MAX_BODY_MB (default 64)
    request_body {
        max_size 64MB
    }

    reverse_proxy 127.0.0.1:8910 {
        # server-sent events (/v1/events) must not be buffered
        flush_interval -1
    }
}
```

Run `caddy run --config Caddyfile`. For a LAN-only hostname without a public domain, use `tls internal` inside the
site block (clients must trust Caddy's local CA). Behind a proxy every request comes from `127.0.0.1`, so the rate
limit is per key, not per client IP: keep API keys on.

Additional hardening if you do expose it: set a low `CLEF_RATE_LIMIT`, keep `CLEF_ALLOW_URL_FETCH=0` (default), keep
`CLEF_LOG_STATE=0` (default; request contents are not stored), and restrict by IP in the proxy where possible.

# Remote access

By default clef binds `127.0.0.1`: only the machine it runs on can call it. This page covers calling it from other
machines on your local network, and what to add if you expose it further.

## Two kinds of keys

- **Env keys** (`CLEF_API_KEY`, `CLEF_API_KEYS`): full admin, no quota, the global rate limit. They are the
  break-glass key and the bootstrap: the first admin key is made with one of them, or with `clef keys create`.
  They are read at startup. A managed key may not share a name with an env key: the server refuses such a key
  (`401`) and logs a warning at startup; run `clef keys revoke NAME` or rename the env key. (The server is the
  authority, since the env of `clef keys` can differ from the server's.)
- **Managed keys** (`clef keys ...` or the console **Keys** tab): a *principal* (a name) with scopes, an optional daily
  quota and rate limit, and one to five keys (`clef_live_<id>_<secret>`). Only the SHA-256 of a key is stored; the
  secret is shown once, when it is created or rotated. Changes made in the console or through `/v1/admin/keys` apply at
  once; changes made with `clef keys` while the server runs apply within about 10 seconds.

Auth is on when an env key exists or any managed key was ever created, and it never turns off by itself: revoking the
last managed key leaves the server locked until you make a new one (`clef keys create` works with the server stopped,
straight on the database).

```bash
clef keys create ci-bot --scope classify,jobs --quota 50000 --expires 90d   # prints the key once
clef keys create ops --scope admin                                         # an admin key for the console
clef keys list
clef keys rotate ci-bot --grace 24h     # new key now; the old ones keep working for 24 h
clef keys revoke ci-bot --key-id ab3...  # one key; without --key-id the principal is disabled for good
clef keys update ci-bot --quota none --scope classify,jobs,ops
clef keys usage ci-bot --days 30
```

Scopes: `classify` (the model routes, `GET /v1/classifiers`, `/v1/models`, `/v1/evaluate`), `jobs` (`/v1/jobs*`),
`classifiers` (create / change / delete saved classifiers), `ops` (stats, log, events), `urgent` (urgent jobs) and
`admin` (everything, plus `/v1/admin/*` and the Keys tab). A key without the route's scope gets `403`. `GET /v1/whoami`
shows the caller's name, scopes and quota use.

**Daily quota.** Counts records (a batch of 50 is 50) per UTC day. Computed records are charged; results served from
the result cache are free. A job is charged in full when it is submitted, and the part it did not process comes back
if it is cancelled or fails. Over the quota: `429` with `Retry-After` (seconds to the next UTC midnight); the OpenAI
routes report `insufficient_quota`. Responses carry `X-Quota-Limit` and `X-Quota-Remaining`.

**Audit log.** Key changes, job cancel / delete and saved-classifier changes are recorded with who, what, when and
IP (`GET /v1/admin/audit`, also in the console). Secrets are never recorded.

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
   `default`. These are env keys (full admin: with several of them, each can see and delete every job, edit every classifier and mint keys, and the server warns at startup); for scoped keys with quotas use `clef keys create` instead of, or next
   to, them. Binding a non-loopback host with no key (env or managed) logs a loud warning at startup; do not ignore it.

3. Allow the port through the firewall for your LAN only (below).

4. Call it from another machine:

   ```bash
   curl http://192.168.1.50:8910/v1/classify -H "X-API-Key: $CLEF_KEY" -H 'Content-Type: application/json' \
     -d '{"input": "Checkout is down", "labels": ["billing", "technical"]}'
   ```

   `Authorization: Bearer <key>` works too. The Python and JS clients take the key as `api_key=` / `apiKey`.

All `/v1/*` routes, and the Hugging Face routes (`/hf/models/*`), then require a key (401 otherwise, 403 for a key
without the route's scope); `/v1/admin/*` always requires an admin key, even when auth is off; the OpenAI SDKs
send it as `Authorization: Bearer` from their `api_key` setting. Open routes: `/`, static files, `/health`, `/livez`, `/readyz`, `/docs`,
`/openapi.json`. The console is served without a key, but its API calls need one: enter it in the console settings. The
console sends the key in the `X-API-Key` header, also for the live event stream. `/v1/events` still accepts `?key=`
for plain `EventSource` clients (they cannot set headers); this puts the key in the URL, so on untrusted networks
use TLS.

## Keys and logging

- Each key has a name. The name (never the key) is stored on every request in `/v1/log` and counted in `/v1/stats`
  (`by_key`), so you can see who is calling.
- Keys are compared in constant time. clef never logs them: `?key=` values are written as `REDACTED` in the access
  log. If you turn on the access log of a reverse proxy, mask the key there too; see
  [Keys in proxy logs](security.md#keys-in-proxy-logs) for Caddy and nginx.
- Rotate a managed key with `clef keys rotate NAME` (or the console): the old key keeps working for the grace
  period (default 24 h), so callers can switch without a gap. Env keys are rotated by editing the keys file and
  restarting `clef serve` (they are read at startup).

## Rate limit

`CLEF_RATE_LIMIT=N` allows N inference requests per minute per key (per client IP when auth is off), sliding 60 s
window. A managed key can have its own limit (`--rate-limit`), which applies even when `CLEF_RATE_LIMIT` is 0. Counted routes: `/v1/systemone`, `/v1/batch`, `/v1/classify`, `/v1/classify/batch`, `/v1/score`,
`POST /v1/classifiers/{name}[/batch]`, `/v1/chat/completions`, `/hf/models/*`, `/v1/evaluate` and `POST /v1/jobs`. Over the limit: `429` with `Retry-After: <seconds>`. `0` (default) disables it.
A batch call counts as one request, so also keep `CLEF_MAX_BATCH` reasonable.

## Jobs, webhooks and keys

Async jobs belong to the principal that submitted them: other non-admin keys get `404` for them and do not see them
in the job list. Admin keys, including every env key, see all jobs. Jobs and classifiers created while auth was off have no owner and are admin-only to read results of, change or delete once auth is on (non-admin keys may still run such a classifier).
Webhooks are off until you set `CLEF_WEBHOOK_ALLOW`; list only the receivers you trust (a private address needs its
IP or CIDR listed explicitly) and see [Jobs & webhooks](jobs.md#webhooks-are-off-by-default-ssrf) and the
[security policy](../SECURITY.md) before enabling them on an exposed server. `clef.db` in the state directory (or your Postgres
database, with `CLEF_DATABASE_URL`) holds your submitted inputs and the webhook secrets in clear text, so protect that directory like the keys file.

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

    # no `log` directive: if you add one, mask X-Api-Key and ?key= (security.md#keys-in-proxy-logs)
    reverse_proxy 127.0.0.1:8910 {
        # server-sent events (/v1/events) must not be buffered
        flush_interval -1
    }
}
```

Run `caddy run --config Caddyfile`. For a LAN-only hostname without a public domain, use `tls internal` inside the
site block (clients must trust Caddy's local CA). Behind a proxy every request comes from `127.0.0.1`, so the rate
limit is per key, not per client IP: keep API keys on.

**Load balancers.** Point the health check at `/readyz` (open, no key). To take a server out of rotation before
it stops, set `CLEF_DRAIN_S` (e.g. `5`): after SIGTERM / Ctrl-C clef keeps serving that long while `/readyz`
answers 503 `draining`, then shuts down; a second signal stops it at once. Pick the probe interval x failure
threshold below `CLEF_DRAIN_S`, and keep `CLEF_DRAIN_S` below your supervisor's stop timeout (10 s for
`docker stop`). With the default `0` the listening socket closes at once and the load balancer only sees refused
connections.

Additional hardening if you do expose it: set a low `CLEF_RATE_LIMIT`, keep `CLEF_ALLOW_URL_FETCH=0` (default), keep
`CLEF_LOG_STATE=0` (default; request contents are not stored), and restrict by IP in the proxy where possible.

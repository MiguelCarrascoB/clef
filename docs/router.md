# Router and Review

A **router** grows the label set of one single-label [saved classifier](classification.md#saved-classifiers). Texts it
cannot place are kept; every 20 of them go to an LLM that proposes new labels; clef then re-classifies those texts
against the proposed set and only a proposal that clef itself confirms is shown to you (or, in auto mode, published as a
new classifier version). Labels are only ever **added**: no merge, split, rename or retire. A router holds at most 63
labels plus `other`.

The router is **off** until you give it an LLM key. Routing itself (`/v1/routers/{name}/route`) works without one.

## Turn it on

Put the key in a `.env` file next to where you run `clef serve`, or in `~/.config/clef/env` (see
[Configuration](configuration.md#env-files)). A variable already set in the environment wins over both files.

```bash
echo 'GEMINI_API_KEY=your-key' >> .env                 # or: >> ~/.config/clef/env
# systemd (deploy/clef.service) reads ~/.config/clef/env as well
# docker compose: the .env file next to docker-compose.yml is passed through by the compose file
```

Restart the server and check with `clef doctor` (it says whether the LLM is on, never the key). By default clef
talks to Gemini's OpenAI-compatible endpoint with `gemini-3.8-flash`. Any OpenAI-compatible server works:

```bash
CLEF_LLM_BASE_URL=http://localhost:11434/v1 CLEF_LLM_MODEL=llama3 CLEF_LLM_API_KEY=ollama clef serve
```

!!! warning "Privacy"
    Setting the key is your consent: up to `CLEF_ROUTER_MAX_INPUT_CHARS` (500) characters of each buffered input are
    sent to the LLM host. The console shows an "external LLM: host" badge on every router. Nothing is sent without a
    key, and the key is never logged.

## Create and use a router

```bash
curl -X PUT localhost:8910/v1/routers/tickets -H 'Content-Type: application/json' -d '{
  "seeds": {"billing": "money, invoices, refunds", "technical": "bugs and outages"},
  "domain_hint": "customer support tickets", "mode": "review"}'

curl -X POST localhost:8910/v1/routers/tickets/route -H 'Content-Type: application/json' \
  -d '{"input": "my password reset email never arrives"}'
```

`seeds` are optional. Without them an existing single-label classifier named like the router (or `classifier`) supplies
the labels; with neither, the router starts **cold**: it answers `other` (reason `cold`) without running the model, and
after 20 items one bigger LLM call proposes a starter set of up to 8 labels. clef needs at least two options, so a
single seed is enough to start (`[seed, other]`).

Every routed result has a `reason`:

| reason | meaning | kept? |
| --- | --- | --- |
| `classified` | confident answer | no |
| `other` | `other` won | yes, buffered |
| `low_confidence` | the top probability is below the threshold for this label count | yes, buffered |
| `ambiguous` | the top two labels are within `margin` (0.15) | yes, for a human only |
| `cold` | the router has no labels yet | yes, buffered |

The threshold rises with the number of options (`other` included): 0.5 up to 10, 0.7 up to 30, 0.85 up to 64. Override
it per router with `gate.thresholds` (`[[10, 0.5], [30, 0.7], [64, 0.85]]`) and `gate.margin`. These are defaults
chosen for the label counts; tune them on your own data with the Evaluate screen's coverage curve.

## Proposals and the re-check

A call happens when `CLEF_ROUTER_BATCH` (20) items are buffered, when the oldest has waited `CLEF_ROUTER_FLUSH_S`
(600 s; at least 2 items, 5 for a cold start), or on `POST /v1/routers/{name}/propose`. The LLM is shown the current
labels (to avoid duplicates) and the numbered texts; it asks for at most 3 new labels. Then clef re-classifies each
proposal's member texts under *current labels + the proposal + other*. A proposal passes when at least half of its
members (and at least 2) land on the new label, and it fails as a duplicate when most land on one existing label. Texts
a person already labelled with an existing label count against the proposal. Failed proposals are kept as `failed`
rows so you can see why. Items that no proposal takes are checked at most twice more, then become `permanent_other`.

## Modes

* **review** (default): confirmed proposals appear in the Review screen. *Approve* publishes them (all, or an edited
  subset with your own descriptions) as a new classifier version; *Reject* returns the texts to the buffer.
* **auto**: a confirmed proposal is published at once, at most `CLEF_ROUTER_MAX_NEW_PER_DAY` (10) new labels per UTC day;
  beyond that it waits in the queue. The starter set of a cold router is exempt from that limit.

After a label set is published, the buffered texts are re-classified; those that now pass the gate are resolved by the
router. Every label set is a router version (`GET /v1/routers/{name}/versions`); *Roll back* publishes an earlier one
again as a new classifier version.

## The Review queue

One table holds router proposals and low-confidence items. Items come from routers, and from any saved classifier with
a `review_threshold` (single-label, 0..1): interactive runs (`POST /v1/classifiers/{name}[/batch]`, not jobs) whose
confidence is below it are queued, in the classifier owner's queue. Only runs by the owner or an admin are captured
(with auth off, every run): when another key runs your classifier, its inputs are never stored for review, so its
text does not reach you. Media inputs are never captured.

Labelling an item re-classifies its text under the active labels **before** the text is purged and records whether clef
agrees (`prediction.recheck`); the label also counts against later proposals that would absorb the same text. Input text
is stored only in review rows: it is dropped when the item is resolved and after `CLEF_REVIEW_TTL_DAYS` (30) for rows
nobody resolved (they become `expired`). Resolved rows keep the label, the input's sha256 and the prediction.

The **Review** screen of the [console](console.md) lists the queue (filters: kind, state, router, classifier), has a
label picker and Discard for items, approve / edit / reject for proposals, and per router: its label set, versions with
roll back, mode, buffer size, the LLM badge and today's calls and tokens. It needs the `classifiers` scope.

## API

| Route | Scope |
| --- | --- |
| `PUT\|GET\|DELETE /v1/routers/{name}`, `GET /v1/routers`, `POST .../propose`, `GET .../versions`, `POST .../versions/{v}/activate` | `classifiers` |
| `POST /v1/routers/{name}/route`, `.../route/batch` | `classify` |
| `GET /v1/review`, `GET /v1/review/{id}`, `POST /v1/review/{id}/resolve` | `classifiers` |

Routers and review rows belong to the key that created them; admins see all. `resolve` takes `{"action": "label",
"label"}` or `{"action": "discard"}` for items, `{"action": "approve", "labels"?}` or `{"action": "reject"}` for
proposals. Errors: 400 bad name, label or cap; 403 scope or owner; 404; 409 a call in flight, the classifier belongs to
another router, or the row is already resolved; 502 the LLM call failed; 503 the LLM is off.
Full request and response shapes: [Architecture](ARCHITECTURE.md) and `/docs`.

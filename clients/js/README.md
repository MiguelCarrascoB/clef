# clef-client (JavaScript)

Dependency-free ESM client for the clef-flash server. Node 18+ and browsers (uses the global `fetch`).
Not published to npm; import it from this folder: `import { ClefClient } from './clients/js/src/index.js'`.

```js
import { ClefClient } from 'clef-client';

const clef = new ClefClient({ baseUrl: 'http://127.0.0.1:8910', apiKey: process.env.CLEF_API_KEY });

const r = await clef.classify('Checkout is down, orders blocked', ['billing', 'technical', 'account']);
console.log(r.label, r.confidence, r.scores);

const tags = await clef.classify(text, { billing: 'Payments, invoices', technical: 'Bugs, outages' },
  { multiLabel: true, threshold: 0.5 });          // tags.labels = every label >= threshold
const s = await clef.score(text, ['low', 'medium', 'high']);          // s.level, s.score, s.distribution
const rows = await clef.classifyMany(texts, ['billing', 'technical']); // one request, results in order

const triage = clef.classifier('support-triage');                      // saved classifier
await triage.save({ labels: ['billing', 'technical', 'account'] });
console.log((await triage.classify(text)).label);
```

## Options

`new ClefClient({ baseUrl, apiKey, retries = 3, backoffMs = 500, timeoutMs, fetch })`

- `baseUrl` / `apiKey` fall back to `CLEF_URL` / `CLEF_API_KEY` when `process.env` exists (Node); default URL `http://127.0.0.1:8910`.
- `fetch` is injectable (tests, proxies, older runtimes). `timeoutMs` aborts each attempt via `AbortController`.
- Retries (exponential backoff with jitter): 503 (model loading / OOM), 429 (waits for `Retry-After`), network errors. Other 4xx are never retried; timeouts are not retried.

## API

| Call | Returns |
| --- | --- |
| `classify(input, labels, { instructions, multiLabel, threshold, images, videos, model })` | `{ label, labels, confidence, scores, multiLabel, threshold, requestId, usage, timing, classifier, raw }` |
| `classifyMany(inputs, labels, { instructions, multiLabel, threshold, model, chunkSize })` | array of the above, in input order |
| `score(input, levels, { instructions, images, videos, model })` | `{ score, level, levelIndex, confidence, distribution, requestId, usage, timing, raw }` |
| `classifier(name)` | handle with `.classify(input, opts)`, `.classifyMany(inputs)`, `.save(def)`, `.get()`, `.delete()` |
| `listClassifiers()`, `saveClassifier(name, def)`, `getClassifier(name)`, `deleteClassifier(name)` | raw server objects |
| `decide(state, questions, { images, videos, model })`, `batch(records)`, `health()`, `stats()` | raw server objects |

Multi-label: `labels` holds every label scoring >= `threshold` (best first, possibly empty); `label` is the top one (or `null`) and `confidence` its score.

Errors are `ClefError` with `status` (null for network/timeout), `detail`, `requestId`, `retryAfter` (seconds).

## Browsers and CORS

Browsers need the server to allow your origin: start it with `CLEF_CORS_ORIGINS=https://your.app`. Do not ship a real API key in public frontend code.

## Tests

`npm test` (Node 18+, mocked fetch, no network).

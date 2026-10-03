import test from 'node:test';
import assert from 'node:assert/strict';
import { ClefClient, ClefError } from '../src/index.js';

const SINGLE = {
  multi_label: false,
  label: 'technical',
  confidence: 0.96,
  scores: { billing: 0.04, technical: 0.96 },
  usage: { input_tokens: 5 },
  timing: { total_ms: 3 },
  request_id: 'r1',
};
const MULTI = {
  multi_label: true,
  labels: ['technical', 'billing'],
  scores: { billing: 0.6, technical: 0.9, account: 0.1 },
  threshold: 0.5,
  request_id: 'r2',
};
const SCORE = {
  score: 1.78,
  level: 'high',
  level_index: 2,
  confidence: 0.86,
  distribution: { low: 0.07, medium: 0.07, high: 0.86 },
  request_id: 'r3',
};

const json = (status, body, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json', ...headers } });

// A mocked fetch that records calls and answers from a queue or a function.
function mock(responder) {
  const calls = [];
  const fetch = async (url, init) => {
    const call = { url, method: init.method, headers: init.headers, body: init.body ? JSON.parse(init.body) : null, signal: init.signal };
    calls.push(call);
    const r = typeof responder === 'function' ? await responder(call, calls.length) : responder;
    if (r instanceof Error) throw r;
    return r;
  };
  return { fetch, calls };
}

const client = (m, extra = {}) => {
  const c = new ClefClient({ fetch: m.fetch, backoffMs: 0, ...extra });
  return c;
};

test('classify single', async () => {
  const m = mock(json(200, SINGLE));
  const r = await client(m, { apiKey: 'k' }).classify('Checkout is down', ['billing', 'technical'], { instructions: 'why?' });
  assert.equal(r.label, 'technical');
  assert.deepEqual(r.labels, ['technical']);
  assert.equal(r.confidence, 0.96);
  assert.equal(r.multiLabel, false);
  assert.equal(r.requestId, 'r1');
  assert.equal(r.raw.label, 'technical');
  assert.equal(m.calls[0].url, 'http://127.0.0.1:8910/v1/classify');
  assert.equal(m.calls[0].headers['X-API-Key'], 'k');
  assert.deepEqual(m.calls[0].body, { input: 'Checkout is down', labels: ['billing', 'technical'], instructions: 'why?', model: 'clef-flash' });
});

test('classify multi-label', async () => {
  const m = mock(json(200, MULTI));
  const r = await client(m).classify('x', { billing: 'pay' }, { multiLabel: true, threshold: 0.5, images: ['data:a'] });
  assert.deepEqual(r.labels, ['technical', 'billing']);
  assert.equal(r.label, 'technical');
  assert.equal(r.confidence, 0.9);
  assert.equal(r.threshold, 0.5);
  assert.equal(m.calls[0].body.multi_label, true);
  assert.deepEqual(m.calls[0].body.images, ['data:a']);
});

test('multi-label with no hits', async () => {
  const m = mock(json(200, { multi_label: true, labels: [], scores: { a: 0.1 } }));
  const r = await client(m).classify('x', ['a'], { multiLabel: true });
  assert.equal(r.label, null);
  assert.equal(r.confidence, null);
});

test('classifyMany keeps order and chunks', async () => {
  const m = mock((call) => json(200, { request_id: 'rb', results: call.body.inputs.map((i) => ({ ...SINGLE, label: i, request_id: undefined })) }));
  const c = client(m);
  const out = await c.classifyMany(['a', 'b', 'c'], ['a', 'b']);
  assert.deepEqual(out.map((r) => r.label), ['a', 'b', 'c']);
  assert.equal(out[0].requestId, 'rb');
  assert.equal(m.calls.length, 1);
  assert.equal(m.calls[0].url.endsWith('/v1/classify/batch'), true);
  const out2 = await c.classifyMany(['a', 'b', 'c'], ['a', 'b'], { chunkSize: 2 });
  assert.equal(out2.length, 3);
  assert.equal(m.calls.length, 3);
});

test('score', async () => {
  const m = mock(json(200, SCORE));
  const r = await client(m).score('x', ['low', 'medium', 'high']);
  assert.equal(r.level, 'high');
  assert.equal(r.levelIndex, 2);
  assert.equal(r.distribution.high, 0.86);
  assert.deepEqual(m.calls[0].body.levels, ['low', 'medium', 'high']);
});

test('classifier handle CRUD', async () => {
  const m = mock((call) => {
    if (call.method === 'PUT') return json(200, { ...call.body, name: 'support-triage' });
    if (call.method === 'DELETE') return json(200, { deleted: 'support-triage' });
    if (call.url.endsWith('/v1/classifiers')) return json(200, { classifiers: [{ name: 'a' }] });
    if (call.url.endsWith('/batch')) return json(200, { results: [SINGLE, SINGLE] });
    if (call.method === 'GET') return json(200, { name: 'support-triage' });
    return json(200, { ...SINGLE, classifier: 'support-triage' });
  });
  const c = client(m);
  const h = c.classifier('support-triage');
  const saved = await h.save({ labels: ['billing', 'technical'], description: 'd' });
  assert.equal(saved.kind, 'classify');
  assert.equal(m.calls[0].url.endsWith('/v1/classifiers/support-triage'), true);
  assert.equal((await h.get()).name, 'support-triage');
  const r = await h.classify('x');
  assert.equal(r.classifier, 'support-triage');
  const many = await h.classifyMany(['a', 'b']);
  assert.equal(many.length, 2);
  assert.equal(many[0].classifier, 'support-triage');
  assert.deepEqual(await c.listClassifiers(), [{ name: 'a' }]);
  assert.deepEqual(await h.delete(), { deleted: 'support-triage' });
  assert.equal((await c.saveClassifier('s', { levels: ['a', 'b'] })).kind, 'score');
});

for (const status of [503, 429]) {
  test(`retries ${status} then succeeds`, async () => {
    const m = mock((_c, n) => (n < 3 ? json(status, { detail: 'busy' }, { 'Retry-After': '0' }) : json(200, SINGLE)));
    const r = await client(m).classify('x', ['a', 'b']);
    assert.equal(r.label, 'technical');
    assert.equal(m.calls.length, 3);
  });
}

test('exhausted retries expose status, requestId, retryAfter', async () => {
  const m = mock(() => json(429, { detail: 'slow', request_id: 'q' }, { 'Retry-After': '0' }));
  await assert.rejects(client(m, { retries: 2 }).health(), (e) => {
    assert.ok(e instanceof ClefError);
    assert.equal(e.status, 429);
    assert.equal(e.requestId, 'q');
    assert.equal(e.retryAfter, 0);
    assert.match(e.message, /slow/);
    return true;
  });
  assert.equal(m.calls.length, 3);
});

test('Retry-After is honoured', async () => {
  const m = mock((_c, n) => (n === 1 ? json(429, { detail: 'x' }, { 'Retry-After': '7' }) : json(200, {})));
  const c = client(m);
  const waits = [];
  c._sleep = async (ms) => waits.push(ms);
  await c.health();
  assert.deepEqual(waits, [7000]);
});

test('no retry on 400', async () => {
  const m = mock(json(400, { detail: 'labels: need 2', request_id: 'z' }));
  await assert.rejects(client(m).classify('x', ['a']), (e) => e.status === 400 && e.detail === 'labels: need 2');
  assert.equal(m.calls.length, 1);
});

test('network errors are retried', async () => {
  const m = mock((_c, n) => (n < 2 ? new TypeError('fetch failed') : json(200, { ok: 1 })));
  assert.deepEqual(await client(m).health(), { ok: 1 });
  assert.equal(m.calls.length, 2);
});

test('timeout aborts via AbortController and is not retried', async () => {
  const fetch = (_u, init) =>
    new Promise((_res, rej) => init.signal.addEventListener('abort', () => rej(Object.assign(new Error('aborted'), { name: 'AbortError' }))));
  const c = new ClefClient({ fetch, timeoutMs: 20, backoffMs: 0 });
  await assert.rejects(c.health(), (e) => e instanceof ClefError && e.status === null && /timeout/.test(e.detail));
});

test('baseUrl trailing slash is trimmed; no apiKey header by default', async () => {
  const m = mock(json(200, {}));
  await new ClefClient({ baseUrl: 'http://h:1/', fetch: m.fetch }).health();
  assert.equal(m.calls[0].url, 'http://h:1/health');
  assert.equal('X-API-Key' in m.calls[0].headers, false);
});

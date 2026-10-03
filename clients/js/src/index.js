// clef-client: dependency-free client for the clef-flash server. Works in Node 18+ and browsers.

export const DEFAULT_URL = 'http://127.0.0.1:8910';
const MAX_RETRY_AFTER_MS = 60000;

export class ClefError extends Error {
  constructor(status, detail, requestId = null, retryAfter = null) {
    super(`[${status}] ${detail}` + (requestId ? ` (request_id=${requestId})` : ''));
    this.name = 'ClefError';
    this.status = status;
    this.detail = detail;
    this.requestId = requestId;
    this.retryAfter = retryAfter; // seconds, or null
  }
}

const clean = (o) => {
  const out = {};
  for (const [k, v] of Object.entries(o)) {
    if (v === undefined || v === null) continue;
    if (Array.isArray(v) && v.length === 0 && k !== 'inputs') continue;
    out[k] = v;
  }
  return out;
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const enc = encodeURIComponent;

export function parseClassification(body, requestId = null) {
  const scores = body.scores || {};
  const multiLabel = !!body.multi_label;
  let label, labels;
  if (multiLabel) {
    labels = body.labels || [];
    label = labels.length ? labels[0] : null;
  } else {
    label = body.label ?? null;
    labels = label !== null ? [label] : [];
  }
  let confidence = body.confidence ?? null;
  if (confidence === null && label !== null) confidence = scores[label] ?? null;
  return {
    label,
    labels,
    confidence,
    scores,
    multiLabel,
    threshold: body.threshold ?? null,
    requestId: body.request_id ?? requestId,
    usage: body.usage || {},
    timing: body.timing || {},
    classifier: body.classifier ?? null,
    raw: body,
  };
}

export function parseScore(body, requestId = null) {
  return {
    score: body.score,
    level: body.level,
    levelIndex: body.level_index,
    confidence: body.confidence ?? null,
    distribution: body.distribution || {},
    requestId: body.request_id ?? requestId,
    usage: body.usage || {},
    timing: body.timing || {},
    classifier: body.classifier ?? null,
    raw: body,
  };
}

// Saved classifiers answer with either shape; `distribution` marks a score.
const parseResult = (body) => ('distribution' in body ? parseScore(body) : parseClassification(body));

function parseBatch(body, classifier = null) {
  return (body.results || []).map((r) => {
    const item = classifier && !r.classifier ? { ...r, classifier } : r;
    const res = parseResult(item);
    if (res.requestId == null) res.requestId = body.request_id ?? null;
    return res;
  });
}

const TERMINAL = new Set(['succeeded', 'failed', 'cancelled']);

export function parseJob(body) {
  const p = body.progress || {};
  return {
    id: body.id,
    kind: body.kind ?? '',
    status: body.status,
    done: p.done ?? 0,
    total: p.total ?? null,
    failed: p.failed ?? 0,
    percent: p.percent ?? null,
    etaS: body.eta_s ?? null,
    error: body.error ?? null,
    result: body.result ?? null,
    metadata: body.metadata ?? null,
    createdAt: body.created_at ?? null,
    startedAt: body.started_at ?? null,
    finishedAt: body.finished_at ?? null,
    webhook: body.webhook ?? null,
    finished: TERMINAL.has(body.status),
    ok: body.status === 'succeeded',
    raw: body,
  };
}

// One result row; `result` is a typed Classification / ScoreResult for classify / score jobs (null on error rows).
export function parseJobItem(row, kind = '') {
  const error = row.error ?? null;
  let result = null;
  if (error === null && kind === 'classify') result = parseClassification({ ...row, multi_label: 'labels' in row });
  else if (error === null && kind === 'score') result = parseScore(row);
  return { index: row.index ?? -1, error, ok: error === null, result, raw: row };
}

function chunks(items, size) {
  if (!size || size <= 0 || items.length <= size) return [items];
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

export class ClefClient {
  /**
   * @param {object} [opts]
   * @param {string} [opts.baseUrl] defaults to CLEF_URL (Node) or http://127.0.0.1:8910
   * @param {string} [opts.apiKey] defaults to CLEF_API_KEY (Node)
   * @param {number} [opts.retries=3] extra attempts on 503, 429 (Retry-After) and network errors
   * @param {number} [opts.backoffMs=500]
   * @param {number} [opts.timeoutMs] per-attempt timeout (AbortController)
   * @param {typeof fetch} [opts.fetch] injectable fetch
   */
  constructor({ baseUrl, apiKey, retries = 3, backoffMs = 500, timeoutMs, fetch: fetchImpl } = {}) {
    const env = (typeof process !== 'undefined' && process.env) || {};
    this.baseUrl = (baseUrl || env.CLEF_URL || DEFAULT_URL).replace(/\/+$/, '');
    this.apiKey = apiKey || env.CLEF_API_KEY || undefined;
    this.retries = Math.max(0, retries);
    this.backoffMs = backoffMs;
    this.timeoutMs = timeoutMs;
    this._fetch = fetchImpl || ((...a) => globalThis.fetch(...a));
    this._sleep = sleep;
  }

  async _once(method, path, body) {
    const headers = { Accept: 'application/json' };
    if (this.apiKey) headers['X-API-Key'] = this.apiKey;
    if (body !== undefined) headers['Content-Type'] = 'application/json';
    const init = { method, headers };
    if (body !== undefined) init.body = JSON.stringify(body);
    let timer;
    if (this.timeoutMs) {
      const ctl = new AbortController();
      init.signal = ctl.signal;
      timer = setTimeout(() => ctl.abort(), this.timeoutMs);
    }
    let resp;
    try {
      resp = await this._fetch(this.baseUrl + path, init);
    } catch (e) {
      const timedOut = e && e.name === 'AbortError';
      const err = new ClefError(null, timedOut ? `timeout after ${this.timeoutMs} ms` : `network error: ${e && e.message}`);
      err.cause = e;
      err.retryable = !timedOut;
      throw err;
    } finally {
      if (timer) clearTimeout(timer);
    }
    const text = await resp.text();
    let data;
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = undefined;
    }
    if (resp.ok) {
      if (data === undefined) throw new ClefError(resp.status, 'invalid JSON in response');
      return data;
    }
    let detail = text;
    let requestId = resp.headers.get('X-Request-ID');
    if (data && typeof data === 'object') {
      if (data.detail !== undefined) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      requestId = data.request_id || requestId;
    }
    const ra = parseFloat(resp.headers.get('Retry-After'));
    const err = new ClefError(resp.status, detail, requestId, Number.isFinite(ra) ? Math.max(0, ra) : null);
    err.retryable = resp.status === 429 || resp.status === 503;
    throw err;
  }

  async _request(method, path, body) {
    for (let attempt = 0; ; attempt++) {
      try {
        return await this._once(method, path, body);
      } catch (err) {
        if (!(err instanceof ClefError) || !err.retryable || attempt >= this.retries) throw err;
        const wait =
          err.retryAfter != null
            ? Math.min(err.retryAfter * 1000, MAX_RETRY_AFTER_MS)
            : this.backoffMs * 2 ** attempt * (0.5 + Math.random() / 2);
        await this._sleep(wait);
      }
    }
  }

  // --- v2 API ---
  decide(state, questions, { images, videos, model = 'clef-flash' } = {}) {
    return this._request('POST', '/v1/systemone', clean({ model, state, questions, images, videos }));
  }
  batch(records) {
    return this._request('POST', '/v1/batch', { batch: records });
  }
  health() {
    return this._request('GET', '/health');
  }
  stats() {
    return this._request('GET', '/v1/stats');
  }

  // --- classification ---
  async classify(input, labels, { instructions, multiLabel = false, threshold, images, videos, model = 'clef-flash' } = {}) {
    const body = clean({
      input,
      labels,
      instructions,
      multi_label: multiLabel ? true : undefined,
      threshold,
      images,
      videos,
      model,
    });
    return parseClassification(await this._request('POST', '/v1/classify', body));
  }

  async classifyMany(inputs, labels, { instructions, multiLabel = false, threshold, model = 'clef-flash', chunkSize } = {}) {
    const out = [];
    for (const part of chunks(inputs, chunkSize)) {
      const body = clean({
        labels,
        instructions,
        multi_label: multiLabel ? true : undefined,
        threshold,
        model,
      });
      body.inputs = part;
      out.push(...parseBatch(await this._request('POST', '/v1/classify/batch', body)));
    }
    return out;
  }

  async score(input, levels, { instructions, images, videos, model = 'clef-flash' } = {}) {
    const body = clean({ input, levels, instructions, images, videos, model });
    return parseScore(await this._request('POST', '/v1/score', body));
  }

  // --- async jobs (large batches; see docs/jobs.md) ---
  /** Queue a job and return at once. `webhook`: a URL string or { url, secret, events }. */
  async submitJob(kind, payload, { webhook, metadata } = {}) {
    const hook = typeof webhook === 'string' ? { url: webhook } : webhook;
    return parseJob(await this._request('POST', '/v1/jobs', clean({ kind, payload, webhook: hook, metadata })));
  }

  /** classifyMany without holding a connection open: `labels` or a saved `classifier` name. */
  classifyJob(inputs, labels, { classifier, instructions, multiLabel = false, threshold, model = 'clef-flash', webhook, metadata } = {}) {
    const payload = clean({
      labels,
      classifier,
      instructions,
      multi_label: multiLabel ? true : undefined,
      threshold,
      model,
    });
    payload.inputs = inputs;
    return this.submitJob('classify', payload, { webhook, metadata });
  }

  async job(id) {
    return parseJob(await this._request('GET', `/v1/jobs/${enc(id)}`));
  }

  /** One page of jobs, newest first. */
  async jobs({ status, kind, limit = 50, offset = 0 } = {}) {
    const q = new URLSearchParams(clean({ status, kind, limit, offset: offset || undefined }));
    return ((await this._request('GET', `/v1/jobs?${q}`)).jobs || []).map(parseJob);
  }

  async cancelJob(id) {
    return parseJob(await this._request('POST', `/v1/jobs/${enc(id)}/cancel`));
  }

  deleteJob(id) {
    return this._request('DELETE', `/v1/jobs/${enc(id)}`);
  }

  /** Poll until succeeded / failed / cancelled (check `job.ok`). Rejects with an Error after `timeoutMs`. */
  async waitJob(id, { timeoutMs, pollMs = 1000, onProgress } = {}) {
    const deadline = timeoutMs == null ? null : Date.now() + timeoutMs;
    for (;;) {
      const job = await this.job(id);
      if (onProgress) onProgress(job);
      if (job.finished) return job;
      if (deadline !== null && Date.now() + pollMs > deadline) {
        throw new Error(`job ${id} still ${job.status} after ${timeoutMs} ms (${job.done}/${job.total})`);
      }
      await this._sleep(pollMs);
    }
  }

  /** Async iterator over every result row currently stored, in input order, page by page. */
  async *jobResults(id, { pageSize = 500, offset = 0 } = {}) {
    for (;;) {
      const q = new URLSearchParams({ offset, limit: pageSize });
      const page = await this._request('GET', `/v1/jobs/${enc(id)}/results?${q}`);
      for (const row of page.items || []) yield parseJobItem(row, page.kind);
      if (page.next_offset == null || !(page.items || []).length) return;
      offset = page.next_offset;
    }
  }

  // --- saved classifiers ---
  classifier(name) {
    return new Classifier(this, name);
  }
  async listClassifiers() {
    return (await this._request('GET', '/v1/classifiers')).classifiers || [];
  }
  saveClassifier(name, { labels, levels, kind, instructions, multiLabel, threshold, description } = {}) {
    kind = kind || (levels && !labels ? 'score' : 'classify');
    const body = clean({ kind, labels, levels, instructions, multi_label: multiLabel, threshold, description });
    return this._request('PUT', `/v1/classifiers/${enc(name)}`, body);
  }
  getClassifier(name) {
    return this._request('GET', `/v1/classifiers/${enc(name)}`);
  }
  deleteClassifier(name) {
    return this._request('DELETE', `/v1/classifiers/${enc(name)}`);
  }
}

export class Classifier {
  constructor(client, name) {
    this.client = client;
    this.name = name;
    this._path = `/v1/classifiers/${enc(name)}`;
  }
  async classify(input, { images, videos, threshold } = {}) {
    const body = clean({ input, images, videos, threshold });
    return parseResult(await this.client._request('POST', this._path, body));
  }
  async classifyMany(inputs, { chunkSize } = {}) {
    const out = [];
    for (const part of chunks(inputs, chunkSize)) {
      const body = await this.client._request('POST', `${this._path}/batch`, { inputs: part });
      out.push(...parseBatch(body, this.name));
    }
    return out;
  }
  save(def = {}) {
    return this.client.saveClassifier(this.name, def);
  }
  get() {
    return this.client.getClassifier(this.name);
  }
  delete() {
    return this.client.deleteClassifier(this.name);
  }
}

// Media helpers (browser + Node 18+): bytes -> data URL.
export function bytesToDataUrl(bytes, mime = 'image/png') {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let bin = '';
  for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode(...u8.subarray(i, i + 0x8000));
  return `data:${mime};base64,${btoa(bin)}`;
}

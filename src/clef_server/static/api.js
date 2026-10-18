// Thin fetch wrapper: injects X-API-Key, parses the {detail, request_id} error shape, measures latency.
const KEY = 'clef.apiKey';

export function getKey() {
  try { return localStorage.getItem(KEY) || ''; } catch { return ''; }
}
export function setKey(v) {
  try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch { /* ignore */ }
  window.dispatchEvent(new CustomEvent('clef:key')); // the shell re-reads /v1/whoami (tabs depend on the scopes)
}

export class ApiError extends Error {
  constructor(message, { status = 0, requestId = null, body = null, retryAfter = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.requestId = requestId;
    this.body = body;
    this.retryAfter = retryAfter;
  }
}

function detailText(d) {
  if (d == null) return '';
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) {
    return d.map((x) => (x && x.msg ? `${(x.loc || []).join('.')}: ${x.msg}` : JSON.stringify(x))).join('; ');
  }
  return JSON.stringify(d);
}

/**
 * request(path, {method, body, signal, okStatuses}) -> {data, ms, status, requestId}
 * Throws ApiError for non-2xx (unless status is in okStatuses) and for network failures.
 */
export async function request(path, { method = 'GET', body, signal, okStatuses = [] } = {}) {
  const headers = { Accept: 'application/json' };
  const key = getKey();
  if (key) headers['X-API-Key'] = key;
  let payload;
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json';
    payload = typeof body === 'string' ? body : JSON.stringify(body);
  }
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(path, { method, headers, body: payload, signal, cache: 'no-store' });
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    throw new ApiError(`Network error: ${e && e.message ? e.message : e}`, { status: 0 });
  }
  const text = await res.text();
  const ms = performance.now() - t0;
  const requestId = res.headers.get('X-Request-ID');
  let data = null;
  if (text) { try { data = JSON.parse(text); } catch { data = null; } }
  if (!res.ok && !okStatuses.includes(res.status)) {
    if (res.status === 401) window.dispatchEvent(new CustomEvent('clef:unauthorized'));
    const detail = data && data.detail !== undefined ? detailText(data.detail) : (text ? text.slice(0, 300) : res.statusText);
    const ra = res.headers.get('Retry-After');
    const retryAfter = ra != null && ra !== '' && !Number.isNaN(Number(ra)) ? Number(ra) : null;
    throw new ApiError(detail || `HTTP ${res.status}`, {
      status: res.status, requestId: (data && data.request_id) || requestId, body: data, retryAfter,
    });
  }
  return { data, ms, status: res.status, requestId };
}

/**
 * Server-sent events over fetch(), so the key travels in the X-API-Key header and never in the URL
 * (EventSource cannot set headers). streamEvents(path, {onOpen, onEvent(name, data), signal}) resolves when
 * the stream ends, rejects with ApiError (status 0 = network) on a failed connect or a lost stream.
 */
export async function streamEvents(path, { onOpen, onEvent, signal } = {}) {
  const headers = { Accept: 'text/event-stream' };
  const key = getKey();
  if (key) headers['X-API-Key'] = key;
  let res;
  try {
    res = await fetch(path, { headers, signal, cache: 'no-store' });
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    throw new ApiError(`Network error: ${e && e.message ? e.message : e}`, { status: 0 });
  }
  if (!res.ok || !res.body) {
    if (res.status === 401) window.dispatchEvent(new CustomEvent('clef:unauthorized'));
    throw new ApiError(`HTTP ${res.status}`, { status: res.status || 0 });
  }
  if (onOpen) onOpen();
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  const dispatch = (block) => {
    let event = 'message';
    const data = [];
    for (const line of block.split('\n')) {
      if (!line || line[0] === ':') continue;
      const i = line.indexOf(':');
      const field = i < 0 ? line : line.slice(0, i);
      let value = i < 0 ? '' : line.slice(i + 1);
      if (value[0] === ' ') value = value.slice(1);
      if (field === 'event') event = value;
      else if (field === 'data') data.push(value);
    }
    if (data.length && onEvent) onEvent(event, data.join('\n'));
  };
  // A chunk may end between the \r and \n of a CRLF: hold a trailing \r back until the next chunk (or the end of
  // the stream) so it is not read as a lone CR line end, which would add a false blank line.
  let carry = '';
  const feed = (text, last) => {
    let raw = carry + text;
    carry = '';
    if (!last && raw.endsWith('\r')) {
      carry = '\r';
      raw = raw.slice(0, -1);
    }
    buf += raw.replace(/\r\n?/g, '\n');
    let cut;
    while ((cut = buf.indexOf('\n\n')) >= 0) {
      dispatch(buf.slice(0, cut));
      buf = buf.slice(cut + 2);
    }
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      feed(dec.decode(value, { stream: true }), false);
    }
    feed(dec.decode(), true);
  } catch (e) {
    if (e && e.name === 'AbortError') throw e;
    throw new ApiError(`Stream lost: ${e && e.message ? e.message : e}`, { status: 0 });
  }
}

export const health = () => request('/health', { okStatuses: [503] });
export const stats = (windowS) => request(`/v1/stats${windowS ? `?window_s=${windowS}` : ''}`);
export const timeseries = (windowS, stepS, signal) => request(`/v1/stats/timeseries?window_s=${windowS}${stepS ? `&step_s=${stepS}` : ''}`, { signal });
export const log = (since) => request(`/v1/log?limit=100${since != null ? `&since=${since}` : ''}`);
export const schemaExample = () => request('/schema-example');
export const systemone = (body, signal) => request('/v1/systemone', { method: 'POST', body, signal });
export const batch = (records, signal) => request('/v1/batch', { method: 'POST', body: { batch: records }, signal });

// ---- classification API -------------------------------------------------------------------------
export const classify = (body, signal) => request('/v1/classify', { method: 'POST', body, signal });
export const listClassifiers = () => request('/v1/classifiers');
export const putClassifier = (name, body) => request(`/v1/classifiers/${encodeURIComponent(name)}`, { method: 'PUT', body });
export const deleteClassifier = (name) => request(`/v1/classifiers/${encodeURIComponent(name)}`, { method: 'DELETE' });
export const classifyBatch = (body, signal) => request('/v1/classify/batch', { method: 'POST', body, signal });
export const classifierBatch = (name, body, signal) => request(`/v1/classifiers/${encodeURIComponent(name)}/batch`, { method: 'POST', body, signal });
export const evaluateMetrics = (body, signal) => request('/v1/evaluate/metrics', { method: 'POST', body, signal });
export const NAME_RE =/^[a-z0-9][a-z0-9_-]{0,63}$/;

// ---- keys and principals (D1) ---------------------------------------------------------------------
export const whoami = () => request('/v1/whoami');
export const listKeys = (disabled) => request(`/v1/admin/keys${disabled ? '?include_disabled=true' : ''}`);
export const createKey = (body) => request('/v1/admin/keys', { method: 'POST', body });
export const updateKey = (name, body, force) => request(`/v1/admin/keys/${encodeURIComponent(name)}${force ? '?force=true' : ''}`, { method: 'PATCH', body });
export const rotateKey = (name, body) => request(`/v1/admin/keys/${encodeURIComponent(name)}/rotate`, { method: 'POST', body });
export const revokeKey = (name, keyId, force) => request(`/v1/admin/keys/${encodeURIComponent(name)}/revoke${force ? '?force=true' : ''}`, { method: 'POST', body: keyId ? { key_id: keyId } : {} });
export const keyUsage = (name, days = 30) => request(`/v1/admin/keys/${encodeURIComponent(name)}/usage?days=${days}`);
export const audit = (before) => request(`/v1/admin/audit?limit=50${before != null ? `&before=${before}` : ''}`);

// ---- router and review queue (E2) -------------------------------------------------------------------
const enc = encodeURIComponent;
export const listRouters = () => request('/v1/routers');
export const putRouter = (name, body) => request(`/v1/routers/${enc(name)}`, { method: 'PUT', body });
export const deleteRouter = (name) => request(`/v1/routers/${enc(name)}`, { method: 'DELETE' });
export const proposeRouter = (name) => request(`/v1/routers/${enc(name)}/propose`, { method: 'POST', okStatuses: [202] });
export const routerVersions = (name) => request(`/v1/routers/${enc(name)}/versions`);
export const activateRouterVersion = (name, v) => request(`/v1/routers/${enc(name)}/versions/${v}/activate`, { method: 'POST' });
export const listReview = (params = {}) => {
  const qs = Object.entries(params).filter(([, v]) => v !== '' && v != null).map(([k, v]) => `${enc(k)}=${enc(v)}`).join('&');
  return request(`/v1/review${qs ? `?${qs}` : ''}`);
};
export const resolveReview = (id, body) => request(`/v1/review/${enc(id)}/resolve`, { method: 'POST', body });

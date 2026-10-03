// Thin fetch wrapper: injects X-API-Key, parses the {detail, request_id} error shape, measures latency.
const KEY = 'clef.apiKey';

export function getKey() {
  try { return localStorage.getItem(KEY) || ''; } catch { return ''; }
}
export function setKey(v) {
  try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch { /* ignore */ }
}

export class ApiError extends Error {
  constructor(message, { status = 0, requestId = null, body = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.requestId = requestId;
    this.body = body;
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
    throw new ApiError(detail || `HTTP ${res.status}`, {
      status: res.status, requestId: (data && data.request_id) || requestId, body: data,
    });
  }
  return { data, ms, status: res.status, requestId };
}

export const health = () => request('/health', { okStatuses: [503] });
export const stats = () => request('/v1/stats');
export const log = (since) => request(`/v1/log?limit=100${since != null ? `&since=${since}` : ''}`);
export const schemaExample = () => request('/schema-example');
export const systemone = (body, signal) => request('/v1/systemone', { method: 'POST', body, signal });
export const batch = (records, signal) => request('/v1/batch', { method: 'POST', body: { batch: records }, signal });

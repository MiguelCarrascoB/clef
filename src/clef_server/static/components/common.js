// Shared presentational components.
import { html, useState, useRef, useEffect } from '../vendor/standalone.module.js';
import { copyText, confidenceOf, confLevel, fmtPct } from '../util.js';

export function CopyButton({ text, label = 'Copy', getText }) {
  const [ok, setOk] = useState(false);
  const t = useRef(null);
  const click = async () => {
    const done = await copyText(getText ? getText() : text);
    setOk(done);
    clearTimeout(t.current);
    t.current = setTimeout(() => setOk(false), 1400);
  };
  return html`<button class="btn sm" type="button" onClick=${click}>${ok ? 'Copied' : label}</button>`;
}

export function ConfBadge({ answer, conf }) {
  const c = conf !== undefined ? conf : confidenceOf(answer);
  const lvl = confLevel(c);
  const text = c == null ? 'n/a' : lvl === 'low' ? `uncertain ${fmtPct(c, 0)}` : `${fmtPct(c, 0)} conf`;
  return html`<span class=${`badge conf-${lvl}`} title="Confidence">${text}</span>`;
}

export function Tabs({ tabs, value, onChange, label }) {
  const onKey = (e) => {
    const i = tabs.findIndex((t) => t.id === value);
    if (e.key === 'ArrowRight') { onChange(tabs[(i + 1) % tabs.length].id); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { onChange(tabs[(i - 1 + tabs.length) % tabs.length].id); e.preventDefault(); }
  };
  return html`<div class="subtabs" role="tablist" aria-label=${label} onKeyDown=${onKey}>
    ${tabs.map((t) => html`<button key=${t.id} role="tab" type="button" aria-selected=${value === t.id}
      tabIndex=${value === t.id ? 0 : -1} class=${`subtab ${value === t.id ? 'active' : ''}`}
      onClick=${() => onChange(t.id)}>${t.label}${t.badge ? html`<span class="count">${t.badge}</span>` : null}</button>`)}
  </div>`;
}

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
export function highlightJson(text) {
  return esc(text).replace(
    /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g,
    (m, str, colon, kw, num) => {
      if (str) return colon ? `<span class="j-key">${str}</span>${colon}` : `<span class="j-str">${str}</span>`;
      if (kw) return `<span class="j-kw">${kw}</span>`;
      if (num) return `<span class="j-num">${num}</span>`;
      return m;
    },
  );
}

export function CodeBlock({ text, json = false, copy = true, label }) {
  return html`<div class="codeblock">
    ${copy ? html`<div class="codeblock-tools"><${CopyButton} text=${text} label=${label || 'Copy'} /></div>` : null}
    ${json
      ? html`<pre class="mono" dangerouslySetInnerHTML=${{ __html: highlightJson(text) }}></pre>`
      : html`<pre class="mono">${text}</pre>`}
  </div>`;
}

export function ErrorBox({ error, title = 'Request failed' }) {
  if (!error) return null;
  if (error.status === 429) {
    return html`<div class="errorbox ratelimit" role="alert">
      <strong>Rate limited (HTTP 429)</strong>
      <div class="err-detail">${error.message}</div>
      <div class="small">${error.retryAfter != null ? html`Retry in <span class="mono">${error.retryAfter}s</span>.` : 'Retry shortly.'}</div>
      ${error.requestId ? html`<div class="muted small">request_id <span class="mono">${error.requestId}</span></div>` : null}
    </div>`;
  }
  return html`<div class="errorbox" role="alert">
    <strong>${title}${error.status ? ` (HTTP ${error.status})` : ''}</strong>
    <div class="mono err-detail">${error.message || String(error)}</div>
    ${error.requestId ? html`<div class="muted small">request_id <span class="mono">${error.requestId}</span></div>` : null}
  </div>`;
}

export function Meter({ value, max = 1, tone = '', label }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return html`<div class=${`meter ${tone}`} role="meter" aria-valuenow=${value} aria-valuemin="0" aria-valuemax=${max} aria-label=${label}>
    <div class="meter-fill" style=${{ width: `${pct}%` }}></div></div>`;
}

export function VramBar({ gpu, compact }) {
  if (!gpu) return null;
  const total = gpu.vram_total_gb, alloc = gpu.vram_allocated_gb, res = gpu.vram_reserved_gb;
  if (!total) return null;
  const ap = Math.min(100, (alloc / total) * 100), rp = Math.min(100, ((res ?? alloc) / total) * 100);
  return html`<div class=${`vram ${compact ? 'compact' : ''}`} title=${`VRAM allocated ${alloc?.toFixed(1)} / reserved ${res?.toFixed(1)} / total ${total.toFixed(1)} GB`}>
    <div class="vram-track"><div class="vram-res" style=${{ width: `${rp}%` }}></div><div class="vram-alloc" style=${{ width: `${ap}%` }}></div></div>
    <span class="mono vram-text">${alloc?.toFixed(1)}/${total.toFixed(1)} GB</span>
  </div>`;
}

export function Tile({ label, value, sub, tone, children }) {
  return html`<div class=${`tile ${tone || ''}`}>
    <div class="tile-label">${label}</div>
    <div class="tile-value mono">${value ?? '-'}</div>
    ${sub ? html`<div class="tile-sub muted small">${sub}</div>` : null}
    ${children}
  </div>`;
}

export function Empty({ children }) { return html`<div class="empty" role="status">${children}</div>`; }

/** Closes on outside click / Escape. */
export function usePopover(onClose, ref) {
  useEffect(() => {
    const down = (e) => { if (ref.current && !ref.current.contains(e.target)) onClose(); };
    const key = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('mousedown', down);
    document.addEventListener('keydown', key);
    return () => { document.removeEventListener('mousedown', down); document.removeEventListener('keydown', key); };
  }, [onClose, ref]);
}

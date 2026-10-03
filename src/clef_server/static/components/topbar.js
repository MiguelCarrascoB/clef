// HealthPill (polls /health), SettingsPopover (API key, theme).
import { html, useState, useEffect, useRef, useCallback } from '../vendor/standalone.module.js';
import * as api from '../api.js';
import { S, set, setTheme, useStore } from '../store.js';
import { VramBar, usePopover } from './common.js';

export function useHealthPolling() {
  useEffect(() => {
    let stop = false, timer = null;
    const tick = async () => {
      try {
        const { data } = await api.health();
        if (stop) return;
        const status = data && data.status ? data.status : 'error';
        set({ health: { state: status, data, error: data && data.error ? data.error : null }, limits: (data && data.limits) || S.limits });
      } catch (e) {
        if (stop) return;
        set({ health: { state: 'error', data: S.health.data, error: e.message } });
      }
      timer = setTimeout(tick, 5000);
    };
    tick();
    return () => { stop = true; clearTimeout(timer); };
  }, []);
}

export function HealthPill() {
  const { health } = useStore();
  const d = health.data;
  const label = { loading: 'Loading model', warming: 'Warming up', ready: 'Ready', error: 'Offline' }[health.state] || health.state;
  const tip = health.error ? `Error: ${health.error}` : d ? `${d.model_path || ''}\n${d.dtype || ''} ${d.device || ''}  v${d.version || '?'}` : '';
  return html`<div class="health" title=${tip}>
    <span class=${`dot dot-${health.state}`}></span>
    <span class="health-label">${label}</span>
    ${d && d.backend ? html`<span class="mono muted small" title=${`backend ${d.backend}`}>${d.backend}</span>` : null}
    ${d && d.gpu && d.gpu.available ? html`<span class="health-gpu muted" title=${d.gpu.name}>${shortGpu(d.gpu.name)}</span><${VramBar} gpu=${d.gpu} compact />` : null}
  </div>`;
}
const shortGpu = (n = '') => n.replace(/^AMD |^NVIDIA |Radeon |GeForce /g, '');

export function Settings({ open, onOpen, onClose }) {
  const { theme } = useStore();
  const [val, setVal] = useState(api.getKey());
  const ref = useRef(null);
  const close = useCallback(() => onClose(), [onClose]);
  usePopover(close, ref);
  useEffect(() => { if (open) setVal(api.getKey()); }, [open]);
  const save = () => { api.setKey(val.trim()); onClose(); };
  const hasKey = !!api.getKey();
  return html`<div class="settings" ref=${ref}>
    <button class="btn icon" type="button" aria-haspopup="dialog" aria-expanded=${open} aria-label="Settings" title="Settings"
      onClick=${() => (open ? onClose() : onOpen())}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg>
      ${hasKey ? html`<span class="keydot" aria-hidden="true"></span>` : null}
    </button>
    ${open ? html`<div class="popover" role="dialog" aria-label="Settings">
      <label class="field"><span>API key <span class="muted">(optional, sent as X-API-Key)</span></span>
        <input type="password" class="input mono" value=${val} autocomplete="off" placeholder="not set"
          onInput=${(e) => setVal(e.target.value)} onKeyDown=${(e) => e.key === 'Enter' && save()} /></label>
      ${S.health.state === 'error' || window.__unauth ? html`<div class="muted small">If the server reports 401, enter the key configured in CLEF_API_KEY.</div>` : null}
      <div class="row gap"><button class="btn primary sm" type="button" onClick=${save}>Save</button>
        <button class="btn sm" type="button" onClick=${() => { setVal(''); api.setKey(''); onClose(); }}>Clear</button></div>
      <hr />
      <div class="field"><span>Theme</span>
        <div class="seg" role="radiogroup" aria-label="Theme">
          ${['auto', 'dark', 'light'].map((t) => html`<button key=${t} type="button" role="radio" aria-checked=${theme === t}
            class=${`seg-btn ${theme === t ? 'active' : ''}`} onClick=${() => setTheme(t)}>${t}</button>`)}
        </div></div>
    </div>` : null}
  </div>`;
}

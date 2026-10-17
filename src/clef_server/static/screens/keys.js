// Keys (admins only): principals with scopes, quota and keys; create / edit / rotate / revoke; the audit log.
// A new secret is shown once in a panel and lives only in this component's state: never in the store,
// the history or localStorage.
import { html, useState, useEffect, useCallback, useRef } from '../vendor/standalone.module.js';
import * as api from '../api.js';
import { fmtDateTime, fmtNum } from '../util.js';
import { Empty, CopyButton, ErrorBox } from '../components/common.js';
import { Sparkline } from '../components/charts.js';

const SCOPES = [
  ['classify', 'run the model: classify, score, systemone, OpenAI / HF routes, evaluate'],
  ['jobs', 'async jobs'],
  ['classifiers', 'create, change and delete saved classifiers'],
  ['ops', 'stats, request log, live events'],
  ['admin', 'everything, including this screen'],
  ['urgent', 'submit urgent jobs'],
];
const GRACES = [['1 hour', 3600], ['24 hours', 86400], ['7 days', 604800], ['none (old keys stop now)', 0]];
const NAME_RE = /^[A-Za-z0-9_.-]{1,64}$/;
const KEY_TONE = { active: 'ok', expiring: 'warn', expired: 'bad', revoked: 'bad' };

const when = (ts) => (ts ? fmtDateTime(ts * 1000) : '-');
const intOrNull = (v) => (String(v).trim() === '' ? null : Number.parseInt(v, 10));

function ScopeChips({ scopes }) {
  return html`<span class="chips">${scopes.map((s) => html`<span key=${s} class=${`chip ${s === 'admin' ? 'scope-admin' : ''}`}>${s}</span>`)}</span>`;
}

function QuotaBar({ used, quota }) {
  if (quota == null) return html`<span class="muted small mono">${fmtNum(used)} today · unlimited</span>`;
  const pct = Math.min(100, (used / quota) * 100);
  const tone = pct >= 100 ? 'bad' : pct >= 80 ? 'warn' : '';
  return html`<div class="qbar-wrap"><div class=${`qbar ${tone}`} role="meter" aria-label="Daily quota used" aria-valuenow=${used} aria-valuemin="0" aria-valuemax=${quota}>
    <div class="qbar-fill" style=${{ width: `${pct}%` }}></div></div>
    <span class="muted small mono">${fmtNum(used)} / ${fmtNum(quota)} today</span></div>`;
}

function SecretOnce({ secret, onDismiss }) {
  return html`<div class="card secret-card" role="alert">
    <div class="card-head"><h3>${secret.verb} <span class="mono">${secret.name}</span></h3></div>
    <p class="small">Copy the key now. Only its hash is stored, so it cannot be shown again.</p>
    <div class="row gap"><code class="mono secret-key" data-testid="secret">${secret.key}</code><${CopyButton} text=${secret.key} label="Copy key" /></div>
    ${secret.note ? html`<p class="muted small">${secret.note}</p>` : null}
    <div class="row gap"><button class="btn primary sm" type="button" onClick=${onDismiss}>I have stored it</button></div>
  </div>`;
}

function Field({ label, hint, children }) {
  return html`<label class="field"><span>${label}${hint ? html` <span class="muted">${hint}</span>` : null}</span>${children}</label>`;
}

function ScopePicker({ value, onChange }) {
  const toggle = (s) => onChange(value.includes(s) ? value.filter((x) => x !== s) : [...value, s]);
  return html`<div class="scope-pick" role="group" aria-label="Scopes">${SCOPES.map(([s, d]) => html`<label key=${s} class="check" title=${d}>
    <input type="checkbox" checked=${value.includes(s)} onChange=${() => toggle(s)} /> <span class="mono">${s}</span> <span class="muted small">${d}</span></label>`)}</div>`;
}

function CreatePanel({ onDone, onCancel }) {
  const [f, setF] = useState({ name: '', scopes: ['classify'], quota: '', rate: '', days: '', note: '' });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const set = (p) => setF((x) => ({ ...x, ...p }));
  const bad = !NAME_RE.test(f.name) || !f.scopes.length;
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setErr(null);
    try {
      const body = { name: f.name, scopes: f.scopes, note: f.note || null };
      const q = intOrNull(f.quota), r = intOrNull(f.rate), d = Number.parseFloat(f.days);
      if (q != null) body.daily_quota = q;
      if (r != null) body.rate_limit = r;
      if (d > 0) body.expires_in_days = d;
      const { data } = await api.createKey(body);
      onDone({ name: data.name, key: data.key, verb: 'New key for' });
    } catch (ex) { setErr(ex); } finally { setBusy(false); }
  };
  return html`<form class="card" onSubmit=${submit} aria-label="New key">
    <div class="card-head"><h3>New key</h3></div>
    <div class="grid2">
      <${Field} label="Name" hint="letters, digits, _ . - (never reused)">
        <input class=${`input mono ${f.name && !NAME_RE.test(f.name) ? 'invalid' : ''}`} value=${f.name} autocomplete="off" required onInput=${(e) => set({ name: e.target.value })} /><//>
      <${Field} label="Note" hint="(optional)"><input class="input" value=${f.note} maxlength="500" onInput=${(e) => set({ note: e.target.value })} /><//>
    </div>
    <${Field} label="Scopes"><${ScopePicker} value=${f.scopes} onChange=${(scopes) => set({ scopes })} /><//>
    <div class="grid3">
      <${Field} label="Daily quota" hint="records per UTC day, blank = unlimited"><input class="input mono" type="number" min="1" value=${f.quota} onInput=${(e) => set({ quota: e.target.value })} /><//>
      <${Field} label="Rate limit" hint="requests / min, blank = server default"><input class="input mono" type="number" min="1" value=${f.rate} onInput=${(e) => set({ rate: e.target.value })} /><//>
      <${Field} label="Expires in" hint="days, blank = never"><input class="input mono" type="number" min="1" value=${f.days} onInput=${(e) => set({ days: e.target.value })} /><//>
    </div>
    <${ErrorBox} error=${err} title="Could not create the key" />
    <div class="row gap"><button class="btn primary sm" type="submit" disabled=${bad || busy}>${busy ? 'Creating…' : 'Create key'}</button>
      <button class="btn sm" type="button" onClick=${onCancel}>Cancel</button></div>
  </form>`;
}

function EditPanel({ p, onDone, onCancel }) {
  const [f, setF] = useState({ scopes: p.scopes, quota: p.daily_quota ?? '', rate: p.rate_limit ?? '', note: p.note || '' });
  const [force, setForce] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const set = (x) => setF((v) => ({ ...v, ...x }));
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setErr(null);
    try {
      const body = {};
      if ([...f.scopes].sort().join() !== [...p.scopes].sort().join()) body.scopes = f.scopes;
      const q = intOrNull(f.quota), r = intOrNull(f.rate);
      if (q !== p.daily_quota) body.daily_quota = q; // an explicit null clears the quota
      if (r !== p.rate_limit) body.rate_limit = r;
      if ((f.note || null) !== (p.note || null)) body.note = f.note || null;
      if (Object.keys(body).length) await api.updateKey(p.name, body, force);
      onDone();
    } catch (ex) { setErr(ex); } finally { setBusy(false); }
  };
  return html`<form class="card" onSubmit=${submit} aria-label=${`Edit ${p.name}`}>
    <div class="card-head"><h3>Edit <span class="mono">${p.name}</span></h3></div>
    <${Field} label="Scopes"><${ScopePicker} value=${f.scopes} onChange=${(scopes) => set({ scopes })} /><//>
    <div class="grid3">
      <${Field} label="Daily quota" hint="blank = unlimited"><input class="input mono" type="number" min="1" value=${f.quota} onInput=${(e) => set({ quota: e.target.value })} /><//>
      <${Field} label="Rate limit" hint="blank = server default"><input class="input mono" type="number" min="1" value=${f.rate} onInput=${(e) => set({ rate: e.target.value })} /><//>
      <${Field} label="Note"><input class="input" value=${f.note} maxlength="500" onInput=${(e) => set({ note: e.target.value })} /><//>
    </div>
    <label class="check small"><input type="checkbox" checked=${force} onChange=${(e) => setForce(e.target.checked)} /> allow changes that remove my own admin scope</label>
    <${ErrorBox} error=${err} title="Could not save" />
    <div class="row gap"><button class="btn primary sm" type="submit" disabled=${busy || !f.scopes.length}>${busy ? 'Saving…' : 'Save'}</button>
      <button class="btn sm" type="button" onClick=${onCancel}>Cancel</button></div>
  </form>`;
}

function RotatePanel({ p, onDone, onCancel }) {
  const [grace, setGrace] = useState(86400);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setErr(null);
    try {
      const { data } = await api.rotateKey(p.name, { grace_s: Number(grace) });
      onDone({ name: data.name, key: data.key, verb: 'Rotated key for', note: Number(grace) ? 'The previous keys keep working until the grace period ends.' : 'The previous keys stopped working.' });
    } catch (ex) { setErr(ex); } finally { setBusy(false); }
  };
  return html`<form class="card" onSubmit=${submit} aria-label=${`Rotate ${p.name}`}>
    <div class="card-head"><h3>Rotate <span class="mono">${p.name}</span></h3></div>
    <${Field} label="Keep the old keys working for">
      <select class="input" value=${grace} onChange=${(e) => setGrace(e.target.value)}>${GRACES.map(([l, s]) => html`<option key=${s} value=${s}>${l}</option>`)}</select><//>
    <${ErrorBox} error=${err} title="Could not rotate" />
    <div class="row gap"><button class="btn primary sm" type="submit" disabled=${busy}>${busy ? 'Rotating…' : 'Rotate'}</button>
      <button class="btn sm" type="button" onClick=${onCancel}>Cancel</button></div>
  </form>`;
}

function RevokePanel({ p, keyId, onDone, onCancel }) {
  const [typed, setTyped] = useState('');
  const [force, setForce] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setErr(null);
    try { await api.revokeKey(p.name, keyId, force); onDone(); } catch (ex) { setErr(ex); } finally { setBusy(false); }
  };
  return html`<form class="card danger-card" onSubmit=${submit} aria-label=${`Revoke ${p.name}`}>
    <div class="card-head"><h3>${keyId ? html`Revoke key <span class="mono">${keyId}</span> of` : 'Disable'} <span class="mono">${p.name}</span></h3></div>
    <p class="small">${keyId ? 'This key stops working at once.' : 'All its keys stop working at once and the name can never be used again.'}</p>
    <${Field} label=${`Type ${p.name} to confirm`}><input class="input mono" value=${typed} autocomplete="off" onInput=${(e) => setTyped(e.target.value)} /><//>
    <label class="check small"><input type="checkbox" checked=${force} onChange=${(e) => setForce(e.target.checked)} /> allow this even if it locks me out</label>
    <${ErrorBox} error=${err} title="Could not revoke" />
    <div class="row gap"><button class="btn danger sm" type="submit" disabled=${typed !== p.name || busy}>${busy ? 'Revoking…' : keyId ? 'Revoke key' : 'Disable principal'}</button>
      <button class="btn sm" type="button" onClick=${onCancel}>Cancel</button></div>
  </form>`;
}

function KeyList({ p, onRevoke }) {
  if (p.source === 'env') return html`<span class="muted small">env key (change it in the environment)</span>`;
  if (!p.keys.length) return html`<span class="muted small">no keys</span>`;
  return html`<ul class="keylist">${p.keys.map((k) => html`<li key=${k.id}>
    <span class="mono small" title=${`created ${when(k.created_at)}`}>${k.display}</span>
    <span class=${`status-pill ${KEY_TONE[k.status] || ''}`}>${k.status}</span>
    <span class="muted small">${k.expires_at ? `${k.status === 'expired' ? 'expired' : 'expires'} ${when(k.expires_at)}` : ''}${k.last_used_at ? ` · used ${when(k.last_used_at)}` : ''}</span>
    ${k.status === 'active' || k.status === 'expiring' ? html`<button class="btn ghost sm" type="button" onClick=${() => onRevoke(p, k.id)} aria-label=${`Revoke key ${k.id}`}>Revoke</button>` : null}
  </li>`)}</ul>`;
}

function PrincipalRow({ p, usage, onAction }) {
  const env = p.source === 'env';
  const disabled = !!p.disabled_at;
  return html`<tr class=${disabled ? 'row-disabled' : ''}>
    <td><strong class="mono">${p.name}</strong>
      <div>${env ? html`<span class="chip">env</span>` : null}${disabled ? html` <span class="status-pill bad">disabled</span>` : null}</div>
      ${p.note && !env ? html`<div class="muted small">${p.note}</div>` : null}</td>
    <td><${ScopeChips} scopes=${env ? ['admin'] : p.scopes} /></td>
    <td>${env ? html`<span class="muted small">no quota</span>` : html`<${QuotaBar} used=${p.used_today} quota=${p.daily_quota} />`}
      ${p.rate_limit ? html`<div class="muted small mono">${p.rate_limit}/min</div>` : null}</td>
    <td class="spark-cell">${!env && usage ? html`<${Sparkline} values=${usage.map((d) => d.records)} label=${`${p.name} records per day`} />` : null}</td>
    <td><${KeyList} p=${p} onRevoke=${(pp, id) => onAction({ mode: 'revoke', p: pp, keyId: id })} /></td>
    <td class="actions">${env || disabled ? null : html`<div class="row gap">
      <button class="btn sm" type="button" onClick=${() => onAction({ mode: 'edit', p })}>Edit</button>
      <button class="btn sm" type="button" onClick=${() => onAction({ mode: 'rotate', p })}>Rotate</button>
      <button class="btn sm danger" type="button" onClick=${() => onAction({ mode: 'revoke', p })}>Disable</button></div>`}</td>
  </tr>`;
}

function AuditLog() {
  const [st, setSt] = useState({ entries: [], next: null, loaded: false, error: null, busy: false });
  const load = useCallback(async (before) => {
    setSt((s) => ({ ...s, busy: true, error: null }));
    try {
      const { data } = await api.audit(before);
      setSt((s) => ({ entries: before ? [...s.entries, ...data.entries] : data.entries, next: data.next_before, loaded: true, error: null, busy: false }));
    } catch (e) { setSt((s) => ({ ...s, error: e, busy: false, loaded: true })); }
  }, []);
  useEffect(() => { load(null); }, [load]);
  return html`<section class="card" aria-label="Audit log">
    <div class="card-head"><h3>Audit log <span class="sub">admin actions, newest first</span></h3>
      <div class="card-tools"><button class="btn sm" type="button" onClick=${() => load(null)} disabled=${st.busy}>Refresh</button></div></div>
    <${ErrorBox} error=${st.error} title="Could not load the audit log" />
    ${st.loaded && !st.entries.length && !st.error ? html`<${Empty}>No audit entries yet.<//>` : null}
    ${st.entries.length ? html`<div class="table-wrap"><table class="table"><thead><tr><th>When</th><th>Actor</th><th>Action</th><th>Target</th><th>Detail</th></tr></thead>
      <tbody>${st.entries.map((e) => html`<tr key=${e.id}><td class="nowrap small">${when(e.ts)}</td><td class="mono">${e.actor}</td><td class="mono">${e.action}</td><td class="mono">${e.target}</td>
        <td class="mono small audit-detail">${e.detail ? JSON.stringify(e.detail) : ''}</td></tr>`)}</tbody></table></div>` : null}
    ${st.next != null ? html`<div class="row gap"><button class="btn sm" type="button" disabled=${st.busy} onClick=${() => load(st.next)}>Load older</button></div>` : null}
  </section>`;
}

export function Keys() {
  const [st, setSt] = useState({ loading: true, error: null, principals: [], usage: {} });
  const [panel, setPanel] = useState(null); // {mode: create|edit|rotate|revoke, p?, keyId?}
  const [secret, setSecret] = useState(null);
  const [showDisabled, setShowDisabled] = useState(false);
  const [auditKey, setAuditKey] = useState(0);
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []); // the secret dies with the screen

  const load = useCallback(async (disabled = showDisabled) => {
    try {
      const { data } = await api.listKeys(disabled);
      const usage = {};
      await Promise.all(data.principals.filter((p) => p.source === 'db' && !p.disabled_at).map(async (p) => {
        try { usage[p.name] = (await api.keyUsage(p.name, 30)).data.days; } catch { /* the sparkline is optional */ }
      }));
      if (alive.current) setSt({ loading: false, error: null, principals: data.principals, usage });
    } catch (e) { if (alive.current) setSt((s) => ({ ...s, loading: false, error: e })); }
  }, [showDisabled]);
  useEffect(() => { load(showDisabled); }, [showDisabled]); // eslint-disable-line react-hooks/exhaustive-deps

  const done = (sec) => { setPanel(null); if (sec) setSecret(sec); setAuditKey((k) => k + 1); load(); };
  const act = (a) => { setPanel(a); setSecret(null); };
  return html`<div class="page keys-page">
    <div class="card-head"><h2>Keys</h2>
      <div class="card-tools">
        <label class="check small"><input type="checkbox" checked=${showDisabled} onChange=${(e) => setShowDisabled(e.target.checked)} /> show disabled</label>
        <button class="btn sm" type="button" onClick=${() => load()}>Refresh</button>
        <button class="btn primary sm" type="button" onClick=${() => act({ mode: 'create' })}>New key</button></div></div>
    ${secret ? html`<${SecretOnce} secret=${secret} onDismiss=${() => setSecret(null)} />` : null}
    ${panel && panel.mode === 'create' ? html`<${CreatePanel} onDone=${done} onCancel=${() => setPanel(null)} />` : null}
    ${panel && panel.mode === 'edit' ? html`<${EditPanel} p=${panel.p} onDone=${() => done()} onCancel=${() => setPanel(null)} />` : null}
    ${panel && panel.mode === 'rotate' ? html`<${RotatePanel} p=${panel.p} onDone=${done} onCancel=${() => setPanel(null)} />` : null}
    ${panel && panel.mode === 'revoke' ? html`<${RevokePanel} p=${panel.p} keyId=${panel.keyId} onDone=${() => done()} onCancel=${() => setPanel(null)} />` : null}
    <${ErrorBox} error=${st.error} title="Could not load keys" />
    <section class="card" aria-label="Principals">
      ${st.loading ? html`<${Empty}>Loading…<//>` : !st.principals.length && !st.error ? html`<${Empty}>No keys yet. Create one with New key.<//>` : html`<div class="table-wrap"><table class="table keys-table">
        <thead><tr><th>Principal</th><th>Scopes</th><th>Daily quota</th><th>30 days</th><th>Keys</th><th><span class="sr-only">Actions</span></th></tr></thead>
        <tbody>${st.principals.map((p) => html`<${PrincipalRow} key=${p.name} p=${p} usage=${st.usage[p.name]} onAction=${act} />`)}</tbody></table></div>`}
    </section>
    <${AuditLog} key=${auditKey} />
  </div>`;
}

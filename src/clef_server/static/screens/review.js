// Review: router proposals and low-confidence items in one queue (accept, reject, relabel), plus the routers
// themselves (label set, versions, mode, LLM status). Needs the `classifiers` scope; the server enforces it.
// Input text shows only while the server still stores it (until resolved, or the review TTL).
import { html, useState, useEffect, useCallback, useRef } from '../vendor/standalone.module.js';
import * as api from '../api.js';
import { fmtDateTime, fmtNum, fmtPct } from '../util.js';
import { Empty, ErrorBox, Meter } from '../components/common.js';

const TONE = { open: 'warn', buffered: '', resolved: 'ok', accepted: 'ok', rejected: 'bad', failed: 'bad', discarded: '', permanent_other: '', expired: '' };
const KINDS = [['', 'all'], ['proposal', 'proposals'], ['item', 'items']];
const STATES = [['open', 'open'], ['buffered', 'buffered'], ['resolved', 'resolved'], ['accepted', 'accepted'], ['rejected', 'rejected'], ['failed', 'failed (re-check)'], ['', 'any']];
const NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const PAGE = 50;

const when = (ts) => (ts ? fmtDateTime(ts * 1000) : '-');
const labelNames = (labels) => (Array.isArray(labels) ? labels : Object.keys(labels || {}));
const textOf = (v) => (v == null ? null : typeof v === 'string' ? v : JSON.stringify(v));

function Field({ label, hint, children }) {
  return html`<label class="field"><span>${label}${hint ? html` <span class="muted">${hint}</span>` : null}</span>${children}</label>`;
}

function Pill({ state }) { return html`<span class=${`status-pill ${TONE[state] || ''}`}>${state}</span>`; }

function ItemCard({ row, labels, onDone }) {
  const hint = row.prediction && row.prediction.llm_hint;
  const [label, setLabel] = useState(hint && labels.includes(hint) ? hint : '');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const pred = row.prediction || {};
  const live = row.state === 'open' || row.state === 'buffered';
  const act = async (body) => {
    setBusy(true); setErr(null);
    try { await api.resolveReview(row.id, body); onDone(); } catch (e) { setErr(e); setBusy(false); }
  };
  const text = textOf(row.input);
  return html`<article class="card review-card" aria-label=${`Review item ${row.id}`}>
    <div class="card-head"><h3><span class="chip">${row.reason}</span> <span class="mono small">${row.router || row.classifier || ''}</span></h3>
      <div class="card-tools"><${Pill} state=${row.state} /><span class="muted small">${when(row.created_at)}</span></div></div>
    ${text != null ? html`<pre class="mono review-input">${text}</pre>` : html`<p class="muted small">The text is no longer stored.</p>`}
    <p class="small">Predicted <strong class="mono">${pred.label ?? '-'}</strong> ${pred.confidence != null ? html`at <span class="mono">${fmtPct(pred.confidence)}</span>` : null}
      ${hint ? html` · LLM suggests <strong class="mono">${hint}</strong>` : null}</p>
    ${row.label && !live ? html`<p class="small">Labelled <strong class="mono">${row.label}</strong> by ${row.resolved_by || '-'}${pred.recheck ? (pred.recheck.agrees ? ' · clef agrees' : pred.recheck.label ? html` · clef says <span class="mono">${pred.recheck.label}</span>` : '') : ''}</p>` : null}
    <${ErrorBox} error=${err} title="Could not resolve" />
    ${live ? html`<div class="row gap">
      <select class="input" value=${label} aria-label="Label" onChange=${(e) => setLabel(e.target.value)} disabled=${busy || !labels.length}>
        <option value="">Choose a label…</option>${labels.map((l) => html`<option key=${l} value=${l}>${l}</option>`)}</select>
      <button class="btn primary sm" type="button" disabled=${!label || busy || text == null} onClick=${() => act({ action: 'label', label })}>Label</button>
      <button class="btn sm" type="button" disabled=${busy} onClick=${() => act({ action: 'discard' })}>Discard</button></div>` : null}
  </article>`;
}

function Acceptance({ a }) {
  if (!a) return null;
  if (a.landed != null) return html`<span class="muted small">${a.landed} items land on it at p ≥ ${fmtNum(a.tau, 2)}</span>`;
  return html`<div class="accept"><${Meter} value=${a.on_new} max=${Math.max(a.members, 1)} tone=${a.passed ? '' : 'bad'} label="Members that land on the new label" />
    <span class="muted small mono">${a.on_new}/${a.members} land on it (need ${a.need}, p ≥ ${fmtNum(a.tau, 2)})${a.human_against ? ` · ${a.human_against} human-labelled otherwise` : ''}${a.duplicate_of ? ` · looks like ${a.duplicate_of}` : ''}</span></div>`;
}

function ProposalCard({ row, onDone }) {
  const entries = (row.proposal && row.proposal.labels) || [];
  const [edit, setEdit] = useState(() => Object.fromEntries(entries.map((e) => [e.name, { on: true, description: e.description }])));
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const stats = (row.proposal && row.proposal.stats) || {};
  const live = row.state === 'open';
  const picked = entries.filter((e) => edit[e.name] && edit[e.name].on);
  const act = async (body) => {
    setBusy(true); setErr(null);
    try { await api.resolveReview(row.id, body); onDone(); } catch (e) { setErr(e); setBusy(false); }
  };
  const approve = () => {
    const changed = picked.length !== entries.length || picked.some((e) => edit[e.name].description !== e.description);
    act(changed ? { action: 'approve', labels: picked.map((e) => ({ name: e.name, description: edit[e.name].description })) } : { action: 'approve' });
  };
  return html`<article class="card review-card" aria-label=${`Proposal ${row.id}`}>
    <div class="card-head"><h3><span class="chip">${row.reason === 'bootstrap' ? 'starter set' : 'new label'}</span> <span class="mono small">${row.router}</span></h3>
      <div class="card-tools"><${Pill} state=${row.state} /><span class="muted small">${when(row.created_at)}</span></div></div>
    ${stats.waiting ? html`<p class="warn small">Waiting for you: ${stats.waiting}.</p>` : null}
    ${stats.error ? html`<p class="err small">${stats.error}</p>` : null}
    <ul class="proposal-labels">${entries.map((e) => html`<li key=${e.name}>
      <label class="check">${live ? html`<input type="checkbox" checked=${edit[e.name].on} onChange=${(ev) => setEdit((s) => ({ ...s, [e.name]: { ...s[e.name], on: ev.target.checked } }))} />` : null}
        <strong class="mono">${e.name}</strong></label>
      ${live ? html`<input class="input" aria-label=${`Description of ${e.name}`} value=${edit[e.name].description} maxlength="300"
          onInput=${(ev) => { const v = ev.target.value; setEdit((s) => ({ ...s, [e.name]: { ...s[e.name], description: v } })); }} />` : html`<span class="small"> ${e.description}</span>`}
      <div class="muted small">${e.member_ids.length} item${e.member_ids.length === 1 ? '' : 's'}${e.nearest_existing ? ` · nearest existing: ${e.nearest_existing}` : ''}${e.rationale ? ` · ${e.rationale}` : ''}</div>
      <${Acceptance} a=${e.acceptance} /></li>`)}</ul>
    ${row.state === 'accepted' ? html`<p class="small">Published: <span class="mono">${[].concat(row.label || []).join(', ')}</span> (${row.resolved_by})</p>` : null}
    <${ErrorBox} error=${err} title="Could not resolve" />
    ${live ? html`<div class="row gap">
      <button class="btn primary sm" type="button" disabled=${busy || !picked.length} onClick=${approve}>Approve ${picked.length === entries.length ? 'all' : `${picked.length} of ${entries.length}`}</button>
      <button class="btn sm danger" type="button" disabled=${busy} onClick=${() => act({ action: 'reject' })}>Reject</button></div>` : null}
  </article>`;
}

function NewRouter({ onDone, onCancel }) {
  const [f, setF] = useState({ name: '', seeds: '', mode: 'review', hint: '' });
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const set = (p) => setF((x) => ({ ...x, ...p }));
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setErr(null);
    try {
      const seeds = {};
      f.seeds.split('\n').map((l) => l.trim()).filter(Boolean).forEach((l) => { const i = l.indexOf(':'); const k = (i < 0 ? l : l.slice(0, i)).trim(); seeds[k] = i < 0 ? k : l.slice(i + 1).trim() || k; });
      const body = { mode: f.mode };
      if (Object.keys(seeds).length) body.seeds = seeds;
      if (f.hint.trim()) body.domain_hint = f.hint.trim();
      await api.putRouter(f.name, body);
      onDone();
    } catch (ex) { setErr(ex); setBusy(false); }
  };
  return html`<form class="card" onSubmit=${submit} aria-label="New router">
    <div class="card-head"><h3>New router</h3></div>
    <div class="grid2">
      <${Field} label="Name" hint="also the classifier it publishes to"><input class=${`input mono ${f.name && !NAME_RE.test(f.name) ? 'invalid' : ''}`} value=${f.name} required autocomplete="off" onInput=${(e) => set({ name: e.target.value })} /><//>
      <${Field} label="Mode"><select class="input" value=${f.mode} onChange=${(e) => set({ mode: e.target.value })}>
        <option value="review">review: proposals wait for you</option><option value="auto">auto: confirmed proposals publish at once</option></select><//>
    </div>
    <${Field} label="Starting labels" hint="one per line, name: description. Empty = start cold (20 items, then the LLM proposes a starter set)">
      <textarea class="input mono" rows="4" value=${f.seeds} onInput=${(e) => set({ seeds: e.target.value })}></textarea><//>
    <${Field} label="Domain" hint="(optional) one line the LLM sees"><input class="input" value=${f.hint} maxlength="200" onInput=${(e) => set({ hint: e.target.value })} /><//>
    <${ErrorBox} error=${err} title="Could not create the router" />
    <div class="row gap"><button class="btn primary sm" type="submit" disabled=${!NAME_RE.test(f.name) || busy}>${busy ? 'Creating…' : 'Create router'}</button>
      <button class="btn sm" type="button" onClick=${onCancel}>Cancel</button></div>
  </form>`;
}

function RouterCard({ rt, onChange }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [note, setNote] = useState(null);
  const [versions, setVersions] = useState(null);
  const run = async (fn, okNote) => {
    setBusy(true); setErr(null); setNote(null);
    try { const r = await fn(); if (okNote) setNote(okNote(r)); onChange(); } catch (e) { setErr(e); } finally { setBusy(false); }
  };
  const llm = rt.llm || {};
  const loadVersions = () => run(async () => { setVersions((await api.routerVersions(rt.name)).data.versions); });
  return html`<article class="card router-card" aria-label=${`Router ${rt.name}`}>
    <div class="card-head"><h3><span class="mono">${rt.name}</span> <span class="muted small">→ classifier ${rt.classifier}${rt.classifier_version ? ` v${rt.classifier_version}` : ''}</span></h3>
      <div class="card-tools">
        <select class="input sm" aria-label="Mode" value=${rt.mode} disabled=${busy} onChange=${(e) => run(() => api.putRouter(rt.name, { mode: e.target.value }))}>
          <option value="review">review</option><option value="auto">auto</option></select>
        <button class="btn sm" type="button" disabled=${busy || !llm.enabled} title=${llm.enabled ? 'Ask the LLM for new labels now' : 'Set GEMINI_API_KEY to enable'}
          onClick=${() => run(() => api.proposeRouter(rt.name), (r) => `${r.data.items} items sent: ${r.data.queued} proposal(s) waiting, ${r.data.activated} published`)}>Propose now</button></div></div>
    <p class="small">${rt.active_version == null ? 'Cold: no labels yet.' : html`Label set v${rt.active_version}`} · <span class="mono">${rt.buffered}</span> buffered · <span class="mono">${rt.open_reviews}</span> waiting for review</p>
    ${rt.active_version != null ? html`<span class="chips">${Object.keys(rt.labels).map((l) => html`<span key=${l} class="chip" title=${rt.labels[l]}>${l}</span>`)}</span>` : null}
    <p class="small ${llm.enabled ? '' : 'muted'}">${llm.enabled
      ? html`<span class="status-pill warn" title="Buffered inputs (first characters only) are sent to this host to propose labels">external LLM: ${llm.host}</span> ${llm.model} · ${fmtNum(llm.calls_today, 0)} calls, ${fmtNum(llm.input_tokens_today, 0)} in / ${fmtNum(llm.output_tokens_today, 0)} out tokens today${llm.errors_today ? ` · ${llm.errors_today} failed` : ''}`
      : 'LLM off: nothing leaves this machine. Set GEMINI_API_KEY to let the router propose labels.'}</p>
    ${note ? html`<p class="ok small" role="status">${note}</p>` : null}
    <${ErrorBox} error=${err} title="Router action failed" />
    <div class="row gap">
      <button class="btn ghost sm" type="button" disabled=${busy} onClick=${() => (versions ? setVersions(null) : loadVersions())}>${versions ? 'Hide versions' : 'Versions'}</button>
      <button class="btn ghost sm danger" type="button" disabled=${busy} onClick=${() => { if (window.confirm(`Delete router ${rt.name}? Its classifier stays; buffered texts are dropped.`)) run(() => api.deleteRouter(rt.name)); }}>Delete</button></div>
    ${versions ? html`<ul class="keylist">${versions.map((v) => html`<li key=${v.version}>
      <span class="mono small">v${v.version}</span> <span class="chip">${v.source}</span>
      <span class="muted small">${Object.keys(v.labels).length - 1} labels · ${when(v.created_at)}${v.created_by ? ` · ${v.created_by}` : ''}</span>
      ${v.version !== rt.active_version ? html`<button class="btn ghost sm" type="button" disabled=${busy} onClick=${() => run(() => api.activateRouterVersion(rt.name, v.version), () => `Rolled back to v${v.version}`).then(loadVersions)}>Roll back to this</button>` : html`<span class="status-pill ok">active</span>`}</li>`)}</ul>` : null}
  </article>`;
}

export function Review() {
  const [filters, setFilters] = useState({ kind: '', state: 'open', router: '', classifier: '' });
  const [q, setQ] = useState({ rows: [], total: 0, loading: true, error: null });
  const [routers, setRouters] = useState([]);
  const [classifiers, setClassifiers] = useState([]);
  const [creating, setCreating] = useState(false);
  const alive = useRef(true);
  useEffect(() => () => { alive.current = false; }, []);

  const loadQueue = useCallback(async (offset = 0) => {
    try {
      const { data } = await api.listReview({ ...filters, limit: PAGE, offset });
      if (alive.current) setQ((s) => ({ rows: offset ? [...s.rows, ...data.items] : data.items, total: data.total, loading: false, error: null }));
    } catch (e) { if (alive.current) setQ((s) => ({ ...s, loading: false, error: e })); }
  }, [filters]);
  const loadRefs = useCallback(async () => {
    try { const r = await api.listRouters(); if (alive.current) setRouters(r.data.routers); } catch { /* shown by the queue's own error */ }
    try { const c = await api.listClassifiers(); if (alive.current) setClassifiers(c.data.classifiers); } catch { /* optional */ }
  }, []);
  useEffect(() => { loadQueue(0); }, [loadQueue]);
  useEffect(() => { loadRefs(); }, [loadRefs]);

  const refresh = () => { loadQueue(0); loadRefs(); };
  const labelsFor = (row) => {
    if (row.router) return labelNames((routers.find((r) => r.name === row.router) || {}).labels);
    return labelNames((classifiers.find((c) => c.name === row.classifier) || {}).labels);
  };
  const set = (p) => setFilters((f) => ({ ...f, ...p }));

  return html`<div class="page review-page">
    <div class="card-head"><h2>Review</h2>
      <div class="card-tools"><button class="btn sm" type="button" onClick=${refresh}>Refresh</button>
        <button class="btn primary sm" type="button" onClick=${() => setCreating(true)}>New router</button></div></div>
    ${creating ? html`<${NewRouter} onDone=${() => { setCreating(false); refresh(); }} onCancel=${() => setCreating(false)} />` : null}
    <section aria-label="Routers" class="stack">
      ${routers.length ? routers.map((rt) => html`<${RouterCard} key=${rt.name} rt=${rt} onChange=${refresh} />`)
        : html`<${Empty}>No routers yet. A router grows a classifier's labels from what it cannot place; create one with New router.<//>`}
    </section>
    <section class="card" aria-label="Queue filters">
      <div class="row gap wrap">
        <${Field} label="Kind"><select class="input" value=${filters.kind} onChange=${(e) => set({ kind: e.target.value })}>${KINDS.map(([v, l]) => html`<option key=${v} value=${v}>${l}</option>`)}</select><//>
        <${Field} label="State"><select class="input" value=${filters.state} onChange=${(e) => set({ state: e.target.value })}>${STATES.map(([v, l]) => html`<option key=${v} value=${v}>${l}</option>`)}</select><//>
        <${Field} label="Router"><select class="input" value=${filters.router} onChange=${(e) => set({ router: e.target.value })}>
          <option value="">all</option>${routers.map((r) => html`<option key=${r.name} value=${r.name}>${r.name}</option>`)}</select><//>
        <${Field} label="Classifier"><select class="input" value=${filters.classifier} onChange=${(e) => set({ classifier: e.target.value })}>
          <option value="">all</option>${classifiers.map((c) => html`<option key=${c.name} value=${c.name}>${c.name}</option>`)}</select><//>
        <span class="muted small" role="status">${q.total} row${q.total === 1 ? '' : 's'}</span></div>
    </section>
    <${ErrorBox} error=${q.error} title="Could not load the review queue" />
    ${q.loading ? html`<${Empty}>Loading…<//>` : !q.rows.length && !q.error ? html`<${Empty}>Nothing here. Items land in this queue when a router cannot place them, or a classifier's result falls below its review_threshold.<//>` : null}
    <section class="stack" aria-label="Queue">${q.rows.map((row) => (row.kind === 'proposal'
      ? html`<${ProposalCard} key=${row.id} row=${row} onDone=${refresh} />`
      : html`<${ItemCard} key=${row.id} row=${row} labels=${labelsFor(row)} onDone=${refresh} />`))}</section>
    ${q.rows.length < q.total ? html`<div class="row gap"><button class="btn sm" type="button" onClick=${() => loadQueue(q.rows.length)}>Load more</button></div>` : null}
  </div>`;
}

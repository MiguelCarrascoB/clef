// Playground "Classify" mode: input, label chips, single/multi toggle, probability bars, save as classifier, snippets.
import { html, useState, useEffect } from '../vendor/standalone.module.js';
import * as api from '../api.js';
import { S, set, setCl, setPgMode, useStore, addHistory } from '../store.js';
import { fmtPct, entropy, classifierSnippets, truncateMedia } from '../util.js';
import { getKey } from '../api.js';
import { Tabs, CodeBlock, ErrorBox, Empty, ConfBadge } from '../components/common.js';
import { Metrics } from '../components/results.js';
import { ProbBars, Uncertainty } from '../components/charts.js';

let abortCtl = null;

export function ModeSwitch() {
  const { pgMode } = useStore();
  return html`<div class="seg" role="radiogroup" aria-label="Playground mode" style=${{ margin: 0 }}>
    ${[['systemone', 'SystemOne'], ['classify', 'Classify']].map(([id, l]) => html`<button key=${id} type="button" role="radio" aria-checked=${pgMode === id}
      class=${`seg-btn ${pgMode === id ? 'active' : ''}`} style=${{ textTransform: 'none' }} onClick=${() => setPgMode(id)}>${l}</button>`)}
  </div>`;
}

/** Labels in the API shape: a list when no descriptions, else {label: description-or-label}. */
export function labelsPayload(labels) {
  const named = labels.filter((l) => l.name.trim());
  if (named.some((l) => l.desc.trim())) return Object.fromEntries(named.map((l) => [l.name.trim(), l.desc.trim() || l.name.trim()]));
  return named.map((l) => l.name.trim());
}

export function validateCl(cl, limits) {
  const errs = [];
  const names = cl.labels.map((l) => l.name.trim());
  if (!cl.input.trim()) errs.push('Enter some input to classify.');
  if (names.some((n) => !n)) errs.push('Every label needs a name.');
  if (new Set(names).size !== names.length) errs.push('Label names must be unique.');
  const min = cl.multi ? 1 : 2;
  if (names.filter(Boolean).length < min) errs.push(cl.multi ? 'Add at least one label.' : 'Add at least two labels (or switch to multi-label).');
  const max = (limits && limits.max_labels) || 64;
  if (names.length > max) errs.push(`At most ${max} labels.`);
  return errs;
}

function buildBody(cl) {
  const body = { input: cl.input, labels: labelsPayload(cl.labels), multi_label: !!cl.multi };
  if (cl.multi) body.threshold = Number(cl.threshold);
  if (cl.instructions.trim()) body.instructions = cl.instructions.trim();
  return body;
}

export async function runClassify() {
  if (S.clRun.running) return;
  const errs = validateCl(S.cl, S.limits);
  if (errs.length) { set({ clRun: { ...S.clRun, error: new api.ApiError(errs[0]), result: null, running: false } }); return; }
  const request = buildBody(S.cl);
  const before = S.clRun.result;
  abortCtl = new AbortController();
  set({ clRun: { running: true, result: null, error: null, request, clientMs: null } });
  const t0 = performance.now();
  try {
    const { data } = await api.classify(request, abortCtl.signal);
    const clientMs = performance.now() - t0;
    set({ clRun: { running: false, result: data, error: null, request, clientMs }, clPrev: before || S.clPrev });
  } catch (e) {
    if (e && e.name === 'AbortError') set({ clRun: { ...S.clRun, running: false } });
    else set({ clRun: { running: false, result: null, error: e, request, clientMs: performance.now() - t0 } });
  } finally { abortCtl = null; }
}

// ---- label chips -----------------------------------------------------------------------------
function LabelEditor() {
  const { cl } = useStore();
  const [name, setName] = useState('');
  const add = () => {
    const n = name.trim();
    if (!n || cl.labels.some((l) => l.name === n)) return;
    setCl({ labels: [...cl.labels, { name: n, desc: '' }] });
    setName('');
  };
  const upd = (i, patch) => setCl({ labels: cl.labels.map((l, j) => (j === i ? { ...l, ...patch } : l)) });
  return html`<div>
    <div class="label-chips" role="list">
      ${cl.labels.map((l, i) => html`<div key=${i} role="listitem" class="lchip">
        <span class="nm" title=${l.name}>${l.name}</span>
        <input aria-label=${`Description of ${l.name}`} placeholder="description (optional)" value=${l.desc} onInput=${(e) => upd(i, { desc: e.target.value })} />
        <button class="x" type="button" aria-label=${`Remove label ${l.name}`} onClick=${() => setCl({ labels: cl.labels.filter((_, j) => j !== i) })}>✕</button>
      </div>`)}
    </div>
    <div class="row gap" style=${{ marginTop: '8px' }}>
      <input class="input grow mono" aria-label="New label" placeholder="Add a label and press Enter" value=${name} onInput=${(e) => setName(e.target.value)} onKeyDown=${(e) => { if (e.key === 'Enter') { e.preventDefault(); add(); } }} />
      <button class="btn sm" type="button" onClick=${add} disabled=${!name.trim() || cl.labels.some((l) => l.name === name.trim())}>Add</button>
    </div>
  </div>`;
}

// ---- saved classifiers -----------------------------------------------------------------------
async function refreshClassifiers() {
  try {
    const { data } = await api.listClassifiers();
    set({ classifiers: { loaded: true, list: (data && data.classifiers) || [], error: null } });
  } catch (e) { set({ classifiers: { loaded: true, list: [], error: e.status === 404 ? 'Saved classifiers are not available on this server version.' : e.message } }); }
}

function SavePanel() {
  const { cl, classifiers } = useStore();
  const [state, setState] = useState({ busy: false, error: null, ok: null });
  useEffect(() => { if (!S.classifiers.loaded) refreshClassifiers(); }, []);
  const name = cl.saveName;
  const nameOk = api.NAME_RE.test(name);
  const errs = validateCl({ ...cl, input: 'x' }, S.limits);
  const save = async () => {
    setState({ busy: true, error: null, ok: null });
    const body = { kind: 'classify', labels: labelsPayload(cl.labels), multi_label: !!cl.multi };
    if (cl.multi) body.threshold = Number(cl.threshold);
    if (cl.instructions.trim()) body.instructions = cl.instructions.trim();
    try {
      await api.putClassifier(name, body);
      setState({ busy: false, error: null, ok: `Saved "${name}".` });
      refreshClassifiers();
    } catch (e) { setState({ busy: false, error: e, ok: null }); }
  };
  const load = (c) => {
    if (c.kind && c.kind !== 'classify') { setState({ busy: false, error: new api.ApiError(`"${c.name}" is a score classifier; the console edits classify definitions only.`), ok: null }); return; }
    const labels = Array.isArray(c.labels) ? c.labels.map((n) => ({ name: n, desc: '' })) : Object.entries(c.labels || {}).map(([n, d]) => ({ name: n, desc: d === n ? '' : d }));
    setCl({ labels, multi: !!c.multi_label, threshold: c.threshold ?? 0.5, instructions: c.instructions || '', saveName: c.name });
    setState({ busy: false, error: null, ok: `Loaded "${c.name}".` });
  };
  const del = async (c) => {
    if (!confirm(`Delete classifier "${c.name}"?`)) return;
    try { await api.deleteClassifier(c.name); refreshClassifiers(); } catch (e) { setState({ busy: false, error: e, ok: null }); }
  };
  return html`<div>
    <div class="row gap">
      <input class=${`input grow mono ${name && !nameOk ? 'invalid' : ''}`} aria-label="Classifier name" placeholder="support-triage" value=${name} aria-invalid=${!!name && !nameOk}
        onInput=${(e) => setCl({ saveName: e.target.value })} onKeyDown=${(e) => e.key === 'Enter' && nameOk && !errs.length && save()} />
      <button class="btn primary sm" type="button" disabled=${!nameOk || errs.length > 0 || state.busy} onClick=${save}>${state.busy ? 'Saving…' : 'Save as classifier'}</button>
    </div>
    <div class=${`hint ${name && !nameOk ? 'err' : ''}`} style=${{ marginTop: '4px' }}>${name && !nameOk ? 'Use lowercase letters, digits, - and _ (start with a letter or digit, max 64).' : 'Saved definitions can be called with just the input: POST /v1/classifiers/<name>.'}</div>
    ${state.ok ? html`<div class="ok small" role="status" style=${{ marginTop: '6px' }}>${state.ok}</div>` : null}
    <${ErrorBox} error=${state.error} title="Could not save" />
    ${classifiers.error ? html`<div class="muted small" style=${{ marginTop: '8px' }}>${classifiers.error}</div>` : null}
    ${classifiers.list.length ? html`<hr /><div class="muted small" style=${{ marginBottom: '6px' }}>Saved classifiers</div>
      <div class="saved-list">${classifiers.list.map((c) => html`<span key=${c.name} class="lchip" style=${{ paddingRight: '4px' }}>
        <button class="btn ghost sm mono" type="button" style=${{ padding: '1px 8px' }} onClick=${() => load(c)} title=${`Load ${c.name}`}>${c.name}</button>
        <button class="x" type="button" aria-label=${`Delete classifier ${c.name}`} onClick=${() => del(c)}>✕</button></span>`)}</div>` : null}
  </div>`;
}

// ---- results -----------------------------------------------------------------------------------
function ClassifyCards({ run, prev }) {
  const r = run.result;
  if (!r) return null;
  const scores = r.scores || {};
  const rows = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const prevScores = prev && !!prev.multi_label === !!r.multi_label ? prev.scores : null;
  if (r.multi_label) {
    const set_ = new Set(r.labels || []);
    return html`<section class="card qres">
      <header class="qres-head"><div class="qres-title"><span class="mono qid">labels</span><span class="chip">multi-label</span></div>
        <span class="badge conf-na" title="Threshold">threshold ${fmtPct(r.threshold ?? 0.5, 0)}</span></header>
      <div class="qres-answer">${(r.labels || []).length
        ? html`<div class="chips">${r.labels.map((l) => html`<span key=${l} class="answer" style=${{ fontSize: '20px' }}>${l}</span>`)}</div>`
        : html`<span class="answer muted">no label above threshold</span>`}</div>
      <${ProbBars} entries=${rows} prev=${prevScores} threshold=${r.threshold ?? 0.5} matchSet=${set_} />
      <div class="hint" style=${{ marginTop: '8px' }}>Each bar is an independent P(true); they do not sum to 1.</div>
    </section>`;
  }
  return html`<section class="card qres">
    <header class="qres-head"><div class="qres-title"><span class="mono qid">label</span><span class="chip">single-label</span></div>
      <${ConfBadge} conf=${r.confidence} /></header>
    <div class="qres-answer"><span class="answer">${r.label}</span></div>
    <${ProbBars} entries=${rows} topKey=${r.label} prev=${prevScores} />
    <${Uncertainty} ent=${entropy(Object.values(scores))} />
  </section>`;
}

function ClassifySnippets({ cl, request }) {
  const nm = api.NAME_RE.test(cl.saveName) ? cl.saveName : 'my-classifier';
  const sn = classifierSnippets(nm, cl.input.trim().slice(0, 200), !!getKey());
  const body = JSON.stringify(request || buildBody(cl), null, 2);
  const curl = `curl -sS ${location.origin}/v1/classify \\\n  -H "Content-Type: application/json" \\${getKey() ? '\n  -H "X-API-Key: <your-key>" \\' : ''}\n  -d @- <<'JSON'\n${body}\nJSON`;
  return html`<div class="snippets">
    ${api.NAME_RE.test(cl.saveName) ? null : html`<div class="muted small">Name the classifier and press Save to use the snippets below; <span class="mono">my-classifier</span> is a placeholder.</div>`}
    ${[['python', 'Python'], ['javascript', 'JavaScript'], ['curl', 'curl (saved classifier)']].map(([k, t]) => html`<div key=${k}><div class="snip-title">${t}</div><${CodeBlock} text=${sn[k]} /></div>`)}
    <div><div class="snip-title">curl (one-off, no saved classifier)</div><${CodeBlock} text=${curl} /></div>
  </div>`;
}

function ClassifyResults() {
  const { clRun: run, clPrev, cl, showPrev } = useStore();
  const [tab, setTab] = useState('result');
  const tabs = [{ id: 'result', label: 'Result' }, { id: 'request', label: 'Request JSON' }, { id: 'response', label: 'Response JSON' }, { id: 'snippets', label: 'Snippets' }];
  const { result, error, request, running } = run;
  return html`<div class="results">
    <${Tabs} tabs=${tabs} value=${tab} onChange=${setTab} label="Classify result views" />
    ${running ? html`<div class="status-line" role="status"><span class="spinner"></span>Classifying…</div>` : null}
    <${ErrorBox} error=${error} />
    ${tab === 'result' ? html`<div>
      ${result ? html`<${Metrics} run=${run} />` : null}
      ${result && clPrev ? html`<label class="row gap small" style=${{ margin: '8px 0 0', cursor: 'pointer' }}><input type="checkbox" checked=${showPrev} onChange=${(e) => set({ showPrev: e.target.checked })} />
        <span>Overlay previous run</span><span class="muted">(ticks and deltas)</span></label>` : null}
      ${!result && !error && !running ? html`<${Empty}><span class="glyph" aria-hidden="true">♭</span><strong>No result yet</strong><span>Add labels, then press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> or Classify.</span><//>` : null}
      ${running && !result ? html`<div class="skel" style=${{ height: '170px', marginTop: '10px' }} aria-hidden="true"></div>` : null}
      <div class="qres-list">${result ? html`<${ClassifyCards} run=${run} prev=${showPrev ? clPrev : null} />` : null}</div>
    </div>` : null}
    ${tab === 'request' ? (request ? html`<${CodeBlock} json text=${JSON.stringify(truncateMedia(request), null, 2)} />` : html`<${Empty}>No request yet.<//>`) : null}
    ${tab === 'response' ? (result ? html`<${CodeBlock} json text=${JSON.stringify(result, null, 2)} />` : html`<${Empty}>No response yet.<//>`) : null}
    ${tab === 'snippets' ? html`<${ClassifySnippets} cl=${cl} request=${request} />` : null}
  </div>`;
}

export function Classify() {
  const { cl, clRun, limits } = useStore();
  useEffect(() => {
    const h = (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); runClassify(); } };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, []);
  const errs = validateCl(cl, limits);
  return html`<div class="split">
    <div class="pane left">
      <div class="pane-head">
        <div class="row gap"><h2>Classify</h2><${ModeSwitch} /></div>
        <div class="row gap">
          ${clRun.running ? html`<button class="btn sm" type="button" onClick=${() => abortCtl && abortCtl.abort()}>Cancel</button>` : null}
          <button class="btn primary" type="button" disabled=${clRun.running || errs.length > 0} onClick=${runClassify} title=${errs[0] || 'Classify (Ctrl+Enter)'}>${clRun.running ? 'Running…' : 'Classify'} <kbd>Ctrl+Enter</kbd></button>
        </div>
      </div>
      <section class="card"><h3>Input</h3>
        <textarea class="input textarea" aria-label="Input to classify" rows="6" placeholder="Text to classify…" value=${cl.input} onInput=${(e) => setCl({ input: e.target.value })}></textarea>
      </section>
      <section class="card"><div class="card-head"><h3>Labels<span class="sub">${cl.labels.length}</span></h3>
          <div class="seg" role="radiogroup" aria-label="Label mode" style=${{ margin: 0 }}>
            ${[[false, 'Single-label'], [true, 'Multi-label']].map(([v, l]) => html`<button key=${l} type="button" role="radio" aria-checked=${cl.multi === v} class=${`seg-btn ${cl.multi === v ? 'active' : ''}`} style=${{ textTransform: 'none' }} onClick=${() => setCl({ multi: v })}>${l}</button>`)}
          </div></div>
        <${LabelEditor} />
        ${cl.multi ? html`<label class="field"><span>Threshold <span class="mono">${Number(cl.threshold).toFixed(2)}</span> <span class="muted">(labels at or above it are returned)</span></span>
          <input type="range" min="0" max="1" step="0.01" value=${cl.threshold} aria-label="Threshold" onInput=${(e) => setCl({ threshold: Number(e.target.value) })} /></label>` : html`<div class="hint" style=${{ marginTop: '8px' }}>Single-label: exactly one label, probabilities sum to 1.</div>`}
        <label class="field"><span>Instructions <span class="muted">(optional question for the model)</span></span>
          <input class="input" value=${cl.instructions} placeholder="Which team should handle this?" onInput=${(e) => setCl({ instructions: e.target.value })} /></label>
        ${errs.length ? html`<ul class="errlist" role="alert">${errs.map((m) => html`<li key=${m}>${m}</li>`)}</ul>` : null}
      </section>
      <section class="card"><h3>Save as classifier</h3><${SavePanel} /></section>
    </div>
    <div class="pane right">
      <div class="pane-head"><h2>Results</h2></div>
      <${ClassifyResults} />
    </div>
  </div>`;
}

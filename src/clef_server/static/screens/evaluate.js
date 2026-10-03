// Evaluate: labelled CSV/JSON -> chunked /v1/classify/batch -> /v1/evaluate/metrics (the one metric implementation)
// -> KPIs, confusion matrix, reliability diagram, coverage curve, per-label table, mistakes, export.
import { html, useRef, useState, useEffect } from '../vendor/standalone.module.js';
import * as api from '../api.js';
import { S, set, useStore } from '../store.js';
import { parseCsv, toCsv, download, fmtPct, fmtNum, fmtDur } from '../util.js';
import { Empty, ErrorBox } from '../components/common.js';
import { ConfusionMatrix, ReliabilityDiagram, CoverageCurve } from '../components/evalcharts.js';
import { ExportMenu } from '../components/charts.js';

const SAMPLE = `ticket,department
"I was charged twice for my March subscription, please refund one payment.",billing
"The invoice PDF shows the wrong VAT number for our company.",billing
"Can I get a copy of last year's receipts for my accountant?",billing
"My card was declined but the money left my account.",billing
"The app crashes every time I open the reports page.",technical
"API requests return 502 errors since this morning.",technical
"Exported CSV files are empty when I filter by date.",technical
"Webhook deliveries stopped arriving about an hour ago.",technical
"I can't log in, the password reset email never arrives.",account
"Please change the email address on my profile.",account
"How do I add a teammate to our workspace?",account
"My two-factor codes are rejected after I changed phones.",account
"Where do I update my payment method before the renewal fails?",billing
"Dashboard loads forever and then shows a blank page.",technical
"I want to delete my account and all my data.",account
"My account was locked after the payment failed, what now?",account
`;

S.ev = {
  fileName: '', columns: [], items: [], parseError: null, dragging: false,
  inputCol: '', goldCol: '', multi: false, sep: ';', labelsText: '', labelsEdited: false,
  instructions: '', classifier: '', threshold: 0.5, bins: 10,
  status: 'idle', done: 0, total: 0, t0: 0, failed: 0, skipped: 0, rows: [], notice: null,
  metrics: null, metricRows: [], metricsError: null, metricsBusy: false,
  ui: { normalize: false, thr: 0.8, cell: null, q: '', sortDir: -1, all: false, setThr: false },
};
const E = () => S.ev;
const upd = (patch) => set({ ev: { ...S.ev, ...patch } });
const updUi = (patch) => set({ ev: { ...S.ev, ui: { ...S.ev.ui, ...patch } } });

// ---- parsing ---------------------------------------------------------------------------------
function parseText(name, text) {
  const t = text.replace(/^﻿/, '').trim();
  if (!t) throw new Error('The file is empty.');
  const isJsonl = /\.(jsonl|ndjson)$/i.test(name) || (t[0] === '{' && t.includes('\n'));
  let items;
  if (isJsonl || t[0] === '[' || t[0] === '{') {
    if (t[0] === '[') {
      try { items = JSON.parse(t); } catch (e) { throw new Error(`Invalid JSON: ${e.message}`); }
    } else {
      items = t.split(/\r?\n/).filter((l) => l.trim()).map((l, i) => {
        try { return JSON.parse(l); } catch (e) { throw new Error(`Line ${i + 1}: ${e.message}`); }
      });
    }
    if (!Array.isArray(items) || items.some((x) => !x || typeof x !== 'object' || Array.isArray(x))) throw new Error('Expected a JSON array (or JSONL) of objects.');
    return { items, columns: [...new Set(items.flatMap((x) => Object.keys(x)))] };
  }
  const rows = parseCsv(t);
  if (rows.length < 2) throw new Error('Need a header row and at least one data row.');
  const [head, ...body] = rows;
  return { items: body.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i] ?? '']))), columns: head };
}

const guess = (cols, re, fallback) => cols.find((c) => re.test(c)) || fallback;

function loadText(name, text) {
  try {
    const { items, columns } = parseText(name, text);
    const goldCol = guess(columns, /^(gold|label|labels|class|category|target|truth|dept|department|y)$/i, columns[columns.length - 1]);
    const inputCol = guess(columns.filter((c) => c !== goldCol), /text|input|message|ticket|body|content|description|query/i, columns.find((c) => c !== goldCol) || columns[0]);
    set({ ev: { ...S.ev, fileName: name, items, columns, parseError: null, inputCol, goldCol, labelsEdited: false, metrics: null, rows: [], status: 'idle', done: 0, total: 0, notice: null, metricsError: null, ui: { ...S.ev.ui, cell: null, setThr: false } } });
    deriveLabels();
  } catch (e) {
    upd({ parseError: e.message, items: [], columns: [], fileName: name });
  }
}

async function readFile(f) {
  if (!f) return;
  if (f.size > 64 * 1024 * 1024) { upd({ parseError: 'File is larger than 64 MB.' }); return; }
  loadText(f.name, await f.text());
}

function parseGold(v, multi, sep) {
  if (Array.isArray(v)) return v.map((x) => String(x).trim()).filter(Boolean);
  const s = v == null ? '' : String(v).trim();
  if (!multi) return s ? [s] : [];
  return s.split(sep || ';').map((x) => x.trim()).filter(Boolean);
}

/** Labels = distinct gold values, most frequent first (unless the user edited the list or a saved classifier owns it). */
function deriveLabels() {
  const e = E();
  if (e.classifier || e.labelsEdited || !e.goldCol) return;
  const count = new Map();
  for (const it of e.items) for (const g of parseGold(it[e.goldCol], e.multi, e.sep)) count.set(g, (count.get(g) || 0) + 1);
  const labels = [...count.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([k]) => k);
  upd({ labelsText: labels.join('\n') });
}

const parseLabels = (text) => [...new Set(text.split(/[\n,]/).map((x) => x.trim()).filter(Boolean))];
const savedDef = () => (E().classifier ? S.classifiers.list.find((c) => c.name === E().classifier) : null);
const labelNames = () => {
  const def = savedDef();
  if (def) return Array.isArray(def.labels) ? def.labels : Object.keys(def.labels || {});
  return parseLabels(E().labelsText);
};
const isMulti = () => { const def = savedDef(); return def ? !!def.multi_label : E().multi; };

async function loadClassifiers() {
  try {
    const { data } = await api.listClassifiers();
    set({ classifiers: { loaded: true, list: (data && data.classifiers) || [], error: null } });
  } catch (e) { set({ classifiers: { loaded: true, list: [], error: e.status === 404 ? 'unavailable' : e.message } }); }
}

// ---- run -------------------------------------------------------------------------------------
let ctl = null;
let metricsToken = 0;
const sleep = (ms, signal) => new Promise((res, rej) => {
  const t = setTimeout(res, ms);
  signal.addEventListener('abort', () => { clearTimeout(t); rej(new DOMException('aborted', 'AbortError')); }, { once: true });
});

function validate() {
  const e = E();
  const names = labelNames();
  const multi = isMulti();
  if (!e.items.length) return 'Load a labelled file first.';
  if (!e.inputCol || !e.goldCol) return 'Pick the input column and the gold-label column.';
  if (e.inputCol === e.goldCol) return 'The input and gold columns must differ.';
  if (names.length < (multi ? 1 : 2)) return multi ? 'Add at least one label.' : 'Add at least two labels.';
  const max = (S.limits && S.limits.max_labels) || 64;
  if (names.length > max) return `At most ${max} labels (server limit).`;
  return null;
}

function prepare() {
  const e = E();
  const names = new Set(labelNames());
  const multi = isMulti();
  const rows = [];
  let skipped = 0;
  e.items.forEach((it, i) => {
    const gold = parseGold(it[e.goldCol], multi, e.sep);
    const ok = multi ? gold.every((g) => names.has(g)) : gold.length === 1 && names.has(gold[0]);
    const input = it[e.inputCol];
    if (!ok || input == null || input === '') { skipped++; return; }
    rows.push({ i, input, gold: multi ? gold : gold[0], scores: null, error: null });
  });
  return { rows, skipped };
}

async function runChunks(indices) {
  const e = E();
  const names = labelNames();
  const multi = isMulti();
  const def = savedDef();
  const size = Math.max(1, Math.min((S.limits && S.limits.max_batch) || 64, 32));
  const chunks = [];
  for (let i = 0; i < indices.length; i += size) chunks.push(indices.slice(i, i + size));
  ctl = new AbortController();
  const rows = E().rows.map((r) => (indices.includes(r.i) ? { ...r, scores: null, error: null } : r));
  upd({ status: 'running', notice: null, rows, done: rows.filter((r) => r.scores).length, total: rows.length, t0: Date.now(), failed: 0, metricsError: null });
  const base = rows.filter((r) => r.scores).length;
  let doneNow = 0;
  try {
    for (const chunk of chunks) {
      const byIndex = new Map(E().rows.map((r, k) => [r.i, k]));
      const inputs = chunk.map((i) => E().rows[byIndex.get(i)].input);
      let ok = false, lastErr = null;
      for (let attempt = 0; attempt < 3 && !ok; attempt++) {
        try {
          if (attempt) await sleep(800 * attempt, ctl.signal);
          const body = def ? { inputs } : { inputs, labels: names, multi_label: multi, ...(e.instructions.trim() ? { instructions: e.instructions.trim() } : {}) };
          if (multi) body.threshold = Number(e.threshold);
          const { data } = await (def ? api.classifierBatch(def.name, { inputs, ...(multi ? { threshold: Number(e.threshold) } : {}) }, ctl.signal) : api.classifyBatch(body, ctl.signal));
          const results = (data && data.results) || [];
          const cur = E().rows.slice();
          chunk.forEach((i, k) => {
            const at = byIndex.get(i);
            cur[at] = { ...cur[at], scores: results[k] ? results[k].scores : null, error: results[k] ? null : 'missing result' };
          });
          doneNow += chunk.length;
          upd({ rows: cur, done: base + doneNow });
          ok = true;
        } catch (err) {
          if (err && err.name === 'AbortError') throw err;
          lastErr = err;
          if (err.status === 400 || err.status === 401 || err.status === 413) break;
        }
      }
      if (!ok) {
        const cur = E().rows.slice();
        const msg = lastErr ? `${lastErr.message}${lastErr.requestId ? ` [${lastErr.requestId}]` : ''}` : 'failed';
        chunk.forEach((i) => { const at = byIndex.get(i); cur[at] = { ...cur[at], error: msg }; });
        upd({ rows: cur, failed: cur.filter((r) => r.error && !r.scores).length, notice: `Some rows failed: ${msg}` });
      }
    }
    upd({ status: 'done' });
  } catch (err) {
    upd({ status: 'cancelled' });
  } finally { ctl = null; }
  if (E().rows.some((r) => r.scores)) await fetchMetrics();
}

function start() {
  const bad = validate();
  if (bad) { upd({ notice: bad }); return; }
  const { rows, skipped } = prepare();
  if (!rows.length) { upd({ notice: 'No usable rows: every row has an empty input or a gold label outside the label list.', skipped }); return; }
  upd({ rows, skipped, metrics: null, metricRows: [], ui: { ...E().ui, cell: null, setThr: false } });
  runChunks(rows.map((r) => r.i));
}
const retryFailed = () => runChunks(E().rows.filter((r) => !r.scores).map((r) => r.i));

async function fetchMetrics() {
  const e = E();
  const used = e.rows.filter((r) => r.scores);
  if (!used.length) return;
  const token = ++metricsToken;
  upd({ metricsBusy: true, metricsError: null });
  try {
    const multi = isMulti();
    const body = { labels: labelNames(), rows: used.map((r) => ({ gold: r.gold, scores: r.scores })), multi_label: multi, bins: Number(e.bins), include_predictions: true };
    if (multi) body.threshold = Number(e.threshold);
    const { data } = await api.evaluateMetrics(body);
    if (token !== metricsToken) return;
    const route = (data.auto_route || []).find((r) => r.target_accuracy === 0.95 && r.threshold != null);
    const ui = E().ui.setThr ? E().ui : { ...E().ui, thr: route ? route.threshold : 0.8 };
    upd({ metrics: data, metricRows: used, metricsBusy: false, ui });
  } catch (err) {
    if (token === metricsToken) upd({ metricsBusy: false, metricsError: err });
  }
}

// ---- export ----------------------------------------------------------------------------------
function exportPredictions() {
  const e = E();
  const m = e.metrics;
  if (!m) return;
  const names = m.labels;
  const head = ['row', 'input', 'gold', 'predicted', 'confidence', 'correct', ...names.map((n) => `p.${n}`)];
  const lines = [head];
  m.predictions.forEach((p, k) => {
    const r = e.metricRows[p.index];
    const conf = typeof p.confidence === 'number' ? p.confidence : '';
    lines.push([r.i + 1, typeof r.input === 'string' ? r.input : JSON.stringify(r.input), [].concat(p.gold).join(';'), [].concat(p.predicted).join(';'), conf, p.correct, ...names.map((n) => (r.scores[n] ?? ''))]);
  });
  download('clef-eval-predictions.csv', toCsv(lines), 'text/csv');
}
function exportMetrics() {
  const m = E().metrics;
  if (!m) return;
  const { predictions, ...rest } = m;
  download('clef-eval-metrics.json', JSON.stringify({ ...rest, dataset: E().fileName, input_column: E().inputCol, gold_column: E().goldCol, instructions: E().instructions || undefined, classifier: E().classifier || undefined }, null, 2), 'application/json');
}

// ---- view parts ------------------------------------------------------------------------------
const eceWord = (v) => (v < 0.05 ? 'well calibrated' : v < 0.1 ? 'fair' : 'over/under-confident');

function Kpi({ label, value, sub, hint }) {
  return html`<div class="kpi kpi-eval"><div class="kpi-label">${label}</div><div class="kpi-value mono">${value}</div><div class="muted small">${sub}</div>${hint ? html`<div class="hint">${hint}</div>` : null}</div>`;
}

function Kpis({ m, failed, skipped }) {
  const multi = m.multi_label;
  const cal = m.calibration;
  return html`<div class="kpis" role="group" aria-label="Headline metrics">
    <${Kpi} label=${multi ? 'Exact match' : 'Accuracy'} value=${fmtPct(multi ? m.exact_match : m.accuracy, 1)} sub=${multi ? `Hamming loss ${fmtNum(m.hamming_loss, 3)}` : `top-2 ${fmtPct(m.top2_accuracy, 1)}`} />
    <${Kpi} label="Macro F1" value=${fmtNum(m.macro_f1, 3)} sub=${`micro ${fmtNum(m.micro_f1, 3)} · weighted ${fmtNum(m.weighted_f1, 3)}`} />
    <${Kpi} label="ECE" value=${fmtNum(cal.ece, 3)} sub=${eceWord(cal.ece)} hint="lower is better" />
    <${Kpi} label="Brier" value=${fmtNum(cal.brier, 3)} sub=${`log-loss ${fmtNum(cal.nll, 3)}`} hint="lower is better" />
    <${Kpi} label="Rows" value=${m.n} sub=${[skipped ? `${skipped} skipped` : null, failed ? `${failed} failed` : null].filter(Boolean).join(' · ') || 'all evaluated'} />
  </div>`;
}

function ConfusionCard({ m, ui }) {
  const cm = m.confusion_matrix;
  if (!cm) {
    return html`<section class="card"><div class="card-head"><h3>Per-label outcomes</h3></div>
      <div class="tablewrap"><table class="table"><thead><tr><th>Label</th><th class="num">TP</th><th class="num">FP</th><th class="num">FN</th><th class="num">TN</th></tr></thead>
        <tbody>${m.per_label.map((r) => html`<tr key=${r.label}><td>${r.label}</td><td class="num mono">${r.tp}</td><td class="num mono">${r.fp}</td><td class="num mono">${r.fn}</td><td class="num mono">${r.tn}</td></tr>`)}</tbody></table></div></section>`;
  }
  return html`<section class="card"><div class="card-head"><h3>Confusion matrix<span class="sub">rows = gold, columns = predicted</span></h3>
      <label class="row gap small"><input type="checkbox" checked=${ui.normalize} onChange=${(e) => updUi({ normalize: e.target.checked })} /> Row-normalised</label></div>
    <${ConfusionMatrix} labels=${cm.labels} matrix=${cm.matrix} normalize=${ui.normalize} selected=${ui.cell} onSelect=${(c) => updUi({ cell: c })} />
    <div class="hint" style=${{ marginTop: '6px' }}>Click a cell to list its rows below.</div></section>`;
}

function ReliabilityCard({ m, bins }) {
  const host = useRef(null);
  const cal = m.calibration;
  return html`<section class="card"><div class="card-head"><h3>Reliability<span class="sub">is a 70% answer right 70% of the time?</span></h3><${ExportMenu} host=${host} name="clef-reliability" /></div>
    <div class="eval-rel"><div ref=${host}><${ReliabilityDiagram} bins=${cal.bins} ece=${cal.ece} mce=${cal.mce} n=${m.n} yLabel=${m.multi_label ? 'observed frequency' : 'accuracy'} /></div>
      <div class="stat-row eval-rel-stats">
        <div class="it"><span class="k">ECE</span><span class="v mono">${fmtNum(cal.ece, 3)}</span></div>
        <div class="it"><span class="k">Max gap</span><span class="v mono">${fmtNum(cal.mce, 3)}</span></div>
        <div class="it"><span class="k">Mean confidence</span><span class="v mono">${fmtPct(cal.mean_confidence, 1)}</span></div>
        <div class="it"><span class="k">${m.multi_label ? 'Positive rate' : 'Accuracy'}</span><span class="v mono">${fmtPct(cal.accuracy, 1)}</span></div>
      </div></div>
    <div class="hint">Bars at or near the dashed diagonal mean the stated confidence can be trusted. ${m.multi_label ? 'Pooled over every (row, label) pair.' : ''}</div></section>`;
}

function CoverageCard({ m, ui }) {
  const host = useRef(null);
  const curve = m.coverage_curve;
  if (!curve) return null;
  const idx = Math.round(ui.thr * 100);
  const cur = curve[idx] || curve[0];
  const routed = cur.n, review = m.n - cur.n;
  const route = (m.auto_route || []).filter((r) => r.threshold != null);
  return html`<section class="card"><div class="card-head"><h3>Coverage vs accuracy<span class="sub">pick an auto-route threshold</span></h3><${ExportMenu} host=${host} name="clef-coverage" /></div>
    <div class="legend" role="list"><span class="lg-item" role="listitem"><span class="lg-sw line" style=${{ background: 'var(--s1)' }}></span>accuracy of rows at or above the threshold</span>
      <span class="lg-item" role="listitem"><span class="lg-sw line" style=${{ background: 'var(--s2)' }}></span>coverage (share of rows at or above)</span></div>
    <div ref=${host}><${CoverageCurve} curve=${curve} threshold=${ui.thr} onThreshold=${(t) => updUi({ thr: t, setThr: true })} /></div>
    <label class="row gap small" style=${{ marginTop: '6px' }}><span class="nowrap">Threshold</span>
      <input type="range" class="grow" min="0" max="1" step="0.01" value=${ui.thr} aria-label="Confidence threshold" onInput=${(e) => updUi({ thr: Number(e.target.value), setThr: true })} />
      <span class="mono">${ui.thr.toFixed(2)}</span></label>
    <div class="readout" role="status" aria-live="polite">At <strong>≥ ${ui.thr.toFixed(2)}</strong> confidence: <strong>${fmtPct(cur.coverage, 1)}</strong> coverage,
      <strong>${cur.accuracy == null ? 'n/a' : fmtPct(cur.accuracy, 1)}</strong> accuracy — ${routed} row${routed === 1 ? '' : 's'} auto-routed, ${review} for human review.</div>
    ${route.length ? html`<div class="chips" style=${{ marginTop: '8px' }}>${route.map((r) => html`<button key=${r.target_accuracy} type="button" class="btn sm" onClick=${() => updUi({ thr: r.threshold, setThr: true })}
      title="Lowest threshold that reaches this accuracy">${Math.round(r.target_accuracy * 100)}% accuracy → ≥${r.threshold.toFixed(2)} (${fmtPct(r.coverage, 0)} covered)</button>`)}</div>`
      : html`<div class="hint" style=${{ marginTop: '8px' }}>No threshold reaches 80% accuracy on this data.</div>`}</section>`;
}

function LabelTable({ m }) {
  return html`<section class="card"><div class="card-head"><h3>Per label</h3></div>
    <div class="tablewrap"><table class="table"><thead><tr><th>Label</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th><th class="num">Predicted</th></tr></thead>
      <tbody>${m.per_label.map((r) => html`<tr key=${r.label}><td>${r.label}</td><td class="num mono">${fmtNum(r.precision, 3)}</td><td class="num mono">${fmtNum(r.recall, 3)}</td>
        <td class="num mono">${fmtNum(r.f1, 3)}</td><td class="num mono">${r.support}</td><td class="num mono ${r.predicted === 0 && r.support > 0 ? 'err' : ''}">${r.predicted}</td></tr>`)}</tbody></table></div></section>`;
}

const confOf = (p) => (typeof p.confidence === 'number' ? p.confidence : Math.max(0, ...Object.values(p.confidence || {})));

function Mistakes({ m, rows, ui }) {
  const all = m.predictions.filter((p) => !p.correct);
  const cell = ui.cell;
  const q = ui.q.trim().toLowerCase();
  const text = (p) => { const r = rows[p.index]; return typeof r.input === 'string' ? r.input : JSON.stringify(r.input); };
  let list = all.filter((p) => (!cell || (p.gold === cell.gold && p.predicted === cell.pred)) && (!q || text(p).toLowerCase().includes(q)));
  list = list.slice().sort((a, b) => (confOf(a) - confOf(b)) * ui.sortDir || a.index - b.index);
  const shown = ui.all ? list : list.slice(0, 100);
  return html`<section class="card"><div class="card-head"><h3>Mistakes<span class="sub">${list.length} of ${all.length}${cell ? ` · gold ${cell.gold} → ${cell.pred}` : ''}</span></h3>
      <div class="row gap wrap">${cell ? html`<button class="btn sm" type="button" onClick=${() => updUi({ cell: null })}>Clear cell filter</button>` : null}
        <input class="input" style=${{ width: '180px' }} type="search" aria-label="Filter mistakes by text" placeholder="Filter text" value=${ui.q} onInput=${(e) => updUi({ q: e.target.value })} /></div></div>
    ${!all.length ? html`<${Empty}><strong>No mistakes</strong><span>Every evaluated row was predicted correctly.</span><//>` : html`<div class="tablewrap"><table class="table">
      <thead><tr><th class="num">#</th><th>Input</th><th>Gold</th><th>Predicted</th>
        <th class="num sortable" tabIndex="0" aria-sort=${ui.sortDir < 0 ? 'descending' : 'ascending'} title="Sort by confidence" onClick=${() => updUi({ sortDir: -ui.sortDir })}
          onKeyDown=${(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), updUi({ sortDir: -ui.sortDir }))}>Conf ${ui.sortDir < 0 ? '▼' : '▲'}</th></tr></thead>
      <tbody>${shown.map((p) => html`<tr key=${p.index}><td class="num mono muted">${rows[p.index].i + 1}</td><td class="state-cell"><div class="clamp">${text(p).slice(0, 220)}</div></td>
        <td class="mono">${[].concat(p.gold).join(', ') || '(none)'}</td><td class="mono err">${[].concat(p.predicted).join(', ') || '(none)'}</td>
        <td class="num mono">${fmtPct(confOf(p), 0)}</td></tr>`)}</tbody></table></div>
      ${!ui.all && list.length > shown.length ? html`<div class="row gap" style=${{ marginTop: '8px' }}><span class="muted small">Showing ${shown.length} of ${list.length}.</span><button class="btn sm" type="button" onClick=${() => updUi({ all: true })}>Show all</button></div>` : null}`}
  </section>`;
}

function Preview({ e }) {
  const rows = e.items.slice(0, 5);
  return html`<div class="tablewrap preview-wrap"><table class="table"><thead><tr>${e.columns.map((c) => html`<th key=${c} class=${c === e.inputCol ? 'col-in' : c === e.goldCol ? 'col-gold' : ''}>${c}${c === e.inputCol ? ' (input)' : c === e.goldCol ? ' (gold)' : ''}</th>`)}</tr></thead>
    <tbody>${rows.map((r, i) => html`<tr key=${i}>${e.columns.map((c) => html`<td key=${c}><div class="clamp">${typeof r[c] === 'string' ? r[c].slice(0, 120) : JSON.stringify(r[c])}</div></td>`)}</tr>`)}</tbody></table></div>`;
}

// ---- view ------------------------------------------------------------------------------------
export function Evaluate() {
  const { ev: e, classifiers, limits } = useStore();
  const file = useRef(null);
  useEffect(() => { if (!S.classifiers.loaded) loadClassifiers(); }, []);
  const running = e.status === 'running';
  const names = labelNames();
  const multi = isMulti();
  const def = savedDef();
  const elapsed = running ? (Date.now() - e.t0) / 1000 : 0;
  const eta = running && e.done > 0 ? (elapsed / e.done) * (e.total - e.done) : null;
  const unknown = e.items.length && e.goldCol && !def ? e.items.filter((it) => { const g = parseGold(it[e.goldCol], multi, e.sep); return !(g.length && g.every((x) => names.includes(x))); }).length : 0;
  const failed = e.rows.filter((r) => !r.scores && r.error).length;
  const onDrop = (ev) => { ev.preventDefault(); upd({ dragging: false }); readFile(ev.dataTransfer.files[0]); };
  const keyOpen = (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); file.current.click(); } };
  const [, setTick] = useState(0);
  useEffect(() => { if (!running) return undefined; const id = setInterval(() => setTick((x) => x + 1), 1000); return () => clearInterval(id); }, [running]);

  return html`<div class="page batch eval">
    <div class="split batch-split">
      <div class="pane left">
        <section class="card"><h3>Dataset</h3>
          <div class=${`dropzone ${e.dragging ? 'drag' : ''}`} role="button" tabIndex="0" aria-label="Upload a labelled CSV, JSON or JSONL file. Drop it here or press Enter to choose."
            onClick=${() => file.current.click()} onKeyDown=${keyOpen}
            onDragOver=${(ev) => { ev.preventDefault(); if (!e.dragging) upd({ dragging: true }); }} onDragLeave=${() => upd({ dragging: false })} onDrop=${onDrop}>
            <strong>${e.fileName ? e.fileName : 'Drop a labelled file here'}</strong>
            <span class="muted small">${e.items.length ? `${e.items.length} rows · ${e.columns.length} columns · click to replace` : 'CSV, JSON (array of objects) or JSONL · click to choose'}</span>
          </div>
          <input ref=${file} type="file" hidden accept=".csv,.json,.jsonl,.ndjson,text/csv,application/json,text/plain" onChange=${(ev) => { const f = ev.target.files[0]; ev.target.value = ''; readFile(f); }} />
          <div class="row gap wrap" style=${{ marginTop: '8px' }}><button class="btn sm" type="button" onClick=${() => loadText('sample-tickets.csv', SAMPLE)}>Load sample (16 tickets)</button>
            <span class="muted small">A larger set lives in the repo: examples/tickets_labelled.csv</span></div>
          ${e.parseError ? html`<div class="errorbox small" role="alert">${e.parseError}</div>` : null}
          ${e.items.length ? html`<div style=${{ marginTop: '10px' }}><${Preview} e=${e} /></div>` : null}
        </section>

        ${e.items.length ? html`<section class="card"><h3>Columns and labels</h3>
          <div class="grid2">
            <label class="field"><span>Input column</span><select class="input" value=${e.inputCol} onChange=${(ev) => upd({ inputCol: ev.target.value })}>${e.columns.map((c) => html`<option key=${c} value=${c}>${c}</option>`)}</select></label>
            <label class="field"><span>Gold-label column</span><select class="input" value=${e.goldCol} onChange=${(ev) => { upd({ goldCol: ev.target.value, labelsEdited: false }); deriveLabels(); }}>${e.columns.map((c) => html`<option key=${c} value=${c}>${c}</option>`)}</select></label>
          </div>
          <div class="row gap wrap"><label class="row gap small"><input type="checkbox" checked=${multi} disabled=${!!def} onChange=${(ev) => { upd({ multi: ev.target.checked, labelsEdited: false }); deriveLabels(); }} /> Multi-label</label>
            ${multi ? html`<label class="row gap small">Separator <input class="input mono num-input" style=${{ width: '56px' }} maxlength="3" value=${e.sep} aria-label="Gold label separator" onInput=${(ev) => { upd({ sep: ev.target.value || ';', labelsEdited: false }); deriveLabels(); }} /></label>
              <label class="row gap small">Threshold <input class="input mono num-input" style=${{ width: '70px' }} type="number" min="0" max="1" step="0.05" value=${e.threshold} onInput=${(ev) => upd({ threshold: ev.target.value })} onChange=${fetchMetrics} /></label>` : null}</div>
          <label class="field"><span>Labels <span class="muted">${def ? `from saved classifier ${def.name}` : 'derived from the gold column · one per line or comma-separated'}</span></span>
            <textarea class="input textarea mono" rows="4" spellcheck="false" disabled=${!!def} aria-label="Labels" value=${def ? names.join('\n') : e.labelsText}
              onInput=${(ev) => upd({ labelsText: ev.target.value, labelsEdited: true })}></textarea></label>
          <div class="row gap wrap small"><span class="muted">${names.length} label${names.length === 1 ? '' : 's'}</span>
            ${!def && e.labelsEdited ? html`<button class="btn ghost sm" type="button" onClick=${() => { upd({ labelsEdited: false }); deriveLabels(); }}>Re-derive from data</button>` : null}</div>
          ${unknown ? html`<div class="warn small" style=${{ marginTop: '6px' }}>${unknown} row${unknown === 1 ? ' has' : 's have'} an empty or unknown gold label and will be skipped.</div>` : null}
        </section>

        <section class="card"><h3>Classifier and run</h3>
          <label class="field"><span>Classifier</span><select class="input" value=${e.classifier} onChange=${(ev) => { upd({ classifier: ev.target.value, labelsEdited: false }); deriveLabels(); }}>
            <option value="">Ad-hoc (labels above + instructions)</option>
            ${classifiers.list.filter((c) => c.kind === 'classify').map((c) => html`<option key=${c.name} value=${c.name}>${c.name}</option>`)}</select></label>
          ${!def ? html`<label class="field"><span>Instructions <span class="muted">optional hint sent with every row</span></span>
            <textarea class="input textarea" rows="2" value=${e.instructions} onInput=${(ev) => upd({ instructions: ev.target.value })}></textarea></label>` : html`<div class="muted small">Instructions and multi-label come from the saved classifier.</div>`}
          <label class="field inline"><span>Reliability bins</span><select class="input num-input" value=${e.bins} onChange=${(ev) => { upd({ bins: Number(ev.target.value) }); fetchMetrics(); }}>${[5, 10, 15, 20].map((b) => html`<option key=${b} value=${b}>${b}</option>`)}</select></label>
          <div class="row gap">
            ${running ? html`<button class="btn danger" type="button" onClick=${() => ctl && ctl.abort()}>Cancel</button>`
              : html`<button class="btn primary" type="button" disabled=${!e.items.length} onClick=${start}>Run evaluation</button>`}
            ${!running && failed ? html`<button class="btn" type="button" onClick=${retryFailed}>Retry ${failed} failed</button>` : null}</div>
          ${e.notice ? html`<div class="warn small" role="status" style=${{ marginTop: '6px' }}>${e.notice}</div>` : null}
        </section>` : null}
      </div>

      <div class="pane right">
        <div class="pane-head"><h2>Results</h2>
          <div class="row gap"><button class="btn sm" type="button" disabled=${!e.metrics} onClick=${exportPredictions}>Export predictions CSV</button>
            <button class="btn sm" type="button" disabled=${!e.metrics} onClick=${exportMetrics}>Export metrics JSON</button></div></div>
        ${e.total ? html`<div class="progress-wrap"><div class="progress" role="progressbar" aria-label="Evaluation progress" aria-valuemin="0" aria-valuemax=${e.total} aria-valuenow=${e.done}>
            <div class="progress-fill" style=${{ width: `${(e.done / e.total) * 100}%` }}></div></div>
          <div class="muted small mono">${e.done}/${e.total} · ${e.status}${eta != null ? ` · ETA ${fmtDur(eta)}` : ''}${failed ? ` · ${failed} failed` : ''}${e.skipped ? ` · ${e.skipped} skipped` : ''}</div></div>` : null}
        ${e.metricsError ? html`<${ErrorBox} error=${e.metricsError} title="Metrics failed" />` : null}
        ${e.metrics ? html`<${Kpis} m=${e.metrics} failed=${failed} skipped=${e.skipped} />
          <${ConfusionCard} m=${e.metrics} ui=${e.ui} />
          <div class="grid2 eval-pair"><${ReliabilityCard} m=${e.metrics} bins=${e.bins} /><${CoverageCard} m=${e.metrics} ui=${e.ui} /></div>
          <${LabelTable} m=${e.metrics} />
          <${Mistakes} m=${e.metrics} rows=${e.metricRows} ui=${e.ui} />`
          : !e.total ? html`<${Empty}><span class="glyph" aria-hidden="true">▦</span><strong>No evaluation yet</strong><span>Load a labelled CSV or JSON, pick the input and gold-label columns, then run. You get accuracy, per-label F1, a confusion matrix and calibration (reliability, ECE, Brier).</span><//>` : null}
      </div>
    </div>
  </div>`;
}

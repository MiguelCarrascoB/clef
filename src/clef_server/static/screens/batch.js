// Batch: CSV / JSONL / pasted input -> sequential chunked /v1/batch calls -> sortable table -> export.
import { html, useRef, useState } from '../vendor/standalone.module.js';
import * as api from '../api.js';
import { S, set, useStore, allPresets } from '../store.js';
import {
  parseCsv, toCsv, download, cardsToContract, validateQuestions, errCount, fmtPct, fmtNum, topAnswer, confidenceOf, confLevel,
} from '../util.js';
import { Empty } from '../components/common.js';
import { BatchSummary, CalibrationCard } from '../components/summary.js';
import { openInPlayground } from './playground.js';

S.batch = {
  fmt: 'csv', text: '', fileName: '', stateMode: 'row', template: '{{text}}', schema: 'playground',
  chunk: 8, status: 'idle', done: 0, total: 0, rows: [], parseError: null, sort: { col: null, dir: 1 }, notice: null, filter: null,
};
const B = () => S.batch;
const upd = (patch) => set({ batch: { ...S.batch, ...patch } });

// ---- parsing ---------------------------------------------------------------------------------
function parseInput(fmt, text) {
  if (!text.trim()) return { items: [], columns: [] };
  if (fmt === 'csv') {
    const rows = parseCsv(text);
    if (rows.length < 1) return { items: [], columns: [] };
    const [head, ...body] = rows;
    const items = body.map((r) => Object.fromEntries(head.map((h, i) => [h, r[i] ?? ''])));
    return { items, columns: head };
  }
  const items = [];
  const lines = text.split(/\r?\n/);
  lines.forEach((ln, i) => {
    if (!ln.trim()) return;
    try { items.push(JSON.parse(ln)); } catch (e) { throw new Error(`Line ${i + 1}: ${e.message}`); }
  });
  const cols = [...new Set(items.flatMap((x) => (x && typeof x === 'object' ? Object.keys(x) : [])))];
  return { items, columns: cols };
}

function stateFor(item, b) {
  if (b.stateMode === 'row') return item;
  const tpl = b.template || '';
  const get = (k) => {
    const v = item && typeof item === 'object' ? item[k.trim()] : undefined;
    if (v === undefined) return '';
    return typeof v === 'string' ? v : JSON.stringify(v);
  };
  return tpl.replace(/\{\{\s*([^}]+?)\s*\}\}/g, (_, k) => get(k));
}

function schemaQuestions(b) {
  if (b.schema === 'playground') return cardsToContract(S.pg.cards);
  const p = allPresets().find((x) => x.name === b.schema);
  return p ? p.questions : {};
}

const stateStr = (st) => (typeof st === 'string' ? st : JSON.stringify(st));

// ---- run -------------------------------------------------------------------------------------
let ctl = null;
const sleep = (ms, signal) => new Promise((res, rej) => {
  const t = setTimeout(res, ms);
  signal.addEventListener('abort', () => { clearTimeout(t); rej(new DOMException('aborted', 'AbortError')); }, { once: true });
});

async function runChunks(indices) {
  const b = B();
  const questions = schemaQuestions(b);
  const errs = validateQuestions(questions, S.limits || {});
  if (errCount(errs)) { upd({ notice: 'Schema has validation errors (fix it in the Playground or pick another).' }); return; }
  const size = Math.max(1, Math.min(Number(b.chunk) || 8, (S.limits && S.limits.max_batch) || 1000));
  const chunks = [];
  for (let i = 0; i < indices.length; i += size) chunks.push(indices.slice(i, i + size));
  ctl = new AbortController();
  const rows = B().rows.map((r) => (indices.includes(r.i) ? { ...r, error: null, response: null } : r));
  upd({ status: 'running', notice: null, rows, done: rows.filter((r) => r.response).length, total: rows.length });
  try {
    for (const chunk of chunks) {
      const records = chunk.map((i) => ({ model: 'clef-flash', state: B().rows[i].state, questions }));
      let lastErr = null, ok = false;
      for (let attempt = 0; attempt < 3 && !ok; attempt++) {
        try {
          if (attempt) await sleep(800 * attempt, ctl.signal);
          const { data } = await api.batch(records, ctl.signal);
          const results = (data && data.results) || [];
          const cur = B().rows.slice();
          chunk.forEach((i, k) => { cur[i] = { ...cur[i], response: results[k] || null, error: results[k] ? null : 'missing result', chunkMs: data.batch_ms }; });
          upd({ rows: cur, done: cur.filter((r) => r.response).length });
          ok = true;
        } catch (e) {
          if (e && e.name === 'AbortError') throw e;
          lastErr = e;
          if (e.status === 400 || e.status === 401 || e.status === 413) break; // retrying will not help
        }
      }
      if (!ok) {
        const cur = B().rows.slice();
        const msg = lastErr ? `${lastErr.message}${lastErr.requestId ? ` [${lastErr.requestId}]` : ''}` : 'failed';
        chunk.forEach((i) => { cur[i] = { ...cur[i], error: msg }; });
        upd({ rows: cur });
      }
    }
    upd({ status: 'done' });
  } catch (e) {
    upd({ status: 'cancelled' });
  } finally { ctl = null; }
}

function start() {
  const b = B();
  let parsed;
  try { parsed = parseInput(b.fmt, b.text); } catch (e) { upd({ parseError: e.message }); return; }
  if (!parsed.items.length) { upd({ parseError: 'No rows to process.' }); return; }
  const rows = parsed.items.map((item, i) => ({ i, item, state: stateFor(item, b), response: null, error: null }));
  upd({ rows, parseError: null, sort: { col: null, dir: 1 }, filter: null });
  runChunks(rows.map((r) => r.i));
}
const retryFailed = () => runChunks(B().rows.filter((r) => !r.response).map((r) => r.i));

// ---- table helpers ---------------------------------------------------------------------------
function cellValue(a) {
  if (!a) return { text: '-', sort: -1, conf: null };
  if (a.type === 'choice') return { text: `${a.choice} ${fmtPct(a.probabilities[a.choice], 0)}`, sort: a.choice, conf: confidenceOf(a), sub: a.probabilities[a.choice] };
  if (a.type === 'score') return { text: fmtNum(a.score), sort: a.score, conf: confidenceOf(a) };
  return { text: `P ${fmtPct(a.noul, 0)}`, sort: a.noul, conf: confidenceOf(a) };
}

function sortRows(rows, sort) {
  if (!sort.col) return rows;
  const val = (r) => {
    if (sort.col === '#') return r.i;
    const a = r.response && r.response.answers && r.response.answers[sort.col];
    return a ? cellValue(a).sort : (sort.dir > 0 ? Infinity : -Infinity);
  };
  return rows.slice().sort((x, y) => {
    const a = val(x), b = val(y);
    const c = typeof a === 'string' || typeof b === 'string' ? String(a).localeCompare(String(b)) : a - b;
    return c * sort.dir || x.i - y.i;
  });
}

function exportCsv(qs) {
  const header = ['row', 'state', 'error'];
  const qcols = Object.entries(qs).map(([id, q]) => {
    const keys = q.type === 'choice' ? Object.keys(q.criteria || {}) : q.type === 'score' ? (q.criteria || []).map((_, i) => String(i)) : [];
    return { id, q, keys };
  });
  for (const { id, q, keys } of qcols) {
    if (q.type === 'choice') header.push(`${id}.choice`, `${id}.confidence`, ...keys.map((k) => `${id}.p.${k}`));
    else if (q.type === 'score') header.push(`${id}.score`, `${id}.confidence`, ...keys.map((k) => `${id}.p.${k}`));
    else header.push(`${id}.noul`);
  }
  const lines = [header];
  for (const r of B().rows) {
    const line = [r.i + 1, stateStr(r.state), r.error || ''];
    for (const { id, q, keys } of qcols) {
      const a = r.response && r.response.answers && r.response.answers[id];
      if (q.type === 'choice') line.push(a ? a.choice : '', a ? a.confidence : '', ...keys.map((k) => (a && a.probabilities ? a.probabilities[k] : '')));
      else if (q.type === 'score') line.push(a ? a.score : '', a ? a.confidence : '', ...keys.map((k) => (a && a.probabilities ? a.probabilities[k] : '')));
      else line.push(a ? a.noul : '');
    }
    lines.push(line);
  }
  download('clef-batch.csv', toCsv(lines), 'text/csv');
}
function exportJsonl() {
  download('clef-batch.jsonl', B().rows.map((r) => JSON.stringify({ index: r.i, state: r.state, response: r.response, error: r.error })).join('\n'), 'application/x-ndjson');
}

// ---- view ------------------------------------------------------------------------------------
export function Batch() {
  const { batch: b, presets, limits } = useStore();
  const file = useRef(null);
  const [pick, setPick] = useState('csv');
  const running = b.status === 'running';
  const questions = schemaQuestions(b);
  const qids = Object.keys(questions);
  let preview = { items: [], columns: [] }, perr = null;
  try { preview = parseInput(b.fmt, b.text); } catch (e) { perr = e.message; }
  const onFile = async (e, fmt) => {
    const f = e.target.files[0]; e.target.value = '';
    if (!f) return;
    upd({ fmt, text: await f.text(), fileName: f.name, parseError: null });
  };
  const idSet = b.filter ? new Set(b.filter.ids) : null;
  const visible = idSet ? b.rows.filter((r) => idSet.has(r.i)) : b.rows;
  const sorted = sortRows(visible, b.sort);
  const done = b.rows.filter((r) => r.response);
  const setSort = (col) => upd({ sort: { col, dir: b.sort.col === col ? -b.sort.dir : 1 } });
  const arrow = (col) => (b.sort.col === col ? (b.sort.dir > 0 ? ' ▲' : ' ▼') : '');
  const failed = b.rows.filter((r) => !r.response && r.error).length;
  const maxBatch = limits && limits.max_batch;

  return html`<div class="page batch">
    <div class="split batch-split">
      <div class="pane left">
        <section class="card"><h3>Input</h3>
          <div class="row gap wrap">
            <button class="btn sm" type="button" onClick=${() => { setPick('csv'); file.current.accept = '.csv,text/csv'; file.current.dataset.fmt = 'csv'; file.current.click(); }}>Upload CSV</button>
            <button class="btn sm" type="button" onClick=${() => { file.current.accept = '.jsonl,.ndjson,.json,text/plain'; file.current.dataset.fmt = 'jsonl'; file.current.click(); }}>Upload JSONL</button>
            <input ref=${file} type="file" hidden onChange=${(e) => onFile(e, e.target.dataset.fmt || 'csv')} />
            <span class="muted small">or paste below${b.fileName ? ` · loaded ${b.fileName}` : ''}</span>
          </div>
          <div class="seg" role="radiogroup" aria-label="Input format">
            ${['csv', 'jsonl'].map((f) => html`<button key=${f} type="button" role="radio" aria-checked=${b.fmt === f} class=${`seg-btn ${b.fmt === f ? 'active' : ''}`} onClick=${() => upd({ fmt: f })}>${f.toUpperCase()}</button>`)}
          </div>
          <textarea class="input textarea mono" rows="8" aria-label="Batch input" spellcheck="false" disabled=${running}
            placeholder=${b.fmt === 'csv' ? 'id,text\n1,"Checkout is down, orders blocked"\n2,"How do I change my invoice address?"' : '{"text": "Checkout is down"}\n{"text": "Invoice question"}'}
            value=${b.text} onInput=${(e) => upd({ text: e.target.value, fileName: '' })}></textarea>
          ${perr ? html`<div class="errorbox small" role="alert">${perr}</div>` : html`<div class="muted small">${preview.items.length} row(s)${preview.columns.length ? ` · columns: ${preview.columns.join(', ')}` : ''}</div>`}
        </section>
        <section class="card"><h3>State per row</h3>
          <div class="seg" role="radiogroup" aria-label="State mode">
            <button type="button" role="radio" aria-checked=${b.stateMode === 'row'} class=${`seg-btn ${b.stateMode === 'row' ? 'active' : ''}`} onClick=${() => upd({ stateMode: 'row' })}>Row as JSON</button>
            <button type="button" role="radio" aria-checked=${b.stateMode === 'template'} class=${`seg-btn ${b.stateMode === 'template' ? 'active' : ''}`} onClick=${() => upd({ stateMode: 'template' })}>Text template</button>
          </div>
          ${b.stateMode === 'template' ? html`<label class="field"><span>Template <span class="muted">use {{column}}</span></span>
            <textarea class="input textarea mono" rows="3" value=${b.template} onInput=${(e) => upd({ template: e.target.value })}></textarea></label>` : null}
          ${preview.items[0] ? html`<div class="muted small">Preview of row 1</div><pre class="mono preview">${stateStr(stateFor(preview.items[0], b)).slice(0, 400)}</pre>` : null}
        </section>
        <section class="card"><h3>Schema and run</h3>
          <label class="field"><span>Schema</span>
            <select class="input" value=${b.schema} onChange=${(e) => upd({ schema: e.target.value })}>
              <option value="playground">Current Playground schema (${S.pg.cards.length} questions)</option>
              ${allPresets().map((p) => html`<option key=${p.name} value=${p.name}>${p.name}</option>`)}
            </select></label>
          <div class="muted small">Questions: ${qids.length ? qids.join(', ') : 'none'}</div>
          <label class="field inline"><span>Chunk size</span>
            <input class="input mono num-input" type="number" min="1" max=${maxBatch || 256} value=${b.chunk} onInput=${(e) => upd({ chunk: e.target.value })} /></label>
          <div class="row gap">
            ${running ? html`<button class="btn danger" type="button" onClick=${() => ctl && ctl.abort()}>Cancel</button>`
              : html`<button class="btn primary" type="button" disabled=${!preview.items.length || !!perr} onClick=${start}>Run batch</button>`}
            ${!running && failed ? html`<button class="btn" type="button" onClick=${retryFailed}>Retry ${failed} failed</button>` : null}
          </div>
          ${b.parseError ? html`<div class="errorbox small" role="alert">${b.parseError}</div>` : null}
          ${b.notice ? html`<div class="warn small">${b.notice}</div>` : null}
        </section>
      </div>
      <div class="pane right">
        <div class="pane-head"><h2>Results</h2>
          <div class="row gap">
            <button class="btn sm" type="button" disabled=${!b.rows.some((r) => r.response)} onClick=${() => exportCsv(schemaQuestions(b))}>Export CSV</button>
            <button class="btn sm" type="button" disabled=${!b.rows.some((r) => r.response)} onClick=${exportJsonl}>Export JSONL</button></div></div>
        ${b.total ? html`<div class="progress-wrap"><div class="progress" role="progressbar" aria-valuemin="0" aria-valuemax=${b.total} aria-valuenow=${b.done}>
            <div class="progress-fill" style=${{ width: `${(b.done / b.total) * 100}%` }}></div></div>
          <div class="muted small mono">${b.done}/${b.total} · ${b.status}${failed ? ` · ${failed} failed` : ''}</div></div>` : null}
        ${done.length ? html`<${BatchSummary} questions=${questions} rows=${b.rows} filter=${b.filter} onFilter=${(f) => upd({ filter: f })} />` : null}
        ${done.length ? html`<${CalibrationCard} questions=${questions} rows=${b.rows} columns=${preview.columns} />` : null}
        ${b.filter ? html`<div class="filter-bar" role="status"><span>Showing <strong>${visible.length}</strong> of ${b.rows.length} rows where <span class="mono">${b.filter.label}</span></span>
          <button class="btn sm" type="button" onClick=${() => upd({ filter: null })}>Clear filter</button></div>` : null}
        ${!b.rows.length ? html`<${Empty}><span class="glyph" aria-hidden="true">▤</span><strong>No batch yet</strong><span>Upload a CSV or JSONL file (or paste rows), then press Run batch. Summary charts appear here.</span><//>` : html`<div class="tablewrap"><table class="table">
          <thead><tr><th class="sortable" onClick=${() => setSort('#')} aria-sort=${b.sort.col === '#' ? (b.sort.dir > 0 ? 'ascending' : 'descending') : 'none'}>#${arrow('#')}</th><th>State</th>
            ${Object.keys(questions).map((id) => html`<th key=${id} class="sortable" tabIndex="0" onClick=${() => setSort(id)} onKeyDown=${(e) => e.key === 'Enter' && setSort(id)}
              aria-sort=${b.sort.col === id ? (b.sort.dir > 0 ? 'ascending' : 'descending') : 'none'}>${id}${arrow(id)}</th>`)}</tr></thead>
          <tbody>${sorted.map((r) => html`<tr key=${r.i} class="clickable" tabIndex="0" title="Open in Playground"
            onClick=${() => openInPlayground({ state: r.state, questions: questions, presetName: b.schema === 'playground' ? '' : b.schema, result: r.response, clientMs: null })}
            onKeyDown=${(e) => e.key === 'Enter' && openInPlayground({ state: r.state, questions, result: r.response })}>
            <td class="num mono muted">${r.i + 1}</td><td class="state-cell"><div class="clamp">${stateStr(r.state).slice(0, 160)}</div>${r.error ? html`<div class="err small">${r.error}</div>` : null}</td>
            ${Object.keys(questions).map((id) => {
              const a = r.response && r.response.answers && r.response.answers[id];
              const c = cellValue(a);
              return html`<td key=${id} class=${`mono ans-cell conf-${confLevel(c.conf)}`}>${c.text}</td>`;
            })}</tr>`)}</tbody></table></div>`}
      </div>
    </div>
  </div>`;
}

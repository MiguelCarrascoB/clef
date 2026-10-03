// History: saved runs, replay, side-by-side comparison with percentage-point deltas.
import { html, useState, useRef } from '../vendor/standalone.module.js';
import { S, set, useStore, removeHistory, clearHistory } from '../store.js';
import { fmtDateTime, topAnswer, fmtPct, fmtNum, confidenceOf, confLevel } from '../util.js';
import { Empty } from '../components/common.js';
import { LineChart, ExportMenu } from '../components/charts.js';
import { openInPlayground } from './playground.js';

const statePreview = (st) => {
  const s = typeof st === 'string' ? st : JSON.stringify(st);
  return s.length > 90 ? `${s.slice(0, 90)}…` : s;
};

function toggleCompare(id) {
  let c = S.compare.slice();
  if (c.includes(id)) c = c.filter((x) => x !== id);
  else { c.push(id); if (c.length > 2) c.shift(); }
  set({ compare: c });
}

function Delta({ a, b, unit = 'pp' }) {
  if (a == null || b == null) return html`<span class="muted mono">-</span>`;
  const d = unit === 'pp' ? (b - a) * 100 : b - a;
  const cls = Math.abs(d) < (unit === 'pp' ? 0.05 : 0.005) ? 'flat' : d > 0 ? 'up' : 'down';
  return html`<span class=${`delta mono ${cls}`}>${d > 0 ? '+' : ''}${d.toFixed(unit === 'pp' ? 1 : 2)}${unit === 'pp' ? ' pp' : ''}</span>`;
}

function CompareQuestion({ id, a, b }) {
  const kind = (a || b).type || 'noul';
  let rows = [];
  if (kind === 'noul') rows = [['P(true)', a && a.noul, b && b.noul]];
  else {
    const keys = [...new Set([...Object.keys((a && a.probabilities) || {}), ...Object.keys((b && b.probabilities) || {})])];
    if (kind === 'score') keys.sort((x, y) => Number(x) - Number(y));
    rows = keys.map((k) => [kind === 'score' && ((a || b).legend || {})[k] ? `${k} · ${(a || b).legend[k]}` : k,
      a && a.probabilities ? a.probabilities[k] : null, b && b.probabilities ? b.probabilities[k] : null]);
  }
  const heads = (x) => (x ? (kind === 'choice' ? x.choice : kind === 'score' ? fmtNum(x.score) : topAnswer(x)) : '-');
  return html`<div class="card cmp-q">
    <div class="qres-head"><div class="qres-title"><span class="mono qid">${id}</span><span class=${`chip type-${kind}`}>${kind}</span></div>
      <span class="muted small mono">${heads(a)} to ${heads(b)}${kind === 'score' && a && b ? html` <${Delta} a=${a.score} b=${b.score} unit="abs" />` : ''}</span></div>
    <table class="table cmp"><thead><tr><th>Option</th><th class="num">A</th><th class="num">B</th><th class="num">Delta</th></tr></thead>
      <tbody>${rows.map(([k, pa, pb]) => html`<tr key=${k}><td>${k}</td>
        <td class="num mono">${fmtPct(pa)}</td><td class="num mono">${fmtPct(pb)}</td><td class="num"><${Delta} a=${pa} b=${pb} /></td></tr>`)}</tbody></table>
  </div>`;
}

function Compare({ A, B }) {
  const ids = [...new Set([...Object.keys(A.response.answers || {}), ...Object.keys(B.response.answers || {})])];
  return html`<div class="compare">
    <div class="cmp-head">
      ${[['A', A], ['B', B]].map(([l, h]) => html`<div key=${l} class="card cmp-run"><strong>${l}</strong> <span class="muted small">${fmtDateTime(h.ts)} · ${h.preset || 'custom'}</span>
        <div class="small mono cmp-state">${statePreview(h.request.state)}</div></div>`)}
    </div>
    ${ids.map((id) => html`<${CompareQuestion} key=${id} id=${id} a=${A.response.answers[id]} b=${B.response.answers[id]} />`)}
  </div>`;
}

function Trend({ history }) {
  const runs = history.slice().reverse(); // chronological
  const qs = new Map();
  runs.forEach((h) => Object.entries(h.response.answers || {}).forEach(([id, a]) => { if (!qs.has(id)) qs.set(id, a.type || (a.noul != null ? 'noul' : 'score')); }));
  const [qSel, setQ] = useState('');
  const [oSel, setO] = useState('');
  const host = useRef(null);
  if (!qs.size || runs.length < 2) return null;
  const qid = qs.has(qSel) ? qSel : [...qs.keys()][0];
  const type = qs.get(qid);
  const opts = type === 'choice' ? [...new Set(runs.flatMap((h) => Object.keys(((h.response.answers || {})[qid] || {}).probabilities || {})))]
    : type === 'score' ? ['expected level'] : ['P(true)'];
  // default to the option the latest run chose (a flat 0% line for an unlikely option says little)
  const latestChoice = type === 'choice' ? ((runs[runs.length - 1].response.answers || {})[qid] || {}).choice : null;
  const opt = opts.includes(oSel) ? oSel : opts.includes(latestChoice) ? latestChoice : opts[0];
  const maxLevel = type === 'score' ? Math.max(1, ...runs.map((h) => Object.keys(((h.response.answers || {})[qid] || {}).probabilities || {}).length - 1)) : 1;
  const points = runs.map((h) => {
    const a = (h.response.answers || {})[qid];
    if (!a) return null;
    const y = type === 'choice' ? (a.probabilities || {})[opt] : type === 'score' ? a.score : a.noul;
    if (y == null) return null;
    return { x: h.ts, y, title: fmtDateTime(h.ts), note: h.preset || 'custom' };
  }).filter(Boolean);
  const d = (t) => new Date(t);
  const fmtX = (t) => `${d(t).getMonth() + 1}/${d(t).getDate()} ${String(d(t).getHours()).padStart(2, '0')}:${String(d(t).getMinutes()).padStart(2, '0')}`;
  return html`<section class="card"><div class="card-head"><h3>Trend<span class="sub">${type === 'choice' ? 'probability' : type === 'score' ? 'expected level' : 'P(true)'} of one question across runs</span></h3>
      <div class="row gap wrap"><select class="input" aria-label="Question" value=${qid} onChange=${(e) => { setQ(e.target.value); setO(''); }}>${[...qs.keys()].map((k) => html`<option key=${k} value=${k}>${k}</option>`)}</select>
        ${type === 'choice' ? html`<select class="input" aria-label="Option" value=${opt} onChange=${(e) => setO(e.target.value)}>${opts.map((k) => html`<option key=${k} value=${k}>${k}</option>`)}</select>` : null}
        <${ExportMenu} host=${host} name=${`clef-trend-${qid}`} /></div></div>
    <div ref=${host}><${LineChart} points=${points} yMin=${0} yMax=${maxLevel} height=${200} yLabel=${type === 'choice' ? `P(${opt})` : opt}
      fmtY=${type === 'score' ? (v) => fmtNum(v, 1) : (v) => fmtPct(v, 0)} fmtX=${fmtX}
      ariaLabel=${`Trend of ${qid}${type === 'choice' ? ` option ${opt}` : ''} across ${points.length} runs, latest ${points.length ? (type === 'score' ? fmtNum(points[points.length - 1].y) : fmtPct(points[points.length - 1].y)) : 'none'}`} /></div>
  </section>`;
}

export function History() {
  const { history, compare } = useStore();
  const sel = compare.map((id) => history.find((h) => h.id === id)).filter(Boolean);
  const replay = (h) => openInPlayground({
    state: h.request.state, questions: h.request.questions, presetName: h.preset, result: h.response, request: h.request, clientMs: h.clientMs,
  });
  return html`<div class="page">
    <div class="pane-head"><h2>History <span class="muted small">${history.length} / 100 runs, stored locally, media reduced to thumbnails</span></h2>
      <div class="row gap">
        <span class="muted small">${sel.length === 2 ? 'Comparing A and B' : 'Tick two runs to compare'}</span>
        <button class="btn sm" type="button" disabled=${!history.length} onClick=${() => { if (confirm('Delete all history?')) clearHistory(); }}>Clear all</button></div></div>
    ${sel.length === 2 ? html`<${Compare} A=${sel[0]} B=${sel[1]} />` : null}
    <${Trend} history=${history} />
    ${!history.length ? html`<${Empty}><span class="glyph" aria-hidden="true">◷</span><strong>No runs yet</strong><span>Runs from the Playground appear here, with a trend chart once you have two.</span><//>` : html`<div class="tablewrap"><table class="table hist">
      <thead><tr><th class="chk" aria-label="Compare"></th><th>Time</th><th>Preset</th><th>State</th><th>Top answers</th><th class="num">ms</th><th></th></tr></thead>
      <tbody>${history.map((h) => {
        const idx = compare.indexOf(h.id);
        return html`<tr key=${h.id} class=${idx >= 0 ? 'selected' : ''}>
          <td class="chk"><input type="checkbox" checked=${idx >= 0} aria-label="Select for compare" onChange=${() => toggleCompare(h.id)} />${idx >= 0 ? html`<span class="mono small ab">${'AB'[idx]}</span>` : null}</td>
          <td class="nowrap mono small">${fmtDateTime(h.ts)}</td>
          <td>${h.preset || html`<span class="muted">custom</span>`}</td>
          <td class="state-cell"><div class="clamp">${statePreview(h.request.state)}</div>
            ${h.media && h.media.length ? html`<div class="mini-thumbs">${h.media.map((m, i) => (m.thumb ? html`<img key=${i} src=${m.thumb} alt=${m.name} title=${m.name} />` : html`<span key=${i} class="chip">${m.kind}</span>`))}</div>` : null}</td>
          <td><div class="chips">${Object.entries(h.response.answers || {}).map(([id, a]) => html`<span key=${id} class=${`chip ans conf-${confLevel(confidenceOf(a))}`}><span class="muted">${id}</span> ${topAnswer(a)}</span>`)}</div></td>
          <td class="num mono">${h.clientMs != null ? Math.round(h.clientMs) : '-'}</td>
          <td class="nowrap"><button class="btn sm" type="button" onClick=${() => replay(h)}>Replay</button>
            <button class="btn icon sm" type="button" aria-label="Delete run" onClick=${() => removeHistory(h.id)}>✕</button></td>
        </tr>`;
      })}</tbody></table></div>`}
  </div>`;
}

// Batch summary charts (per question) and the calibration report.
import { html, useRef, useState } from '../vendor/standalone.module.js';
import { fmtPct, fmtNum, confidenceOf, binValues } from '../util.js';
import { BarChart, StackedBar, Legend, ExportMenu, Reliability, seriesVar } from './charts.js';

const INK = ['#fff', '#fff', '#0b0b0b', '#0b0b0b', '#0b0b0b', '#fff', '#fff', '#0b0b0b'];

function Block({ title, name, children, sub }) {
  const host = useRef(null);
  return html`<div class="sum-card"><h4><span>${title}${sub ? html`<span class="muted" style=${{ fontWeight: 400 }}> ${sub}</span>` : null}</span><${ExportMenu} host=${host} name=${name} /></h4><div ref=${host}>${children}</div></div>`;
}

const pctBinLabel = (b) => `${b.lo.toFixed(1)}`;

/** Build a filter object: rows (by index) belonging to a chart element. */
const mkFilter = (qid, key, label, idx) => ({ qid, key, label, ids: idx });

export function QuestionSummary({ qid, q, rows, filter, onFilter }) {
  const answers = rows.map((r) => ({ i: r.i, a: r.response && r.response.answers && r.response.answers[qid] })).filter((x) => x.a);
  if (!answers.length) return null;
  const type = q.type;
  const selKey = filter && filter.qid === qid ? filter.key : null;
  const pick = (kind) => (b) => onFilter(b ? mkFilter(qid, `${kind}:${b.id}`, `${qid} ${kind === 'conf' ? 'confidence' : kind === 'ptrue' ? 'P(true)' : 'score'} from ${b.label}`, b.idx) : null);
  const confBins = binValues(answers.map(({ i, a }) => ({ v: confidenceOf(a), i })), 0, 1, 10)
    .map((b, k) => ({ id: k, label: pctBinLabel(b), value: b.count, idx: b.idx, pct: fmtPct(b.count / answers.length, 0) }));
  const meanConf = answers.reduce((s, { a }) => s + (confidenceOf(a) || 0), 0) / answers.length;
  const confChart = html`<${Block} title="Confidence" name=${`clef-${qid}-confidence`} sub="rows per bin">
    <${BarChart} bins=${confBins} height=${130} selected=${selKey && selKey.startsWith('conf:') ? Number(selKey.slice(5)) : null} onSelect=${pick('conf')} unit="confidence" yLabel="rows" ariaLabel=${`Confidence histogram for ${qid}, ${answers.length} rows, mean ${fmtPct(meanConf, 0)}`} />
    <div class="hint">mean ${fmtPct(meanConf, 0)}</div><//>`;

  let main = null;
  if (type === 'choice') {
    const order = Object.keys(q.criteria || {});
    const counts = new Map(order.map((k) => [k, []]));
    answers.forEach(({ i, a }) => { if (!counts.has(a.choice)) counts.set(a.choice, []); counts.get(a.choice).push(i); });
    let ent = [...counts.entries()].map(([k, idx], j) => ({ id: k, label: k, value: idx.length, idx, color: seriesVar(j), ink: INK[j % 8] })).filter((s) => s.value > 0);
    ent.sort((x, y) => y.value - x.value);
    // series colors follow the entity's position in the schema, not its rank
    ent = ent.map((s) => { const j = Math.max(0, [...counts.keys()].indexOf(s.id)); return { ...s, color: j < 7 ? seriesVar(j) : 'var(--muted)', ink: j < 7 ? INK[j] : '#fff' }; });
    main = html`<${Block} title="Choice share" name=${`clef-${qid}-choice-share`} sub=${`${answers.length} rows`}>
      <${StackedBar} segments=${ent} selected=${selKey && selKey.startsWith('choice:') ? selKey.slice(7) : null}
        onSelect=${(s) => onFilter(s ? mkFilter(qid, `choice:${s.id}`, `${qid} = ${s.label}`, s.idx) : null)}
        ariaLabel=${`Choice share for ${qid}: ${ent.map((s) => `${s.label} ${Math.round((s.value / answers.length) * 100)}%`).join(', ')}`} />
      <div style=${{ marginTop: '8px' }}><${Legend} items=${ent.map((s) => ({ label: s.label, color: s.color, value: s.value }))}
        onToggle=${(l) => { const s = ent.find((x) => x.label === l); onFilter(selKey === `choice:${s.id}` ? null : mkFilter(qid, `choice:${s.id}`, `${qid} = ${s.label}`, s.idx)); }} off=${{}} /></div><//>`;
  } else if (type === 'score') {
    const n = (q.criteria || []).length || 2;
    const nb = Math.min(10, Math.max(2, (n - 1) * 2));
    const bins = binValues(answers.map(({ i, a }) => ({ v: a.score, i })), 0, Math.max(1, n - 1), nb)
      .map((b, k) => ({ id: k, label: b.lo.toFixed(1), value: b.count, idx: b.idx, pct: fmtPct(b.count / answers.length, 0) }));
    const mean = answers.reduce((s, { a }) => s + a.score, 0) / answers.length;
    main = html`<${Block} title="Score" name=${`clef-${qid}-score`} sub=${`expected level, mean ${fmtNum(mean)}`}>
      <${BarChart} bins=${bins} height=${130} selected=${selKey && selKey.startsWith('score:') ? Number(selKey.slice(6)) : null} onSelect=${pick('score')} unit="level" yLabel="rows" ariaLabel=${`Score histogram for ${qid}, mean ${fmtNum(mean)}`} /><//>`;
  } else {
    const bins = binValues(answers.map(({ i, a }) => ({ v: a.noul, i })), 0, 1, 10)
      .map((b, k) => ({ id: k, label: pctBinLabel(b), value: b.count, idx: b.idx, pct: fmtPct(b.count / answers.length, 0) }));
    const yes = answers.filter(({ a }) => a.noul >= 0.5).length;
    main = html`<${Block} title="P(true)" name=${`clef-${qid}-ptrue`} sub=${`${yes} of ${answers.length} true`}>
      <${BarChart} bins=${bins} height=${130} selected=${selKey && selKey.startsWith('ptrue:') ? Number(selKey.slice(6)) : null} onSelect=${pick('ptrue')} unit="P(true)" yLabel="rows" ariaLabel=${`P(true) histogram for ${qid}: ${yes} of ${answers.length} at or above 0.5`} /><//>`;
  }
  return html`<div style=${{ display: 'contents' }}>${main}${type === 'noul' ? null : confChart}</div>`;
}

export function BatchSummary({ questions, rows, filter, onFilter }) {
  const ids = Object.keys(questions);
  return html`<section class="card"><div class="card-head"><h3>Summary<span class="sub">click a bar or segment to filter the table</span></h3></div>
    ${ids.map((id) => html`<div key=${id} style=${{ marginBottom: '14px' }}>
      <div class="row gap" style=${{ marginBottom: '6px' }}><span class="mono qid">${id}</span><span class="chip">${questions[id].type}</span></div>
      <div class="sum-grid"><${QuestionSummary} qid=${id} q=${questions[id]} rows=${rows} filter=${filter} onFilter=${onFilter} /></div></div>`)}
  </section>`;
}

// ---- calibration ---------------------------------------------------------------------------------
const truthy = (v) => ['true', '1', 'yes', 'y', 't'].includes(String(v).trim().toLowerCase());
const falsy = (v) => ['false', '0', 'no', 'n', 'f'].includes(String(v).trim().toLowerCase());

export function calibrate(rows, qid, q, col) {
  const pts = []; // {conf, hit, brier}
  for (const r of rows) {
    const a = r.response && r.response.answers && r.response.answers[qid];
    const raw = r.item && typeof r.item === 'object' ? r.item[col] : undefined;
    if (!a || raw == null || raw === '') continue;
    if (q.type === 'noul') {
      let y; if (truthy(raw)) y = 1; else if (falsy(raw)) y = 0; else continue;
      pts.push({ conf: a.noul, hit: y, brier: (a.noul - y) ** 2 });
    } else if (q.type === 'choice') {
      const lab = String(raw).trim().toLowerCase();
      const keys = Object.keys(a.probabilities || {});
      if (!keys.some((k) => k.toLowerCase() === lab)) continue;
      const brier = keys.reduce((s, k) => s + ((a.probabilities[k] || 0) - (k.toLowerCase() === lab ? 1 : 0)) ** 2, 0);
      pts.push({ conf: a.probabilities[a.choice], hit: a.choice.toLowerCase() === lab ? 1 : 0, brier });
    }
  }
  const N = pts.length;
  const bins = Array.from({ length: 10 }, (_, i) => ({ label: `${(i / 10).toFixed(1)}-${((i + 1) / 10).toFixed(1)}`, n: 0, conf: 0, acc: 0 }));
  pts.forEach((p) => { const b = bins[Math.min(9, Math.floor(p.conf * 10))]; b.n++; b.conf += p.conf; b.acc += p.hit; });
  bins.forEach((b) => { if (b.n) { b.conf /= b.n; b.acc /= b.n; } });
  const ece = N ? bins.reduce((s, b) => s + (b.n / N) * Math.abs(b.acc - b.conf), 0) : null;
  const brier = N ? pts.reduce((s, p) => s + p.brier, 0) / N : null;
  const acc = N ? pts.reduce((s, p) => s + p.hit, 0) / N : null;
  return { N, bins, ece, brier, acc };
}

export function CalibrationCard({ questions, rows, columns }) {
  const eligible = Object.entries(questions).filter(([, q]) => q.type === 'choice' || q.type === 'noul');
  const [col, setCol] = useState('');
  const [qid, setQid] = useState('');
  const host = useRef(null);
  if (!columns.length || !eligible.length) return null;
  const q = qid && questions[qid] ? qid : eligible[0][0];
  const res = col ? calibrate(rows, q, questions[q], col) : null;
  return html`<section class="card"><div class="card-head"><h3>Calibration<span class="sub">compare predictions with a label column</span></h3>${res && res.N ? html`<${ExportMenu} host=${host} name="clef-reliability" />` : null}</div>
    <div class="row gap wrap">
      <label class="row gap small">Label column <select class="input" aria-label="Label column" value=${col} onChange=${(e) => setCol(e.target.value)}><option value="">none</option>${columns.map((c) => html`<option key=${c} value=${c}>${c}</option>`)}</select></label>
      <label class="row gap small">Question <select class="input" aria-label="Calibration question" value=${q} onChange=${(e) => setQid(e.target.value)}>${eligible.map(([id]) => html`<option key=${id} value=${id}>${id}</option>`)}</select></label>
    </div>
    ${!col ? html`<div class="hint" style=${{ marginTop: '8px' }}>Pick the CSV/JSONL column holding the true label (choice option name, or true/false for noul questions).</div>`
      : !res.N ? html`<div class="warn small" style=${{ marginTop: '8px' }}>No rows with a usable label in column "${col}" yet (needs completed rows whose label matches an option).</div>`
      : html`<div class="grid2" style=${{ marginTop: '10px', alignItems: 'center' }}>
        <div ref=${host}><${Reliability} bins=${res.bins} ariaLabel=${`Reliability diagram for ${q}: ECE ${res.ece.toFixed(3)}, Brier ${res.brier.toFixed(3)} over ${res.N} rows`} /></div>
        <div class="stat-row" style=${{ flexDirection: 'column' }}>
          <div class="it"><span class="k">ECE</span><span class="v mono">${res.ece.toFixed(3)}</span></div>
          <div class="it"><span class="k">Brier</span><span class="v mono">${res.brier.toFixed(3)}</span></div>
          <div class="it"><span class="k">Accuracy</span><span class="v mono">${fmtPct(res.acc, 1)}</span></div>
          <div class="it"><span class="k">Rows</span><span class="v mono">${res.N}</span></div>
        </div></div>`}
  </section>`;
}

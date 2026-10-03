// Result cards, metrics strip and the 5-tab results view.
import { html, useState } from '../vendor/standalone.module.js';
import { fmtPct, fmtNum, sortedProbs, snippets, truncateMedia } from '../util.js';
import { getKey } from '../api.js';
import { ConfBadge, Tabs, CodeBlock, ErrorBox, Empty } from './common.js';

function ChoiceResult({ a }) {
  const rows = sortedProbs(a);
  return html`<div class="bars">
    ${rows.map(([k, p]) => html`<div key=${k} class=${`bar-row ${k === a.choice ? 'top' : ''}`}>
      <div class="bar-label" title=${k}>${k}</div>
      <div class="bar-track" role="img" aria-label=${`${k} ${fmtPct(p)}`}><div class="bar-fill" style=${{ width: `${Math.max(0.5, p * 100)}%` }}></div></div>
      <div class="bar-val mono">${fmtPct(p)}</div>
    </div>`)}
  </div>`;
}

function ScoreResult({ a }) {
  const keys = Object.keys(a.probabilities || {}).sort((x, y) => Number(x) - Number(y));
  const n = keys.length;
  const W = 100, pad = n > 1 ? 100 / (n * 2) : 50;
  const xOf = (v) => (n > 1 ? pad + (v / (n - 1)) * (W - 2 * pad) : 50);
  const maxP = Math.max(...keys.map((k) => a.probabilities[k]), 0.001);
  return html`<div class="score">
    <div class="score-value"><span class="mono big">${fmtNum(a.score)}</span><span class="muted small"> expected level, 0 to ${n - 1}</span></div>
    <div class="score-axis" role="img" aria-label=${`Expected value ${fmtNum(a.score)} on a 0 to ${n - 1} axis`}>
      <svg viewBox="0 0 100 10" preserveAspectRatio="none" class="axis-svg">
        <line x1=${xOf(0)} x2=${xOf(n - 1)} y1="5" y2="5" class="axis-line" vector-effect="non-scaling-stroke"/>
        ${keys.map((k, i) => html`<line key=${k} x1=${xOf(i)} x2=${xOf(i)} y1="2.5" y2="7.5" class="axis-tick" vector-effect="non-scaling-stroke"/>`)}
      </svg>
      <div class="axis-marker" style=${{ left: `${xOf(a.score)}%` }}><span class="axis-marker-label mono">${fmtNum(a.score)}</span></div>
    </div>
    <div class="dist" style=${{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
      ${keys.map((k) => {
        const p = a.probabilities[k];
        const isTop = p === Math.max(...keys.map((x) => a.probabilities[x]));
        return html`<div key=${k} class=${`dist-col ${isTop ? 'top' : ''}`}>
          <div class="dist-pct mono">${fmtPct(p, 0)}</div>
          <div class="dist-bar-wrap"><div class="dist-bar" style=${{ height: `${(p / maxP) * 100}%` }}></div></div>
          <div class="dist-idx mono">${k}</div>
          <div class="dist-legend" title=${(a.legend || {})[k] || ''}>${(a.legend || {})[k] || ''}</div>
        </div>`;
      })}
    </div>
  </div>`;
}

function NoulResult({ a }) {
  const p = a.noul;
  return html`<div class="noul">
    <div class="noul-head"><span class="mono big">${fmtPct(p)}</span><span class="muted small"> P(true)</span>
      <span class=${`verdict ${p >= 0.5 ? 'yes' : 'no'}`}>${p >= 0.5 ? 'TRUE' : 'FALSE'}</span></div>
    <div class="noul-meter" role="meter" aria-valuenow=${p} aria-valuemin="0" aria-valuemax="1" aria-label="P(true)">
      <div class="noul-fill" style=${{ width: `${p * 100}%` }}></div><div class="noul-mid"></div>
    </div>
    <div class="noul-scale muted small mono"><span>0</span><span>0.5</span><span>1</span></div>
  </div>`;
}

export function QuestionResult({ id, a, question }) {
  if (!a) return null;
  const kind = a.type || (a.noul != null ? 'noul' : a.score != null ? 'score' : 'choice');
  const head = kind === 'choice' ? a.choice : kind === 'score' ? fmtNum(a.score) : (a.noul >= 0.5 ? 'true' : 'false');
  return html`<section class="card qres">
    <header class="qres-head">
      <div class="qres-title"><span class="mono qid">${id}</span><span class=${`chip type-${kind}`}>${kind}</span></div>
      <${ConfBadge} answer=${{ ...a, type: kind }} />
    </header>
    ${question && question.instructions ? html`<div class="muted small qinstr">${question.instructions}</div>` : null}
    <div class="qres-answer"><span class="answer">${head}</span></div>
    ${kind === 'choice' ? html`<${ChoiceResult} a=${a} />` : kind === 'score' ? html`<${ScoreResult} a=${a} />` : html`<${NoulResult} a=${a} />`}
  </section>`;
}

export function Metrics({ run }) {
  const r = run.result, t = r && r.timing;
  const item = (k, v, u) => html`<div class="metric"><span class="metric-k">${k}</span><span class="metric-v mono">${v == null ? '-' : v}${v != null && u ? html`<small>${u}</small>` : null}</span></div>`;
  const f = (n) => (n == null ? null : Number(n).toFixed(n < 10 ? 1 : 0));
  return html`<div class="metrics">
    ${item('client', run.clientMs != null ? run.clientMs.toFixed(0) : null, ' ms')}
    ${item('total', t ? f(t.total_ms) : null, ' ms')}
    ${item('forward', t ? f(t.forward_ms) : null, ' ms')}
    ${item('queue', t ? f(t.queue_ms) : null, ' ms')}
    ${item('batch', t ? t.batch_size : null)}
    ${item('tokens in', r && r.usage ? r.usage.input_tokens : null)}
  </div>`;
}

export function ResultsView({ run, questions }) {
  const [tab, setTab] = useState('cards');
  const { running, result, error, request, slow } = run;
  const tabs = [
    { id: 'cards', label: 'Cards' }, { id: 'request', label: 'Request JSON' },
    { id: 'response', label: 'Response JSON' }, { id: 'snippets', label: 'Snippets' },
  ];
  return html`<div class="results">
    <${Tabs} tabs=${tabs} value=${tab} onChange=${setTab} label="Result views" />
    ${running ? html`<div class="status-line" role="status"><span class="spinner"></span>Running…
      ${slow ? html`<span class="warn"> warming up / compiling kernels for a new input shape… this can take a minute the first time.</span>` : null}</div>` : null}
    <${ErrorBox} error=${error} />
    ${tab === 'cards' ? html`<div>
      ${result ? html`<${Metrics} run=${run} />` : null}
      ${!result && !error && !running ? html`<${Empty}>Press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> or Run to get calibrated probabilities.<//>` : null}
      <div class="qres-list">
        ${result ? Object.entries(result.answers || {}).map(([id, a]) => html`<${QuestionResult} key=${id} id=${id} a=${a} question=${(request && request.questions || questions || {})[id]} />`) : null}
      </div>
    </div>` : null}
    ${tab === 'request' ? (request ? html`<${CodeBlock} json text=${JSON.stringify(truncateMedia(request), null, 2)} />` : html`<${Empty}>No request yet.<//>`) : null}
    ${tab === 'response' ? (result ? html`<${CodeBlock} json text=${JSON.stringify(result, null, 2)} />` : html`<${Empty}>No response yet.<//>`) : null}
    ${tab === 'snippets' ? (request ? html`<${Snippets} request=${request} />` : html`<${Empty}>Run a request first.<//>`) : null}
  </div>`;
}

function Snippets({ request }) {
  const s = snippets(request, !!getKey());
  return html`<div class="snippets">
    ${[['curl', 'curl'], ['powershell', 'PowerShell Invoke-RestMethod'], ['python', 'Python requests']].map(([k, t]) => html`<div key=${k}>
      <div class="snip-title">${t}</div><${CodeBlock} text=${s[k]} /></div>`)}
    <div class="muted small">Base64 media is truncated as &lt;base64…&gt; in snippets.</div>
  </div>`;
}

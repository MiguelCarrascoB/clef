// Result cards, metrics strip and the 5-tab results view.
import { html, useState } from '../vendor/standalone.module.js';
import { fmtPct, fmtNum, sortedProbs, snippets, truncateMedia, answerEntropy } from '../util.js';
import { S, set, useStore } from '../store.js';
import { ProbBars, ScoreChart, Gauge, Uncertainty } from './charts.js';
import { getKey } from '../api.js';
import { ConfBadge, Tabs, CodeBlock, ErrorBox, Empty } from './common.js';

function ChoiceResult({ a, prev }) {
  const rows = sortedProbs(a);
  const pp = prev && prev.type === 'choice' && prev.probabilities ? prev.probabilities : null;
  return html`<div>
    <${ProbBars} entries=${rows} topKey=${a.choice} prev=${pp} />
    <${Uncertainty} ent=${answerEntropy(a)} />
  </div>`;
}

function ScoreResult({ a, prev }) {
  const n = Object.keys(a.probabilities || {}).length;
  const pv = prev && prev.type === 'score' && prev.probabilities ? prev : null;
  return html`<div class="score">
    <div class="muted small" style=${{ marginBottom: '4px' }}>Expected level on a 0 to ${n - 1} scale${pv ? ` (previous ${fmtNum(pv.score)})` : ''}</div>
    <${ScoreChart} probs=${a.probabilities} legend=${a.legend} score=${a.score} prevScore=${pv ? pv.score : null} prevProbs=${pv ? pv.probabilities : null} />
    <${Uncertainty} ent=${answerEntropy(a)} />
  </div>`;
}

function NoulResult({ a, prev }) {
  const p = a.noul;
  const pv = prev && prev.noul != null ? prev.noul : null;
  return html`<div class="noul">
    <div class="gauge-wrap">
      <${Gauge} value=${p} prev=${pv} />
      <div>
        <span class=${`verdict ${p >= 0.5 ? 'yes' : 'no'}`}>${p >= 0.5 ? 'TRUE' : 'FALSE'}</span>
        ${pv != null ? html`<div class="small muted" style=${{ marginTop: '6px' }}>previous run ${fmtPct(pv)}</div>` : null}
        <${Uncertainty} ent=${answerEntropy(a)} />
      </div>
    </div>
  </div>`;
}

export function QuestionResult({ id, a, question, prev }) {
  if (!a) return null;
  const kind = a.type || (a.noul != null ? 'noul' : a.score != null ? 'score' : 'choice');
  const head = kind === 'choice' ? a.choice : kind === 'score' ? fmtNum(a.score) : (a.noul >= 0.5 ? 'true' : 'false');
  const p = prev ? { type: prev.type || kind, ...prev } : null;
  return html`<section class="card qres">
    <header class="qres-head">
      <div class="qres-title"><span class="mono qid">${id}</span><span class=${`chip type-${kind}`}>${kind}</span></div>
      <${ConfBadge} answer=${{ ...a, type: kind }} />
    </header>
    ${question && question.instructions ? html`<div class="muted small qinstr">${question.instructions}</div>` : null}
    <div class="qres-answer"><span class="answer">${head}</span></div>
    ${kind === 'choice' ? html`<${ChoiceResult} a=${a} prev=${p} />` : kind === 'score' ? html`<${ScoreResult} a=${a} prev=${p} />` : html`<${NoulResult} a=${a} prev=${p} />`}
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
  const { prevResult, showPrev } = useStore();
  const prevAnswers = showPrev && prevResult && prevResult.answers ? prevResult.answers : null;
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
      ${result && prevResult ? html`<label class="row gap small" style=${{ margin: '8px 0 0', cursor: 'pointer' }}><input type="checkbox" checked=${showPrev} onChange=${(e) => set({ showPrev: e.target.checked })} />
        <span>Overlay previous run</span><span class="muted">(ticks and deltas)</span></label>` : null}
      ${!result && !error && !running ? html`<${Empty}><span class="glyph" aria-hidden="true">♭</span><strong>No result yet</strong><span>Press <kbd>Ctrl</kbd>+<kbd>Enter</kbd> or Run to get calibrated probabilities.</span><//>` : null}
      ${running && !result ? html`<div class="skel" style=${{ height: '170px', marginTop: '10px' }} aria-hidden="true"></div>` : null}
      <div class="qres-list">
        ${result ? Object.entries(result.answers || {}).map(([id, a]) => html`<${QuestionResult} key=${id} id=${id} a=${a} prev=${prevAnswers ? prevAnswers[id] : null} question=${(request && request.questions || questions || {})[id]} />`) : null}
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

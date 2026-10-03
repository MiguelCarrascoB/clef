// Schema builder: question cards + two-way "Schema JSON" tab, with inline validation.
import { html, useState, useEffect } from '../vendor/standalone.module.js';
import {
  newQuestion, cardsToContract, contractToCards, validateCards, validateQuestions, errCount, parseJson,
} from '../util.js';
import { Tabs } from './common.js';

const move = (arr, i, d) => {
  const j = i + d;
  if (j < 0 || j >= arr.length) return arr;
  const c = arr.slice(); [c[i], c[j]] = [c[j], c[i]]; return c;
};

function QuestionCard({ c, index, count, errs, onChange, onMove, onRemove }) {
  const upd = (patch) => onChange({ ...c, ...patch });
  const setType = (type) => {
    const n = newQuestion(type);
    onChange({ ...n, uid: c.uid, id: c.id, instructions: c.instructions });
  };
  const bad = errs && errs.length;
  return html`<div class=${`qcard ${bad ? 'invalid' : ''}`}>
    <div class="qcard-head">
      <input class="input mono id-input" aria-label="Question id" placeholder="question_id" value=${c.id}
        onInput=${(e) => upd({ id: e.target.value })} />
      <select class="input type-select" aria-label="Question type" value=${c.type} onChange=${(e) => setType(e.target.value)}>
        <option value="choice">choice</option><option value="score">score</option><option value="noul">noul (true/false)</option>
      </select>
      <div class="qcard-actions">
        <button class="btn icon sm" type="button" aria-label="Move question up" disabled=${index === 0} onClick=${() => onMove(-1)}>↑</button>
        <button class="btn icon sm" type="button" aria-label="Move question down" disabled=${index === count - 1} onClick=${() => onMove(1)}>↓</button>
        <button class="btn icon sm danger" type="button" aria-label="Delete question" onClick=${onRemove}>✕</button>
      </div>
    </div>
    <input class="input" aria-label="Instructions" placeholder="Instructions (optional)" value=${c.instructions}
      onInput=${(e) => upd({ instructions: e.target.value })} />
    ${c.type === 'choice' ? html`<div class="crit">
      <div class="crit-title muted small">Options (key to description)</div>
      ${c.options.map((o, i) => html`<div class="crit-row" key=${i}>
        <input class="input mono key" aria-label=${`Option ${i + 1} key`} placeholder="option" value=${o.k}
          onInput=${(e) => upd({ options: c.options.map((x, j) => (j === i ? { ...x, k: e.target.value } : x)) })} />
        <input class="input" aria-label=${`Option ${i + 1} description`} placeholder="description" value=${o.d}
          onInput=${(e) => upd({ options: c.options.map((x, j) => (j === i ? { ...x, d: e.target.value } : x)) })} />
        <button class="btn icon sm" type="button" aria-label=${`Remove option ${i + 1}`} disabled=${c.options.length <= 1}
          onClick=${() => upd({ options: c.options.filter((_, j) => j !== i) })}>✕</button>
      </div>`)}
      <button class="btn sm" type="button" onClick=${() => upd({ options: [...c.options, { k: '', d: '' }] })}>+ Option</button>
    </div>` : null}
    ${c.type === 'score' ? html`<div class="crit">
      <div class="crit-title muted small">Levels (ordered, first = score 0)</div>
      ${c.levels.map((l, i) => html`<div class="crit-row" key=${i}>
        <span class="lvl-idx mono">${i}</span>
        <input class="input" aria-label=${`Level ${i} text`} placeholder=${`level ${i}`} value=${l}
          onInput=${(e) => upd({ levels: c.levels.map((x, j) => (j === i ? e.target.value : x)) })} />
        <button class="btn icon sm" type="button" aria-label=${`Move level ${i} up`} disabled=${i === 0} onClick=${() => upd({ levels: move(c.levels, i, -1) })}>↑</button>
        <button class="btn icon sm" type="button" aria-label=${`Move level ${i} down`} disabled=${i === c.levels.length - 1} onClick=${() => upd({ levels: move(c.levels, i, 1) })}>↓</button>
        <button class="btn icon sm" type="button" aria-label=${`Remove level ${i}`} disabled=${c.levels.length <= 1}
          onClick=${() => upd({ levels: c.levels.filter((_, j) => j !== i) })}>✕</button>
      </div>`)}
      <button class="btn sm" type="button" onClick=${() => upd({ levels: [...c.levels, ''] })}>+ Level</button>
    </div>` : null}
    ${c.type === 'noul' ? html`<div class="crit">
      <div class="crit-title muted small">Descriptions (optional)</div>
      <div class="crit-row"><span class="lvl-idx mono">true</span>
        <input class="input" aria-label="True description" placeholder="when is it true?" value=${c.trueDesc} onInput=${(e) => upd({ trueDesc: e.target.value })} /></div>
      <div class="crit-row"><span class="lvl-idx mono">false</span>
        <input class="input" aria-label="False description" placeholder="when is it false?" value=${c.falseDesc} onInput=${(e) => upd({ falseDesc: e.target.value })} /></div>
    </div>` : null}
    ${bad ? html`<ul class="errlist" role="alert">${errs.map((m) => html`<li key=${m}>${m}</li>`)}</ul>` : null}
  </div>`;
}

export function SchemaBuilder({ cards, onChange, limits }) {
  const [tab, setTab] = useState('builder');
  const errs = validateCards(cards, limits);
  const n = errCount(errs);
  const [jsonText, setJsonText] = useState('');
  const [jsonErr, setJsonErr] = useState(null);
  const [dirty, setDirty] = useState(false);

  // Regenerate the JSON text when cards change from elsewhere (not while the user types in the JSON tab).
  useEffect(() => {
    if (!dirty) { setJsonText(JSON.stringify(cardsToContract(cards), null, 2)); setJsonErr(null); }
  }, [cards, tab, dirty]);

  const onJson = (text) => {
    setJsonText(text); setDirty(true);
    const r = parseJson(text);
    if (r.error) { setJsonErr({ message: r.error, line: r.line, col: r.col }); return; }
    if (!r.value || typeof r.value !== 'object' || Array.isArray(r.value)) { setJsonErr({ message: 'questions must be a JSON object' }); return; }
    setJsonErr(null);
    onChange(contractToCards(r.value));
  };
  const jsonContractErrs = jsonErr ? null : validateQuestions(parseJson(jsonText).value, limits);
  const tabs = [{ id: 'builder', label: 'Builder', badge: n ? `${n}!` : cards.length }, { id: 'json', label: 'Schema JSON' }];

  return html`<div class="schema">
    <${Tabs} tabs=${tabs} value=${tab} onChange=${(t) => { setTab(t); setDirty(false); }} label="Schema editor" />
    ${tab === 'builder' ? html`<div class="schema-body">
      ${errs[''] ? html`<ul class="errlist" role="alert">${errs[''].map((m) => html`<li key=${m}>${m}</li>`)}</ul>` : null}
      ${cards.map((c, i) => html`<${QuestionCard} key=${c.uid} c=${c} index=${i} count=${cards.length} errs=${errs[c.id.trim()]}
        onChange=${(nc) => onChange(cards.map((x, j) => (j === i ? nc : x)))}
        onMove=${(d) => onChange(move(cards, i, d))}
        onRemove=${() => onChange(cards.filter((_, j) => j !== i))} />`)}
      <div class="row gap">
        ${['choice', 'score', 'noul'].map((t) => html`<button key=${t} class="btn sm" type="button" onClick=${() => onChange([...cards, newQuestion(t)])}>+ ${t}</button>`)}
      </div>
    </div>` : html`<div class="schema-body">
      <textarea class="input mono textarea json-area" aria-label="Schema JSON" spellcheck="false" value=${jsonText}
        onInput=${(e) => onJson(e.target.value)}></textarea>
      ${jsonErr ? html`<div class="errorbox small" role="alert">Invalid JSON: ${jsonErr.message}${jsonErr.line ? ` (line ${jsonErr.line}, col ${jsonErr.col})` : ''}</div>` : null}
      ${jsonContractErrs && errCount(jsonContractErrs) ? html`<ul class="errlist" role="alert">
        ${Object.entries(jsonContractErrs).flatMap(([id, ms]) => ms.map((m) => html`<li key=${id + m}>${id ? html`<span class="mono">${id}</span>: ` : ''}${m}</li>`))}</ul>` : null}
      <div class="muted small">Edits apply to the builder live. Shape: {"id": {"type": "choice|score|noul", "instructions"?, "criteria"?}}</div>
    </div>`}
  </div>`;
}

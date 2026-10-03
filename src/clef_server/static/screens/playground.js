// Playground: state editor + media + schema builder + presets (left) and results (right).
import { html, useState, useEffect, useRef } from '../vendor/standalone.module.js';
import * as api from '../api.js';
import {
  S, set, setPg, useStore, allPresets, savePreset, deletePreset, addHistory, goTab,
} from '../store.js';
import {
  parseJson, buildRequest, cardsToContract, contractToCards, validateCards, errCount, truncateMedia,
} from '../util.js';
import { MediaZone } from '../components/media.js';
import { SchemaBuilder } from '../components/schema.js';
import { ResultsView } from '../components/results.js';

let abortCtl = null;

function stripMedia(req) {
  const c = truncateMedia(req);
  delete c.images; delete c.videos;
  return c;
}

export async function runPlayground() {
  if (S.run.running) return;
  const { pg } = S;
  const questions = cardsToContract(pg.cards);
  const errs = validateCards(pg.cards, S.limits || {});
  if (errCount(errs)) {
    set({ run: { ...S.run, error: new api.ApiError('Fix the schema validation errors first.'), result: null, running: false } });
    return;
  }
  let request;
  try { request = buildRequest({ stateMode: pg.stateMode, stateText: pg.stateText, media: pg.media, questions }); } catch (e) {
    set({ run: { ...S.run, error: new api.ApiError(e.message), result: null, running: false } });
    return;
  }
  abortCtl = new AbortController();
  set({ run: { running: true, result: null, error: null, request, clientMs: null, slow: false } });
  const slowTimer = setTimeout(() => { if (S.run.running) set({ run: { ...S.run, slow: true } }); }, 5000);
  const t0 = performance.now();
  try {
    const { data } = await api.systemone(request, abortCtl.signal);
    const clientMs = performance.now() - t0;
    set({ run: { running: false, result: data, error: null, request, clientMs, slow: false } });
    addHistory({
      preset: pg.presetName || '', request: stripMedia(request), response: data, clientMs,
      media: pg.media.map((m) => ({ name: m.name, kind: m.kind, size: m.size, thumb: m.thumb || null })),
    });
  } catch (e) {
    if (e && e.name === 'AbortError') set({ run: { ...S.run, running: false, slow: false } });
    else set({ run: { running: false, result: null, error: e, request, clientMs: performance.now() - t0, slow: false } });
  } finally { clearTimeout(slowTimer); abortCtl = null; }
}

/** Load a record (state + questions) into the Playground and switch to it. */
export function openInPlayground({ state, questions, presetName = '', result = null, request = null, clientMs = null, media = [] }) {
  const isObj = state !== null && typeof state === 'object';
  setPg({
    stateMode: isObj ? 'json' : 'text',
    stateText: isObj ? JSON.stringify(state, null, 2) : String(state ?? ''),
    cards: contractToCards(questions || {}),
    media: media, presetName,
  });
  set({ run: { running: false, result, error: null, request: request || (result ? { model: 'clef-flash', state, questions } : null), clientMs, slow: false } });
  goTab('playground');
}

function StateEditor() {
  const { pg } = useStore();
  const isJson = pg.stateMode === 'json';
  const parsed = isJson ? parseJson(pg.stateText) : null;
  const format = () => { if (parsed && !parsed.error) setPg({ stateText: JSON.stringify(parsed.value, null, 2) }); };
  return html`<div class="state-editor">
    <div class="row between">
      <div class="seg" role="radiogroup" aria-label="State format">
        ${['text', 'json'].map((m) => html`<button key=${m} type="button" role="radio" aria-checked=${pg.stateMode === m}
          class=${`seg-btn ${pg.stateMode === m ? 'active' : ''}`} onClick=${() => setPg({ stateMode: m })}>${m === 'text' ? 'Text' : 'JSON'}</button>`)}
      </div>
      ${isJson ? html`<button class="btn sm" type="button" onClick=${format} disabled=${!parsed || !!parsed.error}>Format</button>` : null}
    </div>
    <textarea class=${`input textarea ${isJson ? 'mono' : ''} ${parsed && parsed.error ? 'invalid' : ''}`} aria-label="State" rows="7" spellcheck=${!isJson}
      placeholder=${isJson ? '{"ticket": {"text": "..."}}' : 'Describe the situation the model should judge…'}
      value=${pg.stateText} onInput=${(e) => setPg({ stateText: e.target.value })}></textarea>
    ${parsed && parsed.error ? html`<div class="errorbox small" role="alert">Invalid JSON: ${parsed.error}${parsed.line ? ` (line ${parsed.line}, col ${parsed.col})` : ''}</div>` : null}
    ${isJson && parsed && !parsed.error ? html`<div class="ok small">Valid JSON</div>` : null}
  </div>`;
}

function Presets() {
  const { pg, presets } = useStore();
  const [sel, setSel] = useState('');
  const [saving, setSaving] = useState(false);
  const [name, setName] = useState('');
  const [note, setNote] = useState(null);
  const list = allPresets();
  const cur = list.find((p) => p.name === sel);
  const load = () => {
    if (!cur) return;
    setPg({ stateMode: cur.stateMode || 'text', stateText: cur.stateText || '', cards: contractToCards(cur.questions), presetName: cur.name, media: [] });
    set({ run: { running: false, result: null, error: null, request: null, clientMs: null, slow: false } });
  };
  const doSave = () => {
    const n = name.trim();
    if (!n) return;
    savePreset(n, { stateMode: pg.stateMode, stateText: pg.stateText, questions: cardsToContract(pg.cards) });
    setPg({ presetName: n }); setSel(n); setSaving(false); setName('');
  };
  const example = async () => {
    try {
      const { data } = await api.schemaExample();
      const ex = data && (data.text_json || data.example || data);
      if (!ex || !ex.questions) throw new Error('unexpected /schema-example shape');
      const isObj = ex.state && typeof ex.state === 'object';
      setPg({
        stateMode: isObj ? 'json' : 'text', stateText: isObj ? JSON.stringify(ex.state, null, 2) : String(ex.state ?? ''),
        cards: contractToCards(ex.questions), presetName: 'example', media: [],
      });
      setNote(null);
    } catch (e) { setNote(`Could not load example: ${e.message}`); }
  };
  return html`<div class="presets">
    <div class="row gap wrap">
      <select class="input grow" aria-label="Preset" value=${sel} onChange=${(e) => setSel(e.target.value)}>
        <option value="">Presets…</option>
        <optgroup label="Built-in">${list.filter((p) => p.builtin).map((p) => html`<option key=${p.name} value=${p.name}>${p.name}</option>`)}</optgroup>
        ${presets.length ? html`<optgroup label="Saved">${presets.map((p) => html`<option key=${p.name} value=${p.name}>${p.name}</option>`)}</optgroup>` : null}
      </select>
      <button class="btn sm" type="button" disabled=${!cur} onClick=${load}>Load</button>
      <button class="btn sm" type="button" disabled=${!cur || cur.builtin} onClick=${() => { deletePreset(sel); setSel(''); }}>Delete</button>
      <button class="btn sm" type="button" onClick=${() => { setSaving(!saving); setName(pg.presetName && !allPresets().find((p) => p.name === pg.presetName && p.builtin) ? pg.presetName : ''); }}>Save as…</button>
      <button class="btn sm" type="button" onClick=${example}>Load example</button>
    </div>
    ${saving ? html`<div class="row gap">
      <input class="input grow" aria-label="Preset name" placeholder="Preset name" value=${name} autoFocus
        onInput=${(e) => setName(e.target.value)} onKeyDown=${(e) => e.key === 'Enter' && doSave()} />
      <button class="btn primary sm" type="button" onClick=${doSave} disabled=${!name.trim()}>Save</button></div>` : null}
    ${note ? html`<div class="warn small">${note}</div>` : null}
  </div>`;
}

export function Playground() {
  const { pg, run, limits } = useStore();
  const root = useRef(null);
  useEffect(() => {
    const h = (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); runPlayground(); } };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, []);
  const errs = validateCards(pg.cards, limits || {});
  const stateBad = pg.stateMode === 'json' && parseJson(pg.stateText).error;
  const blocked = errCount(errs) > 0 || !!stateBad;
  return html`<div class="split" ref=${root}>
    <div class="pane left">
      <div class="pane-head">
        <h2>Request <span class="muted small">${pg.presetName ? `· ${pg.presetName}` : ''}</span></h2>
        <div class="row gap">
          ${run.running ? html`<button class="btn sm" type="button" onClick=${() => abortCtl && abortCtl.abort()}>Cancel</button>` : null}
          <button class="btn primary" type="button" disabled=${run.running || blocked} onClick=${runPlayground}
            title=${blocked ? 'Fix validation errors first' : 'Run (Ctrl+Enter)'}>${run.running ? 'Running…' : 'Run'} <kbd>Ctrl+Enter</kbd></button>
        </div>
      </div>
      <section class="card"><h3>Presets</h3><${Presets} /></section>
      <section class="card"><h3>State</h3><${StateEditor} /></section>
      <section class="card"><h3>Media <span class="muted small">optional</span></h3>
        <${MediaZone} media=${pg.media} limits=${limits} onChange=${(media) => setPg({ media })} /></section>
      <section class="card"><h3>Schema</h3>
        <${SchemaBuilder} cards=${pg.cards} limits=${limits} onChange=${(cards) => setPg({ cards })} /></section>
    </div>
    <div class="pane right">
      <div class="pane-head"><h2>Results</h2></div>
      <${ResultsView} run=${run} questions=${cardsToContract(pg.cards)} />
    </div>
  </div>`;
}

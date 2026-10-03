// Tiny shared store (module singleton) + localStorage-backed presets / history / theme.
import { useState, useEffect } from './vendor/standalone.module.js';
import { contractToCards, cardsToContract, uid } from './util.js';

const lsGet = (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch { return false; } };

// ---- built-in presets -----------------------------------------------------------------------
export const BUILTIN_PRESETS = [
  {
    name: 'Support ticket triage', builtin: true, stateMode: 'text',
    stateText: 'Our checkout started returning errors about 20 minutes ago and orders are blocked. About 1,200 customers are affected and we are losing sales.',
    questions: {
      department: { type: 'choice', instructions: 'Which team should handle the message?', criteria: { billing: 'Payments or invoices', technical: 'Bugs or outages', account: 'Login or profile problems', sales: 'Pricing and plans' } },
      urgency: { type: 'score', instructions: 'How urgent is this?', criteria: ['Can wait', 'This week', 'Today'] },
      outage: { type: 'noul', instructions: 'Is a service down?' },
    },
  },
  {
    name: 'Content moderation', builtin: true, stateMode: 'text',
    stateText: 'You people are idiots. Click here to claim your free prize: http://totally-legit.example/win',
    questions: {
      category: { type: 'choice', instructions: 'Primary policy category of the message.', criteria: { safe: 'Acceptable content', harassment: 'Insults or targeted abuse', spam: 'Unsolicited promotion or scams', hate: 'Attacks on protected groups', violence: 'Threats or glorification of violence' } },
      severity: { type: 'score', instructions: 'How severe is the violation?', criteria: ['None', 'Mild - warn', 'Moderate - remove', 'Severe - ban'] },
      needs_human: { type: 'noul', instructions: 'Should a human moderator review this?', criteria: { true: 'Ambiguous or high-impact case', false: 'Clear-cut case' } },
    },
  },
  {
    name: 'Receipt image check', builtin: true, stateMode: 'text',
    stateText: 'Review the attached receipt image for an expense report.',
    questions: {
      legible: { type: 'noul', instructions: 'Is the total amount clearly legible?' },
      doc_type: { type: 'choice', instructions: 'What kind of document is this?', criteria: { receipt: 'Point-of-sale receipt', invoice: 'Formal invoice', other: 'Something else' } },
      quality: { type: 'score', instructions: 'Image quality for automated processing.', criteria: ['Unusable', 'Poor', 'Acceptable', 'Good'] },
    },
  },
];

// ---- state ----------------------------------------------------------------------------------
const draft = lsGet('clef.draft', null);
const first = BUILTIN_PRESETS[0];

export const S = {
  tab: (location.hash || '').replace('#', '') || 'playground',
  theme: lsGet('clef.theme', 'auto'),
  health: { state: 'loading', data: null, error: null },
  limits: null,
  pg: {
    stateMode: (draft && draft.stateMode) || first.stateMode,
    stateText: draft ? draft.stateText : first.stateText,
    media: [],
    cards: contractToCards(draft ? draft.questions : first.questions),
    presetName: (draft && draft.presetName) || first.name,
  },
  run: { running: false, result: null, error: null, request: null, clientMs: null, slow: false },
  compare: [],
  presets: lsGet('clef.presets', []),
  history: lsGet('clef.history', []),
  batchSchema: 'playground',
};

const listeners = new Set();
export function notify() { listeners.forEach((f) => f()); }
export function set(patch) { Object.assign(S, patch); notify(); }
export function setPg(patch) {
  Object.assign(S.pg, patch);
  notify();
  scheduleDraftSave();
}
export function useStore() {
  const [, bump] = useState(0);
  useEffect(() => {
    const f = () => bump((x) => x + 1);
    listeners.add(f);
    return () => listeners.delete(f);
  }, []);
  return S;
}

let draftTimer = null;
function scheduleDraftSave() {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => {
    lsSet('clef.draft', {
      stateMode: S.pg.stateMode, stateText: S.pg.stateText, presetName: S.pg.presetName,
      questions: cardsToContract(S.pg.cards),
    });
  }, 400);
}

// ---- theme ----------------------------------------------------------------------------------
export function applyTheme(t = S.theme) {
  const el = document.documentElement;
  if (t === 'light' || t === 'dark') el.setAttribute('data-theme', t); else el.removeAttribute('data-theme');
}
export function setTheme(t) { lsSet('clef.theme', t); set({ theme: t }); applyTheme(t); }

// ---- tabs -----------------------------------------------------------------------------------
export function goTab(tab) {
  if (location.hash !== `#${tab}`) history.replaceState(null, '', `#${tab}`);
  set({ tab });
}

// ---- presets --------------------------------------------------------------------------------
export const allPresets = () => [...BUILTIN_PRESETS, ...S.presets];
export function savePreset(name, preset) {
  const p = { ...preset, name, builtin: false };
  const list = S.presets.filter((x) => x.name !== name);
  list.push(p);
  lsSet('clef.presets', list);
  set({ presets: list });
}
export function deletePreset(name) {
  const list = S.presets.filter((x) => x.name !== name);
  lsSet('clef.presets', list);
  set({ presets: list });
}

// ---- history --------------------------------------------------------------------------------
export function addHistory(entry) {
  let list = [{ id: uid(), ts: Date.now(), ...entry }, ...S.history].slice(0, 100);
  while (!lsSet('clef.history', list) && list.length > 1) list = list.slice(0, Math.max(1, Math.floor(list.length * 0.7)));
  set({ history: list });
}
export function removeHistory(id) {
  const list = S.history.filter((h) => h.id !== id);
  lsSet('clef.history', list);
  set({ history: list });
}
export function clearHistory() { lsSet('clef.history', []); set({ history: [], compare: [] }); }

/** Make a ~64px JPEG thumbnail from an image data URL. */
export function makeThumb(dataUrl, size = 64) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      try {
        const k = Math.min(1, size / Math.max(img.width, img.height));
        const c = document.createElement('canvas');
        c.width = Math.max(1, Math.round(img.width * k)); c.height = Math.max(1, Math.round(img.height * k));
        c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
        resolve(c.toDataURL('image/jpeg', 0.6));
      } catch { resolve(null); }
    };
    img.onerror = () => resolve(null);
    img.src = dataUrl;
  });
}

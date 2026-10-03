// Pure helpers: formatting, schema <-> contract conversion/validation, CSV, snippets.
export const uid = () => Math.random().toString(36).slice(2, 10);

export const fmtPct = (p, d = 1) => (p == null || Number.isNaN(p) ? '-' : `${(p * 100).toFixed(d)}%`);
export const fmtNum = (n, d = 2) => (n == null || Number.isNaN(n) ? '-' : Number(n).toFixed(d));
export function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1048576).toFixed(2)} MB`;
}
export function fmtDur(s) {
  if (s == null) return '-';
  s = Math.round(s);
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s % 60}s`;
  return `${s}s`;
}
export const fmtTime = (ts) => new Date(ts).toLocaleTimeString([], { hour12: false });
export const fmtDateTime = (ts) => new Date(ts).toLocaleString([], { hour12: false });

// ---- answers -----------------------------------------------------------------------------
export function confidenceOf(a) {
  if (!a) return null;
  if (a.type === 'noul' || (a.noul != null && a.confidence == null)) return Math.abs(a.noul - 0.5) * 2;
  return a.confidence;
}
export function confLevel(c) {
  if (c == null) return 'na';
  if (c >= 0.85) return 'high';
  if (c >= 0.6) return 'mid';
  return 'low';
}
export function sortedProbs(a) {
  const p = a.probabilities || {};
  return Object.entries(p).sort((x, y) => y[1] - x[1]);
}
/** One-line summary of an answer: "technical 96%", "1.78", "P 84%" */
export function topAnswer(a) {
  if (!a) return '-';
  if (a.type === 'choice') {
    const c = a.choice, p = a.probabilities && a.probabilities[c];
    return `${c} ${fmtPct(p, 0)}`;
  }
  if (a.type === 'score') return `${fmtNum(a.score)}`;
  if (a.type === 'noul' || a.noul != null) return `P ${fmtPct(a.noul, 0)}`;
  return '-';
}

// ---- schema cards <-> contract --------------------------------------------------------------
export function newQuestion(type = 'choice') {
  return {
    uid: uid(), id: '', type, instructions: '',
    options: type === 'choice' ? [{ k: '', d: '' }, { k: '', d: '' }] : [],
    levels: type === 'score' ? ['', ''] : [],
    trueDesc: '', falseDesc: '',
  };
}

export function cardsToContract(cards) {
  const out = {};
  for (const c of cards) {
    const q = { type: c.type };
    if (c.instructions && c.instructions.trim()) q.instructions = c.instructions;
    if (c.type === 'choice') {
      q.criteria = {};
      for (const o of c.options) if (o.k.trim() || o.d.trim()) q.criteria[o.k.trim()] = o.d;
    } else if (c.type === 'score') {
      q.criteria = c.levels.slice();
    } else if (c.type === 'noul') {
      const cr = {};
      if (c.trueDesc.trim()) cr.true = c.trueDesc;
      if (c.falseDesc.trim()) cr.false = c.falseDesc;
      if (Object.keys(cr).length) q.criteria = cr;
    }
    out[c.id.trim()] = q;
  }
  return out;
}

export function contractToCards(questions) {
  const cards = [];
  for (const [id, q] of Object.entries(questions || {})) {
    const c = newQuestion(q && q.type === 'score' ? 'score' : q && q.type === 'noul' ? 'noul' : 'choice');
    c.id = id;
    c.instructions = (q && q.instructions) || '';
    const cr = q && q.criteria;
    if (c.type === 'choice') {
      c.options = cr && typeof cr === 'object' && !Array.isArray(cr)
        ? Object.entries(cr).map(([k, d]) => ({ k, d: String(d) })) : [{ k: '', d: '' }];
    } else if (c.type === 'score') {
      c.levels = Array.isArray(cr) ? cr.map(String) : [''];
    } else {
      c.trueDesc = (cr && cr.true) || '';
      c.falseDesc = (cr && cr.false) || '';
    }
    cards.push(c);
  }
  return cards;
}

/** Validate a questions object (contract form). Returns {id: [messages]}; '' key = global. */
export function validateQuestions(qs, limits) {
  limits = limits || {};
  const errs = {};
  const add = (id, m) => { (errs[id] = errs[id] || []).push(m); };
  if (!qs || typeof qs !== 'object' || Array.isArray(qs)) { add('', 'questions must be an object'); return errs; }
  const ids = Object.keys(qs);
  if (!ids.length) add('', 'at least one question is required');
  if (limits.max_questions && ids.length > limits.max_questions) add('', `at most ${limits.max_questions} questions allowed`);
  for (const [id, q] of Object.entries(qs)) {
    if (!id) add(id, 'id is required');
    if (!q || typeof q !== 'object') { add(id, 'must be an object'); continue; }
    if (!['choice', 'score', 'noul'].includes(q.type)) { add(id, 'type must be choice, score or noul'); continue; }
    if (q.instructions !== undefined && typeof q.instructions !== 'string') add(id, 'instructions must be a string');
    const cr = q.criteria;
    if (q.type === 'choice') {
      if (!cr || typeof cr !== 'object' || Array.isArray(cr) || !Object.keys(cr).length) add(id, 'criteria must be a non-empty object');
      else for (const [k, v] of Object.entries(cr)) {
        if (!k) add(id, 'option names cannot be empty');
        if (typeof v !== 'string') add(id, `description of "${k}" must be a string`);
      }
    } else if (q.type === 'score') {
      if (!Array.isArray(cr) || !cr.length) add(id, 'criteria must be a non-empty list');
      else if (cr.some((l) => typeof l !== 'string' || !l.trim())) add(id, 'every level needs text');
    } else if (cr !== undefined) {
      if (!cr || typeof cr !== 'object' || Array.isArray(cr)) add(id, 'criteria must be an object');
      else for (const [k, v] of Object.entries(cr)) {
        if (k !== 'true' && k !== 'false') add(id, `criteria key "${k}" not allowed (only true/false)`);
        else if (typeof v !== 'string') add(id, `criteria.${k} must be a string`);
      }
    }
  }
  return errs;
}

/** Card-level validation, adds duplicate id / duplicate option checks. */
export function validateCards(cards, limits) {
  const errs = validateQuestions(cardsToContract(cards), limits);
  const seen = new Set();
  for (const c of cards) {
    const id = c.id.trim();
    if (seen.has(id)) (errs[id] = errs[id] || []).push('duplicate id');
    seen.add(id);
    if (c.type === 'choice') {
      const ks = c.options.map((o) => o.k.trim()).filter(Boolean);
      if (new Set(ks).size !== ks.length) (errs[id] = errs[id] || []).push('duplicate option names');
    }
  }
  return errs;
}

export const errCount = (errs) => Object.values(errs).reduce((n, a) => n + a.length, 0);

// ---- JSON helpers ---------------------------------------------------------------------------
/** Parse JSON; on failure return {error, line, col}. */
export function parseJson(text) {
  try { return { value: JSON.parse(text) }; } catch (e) {
    const msg = (e && e.message) || String(e);
    const m = /position (\d+)/.exec(msg);
    let line = null, col = null;
    if (m) {
      const pos = Number(m[1]);
      const before = text.slice(0, pos).split('\n');
      line = before.length; col = before[before.length - 1].length + 1;
    } else {
      const m2 = /line (\d+) column (\d+)/.exec(msg);
      if (m2) { line = Number(m2[1]); col = Number(m2[2]); }
    }
    return { error: msg, line, col };
  }
}

// ---- request building -----------------------------------------------------------------------
export function buildRequest({ stateMode, stateText, media, questions }) {
  let state = stateText;
  if (stateMode === 'json') {
    const r = parseJson(stateText);
    if (r.error) throw new Error(`State JSON: ${r.error}`);
    state = r.value;
  }
  const req = { model: 'clef-flash', state, questions };
  const images = media.filter((m) => m.kind === 'image' && m.dataUrl).map((m) => m.dataUrl);
  const videos = media.filter((m) => m.kind === 'video' && m.dataUrl).map((m) => m.dataUrl);
  if (images.length) req.images = images;
  if (videos.length) req.videos = videos;
  return req;
}

export function truncateMedia(req) {
  const c = JSON.parse(JSON.stringify(req));
  for (const k of ['images', 'videos']) {
    if (Array.isArray(c[k])) {
      c[k] = c[k].map((s) => (typeof s === 'string' && s.startsWith('data:') ? `${s.slice(0, s.indexOf(',') + 1)}<base64…>` : s));
    }
  }
  return c;
}

// ---- snippets -------------------------------------------------------------------------------
export function snippets(req, hasKey) {
  const body = JSON.stringify(truncateMedia(req), null, 2);
  const origin = location.origin;
  const curl = [
    `curl -sS ${origin}/v1/systemone \\`,
    '  -H "Content-Type: application/json" \\',
    hasKey ? '  -H "X-API-Key: <your-key>" \\' : null,
    "  -d @- <<'JSON'",
    body,
    'JSON',
  ].filter((x) => x !== null).join('\n');
  const ps = [
    "$body = @'",
    body,
    "'@",
    `$headers = @{${hasKey ? " 'X-API-Key' = '<your-key>' " : ''}}`,
    `Invoke-RestMethod ${origin}/v1/systemone -Method Post -Headers $headers -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 8`,
  ].join('\n');
  const pyJson = body.replace(/: true\b/g, ': True').replace(/: false\b/g, ': False').replace(/: null\b/g, ': None');
  const py = [
    'import requests',
    '',
    `payload = ${pyJson}`,
    '',
    'r = requests.post(',
    `    "${origin}/v1/systemone",`,
    '    json=payload,',
    `    headers={${hasKey ? '"X-API-Key": "<your-key>"' : ''}},`,
    '    timeout=120,',
    ')',
    'r.raise_for_status()',
    'print(r.json()["answers"])',
  ].join('\n');
  return { curl, powershell: ps, python: py };
}

// ---- CSV (RFC 4180) -------------------------------------------------------------------------
export function parseCsv(text) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows = [];
  let row = [], field = '', inQ = false, i = 0;
  const n = text.length;
  while (i < n) {
    const ch = text[i];
    if (inQ) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i += 2; continue; }
        inQ = false; i++; continue;
      }
      field += ch; i++; continue;
    }
    if (ch === '"' && field === '') { inQ = true; i++; continue; }
    if (ch === ',') { row.push(field); field = ''; i++; continue; }
    if (ch === '\r') { i++; continue; }
    if (ch === '\n') { row.push(field); rows.push(row); row = []; field = ''; i++; continue; }
    field += ch; i++;
  }
  if (inQ) throw new Error('Unterminated quoted field');
  if (field !== '' || row.length) { row.push(field); rows.push(row); }
  return rows.filter((r) => !(r.length === 1 && r[0] === ''));
}

export function csvEscape(v) {
  const s = v == null ? '' : String(v);
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
export const toCsv = (rows) => rows.map((r) => r.map(csvEscape).join(',')).join('\r\n');

export function download(name, text, mime = 'text/plain') {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url; a.download = name; document.body.appendChild(a); a.click();
  setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 0);
}

export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch {
    const ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    ta.remove();
    return ok;
  }
}

// Charts: hand-written SVG/HTML marks + a uPlot wrapper for time series. Colors come from CSS tokens
// (--s1..--s8, validated in both themes); every chart has role="img" + an aria-label summary and a hover tooltip.
import { html, useState, useEffect, useRef, useMemo } from '../vendor/standalone.module.js';
import uPlot from '../vendor/uPlot.esm.min.js';
import { useStore } from '../store.js';
import { fmtPct, fmtNum, entropy, uncertaintyLevel, download } from '../util.js';

export const SERIES = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];
export const seriesVar = (i) => `var(${SERIES[i % SERIES.length]})`;
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** Re-render key that changes when the theme (store or OS) changes, so canvas charts can re-read tokens. */
export function useThemeKey() {
  const { theme } = useStore();
  const [dark, setDark] = useState(() => !!(window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches));
  useEffect(() => {
    if (!window.matchMedia) return undefined;
    const m = matchMedia('(prefers-color-scheme: dark)');
    const f = () => setDark(m.matches);
    m.addEventListener('change', f);
    return () => m.removeEventListener('change', f);
  }, []);
  return `${theme}:${dark}`;
}

/** After mount, flips to true on the next frame so CSS width transitions animate from 0. */
export function useMounted() {
  const [m, setM] = useState(false);
  useEffect(() => { const id = requestAnimationFrame(() => requestAnimationFrame(() => setM(true))); return () => cancelAnimationFrame(id); }, []);
  return m;
}

/** Tooltip state bound to a relatively positioned wrapper. */
export function useTip() {
  const [tip, setTip] = useState(null);
  const wrap = useRef(null);
  const show = (e, content) => {
    const r = wrap.current.getBoundingClientRect();
    setTip({ x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, content });
  };
  const hide = () => setTip(null);
  const el = tip ? html`<div class="tip show" role="presentation" style=${{ left: `${tip.x > tip.w * 0.6 ? tip.x - 14 : tip.x + 14}px`, top: `${Math.max(0, tip.y - 12)}px`, transform: tip.x > tip.w * 0.6 ? 'translateX(-100%)' : 'none' }}>${tip.content}</div>` : null;
  return { wrap, show, hide, el };
}

export const TipRow = ({ color, label, value }) => html`<div class="tip-row">${color ? html`<span class="tip-sw" style=${{ background: color }}></span>` : null}<span>${label}</span><b>${value}</b></div>`;

// ---- card + export -------------------------------------------------------------------------------
function inlineStyles(src, dst) {
  const cs = getComputedStyle(src);
  for (const p of ['fill', 'stroke', 'stroke-width', 'stroke-dasharray', 'opacity', 'font-family', 'font-size', 'font-weight', 'text-anchor', 'fill-opacity']) {
    dst.style.setProperty(p, cs.getPropertyValue(p));
  }
  for (let i = 0; i < src.children.length; i++) inlineStyles(src.children[i], dst.children[i]);
}
function svgString(svg) {
  const clone = svg.cloneNode(true);
  inlineStyles(svg, clone);
  const vb = svg.viewBox.baseVal;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('width', vb.width); clone.setAttribute('height', vb.height);
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  bg.setAttribute('x', vb.x); bg.setAttribute('y', vb.y); bg.setAttribute('width', vb.width); bg.setAttribute('height', vb.height);
  bg.setAttribute('fill', cssVar('--panel'));
  clone.insertBefore(bg, clone.firstChild);
  return { text: new XMLSerializer().serializeToString(clone), w: vb.width, h: vb.height };
}
export function exportSvg(svg, name) {
  if (!svg) return;
  download(`${name}.svg`, svgString(svg).text, 'image/svg+xml');
}
export function exportPng(svg, name) {
  if (!svg) return;
  const { text, w, h } = svgString(svg);
  const img = new Image();
  img.onload = () => {
    const c = document.createElement('canvas');
    c.width = w * 2; c.height = h * 2;
    const g = c.getContext('2d'); g.scale(2, 2); g.drawImage(img, 0, 0, w, h);
    c.toBlob((b) => {
      const url = URL.createObjectURL(b);
      const a = document.createElement('a'); a.href = url; a.download = `${name}.png`; document.body.appendChild(a); a.click();
      setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 0);
    }, 'image/png');
  };
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(text)}`;
}

/** Small "Export" menu for the SVG chart inside `host` (a ref to an element containing one svg.chart-svg). */
export function ExportMenu({ host, name }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const d = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const k = (e) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', d); document.addEventListener('keydown', k);
    return () => { document.removeEventListener('mousedown', d); document.removeEventListener('keydown', k); };
  }, [open]);
  const svg = () => host.current && host.current.querySelector('svg.chart-svg');
  return html`<span class="menu" ref=${ref}>
    <button class="btn ghost sm" type="button" aria-haspopup="menu" aria-expanded=${open} onClick=${() => setOpen(!open)}>Export</button>
    ${open ? html`<div class="menu-pop" role="menu">
      <button type="button" role="menuitem" onClick=${() => { exportSvg(svg(), name); setOpen(false); }}>SVG</button>
      <button type="button" role="menuitem" onClick=${() => { exportPng(svg(), name); setOpen(false); }}>PNG</button></div>` : null}
  </span>`;
}

export function Legend({ items, onToggle, off }) {
  return html`<div class="legend" role="list">
    ${items.map((it) => {
      const inner = html`<span class=${`lg-sw ${it.line ? 'line' : ''}`} style=${{ background: it.color }}></span><span>${it.label}</span>${it.value != null ? html`<span class="mono muted">${it.value}</span>` : null}`;
      return onToggle
        ? html`<button key=${it.label} type="button" role="listitem" class=${`lg-item ${off && off[it.label] ? 'off' : ''}`} aria-pressed=${!(off && off[it.label])} onClick=${() => onToggle(it.label)}>${inner}</button>`
        : html`<span key=${it.label} role="listitem" class="lg-item">${inner}</span>`;
    })}
  </div>`;
}

// ---- sparkline -----------------------------------------------------------------------------------
export function Sparkline({ values, label }) {
  const W = 200, H = 34, P = 3;
  const pts = values.map((v, i) => ({ v, i })).filter((p) => p.v != null);
  if (pts.length < 2) return html`<svg class="spark" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${`${label}: collecting data`}><line x1=${P} x2=${W - P} y1=${H - P} y2=${H - P} class="grid-line"/></svg>`;
  const n = values.length;
  const max = Math.max(...pts.map((p) => p.v)), min = Math.min(0, ...pts.map((p) => p.v));
  const x = (i) => P + (i / Math.max(1, n - 1)) * (W - 2 * P);
  const y = (v) => H - P - ((v - min) / (max - min || 1)) * (H - 2 * P - 2);
  const segs = [];
  let cur = [];
  values.forEach((v, i) => { if (v == null) { if (cur.length) segs.push(cur); cur = []; } else cur.push([x(i), y(v)]); });
  if (cur.length) segs.push(cur);
  const last = pts[pts.length - 1];
  return html`<svg class="spark" viewBox=${`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label=${`${label} trend, latest ${fmtNum(last.v)}, range ${fmtNum(Math.min(...pts.map((p) => p.v)))} to ${fmtNum(max)}`}>
    ${segs.map((s, k) => html`<g key=${k}>
      ${s.length > 1 ? html`<polygon class="spark-area" points=${`${s[0][0]},${H - P} ${s.map((p) => p.join(',')).join(' ')} ${s[s.length - 1][0]},${H - P}`} />` : null}
      <polyline class="spark-line" vector-effect="non-scaling-stroke" points=${s.map((p) => p.join(',')).join(' ')} /></g>`)}
  </svg>`;
}

// ---- probability bars (HTML, animated; optional previous-run overlay and threshold line) ---------
export function ProbBars({ entries, topKey, prev, threshold, matchSet, unit = 'p' }) {
  const mounted = useMounted();
  const showDelta = !!prev;
  return html`<div class="pbars" role="list">
    ${entries.map(([k, p]) => {
      const pp = prev && prev[k] != null ? prev[k] : null;
      const d = pp != null ? (p - pp) * 100 : null;
      const isTop = topKey != null ? k === topKey : false;
      const match = matchSet && matchSet.has(k);
      return html`<div key=${k} role="listitem" class=${`pbar ${isTop ? 'top' : ''} ${match ? 'match' : ''} ${showDelta ? 'has-delta' : ''}`}>
        <div class="pbar-label" title=${k}>${k}${match ? html`<span class="tag">match</span>` : null}</div>
        <div class="pbar-track" role="img" aria-label=${`${k} ${fmtPct(p)}${pp != null ? `, previous run ${fmtPct(pp)}` : ''}`}>
          <div class="pbar-fill" style=${{ width: mounted ? `${Math.max(0.6, p * 100)}%` : '0%' }}></div>
          ${pp != null ? html`<div class="pbar-prev" style=${{ left: `${pp * 100}%` }} title=${`previous run ${fmtPct(pp)}`}></div>` : null}
          ${threshold != null ? html`<div class="pbar-thr" style=${{ left: `${threshold * 100}%` }}></div>` : null}
        </div>
        <div class="pbar-val mono">${fmtPct(p)}</div>
        ${showDelta ? html`<div class=${`pbar-delta mono muted`}>${d == null ? '' : `${d > 0 ? '▲' : d < 0 ? '▼' : ''} ${Math.abs(d).toFixed(1)}`}</div>` : null}
      </div>`;
    })}
    ${prev || threshold != null ? html`<div class="key-legend">
      ${prev ? html`<span><span class="tick"></span>previous run</span>` : null}
      ${threshold != null ? html`<span><span class="dash"></span>threshold ${threshold}</span>` : null}</div>` : null}
  </div>`;
}

// ---- uncertainty (entropy) -----------------------------------------------------------------------
export function Uncertainty({ ent }) {
  const mounted = useMounted();
  if (!ent) return null;
  const lv = uncertaintyLevel(ent.norm);
  return html`<div class="unc" role="img" aria-label=${`Uncertainty ${lv.label}, entropy ${ent.bits.toFixed(2)} bits, ${Math.round(ent.norm * 100)} percent of maximum`}>
    <span class="ico" aria-hidden="true">${lv.icon}</span><strong>${lv.label}</strong>
    <span class="unc-track"><span class="unc-fill" style=${{ width: mounted ? `${Math.max(2, ent.norm * 100)}%` : '0%' }}></span></span>
    <span class="mono">${ent.bits.toFixed(2)} bits</span>
  </div>`;
}

// ---- gauge for P(true) ---------------------------------------------------------------------------
export function Gauge({ value, prev, label = 'P(true)' }) {
  const mounted = useMounted();
  const R = 80, CX = 100, CY = 96;
  const arcLen = Math.PI * R;
  const pt = (f, r = R) => [CX - r * Math.cos(Math.PI * f), CY - r * Math.sin(Math.PI * f)];
  const d = `M ${CX - R} ${CY} A ${R} ${R} 0 0 1 ${CX + R} ${CY}`;
  const [mx1, my1] = pt(0.5, R - 12), [mx2, my2] = pt(0.5, R + 12);
  const [px1, py1] = prev != null ? pt(prev, R - 12) : [0, 0], [px2, py2] = prev != null ? pt(prev, R + 10) : [0, 0];
  return html`<svg class="gauge chart-svg" viewBox="0 0 200 112" role="img" aria-label=${`${label} ${fmtPct(value)}${prev != null ? `, previous run ${fmtPct(prev)}` : ''}`}>
    <path d=${d} class="gauge-track" />
    <path d=${d} class="gauge-fill" style=${{ strokeDasharray: arcLen, strokeDashoffset: mounted ? arcLen * (1 - value) : arcLen }} />
    <line x1=${mx1} y1=${my1} x2=${mx2} y2=${my2} class="gauge-tick" />
    ${prev != null ? html`<line x1=${px1} y1=${py1} x2=${px2} y2=${py2} class="gauge-prev" />` : null}
    <text x=${CX} y=${CY - 18} class="gauge-num">${fmtPct(value)}</text>
    <text x=${CX} y=${CY - 2} class="gauge-sub">${label}</text>
    <text x=${CX - R} y=${CY + 12} class="svg-label" text-anchor="middle">0</text>
    <text x=${CX + R} y=${CY + 12} class="svg-label" text-anchor="middle">1</text>
  </svg>`;
}

// ---- score distribution with expected-value marker ----------------------------------------------
export function ScoreChart({ probs, legend, score, prevScore, prevProbs }) {
  const tip = useTip();
  const mounted = useMounted();
  const keys = Object.keys(probs).sort((a, b) => Number(a) - Number(b));
  const n = keys.length;
  const W = 520, H = 180, L = 8, R = 8, T = 36, B = 34;
  const bw = (W - L - R) / n;
  const maxP = Math.max(0.05, ...keys.map((k) => probs[k]), ...(prevProbs ? keys.map((k) => prevProbs[k] || 0) : []));
  const yv = (p) => H - B - (p / maxP) * (H - B - T);
  const cx = (i) => L + bw * i + bw / 2;
  const evx = (v) => (n > 1 ? cx(0) + (v / (n - 1)) * (cx(n - 1) - cx(0)) : cx(0));
  const topK = keys.reduce((a, k) => (probs[k] > probs[a] ? k : a), keys[0]);
  const short = (s) => (s && s.length > Math.max(6, Math.floor(bw / 6.5)) ? `${s.slice(0, Math.max(5, Math.floor(bw / 6.5) - 1))}…` : s || '');
  return html`<div class="chart" ref=${tip.wrap} onPointerLeave=${tip.hide}>
    <svg class="chart-svg" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${`Score distribution over ${n} levels; expected value ${fmtNum(score)}; most likely level ${topK}${legend && legend[topK] ? ` (${legend[topK]})` : ''}`}>
      <line x1=${L} x2=${W - R} y1=${H - B} y2=${H - B} class="axis-line" />
      ${keys.map((k, i) => {
        const p = probs[k], h = H - B - yv(p);
        return html`<g key=${k}>
          <rect class=${`hbar ${k === topK ? 'top' : 'dim'}`} x=${L + bw * i + bw * 0.18} width=${bw * 0.64} y=${mounted ? yv(p) : H - B} height=${mounted ? Math.max(p > 0 ? 1 : 0, h) : 0} rx="3" style=${{ transition: 'y 200ms var(--ease), height 200ms var(--ease)' }} />
          ${prevProbs && prevProbs[k] != null ? html`<line x1=${L + bw * i + bw * 0.16} x2=${L + bw * (i + 1) - bw * 0.16} y1=${yv(prevProbs[k])} y2=${yv(prevProbs[k])} class="ev-prev" />` : null}
          <text x=${cx(i)} y=${yv(p) - 5} text-anchor="middle" class="svg-label halo">${fmtPct(p, 0)}</text>
          <text x=${cx(i)} y=${H - B + 14} text-anchor="middle" class="svg-text-strong">${k}</text>
          <text x=${cx(i)} y=${H - B + 27} text-anchor="middle" class="svg-label">${short(legend && legend[k])}</text>
          <rect class="hit" x=${L + bw * i} y="0" width=${bw} height=${H - B} tabindex="0" role="img" aria-label=${`Level ${k}${legend && legend[k] ? ` ${legend[k]}` : ''}: ${fmtPct(p)}`}
            onPointerMove=${(e) => tip.show(e, html`<div class="tip-h">Level ${k}</div>${legend && legend[k] ? html`<div class="muted">${legend[k]}</div>` : null}<${TipRow} label="probability" value=${fmtPct(p)} />${prevProbs && prevProbs[k] != null ? html`<${TipRow} label="previous run" value=${fmtPct(prevProbs[k])} />` : null}`)} />
        </g>`;
      })}
      ${prevScore != null ? html`<line x1=${evx(prevScore)} x2=${evx(prevScore)} y1=${T - 14} y2=${H - B} class="ev-prev" />` : null}
      <line x1=${evx(score)} x2=${evx(score)} y1=${T - 14} y2=${H - B} class="ev-line" />
      <circle cx=${evx(score)} cy=${T - 14} r="4" style=${{ fill: 'var(--text)', stroke: 'var(--panel)', strokeWidth: 2 }} />
      <text x=${Math.min(W - 30, Math.max(30, evx(score)))} y="10" class="ev-label">E = ${fmtNum(score)}</text>
    </svg>${tip.el}
  </div>`;
}

// ---- histogram / bar chart with click-to-filter --------------------------------------------------
export function BarChart({ bins, height = 150, width = 380, onSelect, selected, ariaLabel, yLabel, color, showValues, unit }) {
  const tip = useTip();
  const W = width, H = height, L = 6, R = 6, T = 16, B = 28;
  const n = bins.length;
  const bw = (W - L - R) / Math.max(1, n);
  const max = Math.max(1, ...bins.map((b) => b.value));
  const y = (v) => H - B - (v / max) * (H - B - T);
  const dense = n > 12;
  const skip = Math.ceil(n / (W / 36));
  return html`<div class="chart" ref=${tip.wrap} onPointerLeave=${tip.hide}>
    <svg class="chart-svg" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${ariaLabel}>
      <line x1=${L} x2=${W - R} y1=${y(max / 2)} y2=${y(max / 2)} class="grid-line" />
      <line x1=${L} x2=${W - R} y1=${H - B} y2=${H - B} class="axis-line" />
      ${bins.map((b, i) => {
        const sel = selected === b.id;
        const h = H - B - y(b.value);
        const bx = L + i * bw + Math.min(3, bw * 0.12);
        const w = bw - 2 * Math.min(3, bw * 0.12);
        return html`<g key=${b.id ?? i}>
          <rect class=${`hbar ${sel ? 'sel' : ''}`} x=${bx} y=${y(b.value)} width=${Math.max(1, w)} height=${Math.max(b.value ? 2 : 0, h)} rx="3" style=${color ? { fill: color } : null} opacity=${selected != null && !sel ? 0.45 : 1} />
          ${b.value && (showValues ?? !dense) ? html`<text x=${bx + w / 2} y=${y(b.value) - 4} text-anchor="middle" class="svg-label">${b.value}</text>` : null}
          ${i % skip === 0 ? html`<text x=${bx + w / 2} y=${H - B + 13} text-anchor="middle" class="svg-label">${b.label}</text>` : null}
          <rect class="hit" x=${L + i * bw} y="0" width=${bw} height=${H - B} tabindex=${onSelect ? 0 : -1} role=${onSelect ? 'button' : 'img'}
            aria-label=${`${b.label}${unit ? ` ${unit}` : ''}: ${b.value}${b.pct != null ? ` (${b.pct})` : ''}`} aria-pressed=${onSelect ? sel : null}
            onPointerMove=${(e) => tip.show(e, html`<div class="tip-h">${b.label}${unit ? ` ${unit}` : ''}</div><${TipRow} label=${yLabel || 'count'} value=${b.value} />${b.pct != null ? html`<${TipRow} label="share" value=${b.pct} />` : null}${onSelect ? html`<div class="muted">click to filter</div>` : null}`)}
            onClick=${() => onSelect && onSelect(sel ? null : b)} onKeyDown=${(e) => { if (onSelect && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onSelect(sel ? null : b); } }} />
        </g>`;
      })}
    </svg>${tip.el}
  </div>`;
}

// ---- stacked share bar -----------------------------------------------------------------------------
export function StackedBar({ segments, onSelect, selected, ariaLabel }) {
  const tip = useTip();
  const total = segments.reduce((a, s) => a + s.value, 0);
  const W = 380, H = 34, G = 2;
  let acc = 0;
  const segW = segments.map((s) => (total ? (s.value / total) * W : 0));
  return html`<div class="chart" ref=${tip.wrap} onPointerLeave=${tip.hide}>
    <svg class="chart-svg" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${ariaLabel}>
      ${segments.map((s, i) => {
        const x = acc; acc += segW[i];
        const sel = selected === s.id;
        const w = Math.max(0, segW[i] - G);
        return html`<g key=${s.id}>
          <rect x=${x + (i ? G / 2 : 0)} y="2" width=${w} height=${H - 4} rx="4" style=${{ fill: s.color }} opacity=${selected != null && !sel ? 0.4 : 1} />
          ${w > 44 ? html`<text x=${x + segW[i] / 2} y=${H / 2 + 4} text-anchor="middle" style=${{ fill: s.ink || '#fff', font: '600 11px var(--mono)' }}>${Math.round((s.value / total) * 100)}%</text>` : null}
          <rect class="hit" x=${x} y="0" width=${Math.max(2, segW[i])} height=${H} tabindex=${onSelect ? 0 : -1} role=${onSelect ? 'button' : 'img'} aria-pressed=${onSelect ? sel : null}
            aria-label=${`${s.label}: ${s.value} of ${total} (${total ? Math.round((s.value / total) * 100) : 0}%)`}
            onPointerMove=${(e) => tip.show(e, html`<div class="tip-h">${s.label}</div><${TipRow} color=${s.color} label="rows" value=${s.value} /><${TipRow} label="share" value=${fmtPct(total ? s.value / total : 0)} />${onSelect ? html`<div class="muted">click to filter</div>` : null}`)}
            onClick=${() => onSelect && onSelect(sel ? null : s)} onKeyDown=${(e) => { if (onSelect && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onSelect(sel ? null : s); } }} />
        </g>`;
      })}
    </svg>${tip.el}
  </div>`;
}

// ---- generic SVG line chart with crosshair (History trend, reliability) ---------------------------
export function LineChart({ points, yMin = 0, yMax = 1, height = 190, ariaLabel, fmtY = (v) => fmtPct(v, 0), fmtX, xLabelCount = 5, yLabel }) {
  const tip = useTip();
  const [hi, setHi] = useState(null);
  const [W, setW] = useState(560);
  useEffect(() => {
    const el = tip.wrap.current;
    if (!el || !window.ResizeObserver) return undefined;
    let raf = 0;
    const ro = new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { if (el.clientWidth > 80) setW(Math.round(el.clientWidth)); }); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = height, L = 40, R = 12, T = 12, B = 26;
  if (!points.length) return html`<div class="chart-empty" style=${{ '--h': `${height}px` }}>No data</div>`;
  const xs = points.map((p) => p.x);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const X = (x) => (x1 === x0 ? (L + W - R) / 2 : L + ((x - x0) / (x1 - x0)) * (W - L - R));
  const Y = (v) => H - B - ((v - yMin) / (yMax - yMin || 1)) * (H - B - T);
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ');
  const area = `${path} L${X(points[points.length - 1].x).toFixed(1)},${H - B} L${X(points[0].x).toFixed(1)},${H - B} Z`;
  const ticks = [0, 0.5, 1].map((f) => yMin + (yMax - yMin) * f);
  const move = (e) => {
    const svg = e.currentTarget.ownerSVGElement;
    const r = svg.getBoundingClientRect();
    const mx = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    points.forEach((p, i) => { if (Math.abs(X(p.x) - mx) < Math.abs(X(points[best].x) - mx)) best = i; });
    setHi(best);
    const p = points[best];
    tip.show(e, html`<div class="tip-h">${p.title || (fmtX ? fmtX(p.x) : p.x)}</div><${TipRow} color="var(--s1)" label=${yLabel || 'value'} value=${fmtY(p.y)} />${p.note ? html`<div class="muted">${p.note}</div>` : null}`);
  };
  const lbl = Array.from({ length: Math.min(xLabelCount, points.length) }, (_, i) => points[Math.round((i / Math.max(1, Math.min(xLabelCount, points.length) - 1)) * (points.length - 1))]);
  return html`<div class="chart" ref=${tip.wrap} onPointerLeave=${() => { tip.hide(); setHi(null); }}>
    <svg class="chart-svg" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${ariaLabel}>
      ${ticks.map((v) => html`<g key=${v}><line x1=${L} x2=${W - R} y1=${Y(v)} y2=${Y(v)} class=${v === yMin ? 'axis-line' : 'grid-line'} /><text x=${L - 6} y=${Y(v) + 3} text-anchor="end" class="svg-label">${fmtY(v)}</text></g>`)}
      <path d=${area} class="line-area" /><path d=${path} class="line-main" />
      ${points.map((p, i) => html`<circle key=${i} cx=${X(p.x)} cy=${Y(p.y)} r=${hi === i ? 5 : 3} class="dot-pt" />`)}
      ${hi != null ? html`<line x1=${X(points[hi].x)} x2=${X(points[hi].x)} y1=${T} y2=${H - B} class="cross" />` : null}
      ${lbl.map((p, i) => html`<text key=${i} x=${X(p.x)} y=${H - 8} text-anchor=${i === 0 ? 'start' : i === lbl.length - 1 ? 'end' : 'middle'} class="svg-label">${fmtX ? fmtX(p.x) : p.x}</text>`)}
      <rect class="hit" x=${L} y=${T} width=${W - L - R} height=${H - B - T} onPointerMove=${move} style=${{ cursor: 'crosshair' }} />
    </svg>${tip.el}
  </div>`;
}

// ---- reliability diagram ---------------------------------------------------------------------------
export function Reliability({ bins, ariaLabel }) {
  const tip = useTip();
  const S = 260, L = 34, B = 38, T = 10, R = 10;
  const X = (v) => L + v * (S - L - R), Y = (v) => S - B - v * (S - B - T);
  const maxN = Math.max(1, ...bins.map((b) => b.n));
  const used = bins.filter((b) => b.n > 0);
  return html`<div class="chart" ref=${tip.wrap} onPointerLeave=${tip.hide} style=${{ maxWidth: '340px' }}>
    <svg class="chart-svg" viewBox=${`0 0 ${S} ${S}`} role="img" aria-label=${ariaLabel}>
      ${[0, 0.5, 1].map((v) => html`<g key=${v}><line x1=${L} x2=${S - R} y1=${Y(v)} y2=${Y(v)} class="grid-line"/><text x=${L - 5} y=${Y(v) + 3} text-anchor="end" class="svg-label">${v}</text><text x=${X(v)} y=${S - B + 13} text-anchor="middle" class="svg-label">${v}</text></g>`)}
      <line x1=${X(0)} y1=${Y(0)} x2=${X(1)} y2=${Y(1)} class="ev-prev" />
      <polyline points=${used.map((b) => `${X(b.conf)},${Y(b.acc)}`).join(' ')} class="line-main" />
      ${used.map((b, i) => html`<g key=${i}><circle cx=${X(b.conf)} cy=${Y(b.acc)} r=${3 + 4 * Math.sqrt(b.n / maxN)} class="dot-pt" style=${{ fillOpacity: 0.9 }} />
        <circle class="hit" cx=${X(b.conf)} cy=${Y(b.acc)} r="12" onPointerMove=${(e) => tip.show(e, html`<div class="tip-h">${b.label}</div><${TipRow} label="mean confidence" value=${fmtPct(b.conf)} /><${TipRow} label="accuracy" value=${fmtPct(b.acc)} /><${TipRow} label="rows" value=${b.n} />`)} /></g>`)}
      <text x=${(L + S - R) / 2} y=${S - 4} text-anchor="middle" class="svg-text">confidence</text>
      <text x="9" y=${(T + S - B) / 2} text-anchor="middle" class="svg-text" transform=${`rotate(-90 9 ${(T + S - B) / 2})`}>accuracy</text>
    </svg>${tip.el}
  </div>`;
}

// ---- uPlot time-series wrapper ---------------------------------------------------------------------
const hexA = (hex, a) => {
  const h = hex.replace('#', '');
  const n = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  return `rgba(${parseInt(n.slice(0, 2), 16)},${parseInt(n.slice(2, 4), 16)},${parseInt(n.slice(4, 6), 16)},${a})`;
};
const pad2 = (n) => String(n).padStart(2, '0');
export const clock = (sec, withSec = true) => { const d = new Date(sec * 1000); return `${pad2(d.getHours())}:${pad2(d.getMinutes())}${withSec ? `:${pad2(d.getSeconds())}` : ''}`; };

/**
 * series: [{key, label, color: '--s1', data: [...], fill?: 0..1 alpha, fmt?: v=>string}]
 * band: optional [upperKey, lowerKey] filled with the lower series' color.
 */
export function TimeChart({ t, series, height = 180, yMin = 0, yMax, band, fmtAxis = (v) => fmtNum(v, v < 10 && v % 1 ? 1 : 0), ariaLabel, stepS = 5, showLegend = true }) {
  const host = useRef(null), tipEl = useRef(null), plot = useRef(null);
  const themeKey = useThemeKey();
  const sig = series.map((s) => s.key).join(',') + (band ? band.join('>') : '') + themeKey + height + (yMax ?? '');
  const latest = useRef({ t, series, stepS });
  latest.current = { t, series, stepS };

  useEffect(() => {
    const el = host.current;
    if (!el) return undefined;
    const tok = (n) => cssVar(n);
    const surface = tok('--panel'), grid = tok('--grid'), axis = tok('--axis'), label = tok('--axis-label');
    const cols = series.map((s) => tok(s.color));
    const mk = (w) => {
      const opts = {
        width: Math.max(120, w), height, legend: { show: false }, padding: [10, 22, 0, 0],
        cursor: { y: false, drag: { x: false, y: false, setScale: false }, points: { size: 8, width: 2, fill: surface, stroke: (u, si) => cols[si - 1] }, focus: { prox: -1 } },
        scales: { x: { time: true }, y: { range: (u, mn, mx) => [yMin, yMax != null ? yMax : Math.max(mx * 1.18, yMin + 1)] } },
        axes: [
          { stroke: label, grid: { stroke: grid, width: 1 }, ticks: { show: false }, border: { show: true, stroke: axis, width: 1 }, font: '11px ui-monospace, Consolas, monospace', size: 26, gap: 4, space: 88,
            values: (u, sp) => sp.map((v) => clock(v, latest.current.stepS < 60 && (u.scales.x.max - u.scales.x.min) < 1200)) },
          { stroke: label, grid: { stroke: grid, width: 1 }, ticks: { show: false }, border: { show: false }, font: '11px ui-monospace, Consolas, monospace', size: 46, gap: 4, values: (u, sp) => sp.map((v) => (v == null ? '' : fmtAxis(v))) },
        ],
        series: [{}, ...series.map((s, i) => ({
          label: s.label, stroke: cols[i], width: 2, spanGaps: false,
          fill: s.fill ? hexA(cols[i], s.fill) : undefined, points: { show: true, size: 5, stroke: cols[i], fill: surface, width: 1.5 },
        }))],
        bands: band ? [{ series: [series.findIndex((s) => s.key === band[0]) + 1, series.findIndex((s) => s.key === band[1]) + 1], fill: hexA(cols[series.findIndex((s) => s.key === band[1])], 0.16) }] : [],
        hooks: {
          setCursor: [(u) => {
            const tp = tipEl.current;
            const idx = u.cursor.idx;
            if (!tp) return;
            if (idx == null || u.cursor.left < 0) { tp.classList.remove('show'); return; }
            const cur = latest.current;
            const rows = cur.series.map((s, i) => {
              const v = s.data[idx];
              return `<div class="tip-row"><span class="tip-sw" style="background:${cols[i]}"></span><span>${esc(s.label)}</span><b>${v == null ? '-' : esc(s.fmt ? s.fmt(v) : fmtNum(v))}</b></div>`;
            }).join('');
            tp.innerHTML = `<div class="tip-h">${clock(cur.t[idx])}</div>${rows}`;
            tp.classList.add('show');
            const ow = u.over.offsetLeft, ot = u.over.offsetTop;
            const left = ow + u.cursor.left;
            const flip = left > u.over.clientWidth * 0.6;
            tp.style.left = `${left + (flip ? -14 : 14)}px`;
            tp.style.top = `${Math.max(0, ot + (u.cursor.top > 0 ? u.cursor.top : 20) - 24)}px`;
            tp.style.transform = flip ? 'translateX(-100%)' : 'none';
          }],
        },
      };
      return new uPlot(opts, [latest.current.t, ...latest.current.series.map((s) => s.data)], el);
    };
    const u = mk(el.clientWidth);
    plot.current = u;
    let raf = 0;
    const ro = new ResizeObserver(() => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => { if (plot.current && el.clientWidth > 0) plot.current.setSize({ width: Math.max(120, el.clientWidth), height }); }); });
    ro.observe(el);
    const leave = () => tipEl.current && tipEl.current.classList.remove('show');
    el.addEventListener('pointerleave', leave);
    return () => { ro.disconnect(); el.removeEventListener('pointerleave', leave); u.destroy(); plot.current = null; };
  }, [sig]);

  useEffect(() => { if (plot.current) plot.current.setData([t, ...series.map((s) => s.data)]); }, [t, series]);

  const last = t.length - 1;
  const summary = `${ariaLabel}. ${series.map((s) => { const v = s.data[last]; return `${s.label} latest ${v == null ? 'none' : s.fmt ? s.fmt(v) : fmtNum(v)}`; }).join('; ')}.`;
  return html`<div class="chart-box">
    ${showLegend ? html`<${Legend} items=${series.map((s) => ({ label: s.label, color: `var(${s.color})`, line: true }))} />` : null}
    <div class="chart" style=${{ '--h': `${height}px` }} role="img" aria-label=${summary}>
      <div class="chart-uplot" ref=${host} style=${{ '--h': `${height}px` }}></div>
      <div class="tip" ref=${tipEl} role="presentation"></div>
    </div>
  </div>`;
}

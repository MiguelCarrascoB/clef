// Evaluation charts: confusion-matrix heatmap, reliability diagram (bars + counts), coverage-vs-accuracy curve.
// Hand-written SVG/HTML on the shared tokens (--seq-*, --s1/--s2, --grid ...); each has an aria summary.
import { html, useState, useEffect } from '../vendor/standalone.module.js';
import { useTip, TipRow } from './charts.js';
import { fmtPct } from '../util.js';

/** Heatmap as a real table: values are printed in the cells, so colour is never the only channel. */
export function ConfusionMatrix({ labels, matrix, normalize, selected, onSelect }) {
  const rowSum = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  const max = Math.max(1, ...matrix.flat());
  const cell = (v, i) => {
    const frac = normalize ? (rowSum[i] ? v / rowSum[i] : 0) : v / max;
    return { frac, text: normalize ? (rowSum[i] ? `${Math.round(frac * 100)}%` : '-') : String(v) };
  };
  const pick = (i, j) => (selected && selected.gold === labels[i] && selected.pred === labels[j] ? null : { gold: labels[i], pred: labels[j] });
  return html`<div class="tablewrap cm-wrap"><table class="cm" aria-label=${`Confusion matrix, ${labels.length} labels, ${normalize ? 'row-normalised percentages' : 'counts'}`}>
    <thead><tr><th class="cm-corner" scope="col"><span class="sr-only">Gold label (rows) by predicted label (columns)</span><span aria-hidden="true">gold / pred</span></th>
      ${labels.map((l) => html`<th key=${l} scope="col" class="cm-col" title=${l}><span>${l}</span></th>`)}</tr></thead>
    <tbody>${matrix.map((row, i) => html`<tr key=${i}><th scope="row" class="cm-row" title=${labels[i]}>${labels[i]}<span class="cm-n">${rowSum[i]}</span></th>
      ${row.map((v, j) => {
        const c = cell(v, i);
        const sel = selected && selected.gold === labels[i] && selected.pred === labels[j];
        const live = v > 0 && onSelect;
        return html`<td key=${j} class=${`cm-cell ${i === j ? 'diag' : ''} ${sel ? 'sel' : ''} ${v === 0 ? 'zero' : ''}`}
          style=${{ '--f': c.frac }} tabIndex=${live ? 0 : -1} role=${live ? 'button' : null} aria-pressed=${live ? !!sel : null}
          aria-label=${`gold ${labels[i]}, predicted ${labels[j]}: ${v} rows${rowSum[i] ? ` (${Math.round((v / rowSum[i]) * 100)}% of gold ${labels[i]})` : ''}`}
          title=${live ? 'click to list these rows' : null}
          onClick=${() => live && onSelect(pick(i, j))}
          onKeyDown=${(e) => { if (live && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); onSelect(pick(i, j)); } }}>${c.text}</td>`;
      })}</tr>`)}</tbody></table></div>`;
}

/** Reliability: bars = observed accuracy per confidence bin, dashed diagonal = perfect calibration, strip = bin sizes. */
export function ReliabilityDiagram({ bins, ece, mce, n, yLabel = 'accuracy' }) {
  const tip = useTip();
  const W = 340, H = 346, L = 38, R = 10, T = 10, PH = 232, GAP = 22, SH = 34;
  const px0 = L, px1 = W - R, pw = px1 - px0;
  const X = (v) => px0 + v * pw;
  const Y = (v) => T + PH - v * PH;
  const maxN = Math.max(1, ...bins.map((b) => b.count));
  const sy0 = T + PH + GAP; // top of the count strip
  const used = bins.filter((b) => b.count);
  const desc = (b) => `confidence ${fmtPct(b.lo, 0)} to ${fmtPct(b.hi, 0)}: ${b.count ? `${b.count} rows, ${yLabel} ${fmtPct(b.accuracy, 0)}` : 'no rows'}`;
  const label = `Reliability diagram: ${used.length} non-empty bins over ${n} rows, ECE ${ece.toFixed(3)}, largest gap ${mce.toFixed(3)}. ${used.map(desc).join('; ')}`;
  return html`<div class="chart" ref=${tip.wrap} onPointerLeave=${tip.hide} style=${{ maxWidth: '420px' }}>
    <svg class="chart-svg" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${label}>
      ${[0, 0.25, 0.5, 0.75, 1].map((v) => html`<g key=${v}><line x1=${px0} x2=${px1} y1=${Y(v)} y2=${Y(v)} class=${v === 0 ? 'axis-line' : 'grid-line'} /><text x=${L - 5} y=${Y(v) + 3} text-anchor="end" class="svg-label">${Math.round(v * 100)}%</text></g>`)}
      ${bins.map((b, i) => {
        const w = (pw / bins.length) - 2;
        const x = X(b.lo) + 1;
        const ch = (b.count / maxN) * SH;
        return html`<g key=${i}>
          ${b.count ? html`<rect class="hbar rel-bar" x=${x} y=${Y(b.accuracy)} width=${w} height=${Math.max(1.5, b.accuracy * PH)} rx="2" />` : null}
          ${b.count ? html`<rect class="hbar rel-count" x=${x} y=${sy0 + SH - ch} width=${w} height=${Math.max(1.5, ch)} rx="1.5" />` : null}
          <rect class="hit" x=${X(b.lo)} y=${T} width=${pw / bins.length} height=${PH + GAP + SH} tabindex=${b.count ? 0 : -1} role="img" aria-label=${desc(b)}
            onPointerMove=${(e) => tip.show(e, b.count ? html`<div class="tip-h">${fmtPct(b.lo, 0)} to ${fmtPct(b.hi, 0)} confidence</div><${TipRow} label="rows" value=${b.count} /><${TipRow} label="mean confidence" value=${fmtPct(b.mean_confidence)} /><${TipRow} label=${yLabel} value=${fmtPct(b.accuracy)} /><${TipRow} label="gap" value=${fmtPct(Math.abs(b.accuracy - b.mean_confidence))} />` : html`<div class="tip-h">${fmtPct(b.lo, 0)} to ${fmtPct(b.hi, 0)}</div><div class="muted">no rows</div>`)} />
        </g>`;
      })}
      <line x1=${X(0)} y1=${Y(0)} x2=${X(1)} y2=${Y(1)} class="ev-prev" />
      ${bins.map((b, i) => (b.count ? html`<circle key=${`p${i}`} cx=${X(b.mean_confidence)} cy=${Y(b.accuracy)} r="3.5" class="dot-pt" />` : null))}
      <line x1=${px0} x2=${px1} y1=${sy0 + SH} y2=${sy0 + SH} class="axis-line" />
      ${[0, 0.5, 1].map((v) => html`<text key=${v} x=${X(v)} y=${H - 16} text-anchor=${v === 0 ? 'start' : v === 1 ? 'end' : 'middle'} class="svg-label">${Math.round(v * 100)}%</text>`)}
      <text x=${px0} y=${sy0 - 4} class="svg-label">rows per bin (max ${maxN})</text>
      <text x=${(px0 + px1) / 2} y=${H - 3} text-anchor="middle" class="svg-text">confidence</text>
    </svg>${tip.el}
  </div>`;
}

/** Accuracy (solid) and coverage (second hue) vs confidence threshold; click/drag the plot, or use the slider. */
export function CoverageCurve({ curve, threshold, onThreshold }) {
  const tip = useTip();
  const [W, setW] = useState(480);
  useEffect(() => {
    const el = tip.wrap.current;
    if (!el || !window.ResizeObserver) return undefined;
    const ro = new ResizeObserver(() => { if (el.clientWidth > 120) setW(Math.round(el.clientWidth)); });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = 210, L = 40, R = 12, T = 10, B = 40;
  const X = (t) => L + t * (W - L - R);
  const Y = (v) => H - B - v * (H - B - T);
  const path = (key) => {
    let d = '', pen = false;
    for (const p of curve) {
      const v = p[key];
      if (v == null) { pen = false; continue; }
      d += `${pen ? 'L' : 'M'}${X(p.threshold).toFixed(1)},${Y(v).toFixed(1)}`;
      pen = true;
    }
    return d;
  };
  const at = (e) => {
    const svg = e.currentTarget.ownerSVGElement;
    const r = svg.getBoundingClientRect();
    const t = (((e.clientX - r.left) / r.width) * W - L) / (W - L - R);
    return Math.round(Math.max(0, Math.min(1, t)) * 100) / 100;
  };
  const down = (e) => { if (e.currentTarget.setPointerCapture) e.currentTarget.setPointerCapture(e.pointerId); onThreshold(at(e)); };
  const move = (e) => { if (e.buttons) onThreshold(at(e)); };
  const cur = curve[Math.round(threshold * 100)] || curve[0];
  const aria = `Coverage and accuracy by confidence threshold. At confidence of ${threshold.toFixed(2)} or more: ${fmtPct(cur.coverage, 1)} coverage, ${cur.accuracy == null ? 'no rows left' : `${fmtPct(cur.accuracy, 1)} accuracy`}.`;
  return html`<div class="chart" ref=${tip.wrap}>
    <svg class="chart-svg" viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${aria}>
      ${[0, 0.25, 0.5, 0.75, 1].map((v) => html`<g key=${v}><line x1=${L} x2=${W - R} y1=${Y(v)} y2=${Y(v)} class=${v === 0 ? 'axis-line' : 'grid-line'} /><text x=${L - 5} y=${Y(v) + 3} text-anchor="end" class="svg-label">${Math.round(v * 100)}%</text></g>`)}
      ${[0, 0.25, 0.5, 0.75, 1].map((t) => html`<text key=${t} x=${X(t)} y=${H - B + 13} text-anchor=${t === 0 ? 'start' : t === 1 ? 'end' : 'middle'} class="svg-label">${t.toFixed(2)}</text>`)}
      <path d=${path('coverage')} class="line-main cov-line" />
      <path d=${path('accuracy')} class="line-main" />
      <line x1=${X(threshold)} x2=${X(threshold)} y1=${T} y2=${H - B} class="cross thr-line" />
      <circle cx=${X(threshold)} cy=${Y(cur.coverage)} r="4" class="dot-pt cov-dot" />
      ${cur.accuracy != null ? html`<circle cx=${X(threshold)} cy=${Y(cur.accuracy)} r="4" class="dot-pt" />` : null}
      <text x=${(L + W - R) / 2} y=${H - B + 27} text-anchor="middle" class="svg-text">confidence threshold</text>
      <rect class="hit" x=${L} y=${T} width=${W - L - R} height=${H - B - T} style=${{ cursor: 'ew-resize', touchAction: 'none' }} onPointerDown=${down} onPointerMove=${move} />
    </svg>${tip.el}
  </div>`;
}

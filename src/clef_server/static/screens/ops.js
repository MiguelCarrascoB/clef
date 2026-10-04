// Ops: KPI tiles with sparklines + deltas, live time-series (uPlot over /v1/stats/timeseries + SSE points),
// latency / batch-size distributions, traffic breakdowns and the live request log.
import { html, useState, useEffect, useRef, useMemo } from '../vendor/standalone.module.js';
import * as api from '../api.js';
import { useStore } from '../store.js';
import { fmtDur, fmtTime, fmtNum, fmtPct, fmtMs } from '../util.js';
import { Empty, VramBar } from '../components/common.js';
import { TimeChart, BarChart, Sparkline, ExportMenu } from '../components/charts.js';

const WINDOWS = [{ s: 300, label: '5m' }, { s: 900, label: '15m' }, { s: 3600, label: '1h' }];
const POINTS = 60; // points per window; step = window / 60 (5 s / 15 s / 60 s)
const COLS = ['t', 'rps', 'p50_ms', 'p95_ms', 'error_rate', 'avg_batch', 'queue_depth', 'padding_ratio', 'mem_used_gb', 'gpu_util_pct', 'gpu_temp_c', 'gpu_power_w'];
const MAX_LOG = 200;
const lsGet = (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } };

const stepFor = (w) => Math.max(1, Math.round(w / POINTS));
const last = (a) => (a && a.length ? a[a.length - 1] : null);

// Memory mode from /health: nothing for the plain default; else what moved to the host / the cap / the quantizer.
function memoryLabel(m) {
  if (!m) return '';
  const parts = [];
  if (m.mode === 'offload') parts.push(m.layers_on_host ? `offload ${m.layers_on_host}/${m.layers_total} layers` : 'offload embeddings');
  if (m.max_device_memory_gb) parts.push(`cap ${m.max_device_memory_gb} GB`);
  if (m.quant && m.quant !== 'none') parts.push(`${m.quant}${m.quant_backend ? ` (${m.quant_backend})` : ''}`);
  return parts.join(' · ');
}
function memoryTip(m) {
  const lines = [`CLEF_OFFLOAD=${m.offload}`, `device weights ${m.device_weights_gb} GB, host weights ${m.host_weights_gb} GB`];
  if (m.embeddings_on_host) lines.push('token and output embeddings on the host');
  return lines.join('\n');
}

/** Append a new point (new t) or replace the last one (same t). */
function applyPoint(ts, p, max) {
  if (!ts || p.t == null) return ts;
  const n = ts.t.length;
  const lt = n ? ts.t[n - 1] : -Infinity;
  if (p.t < lt) return ts;
  const out = {};
  for (const c of COLS) out[c] = (ts[c] || []).slice();
  if (p.t === lt) COLS.forEach((c) => { out[c][n - 1] = p[c] ?? null; });
  else COLS.forEach((c) => { out[c].push(p[c] ?? null); });
  if (out.t.length > max) COLS.forEach((c) => { out[c] = out[c].slice(out[c].length - max); });
  return { ...ts, ...out };
}

/** Aggregate column `key` over [from, to). mode: mean | wmean (weighted by rps) | last. */
function agg(ts, key, from, to, mode) {
  const col = ts[key] || [];
  const w = ts.rps || [];
  let s = 0, d = 0, lastV = null;
  for (let i = Math.max(0, from); i < Math.min(col.length, to); i++) {
    const v = col[i];
    if (v == null) continue;
    lastV = v;
    const wt = mode === 'wmean' ? (w[i] || 0) : 1;
    s += v * wt; d += wt;
  }
  if (mode === 'last') return lastV;
  if (!d) return mode === 'wmean' ? agg(ts, key, from, to, 'mean') : null;
  return s / d;
}

const KPIS = [
  { id: 'rps', label: 'Requests / s', key: 'rps', mode: 'mean', fmt: (v) => fmtNum(v, 2), good: null },
  { id: 'p50', label: 'Latency p50', key: 'p50_ms', mode: 'wmean', fmt: fmtMs, good: 'down' },
  { id: 'p95', label: 'Latency p95', key: 'p95_ms', mode: 'wmean', fmt: fmtMs, good: 'down' },
  { id: 'err', label: 'Error rate', key: 'error_rate', mode: 'wmean', fmt: (v) => fmtPct(v, 1), good: 'down', abs: true },
  { id: 'queue', label: 'Queue depth', key: 'queue_depth', mode: 'mean', fmt: (v) => fmtNum(v, 1), good: 'down' },
  { id: 'batch', label: 'Avg batch', key: 'avg_batch', mode: 'wmean', fmt: (v) => fmtNum(v, 2), good: null },
  { id: 'mem', label: 'Memory', key: 'mem_used_gb', mode: 'last', fmt: (v) => `${fmtNum(v, 1)} GB`, good: null },
  { id: 'gpu', label: 'GPU util', key: 'gpu_util_pct', mode: 'mean', fmt: (v) => `${Math.round(v)}%`, good: null },
];

function Delta({ cur, prev, kpi, label }) {
  if (cur == null || prev == null) return html`<span class="kpi-delta flat" title="No previous window to compare">-</span>`;
  const diff = cur - prev;
  const rel = kpi.abs ? diff * 100 : prev === 0 ? (cur === 0 ? 0 : 100) * Math.sign(diff || 1) : (diff / Math.abs(prev)) * 100;
  const flat = Math.abs(rel) < (kpi.abs ? 0.1 : 1);
  const up = rel > 0;
  const tone = flat || !kpi.good ? 'flat' : (kpi.good === 'down') === !up ? 'good' : 'bad';
  const txt = `${Math.abs(rel).toFixed(Math.abs(rel) < 10 ? 1 : 0)}${kpi.abs ? ' pp' : '%'}`;
  return html`<span class=${`kpi-delta ${tone}`} title=${`vs ${label}`} aria-label=${flat ? `unchanged vs ${label}` : `${up ? 'up' : 'down'} ${txt} vs ${label}`}>${flat ? '=' : up ? '▲' : '▼'} ${flat ? 'flat' : txt}</span>`;
}

function Kpi({ kpi, ts, cw, loading, unavailable, extra, win }) {
  if (loading) return html`<div class="kpi" aria-busy="true"><div class="kpi-label">${kpi.label}</div><div class="skel" style=${{ height: '34px', width: '60%', marginTop: '4px' }}></div><div class="skel kpi-spark" style=${{ marginTop: 'auto' }}></div></div>`;
  const n = ts.t.length;
  const halves = n <= POINTS;
  const per = halves ? Math.floor(n / 2) : cw;
  const cur = agg(ts, kpi.key, n - per, n, kpi.mode);
  const prev = n >= per * 2 ? agg(ts, kpi.key, n - 2 * per, n - per, kpi.mode) : null;
  const spark = ts[kpi.key].slice(-POINTS);
  const prevLabel = `previous ${halves ? Math.round(win / 2 / 60) : Math.round(win / 60)} min`;
  return html`<div class="kpi" role="group" aria-label=${kpi.label}>
    <div class="kpi-label">${kpi.label}</div>
    <div class="kpi-row">
      <div class="kpi-value mono">${unavailable ? html`<span class="muted">n/a</span>` : cur == null ? html`<span class="muted">-</span>` : kpi.fmt(cur)}</div>
      ${unavailable ? null : html`<${Delta} cur=${cur} prev=${prev} kpi=${kpi} label=${prevLabel} />`}
    </div>
    ${extra || null}
    <div class="kpi-spark">${unavailable ? html`<div class="muted small">telemetry unavailable</div>` : html`<${Sparkline} values=${spark} label=${kpi.label} />`}</div>
  </div>`;
}

const statusKind = (s) => (s >= 500 ? 'bad' : s === 429 ? 'serious' : s >= 400 ? 'warn' : 'ok');
const statusIcon = { ok: '●', warn: '▲', serious: '◆', bad: '■' };

function ChartCard({ title, sub, children, tools }) {
  return html`<section class="card"><div class="card-head"><h3>${title}${sub ? html`<span class="sub">${sub}</span>` : null}</h3>${tools || null}</div>${children}</section>`;
}

function Unavailable({ title, reason, height = 200 }) {
  return html`<div class="chart-unavail" style=${{ '--h': `${height}px` }} role="status"><strong>${title}</strong><span class="small">${reason}</span></div>`;
}

const edgeLabel = (v) => (v >= 1000 ? `${v / 1000}k` : String(v));
function latencyBins(hist) {
  if (!hist || !hist.edges) return [];
  const { edges, counts } = hist;
  const tot = counts.reduce((a, b) => a + b, 0) || 1;
  return counts.map((c, i) => {
    const label = i === 0 ? `<${edgeLabel(edges[0])}` : i === edges.length ? `≥${edgeLabel(edges[edges.length - 1])}` : `${edgeLabel(edges[i - 1])}-${edgeLabel(edges[i])}`;
    return { id: i, label, value: c, pct: fmtPct(c / tot, 0) };
  });
}
function batchBins(col) {
  const vals = col.filter((v) => v != null && v > 0).map((v) => Math.round(v));
  if (!vals.length) return [];
  const max = Math.max(4, ...vals);
  const width = Math.ceil(max / 12);
  const nb = Math.ceil(max / width);
  const bins = Array.from({ length: nb }, (_, i) => ({ id: i, label: width === 1 ? String(i + 1) : `${i * width + 1}-${(i + 1) * width}`, value: 0 }));
  vals.forEach((v) => { bins[Math.min(nb - 1, Math.floor((v - 1) / width))].value++; });
  return bins.map((b) => ({ ...b, pct: fmtPct(b.value / vals.length, 0) }));
}

function HList({ data, empty }) {
  const rows = Object.entries(data || {}).sort((a, b) => b[1] - a[1]).slice(0, 8);
  if (!rows.length) return html`<div class="muted small">${empty}</div>`;
  const max = Math.max(...rows.map((r) => r[1]));
  const tot = rows.reduce((a, r) => a + r[1], 0);
  return html`<div class="hlist" role="list">${rows.map(([k, v]) => html`<div class="hrow" role="listitem" key=${k} title=${`${k}: ${v} (${fmtPct(v / tot, 0)})`}>
    <span class="nm mono">${k}</span><span class="tr"><span class="fl" style=${{ display: 'block', width: `${(v / max) * 100}%` }}></span></span><span class="vv mono">${v}</span></div>`)}</div>`;
}

export function Ops() {
  const { health } = useStore();
  const [win, setWin] = useState(() => { const w = lsGet('clef.opsWindow', 300); return WINDOWS.some((x) => x.s === w) ? w : 300; });
  const [ts, setTs] = useState(null);
  const [st, setSt] = useState(null);
  const [avail, setAvail] = useState('loading'); // loading | ok | unavailable
  const [mode, setMode] = useState('connecting'); // sse | poll | connecting
  const [logs, setLogs] = useState([]);
  const [logFilter, setLogFilter] = useState('all');
  const lastId = useRef(null);
  const logSeen = useRef(new Set());
  const histRef = useRef(null), batchRef = useRef(null);
  const step = stepFor(win);
  const w2 = Math.min(win * 2, 3600);
  const maxPts = Math.ceil(w2 / step);

  const setWindow = (w) => { setWin(w); try { localStorage.setItem('clef.opsWindow', JSON.stringify(w)); } catch { /* ignore */ } };

  useEffect(() => {
    let dead = false, es = null, pollTimer = null, retryTimer = null;
    setTs(null); setMode('connecting');
    const addLog = (entries) => {
      if (dead || !entries.length) return;
      lastId.current = entries[entries.length - 1].id;
      setLogs((old) => {
        const fresh = entries.filter((e) => !logSeen.current.has(e.id));
        fresh.forEach((e) => logSeen.current.add(e.id));
        return [...fresh.reverse(), ...old].slice(0, MAX_LOG);
      });
    };
    const loadSeries = async () => {
      const r = await api.timeseries(w2, step);
      if (!dead) { setTs(r.data); setAvail('ok'); }
    };
    const loadStats = async () => { const s = await api.stats(); if (!dead) { setSt(s.data); setAvail('ok'); } };
    const poll = async () => {
      try {
        await Promise.all([loadSeries(), loadStats()]);
        const l = await api.log(lastId.current); addLog((l.data && l.data.entries) || []);
      } catch (e) { if (!dead) setAvail((a) => (a === 'ok' ? a : 'unavailable')); }
      if (!dead) pollTimer = setTimeout(poll, 3000);
    };
    const startPolling = () => { if (dead || pollTimer) return; setMode('poll'); poll(); };
    const startSse = () => {
      const key = api.getKey();
      try { es = new EventSource(`/v1/events?step_s=${step}${key ? `&key=${encodeURIComponent(key)}` : ''}`); } catch { startPolling(); return; }
      let opened = false;
      es.onopen = () => { opened = true; setMode('sse'); };
      es.addEventListener('stats', (e) => {
        try {
          const b = JSON.parse(e.data);
          if (dead) return;
          setSt(b);
          if (b.point) setTs((old) => applyPoint(old, b.point, maxPts));
        } catch { /* ignore */ }
      });
      es.addEventListener('log', (e) => { try { addLog([JSON.parse(e.data)]); } catch { /* ignore */ } });
      es.onerror = () => { if (!opened || es.readyState === 2) { es.close(); startPolling(); } };
    };
    (async () => {
      try {
        await Promise.all([loadSeries(), loadStats()]);
        if (!lastId.current) { const l = await api.log(null); addLog((l.data && l.data.entries) || []); }
        startSse();
      } catch (e) {
        if (dead) return;
        setAvail('unavailable');
        retryTimer = setTimeout(() => { if (!dead) startPolling(); }, 6000);
      }
    })();
    return () => { dead = true; if (es) es.close(); clearTimeout(pollTimer); clearTimeout(retryTimer); };
  }, [win]);

  const hd = health.data || {};
  const tele = hd.telemetry || (st && st.telemetry) || null;
  const gpuInfo = hd.gpu || (st && st.gpu) || null;
  const memTotal = (tele && tele.mem_total_gb) || (gpuInfo && gpuInfo.vram_total_gb) || null;
  const view = useMemo(() => {
    if (!ts) return null;
    const n = ts.t.length, from = Math.max(0, n - POINTS);
    const o = {};
    COLS.forEach((c) => { o[c] = (ts[c] || []).slice(from); });
    return o;
  }, [ts]);
  const anyGpu = ts && ts.gpu_util_pct && ts.gpu_util_pct.some((v) => v != null);
  const gpuUnavailable = ts && !anyGpu && !(tele && tele.available);
  const memAny = ts && ts.mem_used_gb && ts.mem_used_gb.some((v) => v != null);

  if (avail === 'unavailable') {
    return html`<div class="page"><${Empty}><span class="glyph" aria-hidden="true">!</span><strong>Stats unavailable</strong>
      <span>This server does not expose /v1/stats, /v1/stats/timeseries or /v1/events yet (or the API key is wrong). Retrying in the background.</span><//></div>`;
  }
  const loading = !ts;
  const cw = win === 3600 ? 30 : POINTS;
  const s = st || {};
  const lat = s.latency_ms || {};
  const rows = logFilter === 'errors' ? logs.filter((e) => e.status >= 400) : logs;
  const winLabel = WINDOWS.find((x) => x.s === win).label;

  const chartFmt = {
    ms: (v) => fmtMs(v), rps: (v) => `${fmtNum(v, 2)} req/s`, gb: (v) => `${fmtNum(v, 2)} GB`, pct: (v) => `${fmtNum(v, 0)}%`,
  };

  return html`<div class="page ops">
    <div class="ops-head">
      <h2>Operations</h2>
      <div class="ops-meta">
        <span class=${`live ${mode === 'sse' ? 'on' : ''}`} role="status"><span class="dot"></span>${mode === 'sse' ? 'Live' : mode === 'poll' ? 'Polling 3s' : 'Connecting'}</span>
        <div class="seg" role="radiogroup" aria-label="Time window">
          ${WINDOWS.map((w) => html`<button key=${w.s} type="button" role="radio" aria-checked=${win === w.s} class=${`seg-btn ${win === w.s ? 'active' : ''}`} onClick=${() => setWindow(w.s)}>${w.label}</button>`)}
        </div>
      </div>
    </div>

    <div class="statstrip" aria-label="Server summary">
      <div class="it"><span class="k">Status</span><span class="v">${s.status || hd.status || '-'}</span></div>
      <div class="it"><span class="k">Uptime</span><span class="v mono">${s.uptime_s != null ? fmtDur(s.uptime_s) : '-'}</span></div>
      <div class="it"><span class="k">Requests</span><span class="v mono">${s.total ?? '-'}</span></div>
      <div class="it"><span class="k">Errors</span><span class="v mono">${s.errors ?? '-'}</span></div>
      <div class="it"><span class="k">In flight</span><span class="v mono">${s.in_flight ?? '-'}</span></div>
      <div class="it"><span class="k">Backend</span><span class="v mono">${hd.backend ? `${hd.backend} · ${hd.dtype || ''}${hd.quant && hd.quant !== 'none' ? ` · ${hd.quant}` : ''}` : '-'}</span></div>
      ${memoryLabel(hd.memory) ? html`<div class="it"><span class="k">Memory mode</span><span class="v mono" title=${memoryTip(hd.memory)}>${memoryLabel(hd.memory)}</span></div>` : null}
      <div class="it"><span class="k">Padding</span><span class="v mono">${s.forward && s.forward.padding_ratio != null ? fmtNum(s.forward.padding_ratio, 2) : '-'}</span></div>
    </div>

    <div class="kpis">
      ${KPIS.map((k) => {
        const un = k.id === 'gpu' && gpuUnavailable;
        const extra = k.id === 'mem' && memTotal && ts ? html`<div class="vram"><div class="vram-track" style=${{ width: '100%' }}><div class="vram-alloc" style=${{ width: `${Math.min(100, ((agg(ts, 'mem_used_gb', ts.t.length - 1, ts.t.length, 'last') || 0) / memTotal) * 100)}%` }}></div></div></div>` : null;
        return html`<${Kpi} key=${k.id} kpi=${k} ts=${ts} cw=${cw} win=${win} loading=${loading} unavailable=${un} extra=${extra} />`;
      })}
    </div>
    <div class="muted small" style=${{ marginTop: '-8px' }}>Deltas compare the ${win === 3600 ? 'last 30 minutes with the 30 before' : `last ${winLabel} with the ${winLabel} before`}. Latency values are rps-weighted means of per-step percentiles.</div>

    <div class="grid2">
      <${ChartCard} title="Latency" sub=${`p50 and p95 · ${winLabel}`}>
        ${loading ? html`<div class="skel" style=${{ height: '200px' }}></div>` : html`<${TimeChart} t=${view.t} stepS=${step} height=${200} band=${['p95', 'p50']} ariaLabel=${`Latency p50 and p95 over the last ${winLabel}`}
          fmtAxis=${(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}s` : `${Math.round(v)}`)}
          series=${[{ key: 'p50', label: 'p50', color: '--s1', data: view.p50_ms, fmt: chartFmt.ms }, { key: 'p95', label: 'p95', color: '--s2', data: view.p95_ms, fmt: chartFmt.ms }]} />`}
      <//>
      <${ChartCard} title="Throughput" sub=${`requests per second · ${winLabel}`}>
        ${loading ? html`<div class="skel" style=${{ height: '200px' }}></div>` : html`<${TimeChart} t=${view.t} stepS=${step} height=${200} ariaLabel=${`Throughput over the last ${winLabel}`}
          series=${[{ key: 'rps', label: 'Requests / s', color: '--s1', data: view.rps, fill: 0.14, fmt: chartFmt.rps }]} />`}
      <//>
      <${ChartCard} title="Queue depth" sub="max in step">
        ${loading ? html`<div class="skel" style=${{ height: '150px' }}></div>` : html`<${TimeChart} t=${view.t} stepS=${step} height=${150} ariaLabel=${`Queue depth over the last ${winLabel}`}
          series=${[{ key: 'queue', label: 'Queue depth', color: '--s1', data: view.queue_depth, fill: 0.14, fmt: (v) => fmtNum(v, 0) }]} />`}
      <//>
      <${ChartCard} title="Batch size" sub="average records per forward">
        ${loading ? html`<div class="skel" style=${{ height: '150px' }}></div>` : html`<${TimeChart} t=${view.t} stepS=${step} height=${150} ariaLabel=${`Average batch size over the last ${winLabel}`}
          series=${[{ key: 'batch', label: 'Avg batch', color: '--s2', data: view.avg_batch, fill: 0.14, fmt: (v) => fmtNum(v, 2) }]} />`}
      <//>
      <${ChartCard} title=${gpuInfo && gpuInfo.memory_kind === 'unified' ? 'Memory (unified)' : 'Memory'} sub=${memTotal ? `GB used of ${fmtNum(memTotal, 1)}` : 'GB used'}>
        ${loading ? html`<div class="skel" style=${{ height: '150px' }}></div>` : !memAny ? html`<${Unavailable} height=${150} title="No memory samples yet" reason="The server reports memory with each gauge sample (every couple of seconds)." />`
          : html`<${TimeChart} t=${view.t} stepS=${step} height=${150} yMax=${memTotal || undefined} ariaLabel=${`Memory used over the last ${winLabel}`}
            series=${[{ key: 'mem', label: 'Memory used', color: '--s1', data: view.mem_used_gb, fill: 0.14, fmt: chartFmt.gb }]} />`}
      <//>
      <${ChartCard} title="GPU utilization" sub=${tele && tele.source ? `% · ${tele.source}` : '%'}>
        ${loading ? html`<div class="skel" style=${{ height: '150px' }}></div>` : gpuUnavailable
          ? html`<${Unavailable} height=${150} title="GPU telemetry unavailable" reason=${`${hd.backend ? `The ${hd.backend} backend` : 'This backend'} exposes no utilization counters here (for example AMD under WSL, Apple Silicon or CPU). Set CLEF_TELEMETRY=1 and install pynvml/amdsmi where supported.`} />`
          : html`<${TimeChart} t=${view.t} stepS=${step} height=${150} yMax=${100} ariaLabel=${`GPU utilization over the last ${winLabel}`}
            series=${[{ key: 'gpu', label: 'GPU util', color: '--s3', data: view.gpu_util_pct, fill: 0.14, fmt: chartFmt.pct }]} />
            <div class="small muted mono" style=${{ marginTop: '6px' }}>temp ${last(view.gpu_temp_c) != null ? `${fmtNum(last(view.gpu_temp_c), 0)} C` : '-'} · power ${last(view.gpu_power_w) != null ? `${fmtNum(last(view.gpu_power_w), 0)} W` : '-'}</div>`}
      <//>
    </div>

    <div class="grid2">
      <${ChartCard} title="Latency distribution" sub=${`successful requests, last ${winLabel}`} tools=${html`<${ExportMenu} host=${histRef} name="clef-latency-histogram" />`}>
        <div class="pcts">${[['p50', lat.p50], ['p95', lat.p95], ['p99', lat.p99], ['max', lat.max]].map(([k, v]) => html`<div key=${k} class="pct"><span class="muted small">${k}</span><span class="mono">${v != null ? Math.round(v) : '-'}<small> ms</small></span></div>`)}</div>
        <div ref=${histRef}>${lat.hist ? html`<${BarChart} bins=${latencyBins(lat.hist)} width=${560} height=${170} unit="ms" yLabel="requests" ariaLabel=${`Latency histogram in milliseconds; ${(lat.hist.counts || []).reduce((a, b) => a + b, 0)} requests, p50 ${Math.round(lat.p50 || 0)} ms, p95 ${Math.round(lat.p95 || 0)} ms`} />` : html`<div class="skel" style=${{ height: '170px' }}></div>`}</div>
      <//>
      <${ChartCard} title="Batch-size distribution" sub=${`avg batch per ${step}s step · ${winLabel}`} tools=${html`<${ExportMenu} host=${batchRef} name="clef-batch-size-distribution" />`}>
        <div ref=${batchRef}>${loading ? html`<div class="skel" style=${{ height: '170px', marginTop: '46px' }}></div>`
          : batchBins(view.avg_batch).length ? html`<div style=${{ height: '46px' }} class="small muted">Steps with at least one forward, by rounded average batch size. Padding ratio ${s.forward && s.forward.padding_ratio != null ? fmtNum(s.forward.padding_ratio, 2) : '-'}.</div><${BarChart} bins=${batchBins(view.avg_batch)} width=${560} height=${170} unit="records" yLabel="steps" ariaLabel="Distribution of average batch size per step" />`
          : html`<${Unavailable} height=${216} title="No batches in this window" reason="Send some requests; each forward pass reports its batch size." />`}</div>
      <//>
    </div>

    <div class="grid2">
      <${ChartCard} title="Traffic by endpoint" sub=${`requests, last ${winLabel}`}><${HList} data=${s.by_endpoint} empty="No requests in the window." /><//>
      <${ChartCard} title="Traffic by API key" sub="key name, never the key"><${HList} data=${s.by_key} empty="No requests in the window." /><//>
    </div>

    <section class="card">
      <div class="card-head"><h3>Request log<span class="sub">newest first · ${rows.length} shown</span></h3>
        <div class="seg" role="radiogroup" aria-label="Log filter" style=${{ margin: 0 }}>
          ${[['all', 'All'], ['errors', 'Errors']].map(([id, l]) => html`<button key=${id} type="button" role="radio" aria-checked=${logFilter === id} class=${`seg-btn ${logFilter === id ? 'active' : ''}`} onClick=${() => setLogFilter(id)}>${l}</button>`)}
        </div></div>
      ${!rows.length ? html`<${Empty}><span class="glyph" aria-hidden="true">≡</span><strong>${logs.length ? 'No errors logged' : 'No requests yet'}</strong><span>${logs.length ? 'Everything in the buffer succeeded.' : 'Requests appear here as they arrive (try the Playground).'}</span><//>` : html`<div class="tablewrap logwrap"><table class="table log">
        <caption class="sr-only">Recent requests</caption>
        <thead><tr><th>Time</th><th>Endpoint</th><th>Key</th><th>Status</th><th class="num">ms</th><th class="num">Tokens</th><th class="num">Rec</th><th class="num">Q</th><th>Media</th><th>Error / state</th></tr></thead>
        <tbody>${rows.slice(0, 100).map((e) => { const k = statusKind(e.status); return html`<tr key=${e.id} class=${`row-${k}`}>
          <td class="mono nowrap">${fmtTime(e.ts * 1000)}</td><td class="mono">${e.endpoint}</td>
          <td class="mono small">${e.key || html`<span class="muted">-</span>`}</td>
          <td><span class=${`status-pill ${k}`}><span aria-hidden="true">${statusIcon[k]}</span>${e.status}</span></td>
          <td class="num mono">${e.ms != null ? Math.round(e.ms) : '-'}</td>
          <td class="num mono">${e.input_tokens ?? '-'}</td><td class="num mono">${e.n_records ?? '-'}</td><td class="num mono">${e.n_questions ?? '-'}</td>
          <td class="mono small">${e.media ? `${e.media.images || 0}i/${e.media.videos || 0}v` : '-'}</td>
          <td class="small">${e.error ? html`<span class="err">${e.error}</span>` : e.state_preview ? html`<span class="muted clamp">${e.state_preview}</span>` : ''}</td></tr>`; })}</tbody></table></div>`}
    </section>
  </div>`;
}

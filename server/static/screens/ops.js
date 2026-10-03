// Ops: live tiles, latency histogram, throughput sparkline, request log (SSE with polling fallback).
import { html, useState, useEffect, useRef } from '../vendor/standalone.module.js';
import * as api from '../api.js';
import { fmtDur, fmtTime, fmtNum } from '../util.js';
import { Tile, VramBar, Empty } from '../components/common.js';

const MAX_SAMPLES = 150; // about 5 min at 2 s
const MAX_LOG = 200;
const samples = []; // module level so the sparkline survives tab switches

function Sparkline({ data }) {
  const W = 320, H = 56, P = 4;
  if (data.length < 2) return html`<div class="muted small">collecting samples…</div>`;
  const max = Math.max(0.5, ...data.map((d) => d.v));
  const t0 = data[0].t, t1 = data[data.length - 1].t || t0 + 1;
  const x = (t) => P + ((t - t0) / Math.max(1, t1 - t0)) * (W - 2 * P);
  const y = (v) => H - P - (v / max) * (H - 2 * P);
  const pts = data.map((d) => `${x(d.t).toFixed(1)},${y(d.v).toFixed(1)}`).join(' ');
  return html`<svg viewBox=${`0 0 ${W} ${H}`} class="spark" role="img" aria-label=${`Throughput over last ${Math.round((t1 - t0) / 1000)} seconds, max ${max.toFixed(1)} requests per second`}>
    <polyline points=${`${P},${H - P} ${pts} ${W - P},${H - P}`} class="spark-area" />
    <polyline points=${pts} class="spark-line" fill="none" />
    <text x=${W - P} y="10" text-anchor="end" class="svg-label">${max.toFixed(1)} rps max</text>
  </svg>`;
}

function Histogram({ hist }) {
  if (!hist || !hist.edges) return html`<div class="muted small">no histogram data</div>`;
  const { edges, counts } = hist;
  const labels = counts.map((_, i) => (i === 0 ? `<${edges[0]}` : i === edges.length ? `≥${edges[edges.length - 1]}` : `${edges[i - 1]}-${edges[i]}`));
  const max = Math.max(1, ...counts);
  const W = 560, H = 150, L = 8, B = 30, T = 14;
  const bw = (W - 2 * L) / counts.length;
  return html`<svg viewBox=${`0 0 ${W} ${H}`} class="hist" role="img" aria-label="Latency histogram, milliseconds">
    <line x1=${L} x2=${W - L} y1=${H - B} y2=${H - B} class="axis-line" />
    ${counts.map((c, i) => {
      const h = (c / max) * (H - B - T);
      return html`<g key=${i}>
        <title>${labels[i]} ms: ${c}</title>
        <rect x=${L + i * bw + 3} y=${H - B - h} width=${bw - 6} height=${Math.max(c ? 1 : 0, h)} rx="2" class="hist-bar" />
        ${c ? html`<text x=${L + i * bw + bw / 2} y=${H - B - h - 4} text-anchor="middle" class="svg-label">${c}</text>` : null}
        <text x=${L + i * bw + bw / 2} y=${H - B + 13} text-anchor="middle" class="svg-label">${labels[i]}</text>
      </g>`;
    })}
    <text x=${W - L} y=${H - 3} text-anchor="end" class="svg-label">ms</text>
  </svg>`;
}

const statusClass = (s) => (s >= 500 ? 'bad' : s >= 400 ? 'warn' : 'ok');

export function Ops() {
  const [st, setSt] = useState(null);
  const [avail, setAvail] = useState('loading'); // loading | ok | unavailable
  const [mode, setMode] = useState('connecting'); // sse | poll | connecting
  const [logs, setLogs] = useState([]);
  const [series, setSeries] = useState(samples.slice());
  const lastId = useRef(null);

  useEffect(() => {
    let dead = false, es = null, pollTimer = null;
    const pushStats = (s) => {
      if (dead) return;
      setSt(s); setAvail('ok');
      samples.push({ t: Date.now(), v: s.rps || 0 });
      while (samples.length > MAX_SAMPLES) samples.shift();
      setSeries(samples.slice());
    };
    const addLog = (entries) => {
      if (dead || !entries.length) return;
      lastId.current = entries[entries.length - 1].id;
      setLogs((old) => {
        const seen = new Set(old.map((e) => e.id));
        return [...entries.filter((e) => !seen.has(e.id)).reverse(), ...old].slice(0, MAX_LOG);
      });
    };
    const poll = async () => {
      try {
        const s = await api.stats(); pushStats(s.data);
        const l = await api.log(lastId.current); addLog((l.data && l.data.entries) || []);
      } catch (e) { if (!dead) setAvail((a) => (a === 'ok' ? a : 'unavailable')); }
      if (!dead) pollTimer = setTimeout(poll, 2000);
    };
    const startPolling = () => { if (dead || pollTimer) return; setMode('poll'); poll(); };
    const startSse = () => {
      const key = api.getKey();
      try {
        es = new EventSource(`/v1/events${key ? `?key=${encodeURIComponent(key)}` : ''}`);
      } catch { startPolling(); return; }
      let opened = false;
      es.onopen = () => { opened = true; setMode('sse'); };
      es.addEventListener('stats', (e) => { try { pushStats(JSON.parse(e.data)); } catch { /* ignore */ } });
      es.addEventListener('log', (e) => { try { addLog([JSON.parse(e.data)]); } catch { /* ignore */ } });
      es.onerror = () => {
        // EventSource retries on its own for dropped streams; for hard failures (404/401) it closes.
        if (!opened || es.readyState === 2) { es.close(); startPolling(); }
      };
    };
    (async () => {
      try {
        const s = await api.stats(); pushStats(s.data);
        const l = await api.log(null); addLog((l.data && l.data.entries) || []);
        startSse();
      } catch (e) {
        if (dead) return;
        setAvail('unavailable');
        setSt(null);
        pollTimer = setTimeout(() => { pollTimer = null; if (!dead) startPolling(); }, 6000); // retry in case the backend comes up
      }
    })();
    return () => { dead = true; if (es) es.close(); clearTimeout(pollTimer); };
  }, []);

  if (avail === 'unavailable') {
    return html`<div class="page"><${Empty}><strong>Stats unavailable.</strong><br />This server does not expose /v1/stats, /v1/log or /v1/events (or the API key is wrong). Retrying in the background.<//></div>`;
  }
  const s = st || {};
  const lat = s.latency_ms || {};
  const fw = s.forward || {};
  return html`<div class="page ops">
    <div class="pane-head"><h2>Operations</h2>
      <span class="muted small">${mode === 'sse' ? 'live (SSE)' : mode === 'poll' ? 'polling every 2s' : 'connecting…'} · window ${s.window_s || '-'}s</span></div>
    <div class="tiles">
      <${Tile} label="Status" value=${s.status || '-'} tone=${s.status === 'ready' ? 'good' : s.status === 'error' ? 'bad' : ''} />
      <${Tile} label="Uptime" value=${fmtDur(s.uptime_s)} />
      <${Tile} label="Requests" value=${s.total ?? '-'} />
      <${Tile} label="Errors" value=${s.errors ?? '-'} tone=${s.errors > 0 ? 'warn' : ''} />
      <${Tile} label="In flight" value=${s.in_flight ?? '-'} />
      <${Tile} label="Queue depth" value=${s.queue_depth ?? '-'} />
      <${Tile} label="Req / s" value=${s.rps != null ? fmtNum(s.rps) : '-'} />
      <${Tile} label="Avg batch" value=${fw.avg_batch != null ? fmtNum(fw.avg_batch) : '-'} sub=${fw.avg_ms != null ? `${fmtNum(fw.avg_ms, 0)} ms / forward` : null} />
      <${Tile} label="Padding ratio" value=${fw.padding_ratio != null ? fmtNum(fw.padding_ratio, 2) : '-'} sub=${s.tokens ? `${fmtNum(s.tokens.avg_in, 0)} avg tokens in` : null} />
      <${Tile} label="VRAM" value=${s.gpu && s.gpu.vram_total_gb ? `${fmtNum(s.gpu.vram_allocated_gb, 1)} GB` : '-'}><${VramBar} gpu=${s.gpu} /><//>
    </div>
    <div class="grid2">
      <section class="card"><h3>Latency <span class="muted small">window</span></h3>
        <div class="pcts">${[['p50', lat.p50], ['p95', lat.p95], ['p99', lat.p99], ['max', lat.max]].map(([k, v]) => html`<div key=${k} class="pct"><span class="muted small">${k}</span><span class="mono">${v != null ? Math.round(v) : '-'}<small> ms</small></span></div>`)}</div>
        <${Histogram} hist=${lat.hist} /></section>
      <section class="card"><h3>Throughput <span class="muted small">last ~5 min</span></h3><${Sparkline} data=${series.map((d) => ({ t: d.t, v: d.v }))} /></section>
    </div>
    <section class="card"><h3>Request log <span class="muted small">newest first</span></h3>
      ${!logs.length ? html`<${Empty}>No requests logged yet.<//>` : html`<div class="tablewrap logwrap"><table class="table log">
        <thead><tr><th>Time</th><th>Endpoint</th><th class="num">Status</th><th class="num">ms</th><th class="num">Tokens</th><th class="num">Rec</th><th class="num">Q</th><th>Media</th><th>Error / state</th></tr></thead>
        <tbody>${logs.map((e) => html`<tr key=${e.id}>
          <td class="mono nowrap">${fmtTime(e.ts * 1000)}</td><td class="mono">${e.endpoint}</td>
          <td class=${`num mono st-${statusClass(e.status)}`}>${e.status}</td><td class="num mono">${e.ms != null ? Math.round(e.ms) : '-'}</td>
          <td class="num mono">${e.input_tokens ?? '-'}</td><td class="num mono">${e.n_records ?? '-'}</td><td class="num mono">${e.n_questions ?? '-'}</td>
          <td class="mono small">${e.media ? `${e.media.images || 0}i/${e.media.videos || 0}v` : '-'}</td>
          <td class="small">${e.error ? html`<span class="err">${e.error}</span>` : e.state_preview ? html`<span class="muted clamp">${e.state_preview}</span>` : ''}</td></tr>`)}</tbody></table></div>`}
    </section>
  </div>`;
}

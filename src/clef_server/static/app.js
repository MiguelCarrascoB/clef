// App shell: top bar (tabs, health, settings) + screen router.
import { html, render, useState, useEffect, useCallback } from './vendor/standalone.module.js';
import { S, useStore, goTab, applyTheme } from './store.js';
import { HealthPill, Settings, useHealthPolling } from './components/topbar.js';
import { Playground } from './screens/playground.js';
import { History } from './screens/history.js';
import { Batch } from './screens/batch.js';
import { Evaluate } from './screens/evaluate.js';
import { Ops } from './screens/ops.js';

const TABS = [
  { id: 'playground', label: 'Playground', view: Playground },
  { id: 'history', label: 'History', view: History },
  { id: 'batch', label: 'Batch', view: Batch },
  { id: 'evaluate', label: 'Evaluate', view: Evaluate },
  { id: 'ops', label: 'Ops', view: Ops },
];

function App() {
  const { tab, history } = useStore();
  const [settingsOpen, setSettingsOpen] = useState(false);
  useHealthPolling();
  useEffect(() => {
    const h = () => setSettingsOpen(true);
    window.addEventListener('clef:unauthorized', h);
    const hash = () => { const t = location.hash.replace('#', ''); if (TABS.some((x) => x.id === t) && t !== S.tab) goTab(t); };
    window.addEventListener('hashchange', hash);
    return () => { window.removeEventListener('clef:unauthorized', h); window.removeEventListener('hashchange', hash); };
  }, []);
  const cur = TABS.find((t) => t.id === tab) || TABS[0];
  // On narrow screens the tab strip scrolls sideways: keep the active tab visible.
  useEffect(() => {
    const el = document.querySelector('.tab.active');
    if (el && el.scrollIntoView) el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }, [cur.id]);
  const View = cur.view;
  const open = useCallback(() => setSettingsOpen(true), []);
  const close = useCallback(() => setSettingsOpen(false), []);
  return html`<div class="app">
    <header class="topbar">
      <div class="brand"><span class="logo" aria-hidden="true">♭</span><span class="brand-name">Clef Console</span></div>
      <nav class="tabs" aria-label="Screens">
        ${TABS.map((t) => html`<button key=${t.id} type="button" class=${`tab ${cur.id === t.id ? 'active' : ''}`} aria-current=${cur.id === t.id ? 'page' : null}
          onClick=${() => goTab(t.id)}>${t.label}${t.id === 'history' && history.length ? html`<span class="count">${history.length}</span>` : null}</button>`)}
      </nav>
      <div class="topbar-right"><${HealthPill} /><${Settings} open=${settingsOpen} onOpen=${open} onClose=${close} /></div>
    </header>
    <main class="main" id="main"><${View} /></main>
  </div>`;
}

applyTheme();
render(html`<${App} />`, document.getElementById('root'));

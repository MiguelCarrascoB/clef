// Media drop zone: images/videos -> data URLs, thumbnails, size warnings and /health limit enforcement.
import { html, useState, useRef } from '../vendor/standalone.module.js';
import { fmtBytes, uid } from '../util.js';
import { makeThumb } from '../store.js';

export const WARN_ONE = 8 * 1024 * 1024;
export const WARN_TOTAL = 25 * 1024 * 1024;

const readDataUrl = (file) => new Promise((resolve, reject) => {
  const r = new FileReader();
  r.onload = () => resolve(r.result);
  r.onerror = () => reject(r.error || new Error('read failed'));
  r.readAsDataURL(file);
});

export function MediaZone({ media, onChange, limits }) {
  const [drag, setDrag] = useState(false);
  const [msg, setMsg] = useState(null);
  const input = useRef(null);
  const total = media.reduce((n, m) => n + m.size, 0);
  const maxImg = limits && limits.max_images, maxVid = limits && limits.max_videos;
  const maxBodyBytes = limits && limits.max_body_mb ? limits.max_body_mb * 1048576 : null;

  const add = async (files) => {
    const next = media.slice(); const problems = [];
    for (const f of files) {
      const kind = f.type.startsWith('image/') ? 'image' : f.type.startsWith('video/') ? 'video' : null;
      if (!kind) { problems.push(`${f.name}: unsupported type`); continue; }
      const count = next.filter((m) => m.kind === kind).length;
      const cap = kind === 'image' ? maxImg : maxVid;
      if (cap != null && count >= cap) { problems.push(`${f.name}: server allows at most ${cap} ${kind}(s)`); continue; }
      // base64 inflates by ~4/3
      if (maxBodyBytes && (next.reduce((n, m) => n + m.size, 0) + f.size) * 1.34 > maxBodyBytes) {
        problems.push(`${f.name}: would exceed server body limit (${limits.max_body_mb} MB)`); continue;
      }
      try {
        const dataUrl = await readDataUrl(f);
        next.push({ id: uid(), name: f.name, kind, size: f.size, mime: f.type, dataUrl, thumb: kind === 'image' ? await makeThumb(dataUrl) : null });
      } catch { problems.push(`${f.name}: could not be read`); }
    }
    setMsg(problems.length ? problems.join('; ') : null);
    onChange(next);
  };
  const onDrop = (e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files.length) add([...e.dataTransfer.files]); };

  return html`<div class="media">
    <div class=${`drop ${drag ? 'over' : ''}`} tabIndex="0" role="button" aria-label="Add images or videos"
      onDragOver=${(e) => { e.preventDefault(); setDrag(true); }} onDragLeave=${() => setDrag(false)} onDrop=${onDrop}
      onClick=${() => input.current && input.current.click()}
      onKeyDown=${(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), input.current.click())}>
      <span>Drop images / videos here or <u>browse</u></span>
      <span class="muted small">${limits && (maxImg != null || maxVid != null) ? `server limits: ${maxImg ?? '?'} images, ${maxVid ?? '?'} videos` : 'sent inline as base64 data URLs'}</span>
      <input ref=${input} type="file" accept="image/*,video/*" multiple hidden onChange=${(e) => { add([...e.target.files]); e.target.value = ''; }} />
    </div>
    ${msg ? html`<div class="warn small" role="alert">${msg}</div>` : null}
    ${media.length ? html`<ul class="thumbs">
      ${media.map((m) => html`<li key=${m.id} class=${`thumb ${m.size > WARN_ONE ? 'warn-box' : ''}`}>
        <div class="thumb-img">${m.kind === 'image' ? html`<img src=${m.dataUrl} alt=${m.name} />` : html`<video src=${m.dataUrl} muted preload="metadata"></video>`}</div>
        <div class="thumb-meta"><span class="thumb-name" title=${m.name}>${m.name}</span>
          <span class="mono small muted">${m.kind} · ${fmtBytes(m.size)}</span>
          ${m.size > WARN_ONE ? html`<span class="warn small">large file (over 8 MB)</span>` : null}</div>
        <button class="btn icon sm" type="button" aria-label=${`Remove ${m.name}`} onClick=${() => onChange(media.filter((x) => x.id !== m.id))}>✕</button>
      </li>`)}
    </ul>
    <div class=${`small ${total > WARN_TOTAL ? 'warn' : 'muted'}`}>Total ${fmtBytes(total)}${total > WARN_TOTAL ? ' - over 25 MB, request may be rejected' : ''}</div>` : null}
  </div>`;
}

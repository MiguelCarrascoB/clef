"""Media ingestion: data URLs / (optional) SSRF-guarded URL fetch, image + video decode with size caps.

All functions here are blocking CPU/IO work: call them via `anyio.to_thread.run_sync` / `run_in_threadpool`.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import io
import ipaddress
import logging
import os
import re
import socket
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageOps

from .config import Config

log = logging.getLogger("clef.media")

Image.MAX_IMAGE_PIXELS = 128_000_000  # decompression-bomb guard: warn above, DecompressionBombError at 2x
FETCH_TIMEOUT_S = 30.0
_DATA_URL = re.compile(r"^data:([^;,]*)((?:;[^;,=]+(?:=[^;,]*)?)*),(.*)$", re.DOTALL | re.IGNORECASE)
_VIDEO_MIME_EXT = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/matroska": ".mkv",
    "video/x-m4v": ".mp4",
}


class MediaError(ValueError):
    """Invalid, unsupported or too-large media input (HTTP 400)."""


def max_bytes(cfg: Config) -> int:
    return cfg.max_body_mb * 1024 * 1024


# ---------------------------------------------------------------- fetching / decoding of the payload


def decode_data_url(url: str, limit: int) -> tuple[bytes, str]:
    """`data:<mime>;base64,<payload>` -> (bytes, mime). Validates base64 and the decoded-size cap."""
    m = _DATA_URL.match(url.strip())
    if not m:
        raise MediaError("media must be a data: URL (data:<mime>;base64,<bytes>) or an http(s) URL")
    mime, params, payload = m.group(1).lower(), m.group(2).lower(), m.group(3)
    if ";base64" not in params:
        raise MediaError("data URL must be base64-encoded")
    payload = "".join(payload.split())
    if len(payload) * 3 // 4 > limit + 3:
        raise MediaError(f"media exceeds the {limit // (1024 * 1024)} MB decoded size limit")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaError("data URL contains invalid base64") from exc
    if not raw:
        raise MediaError("media is empty")
    if len(raw) > limit:
        raise MediaError(f"media exceeds the {limit // (1024 * 1024)} MB decoded size limit")
    return raw, mime


def _check_public_host(host: str, port: int) -> None:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise MediaError(f"cannot resolve host {host!r}") from exc
    if not infos:
        raise MediaError(f"cannot resolve host {host!r}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise MediaError("URL resolves to a non-public address; refusing to fetch")


def fetch_url(url: str, cfg: Config) -> tuple[bytes, str]:
    """GET an http(s) URL with an SSRF guard, no redirects, a capped read and a total timeout."""
    import httpx

    if not cfg.allow_url_fetch:
        raise MediaError(
            "URL fetching is disabled; send media as data: URLs (server: CLEF_ALLOW_URL_FETCH=1)"
        )
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise MediaError("only http(s) URLs are supported")
    if parts.username or parts.password:
        raise MediaError("URLs with credentials are not allowed")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise MediaError("invalid URL port") from exc
    _check_public_host(parts.hostname, port)
    limit = cfg.url_fetch_max_mb * 1024 * 1024
    deadline = time.monotonic() + FETCH_TIMEOUT_S
    try:
        client = httpx.Client(follow_redirects=False, timeout=httpx.Timeout(10.0), trust_env=False)
        with client, client.stream("GET", url.strip()) as res:
            if 300 <= res.status_code < 400:
                raise MediaError("URL redirects are not followed")
            if res.status_code != 200:
                raise MediaError(f"URL fetch failed with HTTP {res.status_code}")
            declared = res.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise MediaError(f"URL content exceeds the {cfg.url_fetch_max_mb} MB limit")
            chunks: list[bytes] = []
            size = 0
            for chunk in res.iter_bytes(65536):
                size += len(chunk)
                if size > limit:
                    raise MediaError(f"URL content exceeds the {cfg.url_fetch_max_mb} MB limit")
                if time.monotonic() > deadline:
                    raise MediaError("URL fetch timed out")
                chunks.append(chunk)
            mime = res.headers.get("content-type", "").split(";")[0].strip().lower()
    except MediaError:
        raise
    except httpx.HTTPError as exc:
        raise MediaError(f"URL fetch failed: {type(exc).__name__}") from exc
    raw = b"".join(chunks)
    if not raw:
        raise MediaError("media is empty")
    return raw, mime


def load_bytes(ref: str, cfg: Config) -> tuple[bytes, str]:
    ref = ref.strip()
    low = ref[:8].lower()
    if low.startswith("data:"):
        return decode_data_url(ref, max_bytes(cfg))
    if low.startswith(("http://", "https://")):
        return fetch_url(ref, cfg)
    raise MediaError("media must be a data: URL (data:<mime>;base64,<bytes>) or an http(s) URL")


# ---------------------------------------------------------------- images


def fit_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    scale = (max_pixels / (w * h)) ** 0.5
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def decode_image(raw: bytes, max_pixels: int) -> Image.Image:
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise MediaError("image has too many pixels") from exc
    except Exception as exc:  # PIL raises many types (UnidentifiedImageError, OSError, ValueError, ...)
        raise MediaError("could not decode image (unsupported or corrupt data)") from exc
    return fit_pixels(rgb, max_pixels)


# ---------------------------------------------------------------- videos


def detect_container(raw: bytes, mime: str = "") -> str:
    """Return the file extension (with dot) of the video container, from magic bytes then mime."""
    head = raw[:64]
    if head[4:8] == b"ftyp":
        return ".mov" if head[8:12] == b"qt  " else ".mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ".webm" if b"webm" in raw[:256] else ".mkv"
    ext = _VIDEO_MIME_EXT.get(mime)
    if ext:
        return ext
    raise MediaError("unsupported video container (expected mp4, webm, mov or mkv)")


def _sample_indices(total: int, cap: int) -> list[int]:
    if total <= cap:
        return list(range(total))
    return sorted({int(i) for i in np.linspace(0, total - 1, cap)})


def decode_video(raw: bytes, mime: str, cfg: Config) -> np.ndarray:
    """Video bytes -> uint8 (T, H, W, 3): sampled at `video_fps`, evenly capped to `max_frames`, downscaled."""  # noqa: E501
    import imageio.v3 as iio

    ext = detect_container(raw, mime)
    fd, path = tempfile.mkstemp(suffix=ext, prefix="clef-vid-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
        try:
            src_fps = float(iio.immeta(path, plugin="FFMPEG").get("fps") or 0) or cfg.video_fps
        except Exception:
            src_fps = cfg.video_fps
        step = max(1, round(src_fps / cfg.video_fps)) if cfg.video_fps > 0 else 1
        frames: list[np.ndarray] = []
        try:
            for i, frame in enumerate(iio.imiter(path, plugin="FFMPEG")):
                if i % step:
                    continue
                if frame.ndim != 3 or frame.shape[-1] != 3:
                    raise MediaError("unsupported video pixel format")
                img = fit_pixels(Image.fromarray(np.asarray(frame, dtype=np.uint8)), cfg.max_pixels)
                frames.append(np.asarray(img, dtype=np.uint8))
        except MediaError:
            raise
        except Exception as exc:
            if not frames:
                raise MediaError("could not decode video (unsupported or corrupt data)") from exc
            log.warning("video decode stopped early after %d frames: %s", len(frames), exc)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
    if not frames:
        raise MediaError("video contains no frames")
    keep = _sample_indices(len(frames), cfg.max_frames)
    shapes = {frames[i].shape for i in keep}
    if len(shapes) != 1:
        raise MediaError("video frames have inconsistent sizes")
    return np.stack([frames[i] for i in keep])


# ---------------------------------------------------------------- entry point


def load_media(
    images: list[str] | None, videos: list[str] | None, cfg: Config
) -> tuple[list[Any], list[np.ndarray]]:
    """Resolve and decode every image/video reference of ONE request (blocking)."""
    if len(images or []) > cfg.max_images:
        raise MediaError(f"too many images (max {cfg.max_images})")
    if len(videos or []) > cfg.max_videos:
        raise MediaError(f"too many videos (max {cfg.max_videos})")
    out_images: list[Image.Image] = []
    for n, ref in enumerate(images or []):
        try:
            raw, mime = load_bytes(ref, cfg)
            if mime.startswith("video/"):
                raise MediaError("got a video where an image was expected")
            out_images.append(decode_image(raw, cfg.max_pixels))
        except MediaError as exc:
            raise MediaError(f"images[{n}]: {exc}") from exc
    out_videos: list[np.ndarray] = []
    for n, ref in enumerate(videos or []):
        try:
            raw, mime = load_bytes(ref, cfg)
            out_videos.append(decode_video(raw, mime, cfg))
        except MediaError as exc:
            raise MediaError(f"videos[{n}]: {exc}") from exc
    return out_images, out_videos

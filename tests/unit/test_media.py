"""media.py: data URLs, image/video decode, caps, SSRF guard."""

from __future__ import annotations

import base64
import io
import os
import tempfile

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

from clef_server import media
from clef_server.config import Config
from clef_server.media import MediaError, decode_data_url, detect_container, load_bytes, load_media


def png_bytes(w: int = 32, h: int = 16, mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (w, h), "red" if mode == "RGB" else 100).save(buf, "PNG")
    return buf.getvalue()


def data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def mp4_bytes(n: int = 60, fps: int = 10, size: tuple[int, int] = (64, 48)) -> bytes:
    frames = np.random.default_rng(0).integers(0, 255, (n, size[1], size[0], 3), dtype=np.uint8)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        iio.imwrite(path, frames, fps=fps, plugin="FFMPEG")
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


@pytest.fixture(scope="module")
def video() -> bytes:
    return mp4_bytes()


def test_data_url_roundtrip_and_validation():
    raw, mime = decode_data_url(data_url(b"hello", "image/x"), 100)
    assert raw == b"hello" and mime == "image/x"
    with pytest.raises(MediaError, match="invalid base64"):
        decode_data_url("data:image/png;base64,@@@@", 100)
    with pytest.raises(MediaError, match="base64-encoded"):
        decode_data_url("data:image/png,abcd", 100)
    with pytest.raises(MediaError, match="decoded size limit"):
        decode_data_url(data_url(b"x" * 500), 100)
    with pytest.raises(MediaError, match="data: URL"):
        decode_data_url("nonsense", 100)


def test_decoded_cap_from_max_body_mb():
    with pytest.raises(MediaError, match="1 MB"):
        load_bytes(data_url(b"0" * (1024 * 1024 + 10)), Config(max_body_mb=1))


def test_image_decode_converts_rgb():
    imgs, vids = load_media([data_url(png_bytes(mode="L"))], None, Config())
    assert imgs[0].mode == "RGB" and imgs[0].size == (32, 16) and vids == []


def test_image_downscale_keeps_aspect():
    (img,), _ = load_media([data_url(png_bytes(400, 200))], None, Config(max_pixels=1000))
    w, h = img.size
    assert w * h <= 1000 and abs(w / h - 2.0) < 0.15


def test_image_garbage_and_video_mime():
    with pytest.raises(MediaError, match=r"images\[0\].*could not decode image"):
        load_media([data_url(b"not an image")], None, Config())
    with pytest.raises(MediaError, match="video where an image"):
        load_media([data_url(png_bytes(), "video/mp4")], None, Config())


def test_count_caps():
    with pytest.raises(MediaError, match="too many images"):
        load_media([data_url(png_bytes())] * 3, None, Config(max_images=2))
    with pytest.raises(MediaError, match="too many videos"):
        load_media(None, ["x"] * 3, Config(max_videos=2))


def test_detect_container():
    assert detect_container(b"\x00\x00\x00\x18ftypmp42" + b"0" * 20) == ".mp4"
    assert detect_container(b"\x00\x00\x00\x14ftypqt  " + b"0" * 20) == ".mov"
    assert detect_container(b"\x1a\x45\xdf\xa3" + b"\x42\x82webm" + b"0" * 20) == ".webm"
    assert detect_container(b"\x1a\x45\xdf\xa3" + b"\x42\x82matroska" + b"0" * 20) == ".mkv"
    assert detect_container(b"garbage" * 5, "video/webm") == ".webm"
    with pytest.raises(MediaError, match="unsupported video container"):
        detect_container(b"garbage" * 5, "")


def test_video_sampling_and_frame_cap(video):
    # 60 frames @10fps, sampled at 5 fps -> 30 frames, capped evenly to 8
    _, (arr,) = load_media(None, [data_url(video, "video/mp4")], Config(video_fps=5.0, max_frames=8))
    assert arr.dtype == np.uint8 and arr.ndim == 4 and arr.shape[0] == 8 and arr.shape[-1] == 3
    _, (arr2,) = load_media(None, [data_url(video)], Config(video_fps=5.0, max_frames=100))
    assert arr2.shape[0] == 30


def test_video_frames_downscaled(video):
    _, (arr,) = load_media(None, [data_url(video)], Config(video_fps=2.0, max_frames=32, max_pixels=1000))
    assert arr.shape[1] * arr.shape[2] <= 1000


def test_video_garbage():
    junk = b"\x00\x00\x00\x18ftypmp42" + os.urandom(200)
    with pytest.raises(MediaError, match=r"videos\[0\].*could not decode video"):
        load_media(None, [data_url(junk, "video/mp4")], Config())


def test_even_subsample_indices():
    idx = media._sample_indices(100, 5)
    assert idx[0] == 0 and idx[-1] == 99 and len(idx) == 5
    assert media._sample_indices(3, 5) == [0, 1, 2]


# ---- URL fetch / SSRF


def test_url_fetch_disabled_by_default():
    assert Config().allow_url_fetch is False
    with pytest.raises(MediaError, match="disabled"):
        load_bytes("https://example.com/a.png", Config())


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x.png",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost:8910/health",
        "http://10.0.0.5/a",
        "http://192.168.1.1/a",
        "http://[::1]/a",
        "http://[::ffff:127.0.0.1]/a",
        "http://0.0.0.0/a",  # noqa: E501
    ],
)
def test_ssrf_private_addresses_rejected(url):
    with pytest.raises(MediaError, match="non-public"):
        load_bytes(url, Config(allow_url_fetch=True))


def test_url_scheme_and_credentials():
    cfg = Config(allow_url_fetch=True)
    with pytest.raises(MediaError, match="http"):
        load_bytes("ftp://example.com/a", cfg)
    with pytest.raises(MediaError, match="credentials"):
        load_bytes("http://user:pw@example.com/a", cfg)


def test_url_fetch_no_redirects_and_cap(monkeypatch):
    import httpx

    monkeypatch.setattr(media, "_check_public_host", lambda host, port: None)
    real = httpx.Client

    def client_with(handler):
        return lambda **kw: real(transport=httpx.MockTransport(handler), **kw)

    cfg = Config(allow_url_fetch=True, url_fetch_max_mb=1)
    monkeypatch.setattr(
        httpx, "Client", client_with(lambda r: httpx.Response(302, headers={"location": "http://127.0.0.1/"}))
    )
    with pytest.raises(MediaError, match="redirects"):
        load_bytes("http://example.com/a", cfg)
    monkeypatch.setattr(
        httpx, "Client", client_with(lambda r: httpx.Response(200, content=b"0" * (1024 * 1024 + 1)))
    )  # noqa: E501
    with pytest.raises(MediaError, match="exceeds"):
        load_bytes("http://example.com/a", cfg)
    monkeypatch.setattr(
        httpx,
        "Client",
        client_with(
            lambda r: httpx.Response(200, content=png_bytes(), headers={"content-type": "image/png"})
        ),
    )
    (img,), _ = load_media(["http://example.com/a"], None, cfg)
    assert img.size == (32, 16)

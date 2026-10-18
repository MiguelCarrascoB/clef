"""D2 perf flags: CLEF_MASK_FREE and CLEF_FAST_HEAD (CPU only, tiny random models).

The release head used as the oracle is tests/unit/_ref_head.py (a copy of the release's own code).
If CLEF_RELEASE_DIR points at a real release dir, the same parity test also runs against its module.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from clef_server.config import Config
from clef_server.models.base import ModelSpec
from clef_server.models.clef_release import ClefReleaseAdapter, fast_joint_head
from tests.unit._ref_head import JointSchemaHead
from tests.unit.test_engine import FakeModel, fake_collate, fake_encode

HID, WIDTH, VOCAB = 16, 8, 50
SPEC = ModelSpec("clef-flash", "clef-release", "x/y", "0" * 40, 1.0, {})


def make_head(seed: int = 0) -> JointSchemaHead:
    torch.manual_seed(seed)
    head = JointSchemaHead(HID, WIDTH, routing_layers=2, layers=2, heads=2, feedforward=16)
    with torch.no_grad():
        # the scales / gate start at 0 in a fresh head: make every branch matter
        head.prior_logit_scale.fill_(1.3)
        head.joint_logit_scale.fill_(0.7)
        head.residual_gate.fill_(0.4)
    return head.eval()


def make_encoded(rng: torch.Generator, n_q: int, extra: int = 0) -> SimpleNamespace:
    """A fake EncodedRecord: random ids, spans laid out like the real schema (questions then options)."""
    pos, qs = 4 + extra, []
    for i in range(n_q):
        qs_start = pos
        pos += int(torch.randint(1, 4, (1,), generator=rng))
        spans = []
        for _ in range(int(torch.randint(1, 5, (1,), generator=rng))):
            a = pos
            pos += int(torch.randint(1, 4, (1,), generator=rng))
            spans.append((a, pos))
            pos += 1
        qs.append(
            SimpleNamespace(
                question_id=f"q{i}",
                question_type=int(torch.randint(0, 3, (1,), generator=rng)),
                question_span=(qs_start, qs_start + 1),
                option_spans=tuple(spans),
                option_ids=tuple(f"o{j}" for j in range(len(spans))),
            )
        )
    pos += 3
    ids = tuple(int(x) for x in torch.randint(1, VOCAB, (pos,), generator=rng))
    return SimpleNamespace(input_ids=ids, questions=tuple(qs), media=None, record_id="r")


def make_batch(recs: list[SimpleNamespace], pad_to: int, hidden_noise: bool = True):
    n = len(recs)
    ids = torch.zeros(n, pad_to, dtype=torch.long)
    mask = torch.zeros(n, pad_to, dtype=torch.long)
    for i, r in enumerate(recs):
        ids[i, : len(r.input_ids)] = torch.tensor(r.input_ids)
        mask[i, : len(r.input_ids)] = 1
    return ids, mask


def reference_logits(head, hidden, ids, mask, recs, weight):
    with torch.inference_mode():
        return head(hidden, ids, mask, recs, weight)


def assert_close(ref, got, tol=1e-5):
    assert len(ref) == len(got)
    for r_rec, g_rec in zip(ref, got, strict=True):
        assert len(r_rec) == len(g_rec)
        for r, g in zip(r_rec, g_rec, strict=True):
            assert r.shape == g.shape
            assert (r - g.to(r.dtype)).abs().max().item() <= tol


@pytest.mark.parametrize("sizes", [[3], [1, 4, 2], [2, 2, 5, 1]])
def test_fast_head_matches_release_head(sizes):
    rng = torch.Generator().manual_seed(7)
    head = make_head()
    recs = [make_encoded(rng, n, extra=i * 3) for i, n in enumerate(sizes)]
    lengths = [len(r.input_ids) for r in recs]
    pad_to = max(lengths) + 5
    ids, mask = make_batch(recs, pad_to)
    hidden = torch.randn(len(recs), pad_to, HID, generator=rng)
    weight = torch.randn(VOCAB, HID, generator=rng)
    ref = reference_logits(head, hidden, ids, mask, recs, weight)
    with torch.inference_mode():
        got = fast_joint_head(head, hidden, ids, lengths, recs, weight)
    assert_close(ref, got)
    # padded positions never matter: poison them and get the same answer
    poisoned = hidden.clone()
    poisoned[mask == 0] = 1e4 * torch.randn(int((mask == 0).sum()), HID, generator=rng)
    with torch.inference_mode():
        again = fast_joint_head(head, poisoned, ids, lengths, recs, weight)
    assert_close(got, again, tol=0.0)


def test_fast_head_bf16_matches_release_head():
    """bf16 has 8 bits of mantissa; both heads round differently, so the bound is a few bf16 ulps of the
    logit scale (|logit| is O(1-10) here): max |dlogit| <= 0.1 and every softmax within 2e-2."""
    rng = torch.Generator().manual_seed(9)
    head = make_head(2).to(torch.bfloat16)
    recs = [make_encoded(rng, n, extra=i * 3) for i, n in enumerate([2, 4, 1])]
    lengths = [len(r.input_ids) for r in recs]
    pad_to = max(lengths) + 4
    ids, mask = make_batch(recs, pad_to)
    hidden = torch.randn(len(recs), pad_to, HID, generator=rng).to(torch.bfloat16)
    weight = torch.randn(VOCAB, HID, generator=rng).to(torch.bfloat16)
    ref = reference_logits(head, hidden, ids, mask, recs, weight)
    with torch.inference_mode():
        got = fast_joint_head(head, hidden, ids, lengths, recs, weight)
    for r_rec, g_rec in zip(ref, got, strict=True):
        for r, g in zip(r_rec, g_rec, strict=True):
            r, g = r.float(), g.float()
            assert not torch.isnan(g).any()
            assert (r - g).abs().max().item() <= 0.1
            assert (r.softmax(-1) - g.softmax(-1)).abs().max().item() <= 2e-2


def test_validate_rejects_empty_question_id_and_instructions():
    ad = adapter(FakeModel())
    ok = {"state": "s", "questions": {"q": {"type": "noul", "instructions": "Is it?"}}}
    ad.validate(ok)
    ad.validate(
        {"state": "s", "questions": {"q": {"type": "noul", "instructions": ""}}}
    )  # falls back to the id
    for bad in (
        {"state": "s", "questions": {"": {"type": "noul", "instructions": "Is it?"}}},
        {"state": "s", "questions": {" ": {"type": "noul", "instructions": ""}}},
        {"state": "s", "questions": {"q": {"type": "noul", "instructions": "   "}}},
    ):
        with pytest.raises(ValueError, match="questions"):
            ad.validate(bad)


def test_fast_head_no_padding_and_host_output():
    rng = torch.Generator().manual_seed(3)
    head = make_head(1)
    recs = [make_encoded(rng, 2)]
    n = len(recs[0].input_ids)
    ids, mask = make_batch(recs, n)
    hidden = torch.randn(1, n, HID, generator=rng)
    weight = torch.randn(VOCAB, HID, generator=rng)
    with torch.inference_mode():
        got = fast_joint_head(head, hidden, ids, [n], recs, weight)
    assert_close(reference_logits(head, hidden, ids, mask, recs, weight), got)
    assert all(t.device.type == "cpu" for rec in got for t in rec)


@pytest.mark.skipif(not os.environ.get("CLEF_RELEASE_DIR"), reason="CLEF_RELEASE_DIR not set")
def test_fast_head_matches_real_release_module():  # pragma: no cover - needs the release's python file
    path = Path(os.environ["CLEF_RELEASE_DIR"]) / "joint_schema_model.py"
    spec = importlib.util.spec_from_file_location("real_release", path)
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["real_release"] = mod
    spec.loader.exec_module(mod)
    torch.manual_seed(0)
    head = mod.JointSchemaHead(HID, WIDTH, 2, 2, 2, 16).eval()
    rng = torch.Generator().manual_seed(5)
    recs = [make_encoded(rng, n, extra=n) for n in (2, 3)]
    lengths = [len(r.input_ids) for r in recs]
    ids, mask = make_batch(recs, max(lengths) + 3)
    hidden = torch.randn(2, ids.shape[1], HID, generator=rng)
    weight = torch.randn(VOCAB, HID, generator=rng)
    ref = reference_logits(head, hidden, ids, mask, recs, weight)
    with torch.inference_mode():
        got = fast_joint_head(head, hidden, ids, lengths, recs, weight)
    assert_close(ref, got)


# ------------------------------------------------------------------ adapter paths
class TinyText(torch.nn.Module):
    """A causal toy backbone: embedding -> causal running sum -> linear. Honors attention_mask like Qwen's
    linear-attention path does (padded inputs zeroed) and records what it was called with."""

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, HID)
        self.mix = torch.nn.Linear(HID, HID)
        self.masks: list[object] = []

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, **media):
        self.masks.append(attention_mask)
        x = self.embed(input_ids)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        # causal: position t only sees <= t
        x = torch.cumsum(x, dim=1) / torch.arange(1, x.shape[1] + 1).view(1, -1, 1)
        return SimpleNamespace(last_hidden_state=torch.tanh(self.mix(x)))


class TinyInner(torch.nn.Module):
    """base.model: callable as the multimodal wrapper (media batches) and exposes .language_model (text)."""

    def __init__(self, text):
        super().__init__()
        self.language_model = text

    def forward(self, **kw):
        return self.language_model(**kw)


class TinyClef(torch.nn.Module):
    """ClefModel's shape: .language_model.model.language_model (text) and .head."""

    def __init__(self):
        super().__init__()
        text = TinyText()
        inner = TinyInner(text)
        self.language_model = SimpleNamespace(
            model=inner, get_output_embeddings=lambda: SimpleNamespace(weight=text.embed.weight)
        )
        self.text = text
        self.head = make_head()
        self.calls = 0

    def forward(self, batch):  # the release's ClefModel.forward
        self.calls += 1
        out = self.text(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], **(batch.get("media") or {})
        )
        return self.head(
            out.last_hidden_state,
            batch["input_ids"],
            batch["attention_mask"],
            batch["records"],
            self.text.embed.weight,
        )


def adapter(model, **flags) -> ClefReleaseAdapter:
    cfg = dataclasses.replace(Config(), **flags)
    mod = SimpleNamespace(
        encode_record=None,
        collate_records=lambda recs, pad, dev: fake_collate(recs, pad, dev),
        systemone_answer=lambda q, p: p,
    )
    ad = ClefReleaseAdapter.from_module(mod, SPEC, cfg)
    ad._model, ad._mod = model, mod
    return ad


def encoded_batch(ad, n_recs=3, pad_to=64, seed=11):
    rng = torch.Generator().manual_seed(seed)
    recs = [make_encoded(rng, 1 + i, extra=2 * i) for i in range(n_recs)]
    encs = [SimpleNamespace(raw=r, length=len(r.input_ids), is_media=False) for r in recs]
    batch = ad.collate(encs, pad_to)
    return ad.to_device(batch, torch.device("cpu"))


def run(ad, batch):
    with torch.inference_mode():
        return ad.forward(batch)


def test_flags_default_off_and_forward_is_the_release_forward():
    model = TinyClef().eval()
    ad = adapter(model)
    assert not ad._mask_free and not ad._fast_head  # auto (None) until load() knows the backend
    batch = encoded_batch(ad)
    out = run(ad, batch)
    assert model.calls == 1 and model.text.masks[0] is not None
    ref = reference_logits(
        model.head,
        model.text(batch["input_ids"], batch["attention_mask"]).last_hidden_state,
        batch["input_ids"],
        batch["attention_mask"],
        batch["records"],
        model.text.embed.weight,
    )
    assert_close(ref, out, tol=0.0)  # byte-identical


def test_mask_free_drops_mask_and_keeps_results():
    model = TinyClef().eval()
    base = run(adapter(model), encoded_batch(adapter(model)))
    model.text.masks.clear()
    ad = adapter(model, mask_free=True)
    out = run(ad, encoded_batch(ad))
    assert model.text.masks == [None]  # backbone saw no mask; the toy model never calls itself via ClefModel
    assert_close(base, out, tol=1e-6)


def test_mask_free_head_gets_cpu_mask_no_device_sync():
    model = TinyClef().eval()
    seen = []
    real = model.head.forward

    def spy(hidden, ids, mask, recs, weight):
        seen.append(mask)
        return real(hidden, ids, mask, recs, weight)

    model.head.forward = spy
    ad = adapter(model, mask_free=True)
    batch = encoded_batch(ad)
    run(ad, batch)
    assert seen[0].device.type == "cpu"
    assert torch.equal(seen[0], batch["attention_mask"])  # same mask, rebuilt from the CPU lengths


def test_mask_free_never_for_media_or_missing_lengths():
    model = TinyClef().eval()
    ad = adapter(model, mask_free=True)
    batch = encoded_batch(ad)
    run(ad, {**batch, "media": {"pixel_values": torch.zeros(1)}})
    del batch["lengths"]
    run(ad, batch)
    assert all(m is not None for m in model.text.masks)


def test_fast_head_through_adapter_matches_flags_off():
    model = TinyClef().eval()
    base = run(adapter(model), encoded_batch(adapter(model)))
    for flags in ({"fast_head": True}, {"fast_head": True, "mask_free": True}):
        ad = adapter(model, **flags)
        out = run(ad, encoded_batch(ad))
        assert_close(base, out, tol=1e-5)
    # answer() works on the host views it returns
    q = encoded_batch(ad)["records"][0].questions[0]
    probs = ad.answer(q, {}, out[0][0])
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_flags_fall_back_on_foreign_model():
    model = FakeModel()
    ad = adapter(model, mask_free=True, fast_head=True)
    enc = [
        SimpleNamespace(raw=fake_encode(None, {"state": "a b c", "questions": _Q}), length=6, is_media=False)
    ]
    batch = ad.collate(enc, 8)
    out = run(ad, batch)
    assert model.calls and len(out) == 1


_Q = {"d": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}


def test_config_reads_env(monkeypatch):
    monkeypatch.delenv("CLEF_MASK_FREE", raising=False)
    monkeypatch.delenv("CLEF_FAST_HEAD", raising=False)
    assert (Config().mask_free, Config().fast_head) == (None, None)  # auto
    monkeypatch.setenv("CLEF_MASK_FREE", "1")
    monkeypatch.setenv("CLEF_FAST_HEAD", "0")
    assert (Config().mask_free, Config().fast_head) == (True, False)


@pytest.mark.parametrize(
    ("backend", "flags", "want"),
    [
        ("rocm", {}, (True, True)),  # auto: on where measured
        ("cuda", {}, (False, False)),
        ("mps", {}, (False, False)),
        ("rocm", {"mask_free": False}, (False, True)),  # an explicit 0 wins
        ("cuda", {"mask_free": True, "fast_head": True}, (True, True)),
    ],
)
def test_auto_flags_resolve_per_backend(backend, flags, want):
    ad = adapter(TinyClef().eval(), **flags)
    ad._resolve_flags(SimpleNamespace(name=backend))
    assert (ad._mask_free, ad._fast_head) == want

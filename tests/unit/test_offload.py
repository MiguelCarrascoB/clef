"""CPU offload planning and layer streaming (pure logic: CPU tensors, a fake copier, no GPU or weights)."""

from __future__ import annotations

import json

import pytest
import torch

from clef_server import offload as off

GB = 1024**3


def test_spread_is_even_and_bounded():
    assert off.spread(32, 0) == ()
    assert off.spread(32, 40) == tuple(range(32))
    idx = off.spread(32, 8)
    assert len(idx) == 8 and idx == tuple(sorted(idx))
    gaps = {b - a for a, b in zip(idx, idx[1:], strict=False)}
    assert gaps == {4}
    assert len(off.spread(10, 3)) == 3


def test_split_sizes_groups_layers_embeddings_and_rest():
    sizes = {
        "model.language_model.embed_tokens.weight": 100,
        "lm_head.weight": 100,
        "model.language_model.layers.0.mlp.weight": 30,
        "model.language_model.layers.0.attn.weight": 20,
        "model.language_model.layers.1.mlp.weight": 50,
        "model.visual.blocks.0.weight": 7,
        "model.language_model.norm.weight": 1,
    }
    layers, embed, other, _ = off.split_sizes(sizes)
    assert layers == [50, 50]
    assert embed == 200
    assert other == 8


def test_split_sizes_rejects_unknown_layout():
    with pytest.raises(ValueError, match="does not support"):
        off.split_sizes({"foo.weight": 1})


def test_plan_everything_fits_streams_nothing():
    plan = off.plan_layers([GB] * 4, 2 * GB, GB, 10 * GB)
    assert plan.streamed == () and plan.feasible
    assert plan.host_bytes == 2 * GB  # embeddings are always on the host


def test_plan_streams_just_enough_layers():
    layers = [GB] * 32
    # fixed 1 GB, 32 GB of layers, budget 20 GB: 1 + (32 - k) + 3 buffers <= 20 -> k >= 16
    plan = off.plan_layers(layers, 4 * GB, GB, 20 * GB)
    assert plan.feasible
    assert len(plan.streamed) == 16
    assert plan.device_weights_gb <= 20
    assert plan.host_bytes == 4 * GB + 16 * GB
    # one fewer streamed layer would not fit
    resident_without = GB + (32 - 15) * GB + 3 * GB
    assert resident_without > 20 * GB


def test_plan_infeasible_when_budget_below_fixed_cost():
    plan = off.plan_layers([GB] * 4, GB, 5 * GB, 2 * GB)
    assert not plan.feasible and plan.note
    assert plan.streamed == (0, 1, 2, 3)


def test_device_map_places_embeddings_and_streamed_layers_on_host():
    plan = off.plan_layers([GB] * 4, GB, GB, int(3.5 * GB))
    dm = plan.device_map("cuda:0")
    assert dm[off.EMBED_KEY] == "cpu" and dm[off.HEAD_KEY] == "cpu"
    assert dm["model.visual"] == "cuda:0"
    on_host = {i for i in range(4) if dm[f"{off.LAYER_PREFIX}{i}"] == "cpu"}
    assert on_host == set(plan.streamed) and on_host


def test_checkpoint_sizes_reads_headers(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {
            "model.language_model.layers.0.w": torch.zeros(4, 4, dtype=torch.bfloat16),
            "lm_head.weight": torch.zeros(2, 8, dtype=torch.bfloat16),
        },
        str(tmp_path / "a.safetensors"),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.language_model.layers.0.w": "a.safetensors"}})
    )
    sizes = off.checkpoint_sizes(tmp_path)
    assert sizes == {"model.language_model.layers.0.w": 32, "lm_head.weight": 32}


# ------------------------------------------------------------------------------------------ modules
class _FakeCopier:
    """Counts fetches; 'device' tensors are clones tagged by a fresh storage."""

    def __init__(self):
        self.fetched: list[int] = []

    def fetch(self, host):
        self.fetched.append(len(host))
        out = [t.clone() for t in host]

        class _P:
            def wait(_self):
                return out

        return _P()


class _Block(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.fc = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(3) * value)

    def forward(self, x):
        return self.fc(x)


def _stack(n=6):
    return torch.nn.ModuleList([_Block(float(i + 1)) for i in range(n)])


def test_streamer_matches_plain_forward_and_restores_params():
    layers = _stack(6)
    x = torch.ones(1, 3)
    expected = x
    for layer in layers:
        expected = layer(expected)
    originals = {i: layers[i].fc.weight for i in range(6)}
    copier = _FakeCopier()
    streamer = off.LayerStreamer(layers, [1, 3, 4], copier, lookahead=1)
    streamer.install()
    out = x
    for layer in layers:
        out = layer(out)
    assert torch.allclose(out, expected)
    assert all(layers[i].fc.weight is originals[i] for i in range(6))  # restored after the forward
    assert len(copier.fetched) == 3  # each streamed layer copied exactly once (prefetched, then consumed)
    streamer.remove()
    out2 = x
    for layer in layers:
        out2 = layer(out2)
    assert torch.allclose(out2, expected) and len(copier.fetched) == 3


def test_streamer_reset_after_exception():
    layers = _stack(3)

    class Boom(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    layers.insert(2, Boom())
    streamer = off.LayerStreamer(layers, [0, 1], _FakeCopier(), lookahead=2)
    streamer.install()
    originals = [layers[0].fc.weight, layers[1].fc.weight]
    x = torch.ones(1, 3)
    with pytest.raises(RuntimeError):
        for layer in layers:
            x = layer(x)
    # layer 1 ran and was restored; layer 0 too; reset must leave everything on the host copies
    streamer.reset()
    assert layers[0].fc.weight is originals[0] and layers[1].fc.weight is originals[1]
    # and a clean forward still works afterwards
    y = torch.ones(1, 3)
    for layer in list(layers)[:2]:
        y = layer(y)
    assert torch.allclose(y, torch.full((1, 3), 2.0))


def test_streamer_rejects_layers_with_buffers():
    layer = torch.nn.BatchNorm1d(3)
    with pytest.raises(RuntimeError, match="buffers"):
        off.LayerStreamer([layer], [0], _FakeCopier()).install()


def test_host_embedding_gathers_on_host():
    emb = torch.nn.Embedding(10, 4)
    host = off.HostEmbedding(emb, torch.device("cpu"))
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(host(ids), emb(ids))


def test_host_output_embedding_supports_row_lookup_only():
    lin = torch.nn.Linear(4, 10, bias=False)
    head = off.HostOutputEmbedding(lin, torch.device("cpu"))
    ids = torch.tensor([0, 5, 5])
    assert torch.equal(head.weight[ids], lin.weight.data[ids])
    assert head.weight.shape == (10, 4)
    with pytest.raises(RuntimeError):
        head(torch.zeros(1, 4))

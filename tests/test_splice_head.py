"""Trained splice classifier on frozen backbone features (forward + reverse-complement)."""

import torch

from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.model.splice_head import SpliceHead, SplicePredictor


def _predictor():
    return SplicePredictor(SveltoBackbone(model_name="mock", hidden_dim=32, num_layers=2, device="cpu"), SpliceHead(hidden_dim=32, proj_dim=16))


def test_head_maps_bidirectional_features_to_three_classes():
    head = SpliceHead(hidden_dim=32, proj_dim=16)
    out = head(torch.randn(2, 50, 64))
    assert out.shape == (2, 50, 3)


def test_predictor_outputs_probabilities_per_base():
    p = _predictor()
    probs = p.predict(["ACGT" * 25, "ACGT" * 10])
    assert probs.shape == (2, 100, 3)
    assert torch.allclose(probs[0].sum(-1), torch.ones(100), atol=1e-5)
    assert torch.all(probs[1, 40:] == 0)            # padding positions carry no prediction


def test_features_see_downstream_sequence_through_the_reverse_complement_pass():
    """HyenaDNA is causal: forward states never see downstream bases. The RC pass must make
    position 10 react to a change at base 60 (only meaningful on the real causal checkpoint)."""
    import pytest
    try:
        backbone = SveltoBackbone(model_name="LongSafari/hyenadna-small-32k-seqlen-hf")
    except RuntimeError:
        pytest.skip("HyenaDNA checkpoint not reachable")
    p = SplicePredictor(backbone, SpliceHead(hidden_dim=backbone.hidden_dim))
    a = "ACGT" * 25
    b = a[:60] + ("A" if a[60] != "A" else "C") + a[61:]
    fa, fb = p.features([a]), p.features([b])
    D = backbone.hidden_dim
    fwd = (fa[0, 10, :D] - fb[0, 10, :D]).abs().max().item()
    rc = (fa[0, 10, D:] - fb[0, 10, D:]).abs().max().item()
    # Forward half is causal: only FFT round-off reaches it (~3e-3 under fp16 autocast).
    assert fwd < 0.01
    # Reverse-complement half genuinely sees the downstream change (measured 80-300x the leakage).
    assert rc > 20 * fwd


def test_only_the_head_trains():
    p = _predictor()
    feats = p.features(["ACGT" * 25])
    loss = p.head(feats).logsumexp(-1).mean()
    loss.backward()
    assert all(q.grad is None for q in p.backbone.parameters())
    assert any(q.grad is not None for q in p.head.parameters())

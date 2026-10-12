"""
Issue #10 AC1: saliency for the mutated locus — normalized input-gradient scores, computed without
retaining the backward graph (memory returns to baseline).
"""

import pytest
import torch

from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.model.splice_head import SpliceHead, SplicePredictor
from svelto_dna.research.saliency import locus_saliency


def _predictor(device="cpu"):
    torch.manual_seed(0)
    return SplicePredictor(SveltoBackbone(model_name="mock", hidden_dim=32, num_layers=2, device=device), SpliceHead(hidden_dim=32, proj_dim=16))


def test_scores_are_normalized_per_base():
    s = locus_saliency(_predictor(), "ACGT" * 30, position=60, target="donor")
    assert s.shape == (120,)
    assert float(s.min()) >= 0 and float(s.max()) == pytest.approx(1.0)


def test_saliency_is_local_for_a_short_receptive_field():
    """The mock backbone sees ±4 bp per pass; distant bases must get (near-)zero attribution."""
    s = locus_saliency(_predictor(), "ACGT" * 30, position=60, target="acceptor")
    assert float(s[:40].max()) < 1e-6 and float(s[80:].max()) < 1e-6
    assert float(s[55:66].max()) == pytest.approx(1.0)


def test_backbone_stays_frozen_and_no_gradients_are_left_behind():
    p = _predictor()
    locus_saliency(p, "ACGT" * 30, position=60, target="donor")
    assert all(q.grad is None for q in p.backbone.parameters())
    assert all(q.grad is None for q in p.head.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU memory check")
def test_no_backward_memory_retention_on_gpu():
    p = _predictor(device="cuda")
    locus_saliency(p, "ACGT" * 250, position=500, target="donor")           # warm-up allocations
    torch.cuda.synchronize(); before = torch.cuda.memory_allocated()
    for _ in range(5):
        locus_saliency(p, "ACGT" * 250, position=500, target="donor")
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= before

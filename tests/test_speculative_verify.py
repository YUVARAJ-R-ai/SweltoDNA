"""
Issue #4 (approved redesign): speculative in-silico mutagenesis with batched verification.
Draft ranks candidate variants from one reference pass; the top-K are verified in one batched pass.
"""

import numpy as np
import pytest
import torch

from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.model.splice_head import SpliceHead, SplicePredictor
from svelto_dna.speculative.verify import SpeculativeISM, all_candidates, draft_scores


class MotifPredictor:
    """Deterministic stand-in: donor 0.9 at every GT, acceptor 0.9 at every AG (after a pyrimidine)."""
    engine = "motif"

    def predict(self, seqs):
        L = max(len(s) for s in seqs)
        out = torch.zeros(len(seqs), L, 3); out[..., 0] = 1
        for i, s in enumerate(seqs):
            for j in range(len(s) - 1):
                if s[j:j + 2] == "GT":
                    out[i, j] = torch.tensor([0.1, 0.9, 0.0])
                elif s[j:j + 2] == "AG" and j > 0 and s[j - 1] in "CT":
                    out[i, j] = torch.tensor([0.1, 0.0, 0.9])
        return out


SEQ = "C" * 300 + "CAGGTAAGT" + "C" * 300           # acceptor AG at 301, donors GT at 303 and 307; neutral background
WIN = (296, 312)                                      # 16 positions → 48 candidates


def test_candidates_are_every_alternative_base_in_the_window():
    c = all_candidates(SEQ, *WIN)
    assert len(c) == 3 * (WIN[1] - WIN[0])
    assert all(SEQ[p] != alt for p, alt in c)


def test_draft_ranks_the_donor_dinucleotide_first():
    p_ref = MotifPredictor().predict([SEQ])[0].numpy()
    c = all_candidates(SEQ, *WIN)
    s = draft_scores(SEQ, p_ref, c)
    top = {c[i][0] for i in np.argsort(-s)[:6]}
    assert {303, 304} <= top


def test_batched_verification_equals_sequential_and_finds_every_high_impact_variant():
    ism = SpeculativeISM(MotifPredictor(), context=40, delta_window=10)
    exhaustive = ism.exhaustive(SEQ, *WIN, batch_size=1)
    batched = ism.exhaustive(SEQ, *WIN, batch_size=16)
    assert np.allclose(exhaustive.deltas, batched.deltas)
    truth = set(exhaustive.high_impact())
    assert len(truth) == 23
    n = len(exhaustive.candidates)
    spec = ism.speculative(SEQ, *WIN, k=n, batch_size=16)              # K covers everything: identical to exhaustive
    assert np.allclose(spec.deltas, exhaustive.deltas)
    assert set(spec.high_impact()) == truth
    assert spec.acceptance_rate == pytest.approx(23 / n)


def test_recall_is_limited_by_k_and_precision_stays_high():
    ism = SpeculativeISM(MotifPredictor(), context=40, delta_window=10)
    truth = set(ism.exhaustive(SEQ, *WIN).high_impact())
    spec = ism.speculative(SEQ, *WIN, k=12)
    assert len(truth & set(spec.high_impact())) / len(truth) < 1      # the trade-off the benchmark must report
    assert spec.acceptance_rate == pytest.approx(1.0)


def test_unverified_candidates_are_reported_as_unknown_not_zero():
    ism = SpeculativeISM(MotifPredictor(), context=40, delta_window=10)
    spec = ism.speculative(SEQ, *WIN, k=6, batch_size=6)
    assert np.isnan(spec.deltas).sum() == len(spec.candidates) - 6


def test_real_backbone_batched_verification_matches_sequential_in_fp32():
    try:
        bb = SveltoBackbone(model_name="LongSafari/hyenadna-small-32k-seqlen-hf", dtype=torch.float32)
    except RuntimeError:
        pytest.skip("HyenaDNA checkpoint not reachable")
    torch.manual_seed(0)
    ism = SpeculativeISM(SplicePredictor(bb, SpliceHead(hidden_dim=bb.hidden_dim)), context=200, delta_window=50)
    seq = "ACGT" * 150
    a = ism.exhaustive(seq, 290, 296, batch_size=1).deltas
    b = ism.exhaustive(seq, 290, 296, batch_size=18).deltas
    assert np.max(np.abs(a - b)) < 1e-5

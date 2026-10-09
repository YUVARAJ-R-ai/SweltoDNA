"""
Issue #5, AC3: delta scores for real ClinVar pathogenic splice variants match published ground truth.

Fixture (scripts/make_spliceai_fixtures.py): full ±50 bp reference/alternate probability tracks from the
official SpliceAI 5-model ensemble for 8 ClinVar *Pathogenic* canonical-site SNVs (4 per strand, donors
and acceptors), plus the Broad SpliceAI-lookup *published* DS_*/DP_* for the same variants.

Tolerance: the published scores are rounded to 3 decimals, so agreement better than ±0.0005 (plus float
noise between their run and ours) cannot be demonstrated against them.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from svelto_dna.splice.delta import compute_delta_scores

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "spliceai_clinvar_ground_truth.json").read_text())
TOL = 5e-4 + 1e-4
COMPONENTS = {"donor_gain": "DG", "donor_loss": "DL", "acceptor_gain": "AG", "acceptor_loss": "AL"}


def _tracks(t):
    d, a = np.asarray(t["donor"], dtype=np.float32), np.asarray(t["acceptor"], dtype=np.float32)
    return np.stack([1 - d - a, d, a], axis=-1)            # [neither, donor, acceptor]


@pytest.mark.parametrize("v", FIXTURE["variants"], ids=[v["variant"] for v in FIXTURE["variants"]])
def test_delta_scores_match_published_spliceai(v):
    r = compute_delta_scores(_tracks(v["tracks"]["ref"]), _tracks(v["tracks"]["alt"]), window_size=50, variant_pos=v["variant_index"], return_tensors=False)
    for ours, code in COMPONENTS.items():
        got = float(np.asarray(r[ours]).reshape(-1)[0])
        assert abs(got - v["published"][f"DS_{code}"]) <= TOL, f"{v['variant']} DS_{code}: ours {got:.4f} vs published {v['published'][f'DS_{code}']:.3f}"


@pytest.mark.parametrize("v", FIXTURE["variants"], ids=[v["variant"] for v in FIXTURE["variants"]])
def test_peak_component_and_position_match_published(v):
    r = compute_delta_scores(_tracks(v["tracks"]["ref"]), _tracks(v["tracks"]["alt"]), window_size=50, variant_pos=v["variant_index"], return_tensors=False)
    code = COMPONENTS[r["peak_component"]]
    published_best = max(COMPONENTS.values(), key=lambda c: v["published"][f"DS_{c}"])
    assert code == published_best
    assert r["peak_position"] - v["variant_index"] == v["published"][f"DP_{code}"]


def test_fixture_covers_both_strands_and_site_types():
    vs = FIXTURE["variants"]
    assert {v["strand"] for v in vs} == {"+", "-"}
    assert {v["junction_type"] for v in vs} == {"donor", "acceptor"}
    assert all(max(v["published"][f"DS_{c}"] for c in COMPONENTS.values()) > 0.9 for v in vs)

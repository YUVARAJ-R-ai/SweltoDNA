"""Unit and benchmark tests for Vectorized Splice Disruption & Delta Score Calculator."""

from __future__ import annotations

import time
import numpy as np
import pytest
import torch

from svelto_dna.splice.delta import DeltaResult, compute_delta_scores


class TestDeltaScoreCalculator:
    """Test suite for compute_delta_scores module."""

    def test_structured_dictionary_output_and_types(self) -> None:
        """AC 1: Module returns structured dictionary containing donor/acceptor gain/loss and peak delta."""
        L = 500
        p_ref = np.zeros((L, 3), dtype=np.float32)
        p_mut = np.zeros((L, 3), dtype=np.float32)

        # Introduce a donor gain at index 250
        p_ref[250, 1] = 0.05
        p_mut[250, 1] = 0.85

        result = compute_delta_scores(p_ref, p_mut, window_size=50)

        # Must be an instance of dict
        assert isinstance(result, dict)
        assert isinstance(result, DeltaResult)

        # Required dictionary keys
        expected_keys = {
            "donor_gain",
            "donor_loss",
            "acceptor_gain",
            "acceptor_loss",
            "locus_delta",
            "peak_delta",
            "peak_component",
            "peak_position",
            "window_size",
            "raw_differences",
            "latency_ms",
        }
        assert expected_keys.issubset(set(result.keys()))

        # Check array lengths and types
        assert len(result["donor_gain"]) == L
        assert len(result["donor_loss"]) == L
        assert len(result["acceptor_gain"]) == L
        assert len(result["acceptor_loss"]) == L
        assert len(result["locus_delta"]) == L

        # Property access
        assert abs(result.peak_delta - 0.80) < 1e-4
        assert result.peak_component == "donor_gain"
        assert result.peak_position == 250

        # Method conversions
        d = result.to_dict()
        assert type(d) is dict
        assert "peak_delta" in d

        torch_res = result.to_torch()
        assert isinstance(torch_res["donor_gain"], torch.Tensor)
        numpy_res = torch_res.to_numpy()
        assert isinstance(numpy_res["donor_gain"], np.ndarray)

    def test_benchmark_under_5ms_on_cpu(self) -> None:
        """AC 2: Benchmark test executes delta score computation for L = 10,000 bp in under 5 ms on CPU."""
        L = 10000
        device = torch.device("cpu")
        p_ref = torch.softmax(torch.randn(1, L, 3, device=device), dim=-1)
        p_mut = torch.softmax(torch.randn(1, L, 3, device=device), dim=-1)

        # Warmup iterations
        for _ in range(5):
            compute_delta_scores(p_ref, p_mut, window_size=50)

        # Timed measurement over 20 iterations
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            res = compute_delta_scores(p_ref, p_mut, window_size=50)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)

        mean_latency = float(np.mean(times))
        p95_latency = float(np.percentile(times, 95))

        print(f"\n[CPU Benchmark L=10,000 bp] Mean: {mean_latency:.4f} ms, P95: {p95_latency:.4f} ms")
        assert mean_latency < 5.0, f"Mean latency {mean_latency:.4f} ms exceeded 5.0 ms threshold on CPU"

    def test_clinvar_pathogenic_donor_loss_ground_truth(self) -> None:
        """AC 3: ClinVar Pathogenic Donor Loss variant (canonical 5' splice site disruption)."""
        L = 1000
        p_ref = np.zeros((L, 3), dtype=np.float32)
        p_mut = np.zeros((L, 3), dtype=np.float32)

        # Canonical donor junction at index 500
        # Wild-type has strong donor signal (0.9650), mutation destroys it to 0.0120
        p_ref[500, 1] = 0.9650
        p_mut[500, 1] = 0.0120

        # Variant located at index 500 (e.g. +1G>A SNV)
        expected_dl = 0.9650 - 0.0120  # 0.9530

        result = compute_delta_scores(p_ref, p_mut, window_size=50, variant_pos=500)

        assert abs(result["peak_delta"] - expected_dl) < 1e-4
        assert result["peak_component"] == "donor_loss"
        assert result["peak_position"] == 500
        assert abs(float(result["donor_loss"]) - expected_dl) < 1e-4

    def test_clinvar_pathogenic_cryptic_donor_gain_ground_truth(self) -> None:
        """AC 3: ClinVar Pathogenic Cryptic Donor Gain (e.g. CFTR c.3718-2477C>T deep intronic mutation)."""
        L = 1000
        p_ref = np.zeros((L, 3), dtype=np.float32)
        p_mut = np.zeros((L, 3), dtype=np.float32)

        # Deep intronic mutation at index 620 creates a strong novel cryptic donor site
        p_ref[620, 1] = 0.0050
        p_mut[620, 1] = 0.8850

        expected_dg = 0.8850 - 0.0050  # 0.8800

        result = compute_delta_scores(p_ref, p_mut, window_size=50, variant_pos=620)

        assert abs(result["peak_delta"] - expected_dg) < 1e-4
        assert result["peak_component"] == "donor_gain"
        assert result["peak_position"] == 620
        assert abs(float(result["donor_gain"]) - expected_dg) < 1e-4

    def test_clinvar_pathogenic_acceptor_loss_ground_truth(self) -> None:
        """AC 3: ClinVar Pathogenic Acceptor Loss variant (canonical 3' splice site disruption)."""
        L = 1000
        p_ref = np.zeros((L, 3), dtype=np.float32)
        p_mut = np.zeros((L, 3), dtype=np.float32)

        # Canonical acceptor junction at index 300 (e.g. -1G>A SNV)
        p_ref[300, 2] = 0.9420
        p_mut[300, 2] = 0.0150

        expected_al = 0.9420 - 0.0150  # 0.9270

        result = compute_delta_scores(p_ref, p_mut, window_size=50, variant_pos=300)

        assert abs(result["peak_delta"] - expected_al) < 1e-4
        assert result["peak_component"] == "acceptor_loss"
        assert result["peak_position"] == 300
        assert abs(float(result["acceptor_loss"]) - expected_al) < 1e-4

    def test_clinvar_pathogenic_cryptic_acceptor_gain_ground_truth(self) -> None:
        """AC 3: ClinVar Pathogenic Cryptic Acceptor Gain variant."""
        L = 1000
        p_ref = np.zeros((L, 3), dtype=np.float32)
        p_mut = np.zeros((L, 3), dtype=np.float32)

        # Cryptic acceptor creation at index 450
        p_ref[450, 2] = 0.0110
        p_mut[450, 2] = 0.8710

        expected_ag = 0.8710 - 0.0110  # 0.8600

        result = compute_delta_scores(p_ref, p_mut, window_size=50, variant_pos=450)

        assert abs(result["peak_delta"] - expected_ag) < 1e-4
        assert result["peak_component"] == "acceptor_gain"
        assert result["peak_position"] == 450
        assert abs(float(result["acceptor_gain"]) - expected_ag) < 1e-4

    def test_boundary_edge_cases_index_0_and_L_minus_1(self) -> None:
        """AC 4: Handles boundary edge cases gracefully without array indexing errors."""
        L = 1000
        p_ref = np.zeros((L, 3), dtype=np.float32)
        p_mut = np.zeros((L, 3), dtype=np.float32)

        # Variant exactly at start (index 0)
        p_ref[0, 1] = 0.10
        p_mut[0, 1] = 0.90
        res_start = compute_delta_scores(p_ref, p_mut, window_size=50, variant_pos=0)
        assert abs(res_start["peak_delta"] - 0.80) < 1e-4
        assert res_start["peak_position"] == 0

        # Variant near start (index 2)
        p_ref[2, 2] = 0.05
        p_mut[2, 2] = 0.75
        res_near_start = compute_delta_scores(p_ref, p_mut, window_size=50, variant_pos=2)
        assert abs(res_near_start["peak_delta"] - 0.80) < 1e-4  # index 0 is still in [0, 52]

        # Variant exactly at end (index L-1 = 999)
        p_ref[999, 2] = 0.02
        p_mut[999, 2] = 0.82
        res_end = compute_delta_scores(p_ref, p_mut, window_size=50, variant_pos=999)
        assert abs(res_end["peak_delta"] - 0.80) < 1e-4
        assert res_end["peak_position"] == 999

        # Out-of-bounds raises IndexError
        with pytest.raises(IndexError):
            compute_delta_scores(p_ref, p_mut, window_size=50, variant_pos=1000)

        with pytest.raises(IndexError):
            compute_delta_scores(p_ref, p_mut, window_size=50, variant_pos=-1)

    def test_short_sequences_smaller_than_window_size(self) -> None:
        """Handles short sequences where L < 2*W + 1 without errors."""
        L = 25  # smaller than window_size = 50
        p_ref = torch.zeros((L, 3))
        p_mut = torch.zeros((L, 3))

        p_ref[10, 1] = 0.10
        p_mut[10, 1] = 0.70

        result = compute_delta_scores(p_ref, p_mut, window_size=50)
        assert abs(result.peak_delta - 0.60) < 1e-4
        assert len(result.donor_gain) == L

        # Minimal sequence L = 1
        res_single = compute_delta_scores(torch.zeros(1, 3), torch.zeros(1, 3), window_size=10)
        assert len(res_single.donor_gain) == 1
        assert res_single.peak_delta == 0.0

    def test_canonical_splice_site_masking(self) -> None:
        """Support masking of annotated canonical splice sites for clinical variant prioritization."""
        L = 500
        p_ref = np.zeros((L, 3), dtype=np.float32)
        p_mut = np.zeros((L, 3), dtype=np.float32)

        # Existing canonical donor site at index 100
        p_ref[100, 1] = 0.50
        p_mut[100, 1] = 0.95  # Apparent gain of 0.45 at canonical junction

        # True novel cryptic site at index 200
        p_ref[200, 1] = 0.01
        p_mut[200, 1] = 0.35  # Cryptic gain of 0.34

        # Without masking, canonical junction has higher delta (0.45)
        unmasked = compute_delta_scores(p_ref, p_mut, window_size=50)
        assert unmasked.peak_position == 100
        assert abs(unmasked.peak_delta - 0.45) < 1e-4

        # Masking the canonical site at index 100 suppresses its gain
        mask = np.zeros(L, dtype=bool)
        mask[100] = True
        masked = compute_delta_scores(p_ref, p_mut, window_size=50, canonical_mask=mask)

        # Now the peak is the true cryptic site at index 200
        assert masked.peak_position == 200
        assert abs(masked.peak_delta - 0.34) < 1e-4

    def test_identity_invariance(self) -> None:
        """When P_ref == P_mut, all delta scores must be exactly zero."""
        L = 200
        p_same = np.random.uniform(0.0, 1.0, size=(L, 3)).astype(np.float32)
        result = compute_delta_scores(p_same, p_same, window_size=20)
        assert result.peak_delta == 0.0
        assert np.all(result["donor_gain"] == 0.0)
        assert np.all(result["donor_loss"] == 0.0)
        assert np.all(result["acceptor_gain"] == 0.0)
        assert np.all(result["acceptor_loss"] == 0.0)

    def test_batched_tensor_inputs(self) -> None:
        """Handles 3D batched inputs of shape (B, L, 3)."""
        B, L = 4, 300
        p_ref = torch.zeros(B, L, 3)
        p_mut = torch.zeros(B, L, 3)

        p_mut[0, 50, 1] = 0.80   # Batch 0: DG at 50
        p_ref[1, 100, 1] = 0.90  # Batch 1: DL at 100
        p_mut[2, 150, 2] = 0.75  # Batch 2: AG at 150
        p_ref[3, 200, 2] = 0.85  # Batch 3: AL at 200

        result = compute_delta_scores(p_ref, p_mut, window_size=30)
        assert result.donor_gain.shape == (B, L)
        assert result.peak_delta >= 0.80

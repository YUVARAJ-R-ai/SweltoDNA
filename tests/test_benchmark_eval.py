"""
Unit tests for splice benchmark evaluation metrics: Top-1, Top-k, ROC-AUC,
and PR-AUC under severe class imbalance (>100:1).
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from svelto_dna.data.benchmark import (
    LABEL_ACCEPTOR,
    LABEL_DONOR,
    LABEL_NEITHER,
    compute_splice_metrics,
    compute_top_k_accuracy,
    safe_pr_auc,
    safe_roc_auc,
)


class TestSpliceMetrics:
    """Tests for Top-k, ROC-AUC, and PR-AUC calculations."""

    def test_top_k_accuracy_perfect_and_worst(self) -> None:
        y_true = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        # Perfect scores
        perfect_scores = np.array([0.95, 0.1, 0.2, 0.90, 0.05, 0.01, 0.12, 0.02, 0.03, 0.04])
        acc_top_1 = compute_top_k_accuracy(y_true, perfect_scores, k=1.0)
        assert acc_top_1 == 1.0

        # Inverted scores
        worst_scores = 1.0 - perfect_scores
        acc_worst = compute_top_k_accuracy(y_true, worst_scores, k=1.0)
        assert acc_worst == 0.0

        # Expanding k budget
        # Candidate budget becomes round(2.0 * 2) = 4
        partial_scores = np.array([0.95, 0.8, 0.7, 0.6, 0.05, 0.01, 0.12, 0.02, 0.03, 0.04])
        # Index 0 is in top 1, index 3 is 4th (inside top 4)
        acc_top_2 = compute_top_k_accuracy(y_true, partial_scores, k=2.0)
        assert acc_top_2 == 1.0

    def test_severe_class_imbalance_metrics(self) -> None:
        np.random.seed(42)
        n_neither = 1000
        n_donor = 5
        n_acceptor = 5
        n_total = n_neither + n_donor + n_acceptor

        # Construct ground truth
        y_true = np.zeros(n_total, dtype=int)
        donor_indices = [100, 250, 400, 650, 800]
        acceptor_indices = [150, 300, 550, 750, 950]
        y_true[donor_indices] = LABEL_DONOR
        y_true[acceptor_indices] = LABEL_ACCEPTOR

        # Construct probabilities with high separation for true sites
        y_probs = np.zeros((n_total, 3))
        # Background: 98% Neither, small random noise on splice
        y_probs[:, LABEL_NEITHER] = 0.96 + np.random.uniform(0.01, 0.03, n_total)
        y_probs[:, LABEL_DONOR] = np.random.uniform(0.001, 0.01, n_total)
        y_probs[:, LABEL_ACCEPTOR] = np.random.uniform(0.001, 0.01, n_total)

        # True donor sites: high donor prob
        for idx in donor_indices:
            y_probs[idx, LABEL_DONOR] = 0.88
            y_probs[idx, LABEL_NEITHER] = 0.10
            y_probs[idx, LABEL_ACCEPTOR] = 0.02

        # True acceptor sites: high acceptor prob
        for idx in acceptor_indices:
            y_probs[idx, LABEL_ACCEPTOR] = 0.89
            y_probs[idx, LABEL_NEITHER] = 0.09
            y_probs[idx, LABEL_DONOR] = 0.02

        # Normalize rows to sum to 1
        y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)

        report = compute_splice_metrics(y_true, y_probs)

        # Check severe class imbalance ratio
        assert report.imbalance_ratio >= 100.0  # 1000 / 10 = 100.0

        # Check discrimination
        assert report.donor_roc_auc > 0.95
        assert report.acceptor_roc_auc > 0.95
        assert report.donor_pr_auc > 0.80
        assert report.acceptor_pr_auc > 0.80
        assert report.donor_top_1_acc == 1.0
        assert report.acceptor_top_1_acc == 1.0

        # Summary output formatting
        summary = report.summary()
        assert "Severe Imbalance Ratio: 100.0 : 1" in summary
        assert "ROC-AUC" in summary
        assert "PR-AUC" in summary

    def test_single_class_edge_case_safety(self) -> None:
        # All zeros (no donor sites present)
        y_true_empty = np.zeros(100, dtype=int)
        scores = np.random.uniform(0, 1, 100)

        # Must not raise ValueError
        roc = safe_roc_auc(y_true_empty, scores)
        pr = safe_pr_auc(y_true_empty, scores)

        assert np.isnan(roc)
        assert np.isnan(pr)

    def test_json_telemetry_export(self) -> None:
        y_true = np.array([0, 0, 1, 0, 2])
        y_probs = np.array([
            [0.9, 0.05, 0.05],
            [0.85, 0.1, 0.05],
            [0.1, 0.85, 0.05],
            [0.95, 0.02, 0.03],
            [0.05, 0.05, 0.9],
        ])

        report = compute_splice_metrics(y_true, y_probs)
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "telemetry.json"
            report.save_json(json_file)
            assert json_file.exists()

            with open(json_file, "r") as f:
                content = f.read()
            assert "donor_roc_auc" in content
            assert "acceptor_pr_auc" in content

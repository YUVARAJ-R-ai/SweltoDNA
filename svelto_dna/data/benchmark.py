"""
Benchmarking harness and metrics evaluation engine for splice site prediction.
Computes Top-1/Top-k accuracy, ROC-AUC, and PR-AUC (Average Precision)
under severe class imbalance (non-splice to splice ratio > 100:1) with reproducible fixed seeds.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

LABEL_NEITHER: int = 0
LABEL_DONOR: int = 1
LABEL_ACCEPTOR: int = 2


@dataclass
class ThresholdMetrics:
    """Precision, recall, and F1 score at a specific diagnostic threshold."""
    threshold: float
    precision: float
    recall: float
    f1: float


@dataclass
class SpliceMetricsReport:
    """Comprehensive diagnostic metrics report for splice junction prediction."""

    n_total_loci: int
    n_donors: int
    n_acceptors: int
    imbalance_ratio: float  # (neither) / (donors + acceptors)

    donor_roc_auc: float
    acceptor_roc_auc: float
    mean_roc_auc: float

    donor_pr_auc: float
    acceptor_pr_auc: float
    mean_pr_auc: float

    donor_top_1_acc: float
    donor_top_k_acc: float
    acceptor_top_1_acc: float
    acceptor_top_k_acc: float

    donor_threshold_metrics: Dict[str, ThresholdMetrics] = field(default_factory=dict)
    acceptor_threshold_metrics: Dict[str, ThresholdMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to nested dictionary."""
        d = asdict(self)
        # Convert any NaNs or infinities for clean JSON export
        for k, v in d.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                d[k] = None
        return d

    def save_json(self, path: Union[str, Path]) -> Path:
        """Export report to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p

    def summary(self) -> str:
        """Formatted human-readable summary table."""
        d_roc = f"{self.donor_roc_auc:.4f}" if not np.isnan(self.donor_roc_auc) else "N/A"
        a_roc = f"{self.acceptor_roc_auc:.4f}" if not np.isnan(self.acceptor_roc_auc) else "N/A"
        d_pr = f"{self.donor_pr_auc:.4f}" if not np.isnan(self.donor_pr_auc) else "N/A"
        a_pr = f"{self.acceptor_pr_auc:.4f}" if not np.isnan(self.acceptor_pr_auc) else "N/A"

        return (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            " Svelto-DNA Splice Disruption Benchmark Evaluation\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f" Total Evaluated Loci : {self.n_total_loci:,}\n"
            f" Ground Truth Donors  : {self.n_donors:,}\n"
            f" Ground Truth Acceptors: {self.n_acceptors:,}\n"
            f" Severe Imbalance Ratio: {self.imbalance_ratio:.1f} : 1 (non-splice:splice)\n"
            "─────────────────────────────────────────────────────\n"
            f" Metric                Donor (GT)     Acceptor (AG)\n"
            f" ROC-AUC               {d_roc:<14} {a_roc}\n"
            f" PR-AUC (Avg Prec)     {d_pr:<14} {a_pr}\n"
            f" Top-1 Accuracy        {self.donor_top_1_acc * 100:.2f}%{'':<8} {self.acceptor_top_1_acc * 100:.2f}%\n"
            f" Top-k Accuracy (k=2)  {self.donor_top_k_acc * 100:.2f}%{'':<8} {self.acceptor_top_k_acc * 100:.2f}%\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )


def compute_top_k_accuracy(
    y_true_binary: np.ndarray,
    y_scores: np.ndarray,
    k: float = 1.0,
) -> float:
    """
    Computes SpliceAI-style positional Top-k accuracy:
    Given N true splice sites, ranks all positions by predicted probability score
    and computes the recall among the top round(k * N) candidates.

    Args:
        y_true_binary: 1D array of 0s and 1s indicating true splice loci.
        y_scores: 1D array of predicted probabilities.
        k: Multiplier for candidate budget (e.g. 1.0 for Top-1, 2.0 for Top-2).

    Returns:
        Top-k accuracy (recall in top k*N positions), between 0.0 and 1.0.
    """
    n_pos = int(np.sum(y_true_binary))
    if n_pos == 0:
        return float("nan")

    n_candidates = max(1, int(round(k * n_pos)))
    # Top n_candidates indices sorted descending
    top_indices = np.argsort(y_scores)[::-1][:n_candidates]

    captured_pos = np.sum(y_true_binary[top_indices])
    return float(captured_pos / n_pos)


import logging

logger = logging.getLogger(__name__)


def safe_roc_auc(y_true_binary: np.ndarray, y_scores: np.ndarray, default: float = float("nan")) -> float:
    """
    Safely calculates ROC-AUC, catching ValueError if only one class exists in y_true.
    Logs a warning and returns default value (NaN).
    """
    unique = np.unique(y_true_binary)
    if len(unique) < 2:
        logger.warning(
            "Single-class sequence or batch encountered (classes=%s). ROC-AUC is undefined; defaulting to %s.",
            unique,
            default,
        )
        return default
    try:
        return float(roc_auc_score(y_true_binary, y_scores))
    except ValueError as e:
        logger.warning("ValueError encountered during ROC-AUC calculation: %s. Defaulting to %s.", e, default)
        return default
    except Exception as e:
        logger.warning("Unexpected error during ROC-AUC calculation: %s. Defaulting to %s.", e, default)
        return default


def safe_pr_auc(y_true_binary: np.ndarray, y_scores: np.ndarray, default: float = float("nan")) -> float:
    """
    Safely calculates PR-AUC (Average Precision), catching ValueError if 0 positives exist.
    Logs a warning and returns default value (NaN).
    """
    n_pos = int(np.sum(y_true_binary))
    if n_pos == 0 or n_pos == len(y_true_binary):
        logger.warning(
            "Batch contains %d positive sites out of %d total loci. PR-AUC is undefined; defaulting to %s.",
            n_pos,
            len(y_true_binary),
            default,
        )
        return default
    try:
        return float(average_precision_score(y_true_binary, y_scores))
    except ValueError as e:
        logger.warning("ValueError encountered during PR-AUC calculation: %s. Defaulting to %s.", e, default)
        return default
    except Exception as e:
        logger.warning("Unexpected error during PR-AUC calculation: %s. Defaulting to %s.", e, default)
        return default


def compute_threshold_metrics(
    y_true_binary: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """Computes precision, recall, and F1 at a given confidence threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred,
        pos_label=1,
        average="binary",
        zero_division=0,
    )
    return ThresholdMetrics(
        threshold=float(threshold),
        precision=float(p),
        recall=float(r),
        f1=float(f1),
    )


def compute_splice_metrics(
    y_true: Union[Sequence[int], np.ndarray],
    y_probs: Union[Sequence[Sequence[float]], np.ndarray],
    thresholds: Sequence[float] = (0.2, 0.5, 0.8),
    top_k_multiplier: float = 2.0,
) -> SpliceMetricsReport:
    """
    Computes complete splice prediction metrics suite.

    Args:
        y_true: 1D array of ground truth labels (0=Neither, 1=Donor, 2=Acceptor).
        y_probs: 2D array of shape (N, 3) with probabilities for [Neither, Donor, Acceptor].
        thresholds: List of decision thresholds for binary metrics.
        top_k_multiplier: k multiplier for positional top-k evaluation.

    Returns:
        SpliceMetricsReport object.
    """
    y_t = np.asarray(y_true, dtype=int).ravel()
    y_p = np.asarray(y_probs, dtype=float)

    if y_p.ndim == 1:
        raise ValueError("y_probs must be 2D array of shape (N, 3)")
    if y_p.shape[1] != 3:
        raise ValueError(f"y_probs second dimension must be 3, got {y_p.shape[1]}")
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: len(y_true)={len(y_t)} vs len(y_probs)={len(y_p)}")

    n_total = len(y_t)
    donor_true = (y_t == LABEL_DONOR).astype(int)
    acceptor_true = (y_t == LABEL_ACCEPTOR).astype(int)

    n_donors = int(np.sum(donor_true))
    n_acceptors = int(np.sum(acceptor_true))
    total_splice = n_donors + n_acceptors
    n_neither = n_total - total_splice

    imbalance_ratio = float(n_neither / max(1, total_splice))

    donor_probs = y_p[:, LABEL_DONOR]
    acceptor_probs = y_p[:, LABEL_ACCEPTOR]

    # ROC-AUC
    try:
        d_roc = safe_roc_auc(donor_true, donor_probs)
    except ValueError as e:
        logger.warning("ValueError computing donor ROC-AUC: %s. Defaulting to NaN.", e)
        d_roc = float("nan")

    try:
        a_roc = safe_roc_auc(acceptor_true, acceptor_probs)
    except ValueError as e:
        logger.warning("ValueError computing acceptor ROC-AUC: %s. Defaulting to NaN.", e)
        a_roc = float("nan")

    valid_rocs = [r for r in [d_roc, a_roc] if not np.isnan(r)]
    mean_roc = float(np.mean(valid_rocs)) if valid_rocs else float("nan")

    # PR-AUC
    try:
        d_pr = safe_pr_auc(donor_true, donor_probs)
    except ValueError as e:
        logger.warning("ValueError computing donor PR-AUC: %s. Defaulting to NaN.", e)
        d_pr = float("nan")

    try:
        a_pr = safe_pr_auc(acceptor_true, acceptor_probs)
    except ValueError as e:
        logger.warning("ValueError computing acceptor PR-AUC: %s. Defaulting to NaN.", e)
        a_pr = float("nan")

    valid_prs = [r for r in [d_pr, a_pr] if not np.isnan(r)]
    mean_pr = float(np.mean(valid_prs)) if valid_prs else float("nan")

    # Top-1 and Top-k accuracy
    d_top_1 = compute_top_k_accuracy(donor_true, donor_probs, k=1.0)
    d_top_k = compute_top_k_accuracy(donor_true, donor_probs, k=top_k_multiplier)
    a_top_1 = compute_top_k_accuracy(acceptor_true, acceptor_probs, k=1.0)
    a_top_k = compute_top_k_accuracy(acceptor_true, acceptor_probs, k=top_k_multiplier)

    # Threshold metrics
    d_thresh = {
        f"t_{t}": compute_threshold_metrics(donor_true, donor_probs, t)
        for t in thresholds
    }
    a_thresh = {
        f"t_{t}": compute_threshold_metrics(acceptor_true, acceptor_probs, t)
        for t in thresholds
    }

    return SpliceMetricsReport(
        n_total_loci=n_total,
        n_donors=n_donors,
        n_acceptors=n_acceptors,
        imbalance_ratio=imbalance_ratio,
        donor_roc_auc=d_roc,
        acceptor_roc_auc=a_roc,
        mean_roc_auc=mean_roc,
        donor_pr_auc=d_pr,
        acceptor_pr_auc=a_pr,
        mean_pr_auc=mean_pr,
        donor_top_1_acc=d_top_1,
        donor_top_k_acc=d_top_k,
        acceptor_top_1_acc=a_top_1,
        acceptor_top_k_acc=a_top_k,
        donor_threshold_metrics=d_thresh,
        acceptor_threshold_metrics=a_thresh,
    )

"""
Speculative in-silico saturation mutagenesis (issue #4, approved redesign for an attention-free backbone).

HyenaDNA has no attention, so a tree attention mask has nothing to apply to. The speculative idea is
kept at the level of whole variants instead of tokens:

  vanilla      score every candidate variant with its own forward pass (sequential)
  draft        one reference forward pass + a cheap score ranking which variants could matter
  verify       the top-K candidates in one batched forward pass (same numbers as sequential)

Verified scores are exact. Candidates the draft does not select are reported as NaN (unknown), never
as zero; how many truly high-impact variants the draft misses (recall) must be measured against the
exhaustive result, see scripts/benchmark_speculative.py.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

Candidate = Tuple[int, str]


def all_candidates(seq: str, start: int, end: int) -> List[Candidate]:
    """Every alternative base at every position in [start, end)."""
    return [(p, b) for p in range(start, end) if seq[p] in "ACGT" for b in "ACGT" if b != seq[p]]


def draft_scores(seq: str, p_ref: np.ndarray, cands: Sequence[Candidate], radius: int = 10) -> np.ndarray:
    """
    Cheap impact estimate from the reference prediction alone.
      loss: the variant hits a predicted site's dinucleotide (full weight) or its neighbourhood (decays over ~4 bp)
      gain: the alternative base creates a new GT or AG dinucleotide (fixed 0.5)
    """
    site = p_ref[:, 1:].max(axis=-1)
    n = len(seq)
    out = np.zeros(len(cands), dtype=np.float32)
    for k, (p, b) in enumerate(cands):
        lo, hi = max(0, p - radius), min(n, p + radius + 1)
        j = np.arange(lo, hi)
        near = float(np.max(site[lo:hi] * np.exp(-np.abs(j - p) / 4.0))) if hi > lo else 0.0
        on_motif = max(site[p], site[p - 1] if p > 0 else 0.0)
        left, right = seq[p - 1] if p > 0 else "", seq[p + 1] if p + 1 < n else ""
        creates = (left + b in ("GT", "AG")) or (b + right in ("GT", "AG"))
        out[k] = max(on_motif, near, 0.5 if creates else 0.0)
    return out


@dataclass
class ISMResult:
    candidates: List[Candidate]
    deltas: np.ndarray                # max |Δ| (donor/acceptor) within ±delta_window; NaN = not verified
    verified: int
    threshold: float
    timings_ms: Dict[str, float] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> Optional[float]:
        """Share of verified candidates that turned out high-impact (|Δ| ≥ threshold)."""
        v = self.deltas[~np.isnan(self.deltas)]
        return float(np.mean(v >= self.threshold)) if len(v) else None

    def high_impact(self) -> List[Candidate]:
        return [self.candidates[i] for i in np.flatnonzero(np.nan_to_num(self.deltas, nan=-1) >= self.threshold)]


class SpeculativeISM:
    def __init__(self, predictor, context: int = 1000, delta_window: int = 50, threshold: float = 0.5) -> None:
        self.predictor = predictor
        self.context = context
        self.delta_window = delta_window
        self.threshold = threshold

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _slice(self, seq: str, start: int, end: int) -> Tuple[int, str]:
        lo = max(0, start - self.delta_window - self.context)
        hi = min(len(seq), end + self.delta_window + self.context)
        return lo, seq[lo:hi]

    def _reference(self, ref: str) -> np.ndarray:
        return self.predictor.predict([ref])[0].numpy()

    def _verify(self, ref: str, lo: int, p_ref: np.ndarray, cands: Sequence[Candidate], batch_size: int) -> np.ndarray:
        out = np.empty(len(cands), dtype=np.float64)
        for b0 in range(0, len(cands), batch_size):
            chunk = cands[b0:b0 + batch_size]
            muts = [ref[:p - lo] + alt + ref[p - lo + 1:] for p, alt in chunk]
            probs = self.predictor.predict(muts).numpy()
            for k, (p, _) in enumerate(chunk):
                r = p - lo
                a, b = max(0, r - self.delta_window), min(len(ref), r + self.delta_window + 1)
                out[b0 + k] = float(np.max(np.abs(probs[k, a:b, 1:] - p_ref[a:b, 1:])))
        return out

    def exhaustive(self, seq: str, start: int, end: int, batch_size: int = 1) -> ISMResult:
        """Score every candidate. batch_size=1 is the sequential 'vanilla' baseline."""
        cands = all_candidates(seq, start, end)
        lo, ref = self._slice(seq, start, end)
        self._sync(); t0 = time.perf_counter()
        p_ref = self._reference(ref)
        deltas = self._verify(ref, lo, p_ref, cands, batch_size)
        self._sync()
        return ISMResult(cands, deltas, len(cands), self.threshold, {"total": (time.perf_counter() - t0) * 1000})

    def speculative(self, seq: str, start: int, end: int, k: int, batch_size: Optional[int] = None) -> ISMResult:
        """Draft from one reference pass, then verify the top-k candidates in batched passes."""
        cands = all_candidates(seq, start, end)
        lo, ref = self._slice(seq, start, end)
        self._sync(); t0 = time.perf_counter()
        p_ref = self._reference(ref)
        t1 = time.perf_counter()
        scores = draft_scores(ref, p_ref, [(p - lo, b) for p, b in cands])
        top = np.argsort(-scores, kind="stable")[:k]
        t2 = time.perf_counter()
        deltas = np.full(len(cands), np.nan)
        deltas[top] = self._verify(ref, lo, p_ref, [cands[i] for i in top], batch_size or k)
        self._sync(); t3 = time.perf_counter()
        return ISMResult(cands, deltas, len(top), self.threshold,
                         {"reference": (t1 - t0) * 1000, "draft": (t2 - t1) * 1000, "verify": (t3 - t2) * 1000, "total": (t3 - t0) * 1000})

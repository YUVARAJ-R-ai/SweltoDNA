"""Per-connection state: the session's edited sequence and scoring against the reference."""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from svelto_dna.serve.regions import Region


class SpliceSession:
    def __init__(self, predictor, region: Region, context: int = 1000) -> None:
        self.predictor = predictor
        self.region = region
        self.context = context
        self.seq = list(region.seq)
        self._ref_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def apply_edit(self, pos: int, ref: str, alt: str) -> None:
        if not 0 <= pos < len(self.seq):
            raise ValueError(f"position {pos} is outside the region (0..{len(self.seq) - 1})")
        if self.region.seq[pos] != ref:
            raise ValueError(f"reference base at {pos} is {self.region.seq[pos]}, not {ref}")
        if alt not in "ACGT" or len(alt) != 1:
            raise ValueError(f"unsupported base {alt!r}")
        self.seq[pos] = alt

    def reset(self) -> None:
        self.seq = list(self.region.seq)

    def score(self, start: int, end: int) -> Dict[str, Any]:
        """Reference and current probabilities for [start, end), each computed with `context` bp of flank."""
        n = len(self.seq)
        start, end = max(0, start), min(n, end)
        lo, hi = max(0, start - self.context), min(n, end + self.context)
        key = (lo, hi)
        mut = "".join(self.seq[lo:hi])
        ref_str = "".join(self.region.seq[lo:hi])
        if key not in self._ref_cache:
            probs = self.predictor.predict([ref_str, mut]) if mut != ref_str else self.predictor.predict([ref_str])
            self._ref_cache[key] = probs[0].numpy()
            p_mut_full = probs[-1].numpy()
        else:
            p_mut_full = self._ref_cache[key] if mut == ref_str else self.predictor.predict([mut])[0].numpy()
        p_ref = self._ref_cache[key][start - lo:end - lo]
        p_mut = p_mut_full[start - lo:end - lo]
        d = p_mut - p_ref
        r4 = lambda a: np.round(a.astype(float), 4).tolist()  # noqa: E731
        return {
            "window_start": start, "window_end": end, "p_ref": r4(p_ref), "p_mut": r4(p_mut),
            "delta_scores": {"donor_gain": r4(np.clip(d[:, 1], 0, None)), "donor_loss": r4(np.clip(-d[:, 1], 0, None)),
                             "acceptor_gain": r4(np.clip(d[:, 2], 0, None)), "acceptor_loss": r4(np.clip(-d[:, 2], 0, None))},
        }

"""
Speculative Draft Verification Engine for Svelto-DNA.
Provides auxiliary multi-head draft projections (K=3/4) and residual architectures
for sub-second parallel genomic verification.
"""

from svelto_dna.speculative.draft_heads import (
    DraftHead,
    SpeculativeDraftHeads,
    SpeculativeDraftLoss,
)
from svelto_dna.speculative.trainer import DraftHeadTrainer

__all__ = [
    "DraftHead",
    "SpeculativeDraftHeads",
    "SpeculativeDraftLoss",
    "DraftHeadTrainer",
]

"""
Svelto-DNA: Speculative Draft Verification & Genomic Inference Architecture.
"""

__version__ = "0.1.0"

from svelto_dna.core.tokenizer import GenomicTokenizer
from svelto_dna.core.backbone import SveltoBackbone, BackboneOutput
from svelto_dna.splice.delta import compute_delta_scores, DeltaResult

__all__ = [
    "GenomicTokenizer",
    "SveltoBackbone",
    "BackboneOutput",
    "compute_delta_scores",
    "DeltaResult",
]

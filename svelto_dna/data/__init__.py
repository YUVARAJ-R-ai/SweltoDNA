"""
Data pipeline package for Svelto-DNA.
Handles ClinVar splice disruption extraction, SpliceAI-10k ingestion,
strand-aware coordinate resolution, and benchmark evaluation.
"""

from svelto_dna.data.clinvar import (
    ClinVarFilterConfig,
    ClinVarParser,
    generate_synthetic_clinvar_dataset,
    normalize_chrom,
)
from svelto_dna.data.spliceai import (
    SpliceAIDatasetConfig,
    SpliceAIParser,
    StrandCoordinateResolver,
    generate_synthetic_spliceai_dataset,
)
from svelto_dna.data.leakage import (
    DataLeakageError,
    LeakageReport,
    verify_zero_leakage,
)
from svelto_dna.data.benchmark import (
    SpliceMetricsReport,
    compute_splice_metrics,
    compute_top_k_accuracy,
)

__all__ = [
    "ClinVarFilterConfig",
    "ClinVarParser",
    "generate_synthetic_clinvar_dataset",
    "normalize_chrom",
    "SpliceAIDatasetConfig",
    "SpliceAIParser",
    "StrandCoordinateResolver",
    "generate_synthetic_spliceai_dataset",
    "DataLeakageError",
    "LeakageReport",
    "verify_zero_leakage",
    "SpliceMetricsReport",
    "compute_splice_metrics",
    "compute_top_k_accuracy",
]

# Implementation Plan Review: Task 2 (Issue #2)
## [Data-Pipeline] ClinVar Splice Disruption & SpliceAI-10k Benchmark Extraction Suite

## Executive Summary
The implementation plan for Task 2 provides a comprehensive data engineering and benchmarking architecture for ClinVar splice-disrupting SNVs and SpliceAI-10k autosome splits. The top risks are network dependency on multi-gigabyte NCBI ClinVar and SpliceAI FTP/HTTP dumps in sandboxed environments, chromosome nomenclature discrepancies (`chr1` vs `1`), negative-strand coordinate inversion errors, and metric crashes (e.g. `ValueError: Only one class present`) under extreme class sparsity. With the actionable mitigations and architectural refinements detailed below, the plan is rated **🟢 Ready with Minor Fixes** and can proceed directly into implementation.

**Readiness:** 🟢 Ready with Minor Fixes

---

## Outcome Analysis

| Outcome | Trigger Condition | Likelihood | Impact |
| :--- | :--- | :--- | :--- |
| **Full Success** | All pipeline components extract, filter, resolve strand polarity, assert zero chromosome leakage, and compute Top-1/Top-k, ROC-AUC, and PR-AUC with 100% test coverage. | High | Complete, reproducible clinical benchmark suite for Svelto-DNA. |
| **Partial Success — Network Timeout / Remote Failure** | Remote ClinVar release (multi-GB) or SpliceAI server is unreachable or throttled. | Medium | Execution halts unless an automated synthetic/fixture generator and caching layer are included. |
| **Silent Failure — Negative Strand Coordinate Off-by-One** | Locus indexing in reverse-complement window fails to reverse relative to $5' \to 3'$ transcript polarity. | Medium | Evaluates model on wrong flank; metrics collapse silently or produce false negative errors. |
| **Silent Failure — Chromosome Nomenclature Inconsistency** | `chr1` vs `1` in ClinVar vs SpliceAI partitions causes false positive leakage alerts or empty joins. | Medium | Pipeline skips valid variants or fails chromosome partition checks. |
| **Silent Failure — Single-Class Metric Exception** | Windows with zero positive donor/acceptor loci raise `ValueError` in raw `sklearn.metrics.roc_auc_score`. | High | Pipeline crashes on non-splice chunks unless safely guarded. |

---

## Gap Analysis

| Gap | Category | Severity | Recommendation |
| :--- | :--- | :--- | :--- |
| **Remote Download Resilience & Fixture Mode** | Dependency / Reliability | **High** | Implement a deterministic built-in mock/fixture dataset generator (`generate_synthetic_splice_data`) in `clinvar.py` and `spliceai.py` so unit tests and offline benchmarking execute with zero external network calls. |
| **Chromosome Nomenclature Normalization** | Data Integrity | **High** | Implement a strict `normalize_chrom(chrom: str) -> str` utility standardizing `chr1` and `1` into canonical `chr{N}` notation across all data sources. |
| **Negative Strand Polarity Coordinate Resolver** | Domain / Algorithm | **High** | Formulate a dedicated `StrandCoordinateResolver` with strict invariant unit tests verifying that donor (`GT`) and acceptor (`AG`) loci maintain identical relative biological positions in $5' \to 3'$ transcript orientation. |
| **Class Imbalance Guarded Metrics** | Observability / ML | **High** | Wrap `roc_auc_score` and `average_precision_score` in a defensive `compute_guarded_metrics()` helper that returns `NaN` or calibrated baselines ($P(\text{class})$) when only one class is present in a batch. |
| **Parquet Schema Validation & Polars Memory** | Data / Performance | **Medium** | Define explicit PyArrow/Polars schemas for ClinVar and SpliceAI Parquet tables with zstd compression to optimize IO throughput and memory footprint. |

---

## Risk & Assumption Register

### Risk Register
| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **NCBI / Ensembl FTP Unreachable** | Medium | High | Support local file path inputs (`--input-file`) and automated fallback synthetic generation. |
| **High Memory Usage on Full Genome Scanning** | Low | High | Utilize Polars chunked lazy scanning (`pl.scan_parquet()`) and batch window generation. |
| **Severe Class Imbalance (>100:1) Metric Distortion** | Medium | Medium | Track PR-AUC (Average Precision) as the primary evaluation metric alongside ROC-AUC and Top-k accuracy. |

### Assumption Register
| Assumption | Validated? | If Wrong… |
| :--- | :--- | :--- |
| Variants are limited to SNVs within $\pm 50$ bp of junctions | Yes (per Issue #2 technical spec) | Multi-nucleotide or structural indels are excluded by filter. |
| SpliceAI autosome test split uses chr1, chr3, chr5, chr7, chr9 | Yes (canonical SpliceAI literature split) | Leakage verifier validates custom split configurations if overridden. |
| Polars is available and compatible with PyArrow | Yes (verified in `.venv` with `polars==1.44.2`, `pyarrow==25.0.1`) | High-speed columnar serialization operates smoothly. |

---

## Dependency & Integration Gaps

| Dependency | Issue | Severity | Recommendation |
| :--- | :--- | :--- | :--- |
| **`GenomicTokenizer` (`svelto_dna.core.tokenizer`)** | Must be used for reverse complementation and context window extraction | Medium | Directly integrate `GenomicTokenizer` methods rather than reimplementing string manipulation. |
| **`SveltoBackbone` (`svelto_dna.core.backbone`)** | Evaluation CLI needs inference oracle | Medium | Connect `scripts/eval_splice_benchmark.py` directly to `SveltoBackbone` with mock fallback support. |
| **`scikit-learn`** | Required for ROC-AUC and PR-AUC | Low | Verified installed in `.venv`. |

---

## Strengths
- **Rigorous Domain Grounding:** Explicitly models canonical `GT`/`AG` dinucleotide junctions and $\pm 50$ bp clinical impact windows.
- **Strict Separation of Diagnostic Tiers:** ClinVar records partition pathogenic from benign, isolating ambiguous VUS for dedicated holdout testing.
- **Leakage Prevention as a First-Class Citizen:** Chromosome-level disjointness ensures academic validity and zero benchmark contamination.
- **Columnar Efficiency:** Leveraging Polars and Parquet provides order-of-magnitude faster read/write speeds over legacy Pandas/CSV.

---

## Prioritised Recommendations
1. **[High] Add Synthetic Fixture Generation:** Build `generate_synthetic_clinvar_dataset()` and `generate_synthetic_spliceai_dataset()` to guarantee 100% offline testability and instant verification.
2. **[High] Implement Robust Strand Coordinate Resolver:** Add explicit unit tests testing forward (`+`) and reverse (`-`) strand coordinate transformations.
3. **[High] Guarded Class-Imbalance Metrics:** Safely handle single-class batches and compute Top-1, Top-k, ROC-AUC, and PR-AUC.
4. **[Medium] Standardize Chromosome Prefixing:** Apply `normalize_chrom` across all parsing and validation routines.

---

## Reviewer Verdict
**Verdict:** 🟢 Ready with Minor Fixes.  
The plan is approved for immediate implementation incorporating the recommendations above.

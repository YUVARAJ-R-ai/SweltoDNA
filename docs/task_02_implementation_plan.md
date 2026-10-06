# Technical Implementation Plan: Task 2 (Issue #2)
## [Data-Pipeline] ClinVar Splice Disruption & SpliceAI-10k Benchmark Extraction Suite

- **Author:** Computational Biology & ML Engineering Pair
- **Repository:** `YUVARAJ-R-ai/SweltoDNA`
- **Related Issue:** [#2](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/2)
- **Target Sprint:** Sprint 1 (P0, Size M)

---

## 1. Executive Summary & Problem Formulation
Deep genomic foundation models and speculative verification heads require real-world clinical and canonical splice variant benchmarks to quantify variant effect prediction accuracy (ROC-AUC, PR-AUC, and Top-k accuracy). Canonical splice junctions in the human genome are dominated by donor (`GT`) and acceptor (`AG`) dinucleotides. Single-nucleotide variants (SNVs) within $\pm 50$ bp of these junctions disrupt natural splicing or create cryptic splice sites, leading to severe genetic pathology.

The goal of Task 2 is to:
1. Build an automated extraction, parsing, and filtering pipeline for **ClinVar GRCh38** variants and **SpliceAI-10k** autosome datasets into high-performance Parquet format using Polars and PyArrow.
2. Filter for SNVs located within $\pm 50$ bp of canonical donor and acceptor junctions, cleanly segregating pathogenic from benign variants, and routing variants with conflicting classifications or uncertain significance (VUS) into a dedicated holdout evaluation set.
3. Formulate a robust coordinate resolver accounting for strand polarity ($+$ vs $-$ strand transcripts) to prevent inverted coordinate errors and ensure flanking sequence windows ($L \in [1,000, 10,000]$ bp) align with $5' \to 3'$ transcript orientation using `GenomicTokenizer`.
4. Enforce strict chromosome-level data leakage prevention across homologous chromosome groups, ensuring absolute separation between training, validation, and test splits.
5. Implement a comprehensive benchmark evaluation engine (`scripts/eval_splice_benchmark.py` and `svelto_dna/data/benchmark.py`) computing Top-1, Top-k accuracy, ROC-AUC, and PR-AUC under severe class imbalance (>100:1 non-splice vs splice ratio) with reproducible fixed seeds.
6. Provide full unit tests in `tests/test_data_pipeline.py` and `tests/test_benchmark_eval.py`.

---

## 2. Architecture & Module Design

### 2.1 Directory Structure
```
.
├── svelto_dna/
│   ├── core/
│   │   ├── tokenizer.py
│   │   └── backbone.py
│   ├── profiler/
│   │   └── benchmark.py
│   └── data/
│       ├── __init__.py
│       ├── clinvar.py          # ClinVar GRCh38 parser, +-50bp junction filter, VUS partitioner
│       ├── spliceai.py         # SpliceAI-10k split parser, coordinate resolver & window extractor
│       ├── leakage.py          # Chromosome-group split validator & leakage verifier
│       └── benchmark.py        # Metrics calculator: Top-1/Top-k, ROC-AUC, PR-AUC
├── scripts/
│   ├── eval_splice_benchmark.py # Standalone evaluation harness CLI
│   └── extract_splice_data.py   # Dataset extraction & Parquet generation CLI
└── tests/
    ├── test_data_pipeline.py    # Unit tests for ClinVar, SpliceAI, strand polarity, leakage
    └── test_benchmark_eval.py   # Unit tests for ROC-AUC, PR-AUC under severe class imbalance
```

### 2.2 Component Specifications

#### A. ClinVar Extraction & Filtering (`svelto_dna/data/clinvar.py`)
- **Input Sources:** Standard ClinVar VCF or Tab-delimited variant summary files (GRCh38), plus built-in curated canonical clinical test sets for automated offline testing.
- **Filtering Logic:**
  - Variant Type: Strictly Single Nucleotide Variants (SNVs: `len(ref) == 1 and len(alt) == 1`).
  - Proximity: Loci within $\pm 50$ bp of canonical donor (`GT`) or acceptor (`AG`) splice junctions.
  - Clinical Significance Partitioning:
    - `Pathogenic`: Classified as `Pathogenic`, `Likely pathogenic`, or `Pathogenic/Likely pathogenic`.
    - `Benign`: Classified as `Benign`, `Likely benign`, or `Benign/Likely benign`.
    - `Holdout (VUS)`: Classified as `Uncertain significance`, `conflicting interpretations`, or unreviewed.
- **Output:** Polars DataFrame saved to Parquet with schema:
  `['variant_id', 'chrom', 'pos', 'ref', 'alt', 'strand', 'gene_id', 'junction_type', 'dist_to_junction', 'clinical_significance', 'is_pathogenic', 'is_holdout']`.

#### B. SpliceAI-10k Ingestion & Coordinate Resolver (`svelto_dna/data/spliceai.py`)
- **Splice Classification Labels:**
  - `0`: Neither (non-splice locus)
  - `1`: Donor (`GT`)
  - `2`: Acceptor (`AG`)
- **Strand Polarity & Inversion:**
  - For positive-strand genes (`+`):
    - Sequence window extracted symmetrically around locus using `GenomicTokenizer.extract_context_window`.
    - Label coordinates correspond directly to sequence indices.
  - For negative-strand genes (`-`):
    - Window extracted and transformed via `GenomicTokenizer.reverse_complement`.
    - Locus indices in the window are mapped using $i' = L - 1 - i$.
    - Donor and acceptor positions maintain biological $5' \to 3'$ transcript polarity.
- **Context Window Sizes:** Parameterized for $L \in \{1000, 2000, 5000, 10000\}$.

#### C. Chromosome Split Leakage Checker (`svelto_dna/data/leakage.py`)
- **Canonical SpliceAI Partitioning:**
  - **Test Set:** Chromosomes `chr1`, `chr3`, `chr5`, `chr7`, `chr9`.
  - **Validation Set:** Chromosomes `chr2`, `chr4`, `chr6`, `chr8`, `chr10`.
  - **Training Set:** Remaining autosomes `chr11` through `chr22`.
- **Leakage Verifications:**
  - Enforce disjoint chromosome sets:
    $$\mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{val}} = \emptyset, \quad \mathcal{C}_{\text{train}} \cap \mathcal{C}_{\text{test}} = \emptyset, \quad \mathcal{C}_{\text{val}} \cap \mathcal{C}_{\text{test}} = \emptyset$$
  - Verify zero overlapping genes or shared transcript coordinates across split boundaries.

#### D. Metrics & Imbalance-Aware Benchmark Harness (`svelto_dna/data/benchmark.py` & `scripts/eval_splice_benchmark.py`)
- **Severe Class Imbalance Handling:** Non-splice to splice ratio exceeds $100:1$ in natural sequences.
- **Metrics Computed:**
  - **Top-1 and Top-k Accuracy:** Percentage of ground-truth donor/acceptor sites captured within top $k$ predicted positions.
  - **ROC-AUC:** Area under the ROC curve for Donor vs Other and Acceptor vs Other.
  - **PR-AUC (Average Precision):** Area under the Precision-Recall curve, essential for evaluating low prevalence classes.
  - **Precision, Recall, F1 at Diagnostic Thresholds:** Evaluated at clinical confidence thresholds ($T \in \{0.2, 0.5, 0.8\}$).
- **Reproducibility:** Seeded random state (`seed=42`) for deterministic bootstrapping and evaluation.

---

## 3. Step-by-Step Implementation Sequence

1. **Step 1: Data Pipeline Module Layout (`svelto_dna/data/__init__.py`)**
   - Initialize package exports and core data types/dataclasses.
2. **Step 2: ClinVar Processing (`svelto_dna/data/clinvar.py`)**
   - Implement VCF/TSV parser, proximity filtering ($\pm 50$ bp), VUS holdout partitioning, and Polars Parquet serialization.
   - Include synthetic / mock generator for offline test execution.
3. **Step 3: SpliceAI Processing & Strand Resolver (`svelto_dna/data/spliceai.py`)**
   - Implement SpliceAI dataset reader, strand inversion, and context window extractor ($L=1\text{k}$–$10\text{k}$).
4. **Step 4: Data Leakage Verification (`svelto_dna/data/leakage.py`)**
   - Implement chromosome group validation and disjoint assertions.
5. **Step 5: Benchmark Evaluation Engine (`svelto_dna/data/benchmark.py`)**
   - Implement Top-1, Top-k, ROC-AUC, and PR-AUC metric calculations.
6. **Step 6: CLI Scripts (`scripts/extract_splice_data.py`, `scripts/eval_splice_benchmark.py`)**
   - Build command-line utilities for end-to-end dataset extraction and model evaluation.
7. **Step 7: Comprehensive Unit Testing (`tests/test_data_pipeline.py`, `tests/test_benchmark_eval.py`)**
   - Write tests covering all acceptance criteria with 100% passing tests.

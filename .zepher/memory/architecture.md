# Architecture

- **Foundation Backbone Module (`svelto_dna.core.backbone`):**
  - `SveltoBackbone`: Invariant verification oracle wrapping genomic foundation models (HyenaDNA, Nucleotide Transformer, or architecture-faithful Mock fallback).
  - Enforces strict parameter freezing (`trainable_parameters_count == 0`, `model.eval()`).
  - Emits `BackboneOutput(last_hidden_state, penultimate_hidden_state, logits)`, preserving penultimate states for downstream $K$-head speculative verification (Issue #3).
- **Genomic Tokenizer (`svelto_dna.core.tokenizer`):**
  - `GenomicTokenizer`: Single-nucleotide resolution mapping (`<PAD>:0, A:1, C:2, G:3, T:4, N:5`).
  - Supports IUPAC degeneracy resolution (`map_to_n`, `first`), reverse-complement strand inversion, and symmetrical context window extraction up to 10,000 bp.
- **Baseline Profiler (`svelto_dna.profiler.benchmark`):**
  - Measures wall-clock latency (mean, median, p95, min, max), peak memory (CUDA VRAM or CPU RSS), and throughput across $L \in \{1024, 2048, 5000, 10000\}$ bp.
  - Outputs standardized JSON telemetry consumed by the benchmarking HUD.
- **ClinVar Splice Extraction (`svelto_dna.data.clinvar`):**
  - Parses ClinVar GRCh38 variant dumps (VCF/TSV), filters SNVs within +-50 bp of canonical GT/AG junctions.
  - Partitions pathogenic vs benign variants, and routes VUS / conflicting interpretations into holdout evaluation sets.
  - Columnar Parquet serialization with Polars and zstd compression.
- **SpliceAI-10k Dataset & Strand Coordinate Resolver (`svelto_dna.data.spliceai`):**
  - Ingests SpliceAI splits across autosomes with symmetric context windowing $L \in [1\text{k}, 10\text{k}]$.
  - `StrandCoordinateResolver`: Reverses sequences via `GenomicTokenizer.reverse_complement` and correctly reflects splice dinucleotide coordinates to preserve biological 5' -> 3' transcript polarity on negative (-) strands.
- **Chromosome Leakage Prevention (`svelto_dna.data.leakage`):**
  - Enforces disjoint autosome partitions (test: chr1,3,5,7,9; val: chr2,4,6,8,10; train: chr11-22) with assertion-backed zero leakage across chromosomes and genes.
- **Splice Benchmark Evaluation Engine (`svelto_dna.data.benchmark` & `scripts/eval_splice_benchmark.py`):**
  - Implements Top-1, Top-k positional accuracy, ROC-AUC, and PR-AUC (Average Precision) robust to severe class imbalance (>100:1 non-splice vs splice ratio) with guarded single-class safety.


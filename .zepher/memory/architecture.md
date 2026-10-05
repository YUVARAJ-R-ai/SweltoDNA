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
- **Vectorized Delta Score Calculator (`svelto_dna.splice.delta`):**
  - `compute_delta_scores`: Vectorized computation of donor gain/loss and acceptor gain/loss disruption metrics across genomic sequences up to $L = 10,000$ bp.
  - Leverages 1D max pooling (`torch.nn.functional.max_pool1d`) with symmetrical padding ($W = \pm 50$ bp) executing in $< 2$ ms on CPU ($< 1$ ms on GPU).
  - Supports canonical splice site masking to suppress false-positive cryptic alerts at known junctions.
  - Returns `DeltaResult` structured dictionary containing donor/acceptor arrays, locus delta, peak delta, peak component, and peak position.
- **Delta Scoring Benchmark Runner (`benchmarks/benchmark_delta.py`):**
  - Profiles full-sequence and point-locus execution times and nucleotide throughput across $L \in \{1024, 2048, 5000, 10000\}$, producing `delta_benchmark_telemetry.json`.

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

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
- **Speculative Draft Heads Module (`svelto_dna.speculative.draft_heads`):**
  - `DraftHead`: 2-layer residual projection MLP with LayerNorm, SiLU, and linear skip projection ($\text{DraftHead}_k(h_t) = W_2^{(k)} \cdot \text{SiLU}(\text{LayerNorm}(W_1^{(k)} h_t + b_1^{(k)})) + h_t \cdot \mathbb{I}_{\text{res}}$).
  - `SpeculativeDraftHeads`: Parallel container attaching $K \in \{3, 4\}$ heads onto the penultimate hidden states ($H \in \mathbb{R}^{B \times L \times D}$) emitting candidate predictions $(K, B, L, C)$ ($C=3$ for splice classes or $C=4$ for nucleotides).
  - Enforces $< 4\%$ parameter overhead relative to base foundation backbone (2.87% for $K=3$, 3.83% for $K=4$).
  - `SpeculativeDraftLoss`: Multi-target discounted cross-entropy objective ($\mathcal{L}_{\text{draft}} = \sum_{k=1}^K \lambda_k \cdot \text{CE}(\hat{y}_{t+k}^{(k)}, y_{t+k})$ with $\lambda_k = \gamma^{k-1}, \gamma = 0.85$).
  - `DraftHeadTrainer`: Training and parameter isolation harness verifying base backbone remains completely frozen (`p.grad is None`) while draft heads optimize.

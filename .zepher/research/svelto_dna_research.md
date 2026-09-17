# Zepher Research Entry: Svelto-DNA

- **Topic:** Svelto-DNA Speculative Draft Verification & Genomic Inference Architecture
- **Date:** 2026-09-17
- **Key Discoveries:**
  - Standard clinical variant effect predictors (SpliceAI, SpliceAI-lookup) rely on precomputed static SNV tables or slow legacy TensorFlow models (rate-limited to few queries/min); they fail on novel custom constructs and real-time in silico saturation mutagenesis.
  - In silico saturation mutagenesis (ISM) across 1k-10k bp requires thousands of forward passes if done sequentially.
  - Speculative decoding with Medusa-style draft heads ($K=3/4$ residual 2-layer MLPs) and DAG tree attention masking enables parallel candidate verification in a single backbone forward pass, achieving 2.0x-3.5x speedups.
  - Recommended stack: PyTorch 2.3+ with HyenaDNA/Nucleotide Transformer backbone, FastAPI WebSocket streaming server, Next.js 14 virtualized sequence canvas (@tanstack/react-virtual), and Canvas-rendered dual-track splice disruption monitors.
- **Reference Doc:** `docs/research.md`

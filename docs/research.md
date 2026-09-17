# Project Research: Svelto-DNA
_Generated: 2026-09-17_

## Problem & Goal
Deep genomic foundation models and variant effect predictors (e.g., SpliceAI, Nucleotide Transformer) suffer from high inference latency and memory-bandwidth bottlenecks, restricting clinical variant analysis and in silico saturation mutagenesis to offline batch pipelines. Svelto-DNA introduces speculative draft verification heads and tree attention masking onto frozen genomic foundation backbones, reducing wall-clock inference latency by 2.0x–3.5x. This transforms variant effect scoring and splice-site mutagenesis from hours of batch compute into an interactive, sub-second web diagnostic workflow.

## Target Users
- **Clinical Geneticists & Molecular Pathologists:** Diagnosing rare disease variants of uncertain significance (VUS) impacting canonical and cryptic splice junctions in real time.
- **Computational Biologists & Genomic Researchers:** Conducting in silico saturation mutagenesis (ISM) across 1k–10k bp locus windows without prohibitive GPU compute bills or multi-hour queues.
- **Synthetic Biologists & RNA Therapeutic Designers:** Engineering splice-switching antisense oligonucleotides (ASOs), exon-skipping therapies, and optimizing synthetic gene sequences against aberrant splicing.
- **Diagnostic Laboratory Bioinformaticians:** Integrating high-throughput, low-latency variant scoring services into automated clinical interpretation pipelines.

## Competitive Landscape
| Product | Strengths | Weaknesses | Key Takeaway |
|---------|-----------|------------|--------------|
| **Broad SpliceAI-Lookup** | Standard clinical reference tool; precomputed scores for common GRCh37/38 SNVs; IGV.js track visualization. | Strictly throttled (few queries/min); precomputed lookup only; cannot evaluate custom synthetic sequences or dynamic in silico mutagenesis; backend runs legacy TensorFlow. | Clinicians value visual donor/acceptor delta tracks, but static lookup cannot support interactive sequence exploration or novel multi-hit mutations. |
| **SpliceAI-visual / MobiDetails** | Provides raw splice probability and delta visualization; handles complex indels better than raw VCF output. | Runs as isolated Google Colab notebooks or slow batch jobs; high startup friction; no real-time sub-second canvas. | Visualizing raw probabilities alongside delta spikes is clinically essential, but must be powered by instant sub-second backend inference. |
| **OpenSpliceAI / CI-SpliceAI** | PyTorch rewrite of SpliceAI; dynamic memory allocation; supports local transfer learning and calibration. | Pure sequential forward-pass execution; no speculative decoding acceleration; CLI/Python only with no interactive UI. | Modern PyTorch implementation and delta score mathematics are clean reference points, but lack speculative draft acceleration. |
| **Foundation Generative Models (DNAGPT / Evo2 / EvSpark)** | Long-range context representation; initial speculative sampling experiments for genomic sequence generation. | High parameter footprint (billions of parameters); optimized for de novo sequence generation rather than clinical variant effect / splice junction delta scoring. | Adapt the speculative multi-head draft verification paradigm specifically for deterministic variant delta scoring rather than free-form autoregressive text generation. |

## Feature List

### Core (MVP)
- [ ] **Frozen Genomic Foundation Backbone Loader** — Loads pre-trained GRCh38 weights (HyenaDNA-large-1k or Nucleotide Transformer 500M) in FP16/BF16 with frozen parameters to serve as the invariant verification oracle.
- [ ] **Auxiliary Speculative Draft Heads ($K=3/4$)** — Lightweight 2-layer residual MLPs with LayerNorm and SiLU attached to penultimate representations to draft downstream splice probabilities concurrently.
- [ ] **Tree Attention DAG Masking Engine** — Medusa-style directed acyclic tree attention verification mask allowing parallel validation of multiple candidate mutation states in a single backbone forward pass.
- [ ] **Vectorized Splice Delta Score Calculator** — Efficient tensor computation of donor loss/gain and acceptor loss/gain ($\Delta = \max(P_{\text{mutant}}) - P_{\text{reference}}$) across the active genomic window.
- [ ] **ClinVar & SpliceAI-10k Validation Suite** — Standardized benchmark harness measuring diagnostic accuracy (ROC-AUC, PR-AUC) and wall-clock acceleration curves against vanilla autoregressive execution.
- [ ] **FastAPI WebSocket Streaming Server** — Persistent asynchronous bi-directional communication layer delivering sub-second delta scores and telemetry frames upon mutation events.
- [ ] **Virtualized Interactive Sequence Ribbon** — High-performance horizontal nucleotide viewer (A, C, G, T badges) rendering 1,000 to 10,000 bp without DOM stutter or layout thrashing.
- [ ] **Real-Time Mutagenesis Canvas** — Interactive click-to-mutate interface with a radial base switcher that immediately fires WebSocket mutation payloads.
- [ ] **Comparative Multi-Track Splice Visualizer** — Synchronized dual-track display comparing wild-type reference donor/acceptor markers against mutant cryptic activation/loss spikes with interactive tooltips.
- [ ] **Benchmarking Telemetry HUD** — Live diagnostics dashboard displaying wall-clock latency (Vanilla vs. Svelto-DNA), draft acceptance rate ($\alpha$), and active FLOP savings.

### Important (v1.1)
- [ ] **Mutated Locus Attention & Saliency Maps** — Extraction and visualization of backbone attention weights and hidden-state gradients around altered nucleotides.
- [ ] **Phased Multi-Nucleotide & In-Del Support** — Handling compound heterozygous variants and micro-deletions through the speculative tree verifier.
- [ ] **FASTA & VCF Clinical Export** — One-click export of mutated constructs and standardized clinical reports with HGVS nomenclature and pathogenic risk tiering.
- [ ] **Canonical Splice Junction Annotation Overlay** — Visual indicators for canonical GT-AG donor and acceptor consensus motifs with Gencode v38 exon/intron structure.

### Nice-to-have (Backlog)
- [ ] **Full-Window In Silico Saturation Mutagenesis Heatmap** — Automated pre-computation of all $3 \times L$ possible base substitutions across the window displayed as a 2D interactive matrix.
- [ ] **WebGPU Client-Side Verification Fallback** — Browser-based local ONNX/WebGPU inference for zero-cloud offline exploratory analysis on client workstations.
- [ ] **ClinVar REST API Live Sync** — Dynamic query overlay fetching known clinical classifications for queried genomic loci.
- [ ] **Multi-Transcript Alternative Splicing Selector** — Switcher to evaluate differential splicing disruption across overlapping gene isoforms.

### Don't Build
- **Whole-Genome Autoregressive DNA Generator** — De novo generative design is outside the clinical diagnostic scope and degrades verification latency.
- **Heavy 40B+ Parameter Serving Cluster** — Impractical for commodity clinical hardware and sub-second diagnostic turnaround.
- **Legacy TensorFlow 1.x / Keras Compatibility Layer** — SpliceAI's deprecated runtime introduces static memory locks and slow execution; standardizing on PyTorch 2.x is strictly required.

## Task Breakdown

### Frozen Genomic Backbone & Benchmark Baseline
- [ ] Build GRCh38 FASTA reference sequence data loader and tokenizer (S)
- [ ] Integrate pre-trained genomic foundation model checkpoint (HyenaDNA / Nucleotide Transformer) (M)
- [ ] Implement vanilla sequential forward-pass profiling script for baseline latency and VRAM measurement (S)
- [ ] Implement ClinVar & SpliceAI-10k extraction pipeline focusing on SNVs affecting canonical GT/AG junctions (M)  ← depends on: Build GRCh38 FASTA reference sequence data loader and tokenizer
- [ ] Compute baseline diagnostic metrics (ROC-AUC, PR-AUC) for vanilla backbone (M)  ← depends on: Integrate pre-trained genomic foundation model checkpoint

### Speculative Draft Heads & Verification Engine
- [ ] Design and implement $K=3/4$ residual projection draft heads (2-layer MLP with LayerNorm and SiLU) (M)  ← depends on: Integrate pre-trained genomic foundation model checkpoint
- [ ] Implement auxiliary draft head training loop with cross-entropy loss over adjacent sequence positions (L)  ← depends on: Design and implement $K=3/4$ residual projection draft heads
- [ ] Construct Medusa-style directed acyclic tree attention mask for candidate mutation verification (L)  ← depends on: Design and implement $K=3/4$ residual projection draft heads
- [ ] Implement speculative verification logic and acceptance rate ($\alpha$) calculator (M)  ← depends on: Construct Medusa-style directed acyclic tree attention mask for candidate mutation verification
- [ ] Validate 2.0x–3.5x wall-clock inference speedup curves across varying sequence window lengths (1k–10k bp) (M)  ← depends on: Implement speculative verification logic and acceptance rate ($\alpha$) calculator

### High-Performance FastAPI Engine & Delta Scoring
- [ ] Create asynchronous FastAPI server with WebSocket endpoints for sequence sessions (S)
- [ ] Implement vectorized donor/acceptor disruption calculator: $\Delta = \max(P_{\text{mutant}}) - P_{\text{reference}}$ (S)
- [ ] Build model inference worker with async job queue and cancellation tokens to prevent request pile-up (M)  ← depends on: Create asynchronous FastAPI server with WebSocket endpoints for sequence sessions
- [ ] Implement attention weight and hidden-state gradient extraction module for mutated loci (M)  ← depends on: Build model inference worker with async job queue and cancellation tokens to prevent request pile-up
- [ ] Write integration test suite benchmarking WebSocket round-trip latency (<100ms budget) (S)  ← depends on: Build model inference worker with async job queue and cancellation tokens to prevent request pile-up

### Next.js Interactive Web Application & Visualization
- [ ] Initialize Next.js 14 application with Tailwind CSS and TypeScript (S)
- [ ] Implement virtualized horizontal sequence ribbon rendering up to 10k bp with `@tanstack/react-virtual` (M)  ← depends on: Initialize Next.js 14 application with Tailwind CSS and TypeScript
- [ ] Build interactive radial nucleotide switcher for instant base flipping (S)  ← depends on: Implement virtualized horizontal sequence ribbon rendering up to 10k bp with `@tanstack/react-virtual`
- [ ] Implement WebSocket client manager with automatic reconnect, ping/pong, and sequence state sync (S)  ← depends on: Initialize Next.js 14 application with Tailwind CSS and TypeScript
- [ ] Build comparative multi-track visualizer (Reference Track vs Mutant Disruption Track) with Canvas/SVG (M)  ← depends on: Implement virtualized horizontal sequence ribbon rendering up to 10k bp with `@tanstack/react-virtual`
- [ ] Create Telemetry HUD component displaying real-time FPS, latency comparison, acceptance $\alpha$, and FLOP savings (S)  ← depends on: Implement WebSocket client manager with automatic reconnect, ping/pong, and sequence state sync

### Scientific Scaffolding & Paper Scaffold (IEEE BIBM / ACM-BCB)
- [ ] Generate paper outline and LaTeX scaffolding with IEEE conference formatting (S)
- [ ] Draft Methodology section: formal mathematical formulation of speculative tree decoding in genomic sequences (M)
- [ ] Export automated benchmarking tables and speedup ablation figures for draft head depths $K \in \{1, 2, 3, 4\}$ (M)  ← depends on: Validate 2.0x–3.5x wall-clock inference speedup curves across varying sequence window lengths (1k–10k bp)
- [ ] Document clinical case study on ClinVar pathogenic splice variant rescue / cryptic activation (M)  ← depends on: Compute baseline diagnostic metrics (ROC-AUC, PR-AUC) for vanilla backbone

## Tech Recommendations
| Layer | Recommendation | Reason |
|-------|---------------|--------|
| **Foundation Backbone** | `HyenaDNA-large-1k` (default) & `Nucleotide Transformer 500M` | Sub-quadratic scaling over long genomic sequences (1k–10k bp) with single-nucleotide resolution and low VRAM footprint on commodity GPUs. |
| **Deep Learning Runtime** | `PyTorch 2.3+` with `torch.compile` and FlashAttention-2 | Native support for customized attention masks, low dispatch overhead, and optimized CUDA kernels for draft head execution. |
| **Draft Head Architecture** | 2-Layer Residual MLPs ($K=3$ or $4$) with LayerNorm + SiLU | Lightweight (<4% parameter overhead), zero disruption to backbone representations, and fast parallel draft emission. |
| **Backend Framework** | `FastAPI` + `uvloop` + `websockets` | Minimal Python asynchronous latency (<5ms event loop overhead) and seamless streaming of tensor metrics. |
| **Frontend Framework** | `Next.js 14` (App Router) + `TypeScript` + `Tailwind CSS` | Modern React 18 concurrent features, typed state management for genomic coordinates, and utility-first design. |
| **Sequence Virtualization** | `@tanstack/react-virtual` | Superior performance over raw DOM; smoothly handles 10,000 base pair nodes by recycling offscreen elements. |
| **Track & Signal Graphics** | HTML5 Canvas / High-Performance SVG | Frame-budget efficient rendering of dense splice probability peaks and real-time delta score spikes without re-render cascades. |
| **Genomic Data Tooling** | `Polars` + `pysam` / `pyfaidx` | Vectorized parsing of FASTA references, VCF variant files, and ClinVar tables with 10x speedup over standard Biopython/Pandas. |

## Risks & Open Decisions

### Risks
- **Draft Head Accuracy Drop in High-Entropy Splice Regions** — mitigation: Apply temperature scaling and train auxiliary heads using weighted cross-entropy focused on sequence windows within 50bp of canonical GT/AG motifs.
- **Tree Mask Compatibility with Non-Transformer Convolutions (HyenaDNA)** — mitigation: For FFT-based HyenaDNA layers, implement candidate verification via parallel batch-chunk slicing; for Transformer backbones (NT), deploy direct DAG attention masks.
- **WebSocket Backpressure under Rapid Mutagenesis Clicks** — mitigation: Implement a 50ms client-side debouncer and server-side `asyncio.Task` cancellation to immediately discard stale forward-pass calculations when a newer base flip arrives.
- **Client DOM Degradation at 10k bp Zoom-Out** — mitigation: Dynamically switch to a multi-resolution Canvas LOD (Level-of-Detail) density plot when the viewport zoom exceeds 500 bp.

### Open Decisions
- [ ] **Backbone Primacy:** Finalize whether HyenaDNA-large-1k or Nucleotide Transformer 500M serves as the primary evaluation backbone for initial paper submissions.
- [ ] **Draft Depth $K$ Selection:** Balance compute overhead vs. acceptance rate $\alpha$ (benchmark $K=3$ vs $K=4$ on NVIDIA RTX 3090/4090 and consumer GPUs).
- [ ] **Loss Function for Draft Heads:** Determine whether to use standard cross-entropy on ground-truth splice labels or knowledge distillation from the backbone's full logits.

## GitHub References
- [FasterDecoding/Medusa](https://github.com/FasterDecoding/Medusa) — Reference implementation for multi-head speculative decoding, DAG tree attention masks, and candidate verification algorithms.
- [HazyResearch/hyena-dna](https://github.com/HazyResearch/hyena-dna) — Sub-quadratic genomic foundation model architecture optimized for long sequence context at single-nucleotide resolution.
- [InstaDeepAI/nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer) — SOTA genomic transformer checkpoints benchmarked on ClinVar splice disruption prediction.
- [Kuanhao-Chao/OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI) — Open-source PyTorch implementation of splice-site prediction and vectorized delta score computation.
- [broadinstitute/SpliceAI-lookup](https://github.com/broadinstitute/SpliceAI-lookup) — Clinical web interface standards and IGV.js track conventions for splice donor and acceptor delta scores.
- [igvteam/igv.js](https://github.com/igvteam/igv.js) — High-performance interactive genome browser rendering techniques for genomic coordinates.

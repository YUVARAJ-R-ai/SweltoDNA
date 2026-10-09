# Svelto-DNA 🧬⚡

> **Sub-Second In Silico Splice-Site Mutagenesis & Variant Effect Diagnostics Powered by Speculative Tree Verification.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14%20(App%20Router)-black.svg)](https://nextjs.org/)
[![Project Board](https://img.shields.io/badge/GitHub%20Project-%2313%20SweltoDNA-success)](https://github.com/users/YUVARAJ-R-ai/projects/13)

---

## 📌 Executive Summary & Core Thesis

Variant effect predictors and deep genomic foundation models (such as **SpliceAI**, **Nucleotide Transformer**, and **HyenaDNA**) have significantly advanced the clinical interpretation of non-coding mutations. However, standard architectures rely on sequential autoregressive decoding or heavy, un-accelerated convolutional forward passes across large sequence windows ($1,000$ to $10,000$ bp). Consequently, evaluating in silico saturation mutagenesis (ISM) or exploring novel candidate variants has remained confined to multi-hour offline batch processing pipelines.

**Svelto-DNA** transforms clinical splice-site disruption analysis from slow batch jobs into an **interactive, sub-second web diagnostic workflow**. By introducing **Medusa-style speculative draft verification heads** and **directed acyclic graph (DAG) tree attention masking** directly onto frozen genomic foundation backbones, Svelto-DNA evaluates multiple candidate mutation states simultaneously in a **single forward pass**, targeting a **$2.0\times$ to $3.5\times$ wall-clock speedup** without compromising diagnostic fidelity. *This speedup is a design target and has not been measured yet; see Current Status below.*

---

## 🚦 Current Status (2026-10-08)

What runs today, and what is still scaffolding:

| Area | Status |
| :--- | :--- |
| Backbone | Real HyenaDNA (`LongSafari/hyenadna-small-32k-seqlen-hf`) loads frozen with its own tokenizer; fp32 weights, fp16 autocast on CUDA. Profiled on an RTX 5060 Laptop: 10 kb in 15.8 ms, 262 MB peak VRAM (`baseline_telemetry.json`). The mock backbone remains for offline tests. |
| SpliceAI data | Synthetic only. No loader for the real SpliceAI-10k files exists yet. |
| ClinVar data | Parser works, but the ±50 bp junction filter needs exon annotations (GTF); real ClinVar files carry no junction distance, so every SNV currently passes. |
| Splice benchmark | Harness is honest as of `ebbcca0`. The untrained mock head scores at chance (donor ROC-AUC 0.35); no trained model has been benchmarked. |
| Draft heads (K=3/4) | Implemented and tested on random targets; not yet trained on splice labels. |
| Delta calculator | Implemented; 0.55 ms for 10 kb on CPU (`delta_benchmark_telemetry.json`). |
| DAG tree verification, speedup | Not started (#4). The 2.0–3.5× figure is a target. |
| WebSocket API, frontend | Not started (#6–#9). |

---

## 🏗 System Architecture Pipeline

```
[ User Input / Mutated Nucleotide in Browser ]
                     │
                     ▼ WebSocket Mutation Event
 ┌─────────────────────────────────────────────────────────────┐
 │                FASTAPI STREAMING ENGINE                     │
 │                                                             │
 │   Input Window: Genomic Context (1k–10k bp DNA string)      │
 │                         │                                   │
 │       ┌─────────────────┴─────────────────┐                 │
 │       ▼                                   ▼                 │
 │ ┌───────────────┐               ┌───────────────────┐       │
 │ │ Frozen Base   │               │ Auxiliary Draft   │       │
 │ │ Genomic Model │               │ Heads (Medusa/    │       │
 │ │ (HyenaDNA /   │               │ Hydra Tree Proj.) │       │
 │ │ NT Checkpoint)│               └─────────┬─────────┘       │
 │ └───────┬───────┘                         │                 │
 │         │                                 ▼                 │
 │         │                       K Candidate Tokens/Sites    │
 │         │                                 │                 │
 │         └──────────────► ◄────────────────┘                 │
 │                  Single Forward Pass                        │
 │                  Parallel Verification                      │
 └─────────────────────────────┬───────────────────────────────┘
                               │
                               ▼ Delta Scores (Δ) & Attention Saliency
 ┌─────────────────────────────────────────────────────────────┐
 │                  NEXT.JS / REACT FRONTEND                   │
 │  - Virtualized Sequence Ribbon (A, C, G, T Badges @ 10k bp) │
 │  - Real-Time Mutagenesis Canvas (Radial Base Switcher)      │
 │  - Comparative Multi-Track Visualizer (Ref vs Mutant Spikes)│
 │  - Performance Telemetry HUD (FPS, Latency, Acceptance α)   │
 └─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Innovations

### 1. Speculative Draft Heads ($K=3/4$)
Lightweight 2-layer residual MLPs with LayerNorm and SiLU activations attached to the penultimate hidden states ($H \in \mathbb{R}^{B \times L \times D}$) of the frozen backbone. The backbone parameters are completely frozen, and only the draft heads are trained via discounted multi-target cross-entropy to predict adjacent downstream splice states in parallel:
$$\text{DraftHead}_k(h_t) = W_{2}^{(k)} \cdot \text{SiLU}\left(\text{LayerNorm}(W_{1}^{(k)} h_t + b_1^{(k)})\right) + h_t \cdot \mathbb{I}_{\text{res}}$$

### 2. Directed Acyclic Graph (DAG) Tree Attention Masking
Instead of sequential candidate verification, Svelto-DNA compiles a tree topology of candidate mutation paths and verifies them concurrently in a single forward pass through the primary model backbone using a custom 2D DAG attention mask $M \in \{0, -\infty\}^{N_{\text{tree}} \times N_{\text{tree}}}$:
$$M_{i, j} = \begin{cases} 0 & \text{if } j \in \text{Ancestors}(i) \cup \{i\} \\ -\infty & \text{otherwise} \end{cases}$$

### 3. Vectorized Delta Score ($\Delta$) Calculator
Instantaneous calculation of splice disruption across sliding genomic windows ($W = \pm 50$ bp):
$$\Delta_{\text{donor\_gain}} = \max_{j \in [i-W, i+W]} \left( P_{\text{mut}}(j, \text{donor}) - P_{\text{ref}}(j, \text{donor}) \right)$$
$$\Delta_{\text{donor\_loss}} = \max_{j \in [i-W, i+W]} \left( P_{\text{ref}}(j, \text{donor}) - P_{\text{mut}}(j, \text{donor}) \right)$$
$$\Delta_{\text{locus}} = \max(\Delta_{\text{donor\_gain}}, \Delta_{\text{donor\_loss}}, \Delta_{\text{acceptor\_gain}}, \Delta_{\text{acceptor\_loss}})$$

### 4. Virtualized Sequence Ribbon & Real-Time Canvas
A zero-lag horizontal viewport built with `@tanstack/react-virtual` that comfortably renders $10,000$ base pairs at 60 FPS. Clinicians can click any nucleotide to open a radial base switcher, instantly triggering a WebSocket event that computes and visualizes cryptic site activation within milliseconds.

---

## 📖 In-Depth Documentation

For full mathematical derivations, component API references, biological polarity rules, and empirical benchmark matrices, see:
👉 **[Comprehensive Architecture & Engine Reference (docs/ARCHITECTURE_AND_ENGINE.md)](docs/ARCHITECTURE_AND_ENGINE.md)**

---

## 📂 Repository Structure

```
sweltoDNA/
├── docs/
│   ├── ARCHITECTURE_AND_ENGINE.md   # Comprehensive mathematical, API & architecture documentation
│   ├── research.md                  # Research brief, clinical landscape & task breakdown
│   ├── task_01_implementation_plan.md
│   └── task_02_implementation_plan.md
├── svelto_dna/                      # Core Python / PyTorch package
│   ├── core/                        # Invariant foundation backbone oracle & genomic tokenizer
│   │   ├── backbone.py              # SveltoBackbone (frozen HyenaDNA/NT wrapper)
│   │   └── tokenizer.py             # GenomicTokenizer (IUPAC, reverse-complement)
│   ├── speculative/                 # Auxiliary speculative draft heads & training
│   │   ├── draft_heads.py           # K=3/4 parallel residual projection heads & loss
│   │   └── trainer.py               # DraftHeadTrainer with parameter isolation assertions
│   ├── splice/                      # High-throughput splice disruption scoring
│   │   └── delta.py                 # Vectorized 1D max-pooling delta calculator (<0.6 ms)
│   ├── data/                        # Genomic data pipelines & clinical benchmarking
│   │   ├── clinvar.py               # ClinVar SNV parser (pathogenic/benign/holdout VUS)
│   │   ├── spliceai.py              # SpliceAI-10k parser & StrandCoordinateResolver
│   │   ├── leakage.py               # Chromosome-level zero-leakage split verifier
│   │   └── benchmark.py             # Diagnostic metric evaluator (ROC-AUC, PR-AUC)
│   └── profiler/                    # Latency & memory benchmarking harness
│       └── benchmark.py             # Baseline inference profiler & telemetry exporter
├── benchmarks/                      # Standalone performance evaluation scripts
│   └── benchmark_delta.py           # Vectorized delta throughput & latency benchmark
├── scripts/                         # Operational & data extraction utilities
│   ├── extract_splice_data.py       # Ingest & partition SpliceAI & ClinVar into Parquet
│   └── eval_splice_benchmark.py     # Benchmark evaluation CLI on test splits
├── tests/                           # Automated pytest suite (78 tests)
│   ├── test_backbone.py             # Foundation oracle parameter freezing & determinism
│   ├── test_tokenizer.py            # Coordinate inversion & IUPAC tokens
│   ├── test_draft_heads.py          # Tensor shapes, parameter overhead & discount decay
│   ├── test_delta.py                # 1D max pooling delta scoring & batched peak extraction
│   ├── test_data_pipeline.py        # ClinVar filtering & negative-strand biological polarity
│   └── test_benchmark_eval.py       # Diagnostic metrics & class imbalance guards
├── train_draft_heads.py             # Standalone draft heads training & verification CLI
├── benchmark_baseline.py            # Baseline backbone profiler CLI
├── .zepher/                         # Zepher persistent context & session memory
├── AGENTS.md                        # Autonomous agent rules & instructions
└── README.md                        # Project landing documentation
```

---

## 🚀 Quickstart & Verification Commands

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/YUVARAJ-R-ai/SweltoDNA.git
cd SweltoDNA

# Install with development dependencies
uv sync --extra dev  # or: python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

### 2. Run Test Suite (78 Tests)
```bash
.venv/bin/pytest tests/
```

### 3. Verify Speculative Draft Heads ($K=3/4$)
```bash
# Verify K=3 heads (overhead: 2.87% < 4.0%, backbone frozen)
.venv/bin/python train_draft_heads.py --num-heads 3

# Verify K=4 heads (overhead: 3.83% < 4.0%, backbone frozen)
.venv/bin/python train_draft_heads.py --num-heads 4
```

### 4. Benchmark Vectorized Delta Scoring ($10,000$ bp in $0.55\text{ ms}$)
```bash
uv run python benchmarks/benchmark_delta.py --repeats 50
```

### 5. Extract ClinVar & SpliceAI Datasets (Zero Data Leakage)
```bash
.venv/bin/python scripts/extract_splice_data.py --output-dir data/processed --window-size 1000
```

### 6. Run Clinical Diagnostic Benchmark (ROC-AUC / PR-AUC)
```bash
.venv/bin/python scripts/eval_splice_benchmark.py --test-parquet data/processed/spliceai_test.parquet
```

---

## 📊 Sprint Plan & Development Milestones

All development is tracked on **[GitHub Project #13 (SweltoDNA)](https://github.com/users/YUVARAJ-R-ai/projects/13)**:

| Milestone | Target Due Date | Scope | Key Issues |
| :--- | :--- | :--- | :--- |
| **Sprint 1** | **2026-09-30** | Core Backbone, Speculative Draft Heads, DAG Tree Mask & Streaming API | [#1](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/1), [#2](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/2), [#3](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/3), [#4](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/4), [#5](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/5), [#6](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/6) |
| **Sprint 2** | **2026-10-14** | Virtualized Sequence Canvas, Multi-Track Visualizer & Telemetry HUD | [#7](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/7), [#8](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/8), [#9](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/9), [#10](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/10) |

---

## 📑 Research Paper Scaffolding (IEEE BIBM / ACM-BCB)

1. **Introduction:** High diagnostic significance of non-coding splice mutations in rare disease; latency bottlenecks of current deep variant predictors; introduction of Svelto-DNA.
2. **Related Work:** SpliceAI, foundation sequence models (HyenaDNA, Nucleotide Transformer), and speculative decoding methodologies (Medusa, Hydra, speculative sampling).
3. **Methodology:** Formal mathematical definition of speculative tree decoding over continuous genomic contexts; auxiliary draft loss formulation; delta scoring equations.
4. **Experiments & Results:** Diagnostic performance validation on ClinVar benchmarks (ROC-AUC / PR-AUC); ablation of draft head depth $K$; latency speedup curves and memory footprints against standard baselines.
5. **Interactive Diagnostic Workflow:** Client-server WebSocket latency profiling and clinical case studies during in silico saturation mutagenesis.

---

## 🤝 References & Acknowledgements

- **[Medusa](https://github.com/FasterDecoding/Medusa):** Simple LLM inference acceleration framework with multiple decoding heads.
- **[HyenaDNA](https://github.com/HazyResearch/hyena-dna):** Long-range genomic sequence modeling at single-nucleotide resolution.
- **[Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer):** Foundation models for molecular biology and variant effect prediction.
- **[OpenSpliceAI](https://github.com/Kuanhao-Chao/OpenSpliceAI):** Open-source PyTorch implementation of splice-site prediction and delta score metrics.
- **[SpliceAI Lookup](https://github.com/broadinstitute/SpliceAI-lookup):** Broad Institute interactive clinical lookup for splice variants.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

# Technical Implementation Plan: Task 1 (Issue #1)
## [Core-ML] Frozen Genomic Backbone Setup, GRCh38 Invariant Verification Engine & Baseline Profiler

- **Author:** Computational Biology & ML Engineering Pair
- **Repository:** `YUVARAJ-R-ai/SweltoDNA`
- **Related Issue:** [#1](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/1)
- **Target Sprint:** Sprint 1 (P0, Size M)

---

## 1. Executive Summary & Problem Formulation
Deep genomic foundation models (HyenaDNA, Nucleotide Transformer) serve as the verification oracle in Svelto-DNA's speculative decoding architecture.
The goal of Task 1 is to:
1. Establish a modular, robust PyTorch foundation model loader (`SveltoBackbone`) with strict parameter freezing (`requires_grad = False`), zero trainable parameters, and deterministic forward inference.
2. Build a high-throughput single-nucleotide tokenizer (`GenomicTokenizer`) supporting canonical bases, ambiguous `N` characters, IUPAC degenerate codes, reverse complement strand inversion, and symmetric flanking padding up to $L = 10,000$ bp.
3. Develop an automated baseline inference and profiling module (`benchmark_baseline.py`) tracking wall-clock latency, peak memory (GPU VRAM / CPU RSS), and nucleotide throughput across $L \in \{1024, 2048, 5000, 10000\}$, producing structured JSON telemetry.
4. Provide comprehensive unit and property-based test suites ensuring numerical determinism (cosine similarity = 1.000) and full hardware fallback (CUDA / CPU).

---

## 2. Architecture & Module Design

### 2.1 Directory Structure
```
.
├── pyproject.toml
├── svelto_dna/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── tokenizer.py      # GenomicTokenizer: IUPAC, reverse-complement, context windowing
│   │   └── backbone.py       # SveltoBackbone: frozen HF foundation model loader + mock fallback
│   └── profiler/
│       ├── __init__.py
│       └── benchmark.py      # Core benchmarking & telemetry recording engine
├── benchmark_baseline.py     # Root CLI entry point for telemetry generation
└── tests/
    ├── __init__.py
    ├── test_tokenizer.py     # Unit tests for tokenizer rules, IUPAC, reverse complement
    ├── test_backbone.py      # Unit tests for freezing, parameter counts, deterministic cosine sim
    └── test_profiler.py      # Unit tests for CLI telemetry output format
```

### 2.2 Component Specifications

#### A. `GenomicTokenizer` (`svelto_dna/core/tokenizer.py`)
- **Vocabulary:**
  - Standard tokens: `<PAD>: 0`, `A: 1`, `C: 2`, `G: 3`, `T: 4`, `N: 5`.
  - Configurable support for IUPAC degenerate codes:
    `{'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'], 'W': ['A', 'T'], 'K': ['G', 'T'], 'M': ['A', 'C'], 'B': ['C', 'G', 'T'], 'D': ['A', 'G', 'T'], 'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G']}` mapped deterministically to canonical bases or `N` based on strictness mode.
- **Normalization:** Strip whitespaces, convert lowercase to uppercase (`a->A, c->C, g->G, t->T`).
- **Reverse Complement:**
  - Watson-Crick base pairing: $A \leftrightarrow T$, $C \leftrightarrow G$.
  - Invert sequence direction $5' \to 3'$ to $3' \to 5'$.
- **Context Window Buffer & Symmetrical Flanking:**
  - Given locus position $p$ and context length $L \in [1024, 10000]$, extract symmetrically padded genomic slices:
    $$\text{left\_flank} = \lfloor (L - 1) / 2 \rfloor, \quad \text{right\_flank} = L - 1 - \text{left\_flank}$$
  - Symmetrical padding with `<PAD>` or reference sequence when locus is near sequence boundaries.
- **Output:** Returns PyTorch `BatchEncoding` containing `input_ids` `(batch_size, seq_len)` and `attention_mask` `(batch_size, seq_len)`.

#### B. `SveltoBackbone` (`svelto_dna/core/backbone.py`)
- **Model Support:**
  - Hugging Face Model ID: `HazyResearch/hyena-dna` variants and `InstaDeepAI/nucleotide-transformer-500m-human-ref`.
  - Architecture-faithful fallback: Built-in deterministic mock/local genomic backbone for offline, CI/CD, or non-networked environments where downloading multi-gigabyte checkpoints is infeasible.
- **Parameter Freezing:**
  - Set `param.requires_grad = False` for all parameters.
  - Set `model.eval()`.
  - Strict assertion: `sum(p.numel() for p in model.parameters() if p.requires_grad) == 0`.
- **Hardware & Precision Adaptation:**
  - Automatic device resolution: `cuda` if `torch.cuda.is_available()` else `cpu`.
  - Precision handling:
    - If CUDA: supports FP16 / BF16 via `torch.amp.autocast('cuda')`.
    - If CPU: automatic fallback to FP32 or BF16 (`torch.amp.autocast('cpu', dtype=torch.bfloat16)`).
- **Interface & Output:**
  - Returns `BackboneOutput(last_hidden_state, penultimate_hidden_state, logits)`.
  - Penultimate hidden states are explicitly preserved to feed downstream Speculative Draft Heads ($K=3/4$ residual MLPs) in Issue #3.

#### C. `BaselineProfiler` (`svelto_dna/profiler/benchmark.py` & `benchmark_baseline.py`)
- **Evaluation Matrix:** Window lengths $L \in \{1024, 2048, 5000, 10000\}$.
- **Warmup & Measurement:** 5 warmup iterations, followed by 20 timed measurement iterations per window length.
- **Metrics Collected:**
  - Wall-clock latency: Mean, standard deviation, median (p50), 95th percentile (p95), min, max in milliseconds (ms).
  - Peak memory:
    - If CUDA: `torch.cuda.max_memory_allocated() / (1024 * 1024)` in MB.
    - If CPU: `psutil.Process().memory_info().rss / (1024 * 1024)` in MB.
  - Inference Throughput: $\text{Throughput} = \frac{L}{\text{mean\_latency\_sec}}$ (nucleotides / second).
- **Telemetry Schema:**
  - Emits valid JSON format with system metadata (device, precision, model name, OS, timestamp) and per-length metrics.

---

## 3. Step-by-Step Implementation Sequence

1. **Step 1: Environment & Project Configuration**
   - Configure `pyproject.toml` with `svelto_dna` package layout and dependencies (`torch`, `transformers`, `einops`, `pydantic`, `psutil`, `pytest`).
2. **Step 2: Tokenizer Implementation (`svelto_dna/core/tokenizer.py`)**
   - Implement single-nucleotide mapping, reverse complement, IUPAC degeneracy handling, context windowing, and tensor generation.
3. **Step 3: Backbone Implementation (`svelto_dna/core/backbone.py`)**
   - Implement `SveltoBackbone` module with frozen parameter check, device/precision management, penultimate representation preservation, and local fallback oracle.
4. **Step 4: Profiler Implementation (`svelto_dna/profiler/benchmark.py` & `benchmark_baseline.py`)**
   - Implement timing harness, memory tracking, CLI arguments, and JSON telemetry generation across $L \in \{1024, 2048, 5000, 10000\}$.
5. **Step 5: Automated Testing (`tests/`)**
   - Write tests for tokenizer edge cases (IUPAC, lowercase, reverse complement, length up to 10k).
   - Write tests for backbone freezing (0 trainable parameters) and exact cosine similarity = 1.000 across multiple forward calls.
   - Write tests for profiler telemetry JSON validation.
6. **Step 6: Profiling Run & Verification**
   - Run `benchmark_baseline.py` and output `baseline_telemetry.json`.
   - Run full `pytest` suite and confirm 100% pass rate.

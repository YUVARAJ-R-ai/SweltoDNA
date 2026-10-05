# Implementation Plan Review: Task 1 (Issue #1)
## [Core-ML] Frozen Genomic Backbone Setup, GRCh38 Invariant Verification Engine & Baseline Profiler

## Executive Summary
The implementation plan for Task 1 establishes a robust, modular architecture for loading frozen genomic foundation models (`SveltoBackbone`), single-nucleotide tokenization (`GenomicTokenizer`), and baseline telemetry profiling (`benchmark_baseline.py`). The top risks are network dependency on multi-gigabyte Hugging Face checkpoints in restricted environments, tokenization vocabulary mismatches across foundation architectures (e.g., character-level HyenaDNA vs. 6-mer Nucleotide Transformer), and hardware divergence (CUDA vs. CPU). With the mitigations and architectural refinements detailed below, the plan is ready for immediate execution.

**Readiness:** 🟢 Ready with Minor Fixes

---

## Outcome Analysis

| Outcome | Trigger Condition | Likelihood | Impact |
| :--- | :--- | :--- | :--- |
| **Full Success** | All modules pass unit tests; parameters frozen; deterministic cosine similarity = 1.000; JSON telemetry generated for $L \in \{1024, 2048, 5000, 10000\}$. | High | Clean ground-truth oracle established for downstream speculative draft heads. |
| **Partial Success — Network Timeout** | Hugging Face download fails or is rate-limited when pulling remote weights. | Medium | Blocks tests and execution unless a local mock/fallback backbone architecture is provided. |
| **Partial Success — Device Failure** | Profiler or forward pass hardcodes `torch.cuda` calls on a CPU-only environment. | High | Fatal crash in `autocast()` or `max_memory_allocated()`. |
| **Silent Failure — Stochastic Drift** | Backbone is not explicitly set to `.eval()` or stochastic layers (dropout) activate during verification. | Low | Cosine similarity deviates from 1.000 intermittently, compromising verification integrity. |
| **Silent Failure — IUPAC Information Loss** | Degenerate IUPAC bases (e.g., `R`, `Y`, `S`) stripped or mapped unpredictably, corrupting downstream splice motif recognition. | Medium | Invariant verification produces inaccurate donor/acceptor delta scores. |

---

## Gap Analysis

| Gap | Category | Severity | Recommendation |
| :--- | :--- | :--- | :--- |
| **Tokenizer Vocabulary Mismatch** | Architecture / ML | **High** | Nucleotide Transformer uses 6-mer k-mer tokenization, whereas HyenaDNA and Svelto-DNA require single-nucleotide resolution. Implement an explicit Tokenizer Adapter within `SveltoBackbone` to bridge single-nucleotide inputs with underlying model token IDs. |
| **Offline / Network Dependency** | Dependency | **High** | Hugging Face checkpoints are 500MB–2GB. Implement a built-in `MockGenomicBackbone` that implements the exact forward contract (`BackboneOutput`) and deterministic weights so tests and offline runs execute without network latency or failures. |
| **Hardware Agnostic Profiling** | Observability | **High** | Ensure `benchmark_baseline.py` detects CUDA availability dynamically, using `torch.cuda.max_memory_allocated()` on GPU and `psutil.Process().memory_info().rss` on CPU, documenting memory metric type in telemetry JSON. |
| **Edge-Case Window Slicing** | Testing / Data | **Medium** | Genomic loci near chromosome ends or sequence boundaries need deterministic padding (left flank vs right flank) and validation against empty or out-of-bounds loci. |
| **Penultimate State Retention** | Architecture / Integration | **Medium** | Downstream Issue #3 requires penultimate layer representations for draft heads. Ensure `SveltoBackbone` exposes `penultimate_hidden_state` alongside `last_hidden_state`. |

---

## Risk & Assumption Register

### Risk Register
| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **HF Hub Download Failure** | Medium | High | Support `model_name="mock"` or `--mock` mode with architecture-faithful synthetic backbone for offline development and continuous verification. |
| **CPU Out-of-Memory at $L=10,000$** | Low | High | Enforce default evaluation batch size of 1 for 10k window lengths, and invoke Python garbage collection between benchmark iterations. |
| **Dropout / Non-Determinism** | Low | Critical | Call `model.eval()`, explicitly disable dropout modules if present, and execute inference within `torch.no_grad()`. |

### Assumption Register
| Assumption | Validated? | If Wrong… |
| :--- | :--- | :--- |
| Single-nucleotide resolution is sufficient for verification oracle | Yes (per Issue #1 & research doc) | Downstream speculative decoding requires base-level resolution. |
| Local environment supports PyTorch on CPU | Yes (verified: PyTorch 2.14.1 installed in `.venv`) | CPU execution works cleanly. |
| Telemetry schema can be consumed by Task #9 HUD | Yes (JSON dictionary with latency, memory, throughput) | Standardized schema avoids future breaking changes. |

---

## Dependency & Integration Gaps

| Dependency | Issue | Severity | Recommendation |
| :--- | :--- | :--- | :--- |
| **Task #3 (Speculative Draft Heads)** | Requires penultimate layer embeddings from backbone | High | Output explicit `BackboneOutput(last_hidden_state, penultimate_hidden_state, logits)`. |
| **Task #4 (DAG Tree Mask Engine)** | Requires sequence length dimension alignment | Medium | Maintain standard `(batch_size, seq_len, hidden_dim)` tensor contracts across all window lengths. |
| **`psutil` / `torch`** | Must be installed in active environment | Low | Verified installed in `.venv`. |

---

## Strengths
- **Clean Separation of Concerns:** Tokenizer, backbone model, and benchmark profiler are decoupled into dedicated submodules.
- **Strict Freezing Guarantees:** Assertion-backed parameter freezing ensures zero parameter updates and 100% oracle invariance.
- **Comprehensive Profiling Matrix:** Systematically benchmarks all four target sequence lengths ($1\text{k}, 2\text{k}, 5\text{k}, 10\text{k}$) with rich statistical summaries.
- **Full Coverage of Genomic Edge Cases:** Explicit support for lowercase normalization, IUPAC ambiguity codes, and reverse complement strand inversion.

---

## Prioritised Recommendations
1. **[High] Implement Tokenizer Adapter & Mock Backbone:** Add a self-contained, lightweight genomic backbone mode to ensure offline execution, rapid unit testing, and guaranteed reproducibility without network bottlenecks.
2. **[High] Dynamic Hardware & Memory Detection:** Automatically switch between CUDA VRAM metrics and CPU RSS metrics, ensuring `benchmark_baseline.py` runs on any workstation or server.
3. **[Medium] Expose Penultimate Representations:** Include `penultimate_hidden_state` in the return structure to fulfill interface contracts for upcoming Sprint 1 tasks (Issue #3).
4. **[Medium] Add Boundary Padding Unit Tests:** Validate locus window extraction with asymmetric margins, sequence boundary overruns, and degenerate IUPAC codes.

---

## Reviewer Verdict
**Verdict:** 🟢 Ready with Minor Fixes.
The plan is approved for immediate implementation incorporating the recommendations above.

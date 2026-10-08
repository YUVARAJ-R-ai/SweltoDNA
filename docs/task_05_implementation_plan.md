# Technical Implementation Plan: Task 5 (Issue #5)
## [Backend-Core] Vectorized Splice Disruption & Delta Score (Δ) Calculator

- **Author:** Computational Biology & ML Engineering Pair
- **Repository:** `YUVARAJ-R-ai/SweltoDNA`
- **Related Issue:** [#5](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/5)
- **Target Sprint:** Sprint 1 (P0, Size S)
- **Status:** Planning & Ready

---

## 1. Executive Summary & Problem Formulation
Clinical interpretation of non-coding single-nucleotide variants (SNVs) and indels relies critically on detecting alterations in pre-mRNA splicing mechanisms:
1. **Donor Loss:** Disruption of canonical 5' splice donor motifs (`GT`).
2. **Donor Gain:** Activation of novel/cryptic donor splice sites.
3. **Acceptor Loss:** Disruption of canonical 3' splice acceptor motifs (`AG`).
4. **Acceptor Gain:** Activation of novel/cryptic acceptor splice sites.

The delta score metric ($\Delta$) quantifies the maximum probability change induced by a mutation within a sliding genomic window ($W = \pm 50$ bp) around the variant locus. In high-throughput clinical diagnostics and saturation mutagenesis across long sequences ($L = 10,000$ bp), looping sequentially over nucleotides induces severe latency bottlenecks.

The objective of Task 5 is to:
1. Build a high-performance, fully vectorized PyTorch and NumPy computation engine (`compute_delta_scores`) calculating donor gain/loss, acceptor gain/loss, and peak locus impact in sub-millisecond time.
2. Utilize 1D max pooling (`torch.nn.functional.max_pool1d`) with symmetrical padding and sliding window indexing to process $L = 10,000$ bp in $< 2$ ms (target: $< 5$ ms on CPU, $< 1$ ms on GPU).
3. Implement canonical splice site masking to suppress false positive cryptic alerts at existing known junctions.
4. Robustly handle boundary edge cases ($i < W$, $i > L - 1 - W$, short sequences, batch dimensions).
5. Provide comprehensive unit tests validating known ClinVar pathogenic splice variant mutations within $10^{-4}$ tolerance.

---

## 2. Mathematical Formulation

Let $L$ be the sequence context length.
Given reference splice probabilities $P_{\text{ref}} \in [0, 1]^{L \times 3}$ and mutant splice probabilities $P_{\text{mut}} \in [0, 1]^{L \times 3}$, where:
- Class index $0$: Non-splice / Background
- Class index $1$: Splice Donor ($5'$ junction)
- Class index $2$: Splice Acceptor ($3'$ junction)

For any position $j \in [0, L-1]$:
$$\text{Diff}_{\text{DG}}(j) = \max(0, P_{\text{mut}}(j, 1) - P_{\text{ref}}(j, 1))$$
$$\text{Diff}_{\text{DL}}(j) = \max(0, P_{\text{ref}}(j, 1) - P_{\text{mut}}(j, 1))$$
$$\text{Diff}_{\text{AG}}(j) = \max(0, P_{\text{mut}}(j, 2) - P_{\text{ref}}(j, 2))$$
$$\text{Diff}_{\text{AL}}(j) = \max(0, P_{\text{ref}}(j, 2) - P_{\text{mut}}(j, 2))$$

For a variant at coordinate $i$ with window radius $W$ (default $W = 50$ bp):
$$\Delta_{\text{donor\_gain}}(i) = \max_{j \in [\max(0, i-W), \min(L-1, i+W)]} \text{Diff}_{\text{DG}}(j)$$
$$\Delta_{\text{donor\_loss}}(i) = \max_{j \in [\max(0, i-W), \min(L-1, i+W)]} \text{Diff}_{\text{DL}}(j)$$
$$\Delta_{\text{acceptor\_gain}}(i) = \max_{j \in [\max(0, i-W), \min(L-1, i+W)]} \text{Diff}_{\text{AG}}(j)$$
$$\Delta_{\text{acceptor\_loss}}(i) = \max_{j \in [\max(0, i-W), \min(L-1, i+W)]} \text{Diff}_{\text{AL}}(j)$$
$$\Delta_{\text{locus}}(i) = \max(\Delta_{\text{donor\_gain}}(i), \Delta_{\text{donor\_loss}}(i), \Delta_{\text{acceptor\_gain}}(i), \Delta_{\text{acceptor\_loss}}(i))$$

Across the entire sequence length $L$, the windowed maximum map is computed globally via 1D max pooling:
$$\text{Kernel Size} = 2W + 1, \quad \text{Stride} = 1, \quad \text{Padding} = W$$

---

## 3. Architecture & Module Design

### 3.1 Directory Structure
```
svelto_dna/
├── core/
│   ├── __init__.py
│   ├── tokenizer.py
│   └── backbone.py
├── splice/
│   ├── __init__.py
│   └── delta.py              # compute_delta_scores, DeltaResult, DeltaCalculator
├── profiler/
│   ├── __init__.py
│   └── benchmark.py
benchmarks/
└── benchmark_delta.py        # Dedicated benchmark runner for delta scoring
tests/
├── test_tokenizer.py
├── test_backbone.py
├── test_profiler.py
└── test_delta.py             # Unit tests: ClinVar ground-truth, boundary conditions, masking
```

### 3.2 Component Specifications

#### A. `DeltaResult` (Pydantic / Dataclass Model)
Structured return type containing:
- `donor_gain`: `np.ndarray` or `torch.Tensor`
- `donor_loss`: `np.ndarray` or `torch.Tensor`
- `acceptor_gain`: `np.ndarray` or `torch.Tensor`
- `acceptor_loss`: `np.ndarray` or `torch.Tensor`
- `locus_delta`: `np.ndarray` or `torch.Tensor`
- `peak_delta`: `float`
- `peak_component`: `Literal["donor_gain", "donor_loss", "acceptor_gain", "acceptor_loss"]`
- `peak_position`: `int`
- `to_dict()`: method returning clean dictionary matching the acceptance criteria

#### B. `compute_delta_scores` (`svelto_dna/splice/delta.py`)
- **Inputs:**
  - `p_ref`: Tensor or ndarray of shape `(L, 3)` or `(B, L, 3)`.
  - `p_mut`: Tensor or ndarray of shape `(L, 3)` or `(B, L, 3)`.
  - `window_size: int = 50`: Flanking window radius $W$.
  - `variant_pos: Optional[int] = None`: Specific variant coordinate to extract locus-specific metrics. If None, returns full sequence delta maps.
  - `canonical_mask: Optional[Union[np.ndarray, torch.Tensor]] = None`: Boolean mask of canonical splice junctions to suppress.
  - `clamp_non_negative: bool = True`: Enforce $\Delta \ge 0$ per SpliceAI clinical convention.
- **Optimized Kernel:**
  - Converts inputs to torch tensor on CPU/CUDA if ndarray.
  - Computes 4-channel difference tensor `(B, 4, L)`.
  - Applies canonical mask if provided.
  - Executes `F.max_pool1d(diffs, kernel_size=2*W+1, stride=1, padding=W)`.
  - Computes peak delta, component argmax, and position coordinates.

#### C. Boundary Handling
- When $i < W$, window starts at index 0.
- When $i + W \ge L$, window terminates at index $L-1$.
- `F.max_pool1d` with zero padding matches non-negative clamped differences, avoiding edge distortion.

---

## 4. Step-by-Step Implementation Sequence

1. **Step 1: Module Creation (`svelto_dna/splice/delta.py`)**
   - Implement `compute_delta_scores`, `DeltaResult`, and tensor/array helper conversions.
   - Support both single-sequence `(L, 3)` and batched `(B, L, 3)` inputs.
   - Implement canonical splice site masking.
2. **Step 2: Package Exports (`svelto_dna/splice/__init__.py` & `svelto_dna/__init__.py`)**
   - Re-export `compute_delta_scores` and `DeltaResult`.
3. **Step 3: Unit Testing Suite (`tests/test_delta.py`)**
   - ClinVar ground-truth validation:
     - Canonical donor loss: `+1G>A` splice site destruction ($\Delta_{\text{DL}} \approx 0.93$).
     - Cryptic donor activation: deep intronic SNV ($\Delta_{\text{DG}} \approx 0.87$).
     - Canonical acceptor loss: `-1G>C` disruption ($\Delta_{\text{AL}} \approx 0.90$).
     - Cryptic acceptor activation: ($\Delta_{\text{AG}} \approx 0.85$).
   - Boundary tests: variant at index 0, variant at index $L-1$, window larger than sequence.
   - Invariant / identity test: $P_{\text{ref}} == P_{\text{mut}} \implies \text{all deltas} == 0.0$.
   - Canonical masking test: ensure masked canonical site does not register as cryptic gain.
4. **Step 4: Benchmarking Suite (`benchmarks/benchmark_delta.py`)**
   - Run 100 iterations on $L = 10,000$ bp on CPU (and GPU if available).
   - Assert mean execution time $< 5.0$ ms on CPU and $< 1.0$ ms on GPU.
5. **Step 5: Full Test Suite Verification**
   - Run `uv run --extra dev pytest` and ensure 100% pass rate.

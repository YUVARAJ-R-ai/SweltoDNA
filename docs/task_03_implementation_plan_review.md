# Implementation Plan Review: Task 3 (Issue #3)
## [Speculative-ML] Auxiliary Speculative Draft Heads (K=3/4) & Residual Projection Architecture

- **Reviewer:** Implementation Plan Reviewer & System Architect
- **Plan Under Review:** `docs/task_03_implementation_plan.md`
- **Date:** 2026-10-05
- **Verdict:** 🟢 Ready (with structured enhancements)

---

## 1. Outcome Analysis

| Outcome | Trigger Condition | Likelihood | Impact |
|---------|-------------------|------------|--------|
| **Full Success** | All acceptance criteria satisfied: $K \in \{3, 4\}$ parallel residual heads instantiate, parameter overhead $< 4\%$, shape $(K, B, L, C)$ conformant, discounted loss verified, backbone strictly frozen, and standalone training updates heads. | High | High (Enables speculative drafting and downstream tree decoding) |
| **Partial Failure — Overhead Violation** | Proj dimension $D_{\text{proj}}$ is chosen too wide (e.g. $D_{\text{proj}} \ge 256$ with $K=4$), exceeding 4% parameter threshold against the 3.62M mock backbone. | Low | Medium (Resolved by defaulting $D_{\text{proj}} = \min(D, 128)$ or providing configurable dimension) |
| **Partial Failure — Gradient Bleed** | Backbone parameters accidentally get gradient updates if `p.requires_grad` is not verified before optimizer creation. | Very Low | Critical (Mitigated by assertion in `SveltoBackbone.freeze()` and explicit assertion in `train_draft_heads.py`) |
| **Silent Failure — Index Offsets in Loss** | In `SpeculativeDraftLoss`, head $k$ slices target with off-by-one or mismatched lengths, training draft heads on misaligned positions. | Low | High (Mitigated by explicit tensor slice boundary assertions and unit tests comparing synthetic manual predictions) |

---

## 2. Gap Analysis

| Gap | Category | Severity | Mitigation / Resolution |
|-----|----------|----------|-------------------------|
| **Configurable Projection Dim ($D_{\text{proj}}$)** | Scalability | Medium | Allow `proj_dim` to be passed or default intelligently to 128 to stay well within the $< 4\%$ parameter budget for both mock and full-scale backbones. |
| **Sequence Boundary Handling in Loss** | Edge Cases | Medium | When predicting $k$ positions ahead near sequence end ($t > L - k$), truncate prediction and target tensors symmetrically to length $L - k$, handling padding masks if present. |
| **Residual Connection Flexibility** | Architecture | Low | Support both output projection residual and hidden space residual connection via a clean `residual_mode` flag, defaulting to direct output projection matching the issue specification: $\text{DraftHead}_k(h_t) = W_2^{(k)} \text{SiLU}(\text{LN}(W_1 h + b)) + h \mathbb{I}_{\text{res}}$. |
| **Device & Precision Support** | Testing / Environment | Low | Ensure draft heads inherit device and dtype from backbone outputs seamlessly (`bfloat16`, `float16`, `float32`). |

---

## 3. Domain Checklist: ML Pipeline & Modeling

- [x] **Model Architecture & Dimensionality:** Input $(B, L, D)$, intermediate $D \to D_{\text{proj}}$, output $(K, B, L, C)$ where $C \in \{3, 4\}$.
- [x] **Parameter Budget:** Auxiliary draft heads add $< 4\%$ overhead relative to base backbone.
- [x] **Gradient Isolation:** Backbone frozen with `param.requires_grad = False`; draft heads trained with independent optimizer (`AdamW`).
- [x] **Objective Formulation:** Multi-target cross-entropy with geometric discount factor $\gamma = 0.85$ ($\lambda_k = \gamma^{k-1}$).
- [x] **Residual Skip Connection:** LayerNorm and SiLU non-linearity with residual path to prevent vanishing gradients.
- [x] **Verification Script:** Standalone training script verifying loss convergence and zero backbone gradients.

---

## 4. Verdict & Recommendations
**Verdict:** 🟢 **Ready for Implementation**
The implementation plan is sound, adheres to the issue specifications, and covers all acceptance criteria. Proceed with Step 1.

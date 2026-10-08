# Technical Implementation Plan: Task 3 (Issue #3)
## [Speculative-ML] Auxiliary Speculative Draft Heads (K=3/4) & Residual Projection Architecture

- **Author:** Computational Biology & ML Engineering Pair
- **Repository:** `YUVARAJ-R-ai/SweltoDNA`
- **Related Issue:** [#3](https://github.com/YUVARAJ-R-ai/SweltoDNA/issues/3)
- **Target Sprint:** Sprint 1 (P0, Size M)

---

## 1. Executive Summary & Problem Formulation
In high-throughput genomic variant effect prediction and in silico saturation mutagenesis (ISM), vanilla foundation backbones require sequential forward passes for every nucleotide shift or candidate substitution.
To break this sequential dependency and achieve 2.0x–3.5x wall-clock inference acceleration, Svelto-DNA introduces a Medusa-style speculative decoding paradigm tailored for genomic sequences.

The objective of Task 3 is to:
1. Implement independent residual MLP draft heads (`DraftHead`) attaching to the penultimate hidden representations ($H \in \mathbb{R}^{B \times L \times D}$) of the frozen genomic foundation backbone.
2. Group $K \in \{3, 4\}$ parallel draft heads into a unified container module (`SpeculativeDraftHeads`) emitting simultaneous candidate predictions $(K, B, L, C)$, where $C=3$ for splice classifications (`[None, Donor, Acceptor]`) or $C=4$ for nucleotide tokens (`[A, C, G, T]`).
3. Enforce a strict parameter overhead budget: the auxiliary heads must comprise $< 4\%$ total parameters relative to the base backbone.
4. Implement a multi-target discounted cross-entropy loss module (`SpeculativeDraftLoss`) applying discount weighting $\lambda_k = \gamma^{k-1}$ ($\gamma = 0.85$) across future prediction offsets $t+k$.
5. Provide a standalone training and verification script (`train_draft_heads.py`) that strictly guarantees the base backbone remains completely frozen (`p.grad is None`) while draft head weights update and converge.
6. Provide comprehensive unit tests covering tensor shapes, gradient propagation, parameter overhead, and device compatibility.

---

## 2. Mathematical Formulation & Architecture

### 2.1 Single Draft Head Architecture ($k \in \{1, \dots, K\}$)
For token position $t$ with penultimate hidden state $h_t \in \mathbb{R}^D$:
$$\text{DraftHead}_k(h_t) = W_2^{(k)} \cdot \text{SiLU}\left(\text{LayerNorm}(W_1^{(k)} h_t + b_1^{(k)})\right) + h_t \cdot \mathbb{I}_{\text{res}}$$

Where:
- $W_1^{(k)} \in \mathbb{R}^{D_{\text{proj}} \times D}, b_1^{(k)} \in \mathbb{R}^{D_{\text{proj}}}$
- $\text{LayerNorm}: \mathbb{R}^{D_{\text{proj}}} \to \mathbb{R}^{D_{\text{proj}}}$
- $\text{SiLU}(z) = z \cdot \sigma(z)$
- $W_2^{(k)} \in \mathbb{R}^{C \times D_{\text{proj}}}, b_2^{(k)} \in \mathbb{R}^C$
- $\mathbb{I}_{\text{res}} \in \mathbb{R}^{D \times C}$ is a residual skip projection (or residual feature connection), ensuring stable gradient propagation directly from the target loss to the penultimate representation space without vanishing gradients.

### 2.2 Speculative Draft Heads Module (`SpeculativeDraftHeads`)
- Takes penultimate representations $H \in \mathbb{R}^{B \times L \times D}$ from `SveltoBackbone`.
- Evaluates $K$ heads in parallel:
  $$\hat{Y}^{(k)} = \text{DraftHead}_k(H) \in \mathbb{R}^{B \times L \times C}$$
- Stacks outputs along dimension 0:
  $$\hat{\mathbf{Y}} \in \mathbb{R}^{K \times B \times L \times C}$$
- Provides method `parameter_overhead_ratio(backbone_total_params: int) -> float` asserting $< 0.04$.

### 2.3 Discounted Multi-Target Cross-Entropy Loss (`SpeculativeDraftLoss`)
$$\mathcal{L}_{\text{draft}} = \sum_{k=1}^{K} \lambda_k \cdot \mathcal{L}_{\text{CE}}\left(\hat{y}_{t+k}^{(k)}, y_{t+k}\right)$$
$$\lambda_k = \gamma^{k-1}, \quad \gamma = 0.85$$

For head $k$ (predicting $k$ positions ahead):
- Predictions slice: $\hat{Y}^{(k)}[:, :L-k, :] \in \mathbb{R}^{B \times (L-k) \times C}$
- Ground truth target slice: $Y[:, k:] \in \mathbb{R}^{B \times (L-k)}$
- Mask handling: If an `attention_mask` is provided, valid tokens satisfy $\text{mask}[:, :L-k] \land \text{mask}[:, k:]$.

---

## 3. Directory Layout & Module Plan

```
.
├── svelto_dna/
│   ├── speculative/
│   │   ├── __init__.py
│   │   ├── draft_heads.py        # DraftHead, SpeculativeDraftHeads, SpeculativeDraftLoss
│   │   └── trainer.py            # Head training loop & validation harness
├── train_draft_heads.py          # Standalone training & verification script
└── tests/
    └── test_draft_heads.py       # Comprehensive unit tests for shapes, overhead, gradients
```

---

## 4. Acceptance Criteria & Test Strategy

| Acceptance Criterion | Verification Method |
|----------------------|---------------------|
| PyTorch module `SpeculativeDraftHeads` instantiates $K \in \{3, 4\}$ parallel residual MLP projection heads attached to penultimate representations | Unit test `test_instantiation_k3_k4` checking module structure and layer types |
| Auxiliary draft heads add less than 4% parameter overhead relative to base genomic model | Unit test `test_parameter_overhead_budget` comparing against `SveltoBackbone` (3.62M params) |
| Standalone training script verifies base model parameters remain completely frozen (`grad == None`) while draft head weights update successfully | Integration test `test_standalone_training_freezing` running `train_draft_heads.py` |
| Unit tests verify forward pass tensor shape conformity: input $(B, L, D) \to \text{output } (K, B, L, C)$ | Unit test `test_forward_shape_conformity` parameterized over multiple $B, L, D, K, C$ |
| Residual connection ensures gradient flow | Unit test `test_residual_gradient_flow` verifying gradient presence in both MLP and skip paths |
| Discounted loss follows $\lambda_k = 0.85^{k-1}$ | Unit test `test_discounted_loss_computation` verifying exact numeric match against analytical weights |

---

## 5. Execution Steps
1. Create `svelto_dna/speculative/` package with `__init__.py`.
2. Implement `DraftHead`, `SpeculativeDraftHeads`, and `SpeculativeDraftLoss` in `svelto_dna/speculative/draft_heads.py`.
3. Implement `train_draft_heads.py` and `svelto_dna/speculative/trainer.py`.
4. Implement `tests/test_draft_heads.py`.
5. Run tests with pytest in `.venv`.
6. Run `train_draft_heads.py` CLI and verify parameter freeze and convergence.
7. Update Zepher memory and handoff.

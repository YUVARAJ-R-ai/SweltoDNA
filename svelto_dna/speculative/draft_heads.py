"""
Auxiliary Speculative Draft Heads & Residual Projection Architecture for Svelto-DNA.
Implements K-parallel residual MLP draft heads attached to the penultimate hidden states
of frozen genomic foundation backbones with discounted multi-target cross-entropy loss.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DraftHead(nn.Module):
    """
    Individual residual MLP draft head for speculative genomic token/splice prediction.

    Architecture for head k at position t:
        DraftHead_k(h_t) = W_2^{(k)} * SiLU(LayerNorm(W_1^{(k)} h_t + b_1^{(k)})) + h_t * I_res

    Projects representation:
        Hidden dim D -> Proj dim D_proj -> Class dim C
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        proj_dim: Optional[int] = None,
        num_classes: int = 4,
        use_residual: bool = True,
        residual_mode: str = "output",
        head_index: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim if proj_dim is not None else min(hidden_dim, 128)
        self.num_classes = num_classes
        self.use_residual = use_residual
        self.residual_mode = residual_mode.lower()
        self.head_index = head_index

        if self.residual_mode not in ("output", "hidden"):
            raise ValueError(
                f"Unsupported residual_mode '{residual_mode}'. Must be 'output' or 'hidden'."
            )

        # First projection: W_1: D -> D_proj
        self.fc1 = nn.Linear(self.hidden_dim, self.proj_dim)
        # Normalization and activation
        self.norm = nn.LayerNorm(self.proj_dim)
        self.act = nn.SiLU()

        # Second projection: W_2: D_proj -> C
        self.fc2 = nn.Linear(self.proj_dim, self.num_classes)

        # Residual skip connection I_res
        if self.use_residual:
            if self.residual_mode == "output":
                # Linear skip projection directly into class space C
                self.res_proj = nn.Linear(self.hidden_dim, self.num_classes, bias=False)
            else:
                # Hidden residual projection into D_proj
                if self.hidden_dim != self.proj_dim:
                    self.res_proj = nn.Linear(self.hidden_dim, self.proj_dim, bias=False)
                else:
                    self.res_proj = nn.Identity()
        else:
            self.res_proj = None

    @property
    def parameter_count(self) -> int:
        """Total parameter count in this draft head."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single draft head.

        Args:
            h: Penultimate hidden representation of shape (..., D).

        Returns:
            Output logits of shape (..., C).
        """
        if self.residual_mode == "output":
            # DraftHead_k(h_t) = W_2 * SiLU(LayerNorm(W_1 * h_t + b_1)) + h_t * I_res
            mlp_feature = self.act(self.norm(self.fc1(h)))
            logits = self.fc2(mlp_feature)
            if self.use_residual and self.res_proj is not None:
                logits = logits + self.res_proj(h)
            return logits
        else:
            # Hidden residual mode: W_2 * (SiLU(LayerNorm(W_1 * h_t + b_1)) + h_t * I_res)
            mlp_feature = self.act(self.norm(self.fc1(h)))
            if self.use_residual and self.res_proj is not None:
                mlp_feature = mlp_feature + self.res_proj(h)
            return self.fc2(mlp_feature)


class SpeculativeDraftHeads(nn.Module):
    """
    Multi-head speculative draft module attaching K parallel residual projection heads
    onto the penultimate hidden states (H in R^{B x L x D}) of the frozen genomic backbone.

    Emits candidate predictions in parallel:
        Output tensor shape: (K, B, L, C)
    """

    def __init__(
        self,
        num_heads: int = 3,
        hidden_dim: int = 256,
        proj_dim: Optional[int] = None,
        num_classes: int = 4,
        use_residual: bool = True,
        residual_mode: str = "output",
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be at least 1, got {num_heads}")

        self._num_heads = num_heads
        self._hidden_dim = hidden_dim
        self._proj_dim = proj_dim if proj_dim is not None else min(hidden_dim, 128)
        self._num_classes = num_classes
        self._use_residual = use_residual
        self._residual_mode = residual_mode

        # Initialize K independent projection heads
        self.heads = nn.ModuleList([
            DraftHead(
                hidden_dim=self._hidden_dim,
                proj_dim=self._proj_dim,
                num_classes=self._num_classes,
                use_residual=self._use_residual,
                residual_mode=self._residual_mode,
                head_index=k + 1,
            )
            for k in range(self._num_heads)
        ])

    @property
    def num_heads(self) -> int:
        """Number of parallel speculative draft heads (K)."""
        return self._num_heads

    @property
    def hidden_dim(self) -> int:
        """Penultimate hidden state feature dimension (D)."""
        return self._hidden_dim

    @property
    def proj_dim(self) -> int:
        """Intermediate projection dimension (D_proj)."""
        return self._proj_dim

    @property
    def num_classes(self) -> int:
        """Target prediction classes (C, e.g., 3 for splice, 4 for nucleotides)."""
        return self._num_classes

    @property
    def total_parameters_count(self) -> int:
        """Total trainable parameter count across all auxiliary heads."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_overhead_ratio(self, backbone_total_params: int) -> float:
        """
        Computes parameter overhead relative to base foundation backbone.
        Must remain strictly < 0.04 (4%).
        """
        if backbone_total_params <= 0:
            raise ValueError("backbone_total_params must be greater than 0.")
        return self.total_parameters_count / float(backbone_total_params)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parallel forward pass across all K draft heads.

        Args:
            hidden_states: Penultimate representations tensor of shape (B, L, D).

        Returns:
            Candidate draft predictions tensor of shape (K, B, L, C).
        """
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden_states of shape (B, L, D), got {hidden_states.shape}"
            )
        if hidden_states.shape[-1] != self._hidden_dim:
            raise ValueError(
                f"Hidden dimension mismatch: expected {self._hidden_dim}, got {hidden_states.shape[-1]}"
            )

        # Compute predictions from all K heads
        head_outputs: List[torch.Tensor] = [head(hidden_states) for head in self.heads]

        # Stack along dim 0 to produce (K, B, L, C)
        return torch.stack(head_outputs, dim=0)

    def get_candidate_probabilities(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Computes candidate draft probabilities across all K heads via temperature-scaled softmax.

        Args:
            hidden_states: Penultimate representations tensor of shape (B, L, D).
            temperature: Softmax temperature parameter (must be strictly > 0.0).

        Returns:
            Candidate probabilities tensor of shape (K, B, L, C).
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive, got {temperature}")

        logits = self.forward(hidden_states)
        if temperature != 1.0:
            logits = logits / temperature
        return F.softmax(logits, dim=-1)

    def predict_candidates(
        self,
        hidden_states: torch.Tensor,
        top_k: int = 1,
    ) -> torch.Tensor:
        """
        Extracts top-k candidate token/class indices across all K heads for tree decoding.

        Args:
            hidden_states: Penultimate representations tensor of shape (B, L, D).
            top_k: Number of highest-probability candidate classes to retain.

        Returns:
            Candidate token indices tensor of shape (K, B, L, top_k).
        """
        if top_k < 1 or top_k > self._num_classes:
            raise ValueError(
                f"top_k must be between 1 and num_classes ({self._num_classes}), got {top_k}"
            )

        logits = self.forward(hidden_states)
        _, indices = torch.topk(logits, k=top_k, dim=-1)
        return indices


class SpeculativeDraftLoss(nn.Module):
    """
    Multi-target discounted cross-entropy loss for auxiliary speculative draft heads:

        L_draft = sum_{k=1}^K lambda_k * CrossEntropy(y_hat_{t+k}^{(k)}, y_{t+k})

    where lambda_k = gamma^{k-1} with discount factor gamma = 0.85.
    """

    def __init__(
        self,
        gamma: float = 0.85,
        reduction: str = "mean",
        ignore_index: Optional[int] = 0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def get_lambdas(
        self,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Returns tensor of geometric discount weights lambda_k = gamma^{k-1} for k in {1, ..., K}.
        """
        powers = torch.arange(num_heads, dtype=torch.float32, device=device)
        lambdas = torch.pow(self.gamma, powers)
        if dtype is not None:
            lambdas = lambdas.to(dtype)
        return lambdas

    def forward(
        self,
        draft_logits: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes discounted multi-target cross entropy loss.

        Args:
            draft_logits: Tensor of shape (K, B, L, C).
            target_ids: Ground truth targets. Either LongTensor of shape (B, L)
                        or probability distribution tensor of shape (B, L, C).
            attention_mask: Optional mask tensor of shape (B, L) where 1 indicates valid token.

        Returns:
            Tuple of (total_loss, metrics_dictionary).
        """
        if draft_logits.ndim != 4:
            raise ValueError(
                f"Expected draft_logits of shape (K, B, L, C), got {draft_logits.shape}"
            )

        num_heads, batch_size, seq_len, num_classes = draft_logits.shape

        if seq_len <= num_heads:
            raise ValueError(
                f"Sequence length L ({seq_len}) must be strictly greater than num_heads K ({num_heads})."
            )

        device = draft_logits.device
        lambdas = self.get_lambdas(num_heads, device=device, dtype=draft_logits.dtype)

        total_loss = torch.tensor(0.0, device=device, dtype=draft_logits.dtype)
        metrics: Dict[str, torch.Tensor] = {}

        is_soft_target = target_ids.ndim == 3 and target_ids.shape[-1] == num_classes

        for k in range(1, num_heads + 1):
            # Head k is at index k-1 in draft_logits
            # Predicts position t + k given representation at position t
            head_idx = k - 1
            lambda_k = lambdas[head_idx]

            # Slice logits for t in [0, seq_len - k - 1]
            head_logits = draft_logits[head_idx, :, :seq_len - k, :]  # (B, L - k, C)

            # Slice target for positions [k, seq_len - 1]
            if is_soft_target:
                head_targets = target_ids[:, k:, :]  # (B, L - k, C)
            else:
                head_targets = target_ids[:, k:]  # (B, L - k)

            # Flatten batch and sequence dimensions
            flat_logits = head_logits.reshape(-1, num_classes)
            flat_targets = head_targets.reshape(-1, num_classes) if is_soft_target else head_targets.reshape(-1)

            ce_kwargs = {"reduction": self.reduction}
            if not is_soft_target and self.ignore_index is not None:
                ce_kwargs["ignore_index"] = self.ignore_index

            if attention_mask is not None:
                # Target is valid only if both source t and target t+k are unmasked
                valid_mask = (attention_mask[:, :seq_len - k] & attention_mask[:, k:]).reshape(-1).bool()
                if valid_mask.sum() > 0:
                    selected_logits = flat_logits[valid_mask]
                    selected_targets = flat_targets[valid_mask]
                    if (
                        not is_soft_target
                        and self.ignore_index is not None
                        and (selected_targets != self.ignore_index).sum() == 0
                    ):
                        loss_k = torch.tensor(0.0, device=device, dtype=draft_logits.dtype)
                    else:
                        loss_k = F.cross_entropy(selected_logits, selected_targets, **ce_kwargs)
                else:
                    loss_k = torch.tensor(0.0, device=device, dtype=draft_logits.dtype)
            else:
                if (
                    not is_soft_target
                    and self.ignore_index is not None
                    and (flat_targets != self.ignore_index).sum() == 0
                ):
                    loss_k = torch.tensor(0.0, device=device, dtype=draft_logits.dtype)
                else:
                    loss_k = F.cross_entropy(flat_logits, flat_targets, **ce_kwargs)

            weighted_loss_k = lambda_k * loss_k
            total_loss = total_loss + weighted_loss_k

            metrics[f"loss_head_{k}"] = loss_k.detach()
            metrics[f"lambda_{k}"] = lambda_k.detach()
            metrics[f"weighted_loss_head_{k}"] = weighted_loss_k.detach()

        metrics["loss_total"] = total_loss.detach()
        return total_loss, metrics

"""
Training and Invariant Verification Harness for Speculative Draft Heads.
Guarantees base foundation backbone remains strictly frozen (grad is None)
while draft head parameters update across multi-target cross-entropy objectives.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW

from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.speculative.draft_heads import SpeculativeDraftHeads, SpeculativeDraftLoss

logger = logging.getLogger(__name__)


class DraftHeadTrainer:
    """
    Orchestrates training and parameter isolation verification for Speculative Draft Heads.
    """

    def __init__(
        self,
        backbone: SveltoBackbone,
        draft_heads: SpeculativeDraftHeads,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[SpeculativeDraftLoss] = None,
        ignore_index: Optional[int] = -100,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.backbone = backbone
        self.draft_heads = draft_heads

        # Strictly verify backbone parameter freezing
        self.verify_backbone_frozen()

        # Resolve device
        if device is None:
            self.device = self.backbone.target_device
        else:
            self.device = torch.device(device)

        self.backbone.to(self.device)
        self.draft_heads.to(self.device)

        # Setup loss function with gamma=0.85
        self.loss_fn = (
            loss_fn
            if loss_fn is not None
            else SpeculativeDraftLoss(gamma=0.85, ignore_index=ignore_index)
        )

        # Setup optimizer only on draft head parameters
        if optimizer is None:
            self.optimizer = AdamW(
                self.draft_heads.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optimizer

    def verify_backbone_frozen(self) -> None:
        """
        Strict invariant verification ensuring zero trainable parameters in base backbone.
        """
        trainable = self.backbone.trainable_parameters_count
        if trainable != 0:
            raise RuntimeError(
                f"Invariant violation: Foundation backbone has {trainable} trainable parameters! "
                f"Must strictly equal 0."
            )
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                raise RuntimeError(
                    f"Backbone parameter '{name}' has requires_grad=True!"
                )

    def train_step(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Executes a single optimization step for the speculative draft heads.

        Args:
            input_ids: Input nucleotide token IDs of shape (B, L).
            target_ids: Target token IDs of shape (B, L). If None, defaults to input_ids.
            attention_mask: Optional attention mask of shape (B, L).

        Returns:
            Dictionary of loss values and gradient telemetry.
        """
        input_ids = input_ids.to(self.device)
        if target_ids is None:
            target_ids = input_ids.clone()
        else:
            target_ids = target_ids.to(self.device)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # 1. Forward pass through frozen backbone (no gradient graph)
        with torch.no_grad():
            backbone_out = self.backbone(input_ids, attention_mask=attention_mask)
            penultimate = backbone_out.penultimate_hidden_state.detach()

        # 2. Forward pass through speculative draft heads
        self.draft_heads.train()
        draft_logits = self.draft_heads(penultimate)  # (K, B, L, C)

        # 3. Compute discounted multi-target loss
        loss, loss_metrics = self.loss_fn(
            draft_logits,
            target_ids,
            attention_mask=attention_mask,
        )

        # 4. Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # 5. Invariant assertion: Backbone parameters MUST have grad == None
        for name, param in self.backbone.named_parameters():
            if param.grad is not None:
                raise RuntimeError(
                    f"Invariant violation: Backbone parameter '{name}' received a gradient! "
                    f"Backbone must remain strictly frozen."
                )

        # 6. Verification: Draft heads MUST receive gradients
        draft_head_grads = [
            p.grad.abs().sum().item()
            for p in self.draft_heads.parameters()
            if p.grad is not None
        ]
        if not draft_head_grads or sum(draft_head_grads) == 0.0:
            raise RuntimeError(
                "Optimization error: Speculative draft heads received zero gradients during backward pass!"
            )

        # 7. Optimizer step
        self.optimizer.step()

        # Package metrics
        result: Dict[str, float] = {
            "loss": float(loss.item()),
            "total_grad_norm": float(sum(draft_head_grads)),
        }
        for k, v in loss_metrics.items():
            result[k] = float(v.item()) if isinstance(v, torch.Tensor) else float(v)

        return result

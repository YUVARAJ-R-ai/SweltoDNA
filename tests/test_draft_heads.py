"""
Unit tests for Auxiliary Speculative Draft Heads, Residual Projection Architecture,
and Multi-Target Discounted Cross-Entropy Loss (Issue #3).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.speculative.draft_heads import (
    DraftHead,
    SpeculativeDraftHeads,
    SpeculativeDraftLoss,
)
from svelto_dna.speculative.trainer import DraftHeadTrainer


class TestDraftHead:
    """Tests for single DraftHead module."""

    def test_single_head_forward_shape(self):
        B, L, D, C = 2, 64, 256, 4
        head = DraftHead(hidden_dim=D, proj_dim=128, num_classes=C, use_residual=True)
        x = torch.randn(B, L, D)
        out = head(x)
        assert out.shape == (B, L, C)

    def test_single_head_without_residual(self):
        B, L, D, C = 2, 32, 256, 3
        head = DraftHead(hidden_dim=D, proj_dim=128, num_classes=C, use_residual=False)
        assert head.res_proj is None
        x = torch.randn(B, L, D)
        out = head(x)
        assert out.shape == (B, L, C)

    def test_single_head_hidden_residual_mode(self):
        B, L, D, C = 2, 32, 256, 4
        head = DraftHead(
            hidden_dim=D,
            proj_dim=128,
            num_classes=C,
            use_residual=True,
            residual_mode="hidden",
        )
        x = torch.randn(B, L, D)
        out = head(x)
        assert out.shape == (B, L, C)

    def test_invalid_residual_mode(self):
        with pytest.raises(ValueError, match="Unsupported residual_mode"):
            DraftHead(hidden_dim=256, residual_mode="invalid")


class TestSpeculativeDraftHeads:
    """Tests for SpeculativeDraftHeads multi-head container."""

    @pytest.mark.parametrize("k", [3, 4])
    @pytest.mark.parametrize("c", [3, 4])
    def test_instantiation_k3_k4(self, k: int, c: int):
        heads = SpeculativeDraftHeads(
            num_heads=k,
            hidden_dim=256,
            proj_dim=128,
            num_classes=c,
        )
        assert heads.num_heads == k
        assert heads.num_classes == c
        assert len(heads.heads) == k
        assert heads.total_parameters_count > 0

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError, match="num_heads must be at least 1"):
            SpeculativeDraftHeads(num_heads=0)

    @pytest.mark.parametrize(
        "b, l, d, k, c",
        [
            (1, 32, 256, 3, 4),
            (2, 64, 256, 4, 4),
            (4, 128, 256, 3, 3),
            (2, 50, 128, 4, 3),
        ],
    )
    def test_forward_shape_conformity(self, b: int, l: int, d: int, k: int, c: int):
        """
        Acceptance Criteria: Unit tests verify forward pass tensor shape conformity:
        input (B, L, D) -> output (K, B, L, C).
        """
        module = SpeculativeDraftHeads(
            num_heads=k,
            hidden_dim=d,
            proj_dim=64 if d == 128 else 128,
            num_classes=c,
        )
        hidden_states = torch.randn(b, l, d)
        output = module(hidden_states)

        assert output.shape == (k, b, l, c), (
            f"Shape mismatch: expected ({k}, {b}, {l}, {c}), got {output.shape}"
        )

    def test_forward_dimension_mismatch(self):
        module = SpeculativeDraftHeads(num_heads=3, hidden_dim=256)
        with pytest.raises(ValueError, match="Hidden dimension mismatch"):
            module(torch.randn(2, 32, 128))

    def test_forward_invalid_ndim(self):
        module = SpeculativeDraftHeads(num_heads=3, hidden_dim=256)
        with pytest.raises(ValueError, match="Expected hidden_states of shape"):
            module(torch.randn(2, 256))

    def test_parameter_overhead_budget(self):
        """
        Acceptance Criteria: Auxiliary draft heads add less than 4% parameter overhead
        relative to the base genomic model.
        """
        backbone = SveltoBackbone(model_name="mock", hidden_dim=256)
        backbone_params = backbone.total_parameters_count

        # Test K=3
        heads_k3 = SpeculativeDraftHeads(num_heads=3, hidden_dim=256, proj_dim=128, num_classes=4)
        overhead_k3 = heads_k3.parameter_overhead_ratio(backbone_params)
        assert overhead_k3 < 0.04, f"K=3 overhead {overhead_k3 * 100:.3f}% exceeds 4% budget!"

        # Test K=4
        heads_k4 = SpeculativeDraftHeads(num_heads=4, hidden_dim=256, proj_dim=128, num_classes=4)
        overhead_k4 = heads_k4.parameter_overhead_ratio(backbone_params)
        assert overhead_k4 < 0.04, f"K=4 overhead {overhead_k4 * 100:.3f}% exceeds 4% budget!"

        # Test splice classes C=3
        heads_c3 = SpeculativeDraftHeads(num_heads=4, hidden_dim=256, proj_dim=128, num_classes=3)
        overhead_c3 = heads_c3.parameter_overhead_ratio(backbone_params)
        assert overhead_c3 < 0.04, f"K=4 (C=3) overhead {overhead_c3 * 100:.3f}% exceeds 4% budget!"

    def test_residual_gradient_flow(self):
        """
        Verify gradient flows properly through both MLP branch and residual skip connection.
        """
        module = SpeculativeDraftHeads(num_heads=2, hidden_dim=64, proj_dim=32, num_classes=4, use_residual=True)
        h = torch.randn(2, 16, 64, requires_grad=True)
        out = module(h)
        loss = out.sum()
        loss.backward()

        assert h.grad is not None
        assert torch.isfinite(h.grad).all()

        for head in module.heads:
            assert head.fc1.weight.grad is not None
            assert head.fc2.weight.grad is not None
            assert head.res_proj.weight.grad is not None

    def test_candidate_probabilities(self):
        """Verify softmax candidate probabilities helper."""
        module = SpeculativeDraftHeads(num_heads=3, hidden_dim=64, proj_dim=32, num_classes=4)
        h = torch.randn(2, 16, 64)
        probs = module.get_candidate_probabilities(h, temperature=0.7)
        assert probs.shape == (3, 2, 16, 4)
        assert torch.all(probs >= 0.0)
        assert torch.all(probs <= 1.0)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(3, 2, 16), atol=1e-5)

        with pytest.raises(ValueError, match="temperature must be strictly positive"):
            module.get_candidate_probabilities(h, temperature=0.0)

    def test_predict_candidates_top_k(self):
        """Verify top-k candidate prediction indices for downstream tree decoding."""
        module = SpeculativeDraftHeads(num_heads=3, hidden_dim=64, proj_dim=32, num_classes=4)
        h = torch.randn(2, 16, 64)
        top1 = module.predict_candidates(h, top_k=1)
        assert top1.shape == (3, 2, 16, 1)

        top2 = module.predict_candidates(h, top_k=2)
        assert top2.shape == (3, 2, 16, 2)

        with pytest.raises(ValueError, match="top_k must be between 1 and"):
            module.predict_candidates(h, top_k=0)
        with pytest.raises(ValueError, match="top_k must be between 1 and"):
            module.predict_candidates(h, top_k=5)


class TestSpeculativeDraftLoss:
    """Tests for SpeculativeDraftLoss discounted multi-target cross-entropy."""

    def test_geometric_discount_factors(self):
        loss_fn = SpeculativeDraftLoss(gamma=0.85)
        lambdas = loss_fn.get_lambdas(num_heads=4)
        expected = torch.tensor([1.0, 0.85, 0.85**2, 0.85**3])
        assert torch.allclose(lambdas, expected, atol=1e-5)

    def test_loss_computation_integer_targets(self):
        K, B, L, C = 3, 2, 16, 4
        loss_fn = SpeculativeDraftLoss(gamma=0.85)
        draft_logits = torch.randn(K, B, L, C, requires_grad=True)
        target_ids = torch.randint(0, C, (B, L))

        total_loss, metrics = loss_fn(draft_logits, target_ids)

        assert total_loss.item() > 0
        assert "loss_head_1" in metrics
        assert "loss_head_2" in metrics
        assert "loss_head_3" in metrics
        assert "loss_total" in metrics

        # Verify backward pass propagates to draft_logits
        total_loss.backward()
        assert draft_logits.grad is not None
        assert torch.isfinite(draft_logits.grad).all()

    def test_loss_with_attention_mask(self):
        K, B, L, C = 3, 2, 16, 4
        loss_fn = SpeculativeDraftLoss(gamma=0.85)
        draft_logits = torch.randn(K, B, L, C)
        target_ids = torch.randint(0, C, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        attention_mask[:, 12:] = 0  # Mask out tail padding

        total_loss, metrics = loss_fn(draft_logits, target_ids, attention_mask=attention_mask)
        assert total_loss.item() > 0

    def test_loss_soft_probability_targets(self):
        K, B, L, C = 3, 2, 16, 3
        loss_fn = SpeculativeDraftLoss(gamma=0.85)
        draft_logits = torch.randn(K, B, L, C)
        # Soft probabilities for splice classes [None, Donor, Acceptor]
        target_probs = F.softmax(torch.randn(B, L, C), dim=-1)

        total_loss, metrics = loss_fn(draft_logits, target_probs)
        assert total_loss.item() > 0

    def test_loss_sequence_length_too_short(self):
        loss_fn = SpeculativeDraftLoss(gamma=0.85)
        draft_logits = torch.randn(4, 2, 3, 4)  # L=3 <= K=4
        target_ids = torch.randint(0, 4, (2, 3))
        with pytest.raises(ValueError, match="Sequence length L.*must be strictly greater"):
            loss_fn(draft_logits, target_ids)

    def test_loss_with_ignore_index(self):
        """Verify tokens with ignore_index (e.g. 0 for <PAD>) are excluded from loss."""
        K, B, L, C = 2, 1, 8, 4
        draft_logits = torch.randn(K, B, L, C, requires_grad=True)
        # All target tokens set to 0 (padding)
        target_ids = torch.zeros((B, L), dtype=torch.long)

        loss_fn_ignore = SpeculativeDraftLoss(gamma=0.85, ignore_index=0)
        total_loss, _ = loss_fn_ignore(draft_logits, target_ids)
        # When all tokens are padding, loss should be 0.0 (no NaN)
        assert total_loss.item() == 0.0

        # Partial padding
        target_ids_partial = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 1]])
        loss_partial, _ = loss_fn_ignore(draft_logits, target_ids_partial)
        assert loss_partial.item() > 0.0

        # Without ignore_index, class 0 is included
        loss_fn_no_ignore = SpeculativeDraftLoss(gamma=0.85, ignore_index=None)
        loss_all_zeros, _ = loss_fn_no_ignore(draft_logits, target_ids)
        assert loss_all_zeros.item() > 0.0


class TestDraftHeadTrainer:
    """Tests for DraftHeadTrainer and parameter isolation invariants."""

    def test_backbone_remains_frozen_and_heads_update(self):
        """
        Acceptance Criteria: Standalone training script verifies base model parameters
        remain completely frozen (grad == None) while draft head weights update successfully.
        """
        backbone = SveltoBackbone(model_name="mock", hidden_dim=256)
        draft_heads = SpeculativeDraftHeads(num_heads=3, hidden_dim=256, proj_dim=128, num_classes=4)

        trainer = DraftHeadTrainer(
            backbone=backbone,
            draft_heads=draft_heads,
            lr=1e-3,
        )

        dummy_input = torch.randint(1, 5, (2, 64))
        dummy_target = torch.randint(0, 4, (2, 64))

        # Check initial weights of draft heads
        initial_weight = draft_heads.heads[0].fc1.weight.clone()

        # Step 1
        metrics_1 = trainer.train_step(dummy_input, dummy_target)
        loss_1 = metrics_1["loss"]

        # Verify backbone parameters strictly have grad == None
        for name, param in backbone.named_parameters():
            assert param.grad is None, f"Backbone parameter {name} has non-None grad!"
            assert not param.requires_grad

        # Verify draft heads updated
        updated_weight = draft_heads.heads[0].fc1.weight
        assert not torch.equal(initial_weight, updated_weight), "Draft head weights did not change!"

        # Step 2
        metrics_2 = trainer.train_step(dummy_input, dummy_target)
        assert metrics_2["loss"] < loss_1 or metrics_2["total_grad_norm"] > 0

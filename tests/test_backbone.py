import pytest
import torch
import torch.nn.functional as F
from svelto_dna.core.backbone import SveltoBackbone, BackboneOutput
from svelto_dna.core.tokenizer import GenomicTokenizer


class TestSveltoBackbone:
    @pytest.fixture
    def backbone(self):
        return SveltoBackbone(model_name="mock", hidden_dim=128, num_layers=2)

    @pytest.fixture
    def tokenizer(self):
        return GenomicTokenizer()

    def test_parameter_freezing(self, backbone):
        # Must have exactly 0 trainable parameters
        assert backbone.trainable_parameters_count == 0
        assert backbone.total_parameters_count > 0
        for name, param in backbone.named_parameters():
            assert not param.requires_grad, f"Parameter {name} is not frozen!"

    def test_deterministic_cosine_similarity(self, backbone, tokenizer):
        seq = "ACGT" * 256  # 1024 bp
        encoded = tokenizer.encode(seq, return_tensors="pt")
        input_ids = encoded["input_ids"]

        out1 = backbone(input_ids)
        out2 = backbone(input_ids)

        assert isinstance(out1, BackboneOutput)
        assert isinstance(out2, BackboneOutput)

        # Exact cosine similarity = 1.000 across multiple invocations
        cos_sim = F.cosine_similarity(
            out1.last_hidden_state.flatten(),
            out2.last_hidden_state.flatten(),
            dim=0,
        )
        assert float(cos_sim) >= 0.999999, f"Deterministic check failed with cosine similarity {float(cos_sim)}"

    @pytest.mark.parametrize("length", [1024, 2048, 5000, 10000])
    def test_forward_dimensions_across_context_windows(self, backbone, tokenizer, length):
        seq = "ACGTN" * (length // 5) + "A" * (length % 5)
        encoded = tokenizer.encode(seq, max_length=length, return_tensors="pt")

        out = backbone(encoded["input_ids"], attention_mask=encoded["attention_mask"])

        assert out.last_hidden_state.shape == (1, length, backbone.hidden_dim)
        assert out.penultimate_hidden_state.shape == (1, length, backbone.hidden_dim)
        assert out.logits.shape == (1, length, 6)

    def test_fallback_to_mock_on_invalid_checkpoint(self):
        # Should gracefully fall back to MockGenomicBackbone when remote checkpoint does not exist
        backbone = SveltoBackbone(
            model_name="nonexistent-repo/does-not-exist-genomic-model",
            fallback_to_mock=True,
            hidden_dim=64,
            num_layers=1,
        )
        assert backbone.is_mock is True
        assert backbone.trainable_parameters_count == 0

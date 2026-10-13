"""
Issue #1: load a real, frozen HyenaDNA checkpoint with its own tokenizer.
Skipped automatically when the Hugging Face hub is unreachable and the model is not cached.
"""

import pytest
import torch
import torch.nn.functional as F

from svelto_dna.core.backbone import SveltoBackbone

MODEL_ID = "LongSafari/hyenadna-small-32k-seqlen-hf"


def _available() -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.py", "*.safetensors"])
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="HyenaDNA checkpoint not reachable")


@pytest.fixture(scope="module")
def backbone():
    return SveltoBackbone(model_name=MODEL_ID)


def test_loads_real_model_frozen(backbone):
    assert backbone.is_mock is False
    assert backbone.trainable_parameters_count == 0
    assert backbone.total_parameters_count > 1_000_000
    assert backbone.hidden_dim == 256


def test_uses_the_models_own_tokenizer_without_special_tokens(backbone):
    ids = backbone.encode("ACGTN")["input_ids"]
    assert ids.tolist() == [[7, 8, 9, 10, 11]]


def test_handles_lowercase_and_iupac_like_the_genomic_tokenizer(backbone):
    assert backbone.encode("acgtR")["input_ids"].tolist() == backbone.encode("ACGTN")["input_ids"].tolist()


@pytest.mark.parametrize("length", [1024, 10000])
def test_one_output_position_per_base(backbone, length):
    enc = backbone.encode("ACGT" * (length // 4))
    out = backbone(enc["input_ids"], attention_mask=enc["attention_mask"])
    assert out.last_hidden_state.shape == (1, length, 256)
    assert out.penultimate_hidden_state.shape == (1, length, 256)


def test_padded_batch_does_not_crash_and_masks_padding(backbone):
    enc = backbone.encode(["ACGTACGT", "ACGT"])
    out = backbone(enc["input_ids"], attention_mask=enc["attention_mask"])
    assert out.last_hidden_state.shape == (2, 8, 256)
    assert torch.all(out.last_hidden_state[1, 4:] == 0)


def test_deterministic(backbone):
    ids = backbone.encode("ACGT" * 256)["input_ids"]
    a, b = backbone(ids).last_hidden_state, backbone(ids).last_hidden_state
    assert F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0) >= 0.999999


def test_unknown_checkpoint_fails_loudly_by_default():
    with pytest.raises(RuntimeError, match="Failed to load backbone"):
        SveltoBackbone(model_name="nonexistent-org/definitely-not-a-model")

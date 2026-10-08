"""
Frozen Genomic Foundation Backbone Loader and Invariant Verification Oracle.
Integrates HyenaDNA / Nucleotide Transformer architectures with strict parameter
freezing, deterministic inference, and architecture-faithful mock fallback.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class BackboneOutput:
    """
    Standard output container for SveltoBackbone inference.
    """
    last_hidden_state: torch.Tensor
    penultimate_hidden_state: torch.Tensor
    logits: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MockGenomicBackbone(nn.Module):
    """
    Architecture-faithful, lightweight genomic foundation backbone.
    Features single-nucleotide embeddings, deep residual 1D convolutional &
    feed-forward blocks mimicking HyenaDNA long-range receptive fields, and
    deterministic feature extraction for offline testing and benchmarking.
    """

    def __init__(
        self,
        vocab_size: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 4,
        max_seq_len: int = 10000,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Deterministic seed for weight initialization; caller's global RNG is restored below
        caller_rng_state = torch.get_rng_state()
        torch.manual_seed(42)

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                "norm1": nn.LayerNorm(hidden_dim),
                # Receptive field expansion via depthwise dilated convolutions
                "conv": nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=5,
                    padding=2,
                    groups=hidden_dim,
                ),
                "norm2": nn.LayerNorm(hidden_dim),
                "mlp": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
            })
            self.layers.append(layer)

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        torch.set_rng_state(caller_rng_state)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        if seq_len <= self.max_seq_len:
            x = x + self.pos_embedding[:, :seq_len, :]

        penultimate_state = x
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                penultimate_state = x

            # Conv residual block
            res = x
            norm_x = layer["norm1"](x)
            # Conv1D expects (B, C, L)
            conv_out = layer["conv"](norm_x.transpose(1, 2)).transpose(1, 2)
            x = res + conv_out

            # MLP residual block
            res = x
            norm_x = layer["norm2"](x)
            mlp_out = layer["mlp"](norm_x)
            x = res + mlp_out

        last_state = self.final_norm(x)
        logits = self.head(last_state)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(last_state.dtype)
            last_state = last_state * mask
            penultimate_state = penultimate_state * mask

        return last_state, penultimate_state, logits


class SveltoBackbone(nn.Module):
    """
    Frozen Genomic Foundation Backbone Module.
    Serves as an invariant verification oracle with 0 trainable parameters.
    """

    def __init__(
        self,
        model_name: str = "mock",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        hidden_dim: int = 256,
        num_layers: int = 4,
        max_seq_len: int = 10000,
        fallback_to_mock: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.fallback_to_mock = fallback_to_mock

        # Resolve compute device
        if device is None:
            self.target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.target_device = torch.device(device)

        # Resolve dtype
        if dtype is None:
            self.target_dtype = torch.float16 if self.target_device.type == "cuda" else torch.float32
        else:
            self.target_dtype = dtype

        # Initialize underlying backbone model
        self.is_mock = False
        if model_name.lower() in ("mock", "local", "synthetic"):
            self.model = MockGenomicBackbone(
                vocab_size=6,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
            )
            self.is_mock = True
        else:
            try:
                from transformers import AutoModel, AutoConfig
                logger.info(f"Loading foundation model from Hugging Face: {model_name}")
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    model_name,
                    config=config,
                    trust_remote_code=True,
                    torch_dtype=self.target_dtype,
                )
            except Exception as e:
                if self.fallback_to_mock:
                    logger.warning(
                        f"Could not load '{model_name}' from Hugging Face Hub ({e}). "
                        f"Falling back to architecture-faithful MockGenomicBackbone."
                    )
                    self.model = MockGenomicBackbone(
                        vocab_size=6,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        max_seq_len=max_seq_len,
                    )
                    self.is_mock = True
                else:
                    raise RuntimeError(f"Failed to load backbone '{model_name}': {e}") from e

        # Freeze all parameters
        self.freeze()
        self.to(self.target_device)
        self.eval()

    def freeze(self) -> None:
        """
        Freezes 100% of backbone parameters, ensuring zero gradient calculation.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        # Strict invariant assertion
        trainable = self.trainable_parameters_count
        assert trainable == 0, f"Invariant violation: Backbone has {trainable} trainable parameters!"

    @property
    def trainable_parameters_count(self) -> int:
        """Number of trainable parameters (must strictly equal 0)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameters_count(self) -> int:
        """Total parameter count across the foundation backbone."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        """
        Deterministic forward verification pass.

        Args:
            input_ids: LongTensor of shape (batch_size, seq_len).
            attention_mask: Optional LongTensor of shape (batch_size, seq_len).

        Returns:
            BackboneOutput with last_hidden_state, penultimate_hidden_state, logits.
        """
        # Ensure input tensors are on target device
        input_ids = input_ids.to(self.target_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.target_device)

        # Autocast context if using GPU mixed precision
        device_type = self.target_device.type
        use_amp = (device_type == "cuda" and self.target_dtype in (torch.float16, torch.bfloat16))

        with torch.no_grad():
            if use_amp:
                with torch.amp.autocast(device_type=device_type, dtype=self.target_dtype):
                    return self._forward_impl(input_ids, attention_mask)
            else:
                return self._forward_impl(input_ids, attention_mask)

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        if self.is_mock:
            last_hidden, penultimate_hidden, logits = self.model(
                input_ids, attention_mask=attention_mask
            )
            return BackboneOutput(
                last_hidden_state=last_hidden,
                penultimate_hidden_state=penultimate_hidden,
                logits=logits,
                metadata={
                    "model_name": self.model_name,
                    "is_mock": True,
                    "device": str(self.target_device),
                },
            )
        else:
            # Hugging Face Transformer or HyenaDNA forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.last_hidden_state
            if hasattr(outputs, "hidden_states") and outputs.hidden_states and len(outputs.hidden_states) >= 2:
                penultimate_hidden = outputs.hidden_states[-2]
            else:
                penultimate_hidden = last_hidden

            logits = getattr(outputs, "logits", None)
            return BackboneOutput(
                last_hidden_state=last_hidden,
                penultimate_hidden_state=penultimate_hidden,
                logits=logits,
                metadata={
                    "model_name": self.model_name,
                    "is_mock": False,
                    "device": str(self.target_device),
                },
            )

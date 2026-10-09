"""
Splice-site classifier on a frozen genomic backbone.

HyenaDNA is causal, so each position's forward state only sees upstream sequence, but a donor depends
on the downstream intron. Features therefore concatenate the forward pass with the reverse-complement
pass (flipped back into forward coordinates): every position sees both flanks.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.core.tokenizer import GenomicTokenizer

CLASSES = ("neither", "donor", "acceptor")


class SpliceHead(nn.Module):
    """[h_fwd ; h_rc] (2D) → 3 classes per position."""

    def __init__(self, hidden_dim: int = 256, proj_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),
            nn.Linear(2 * hidden_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, len(CLASSES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SplicePredictor:
    """Frozen backbone + trainable head. `predict` returns per-base probabilities [neither, donor, acceptor]."""

    def __init__(self, backbone: SveltoBackbone, head: SpliceHead) -> None:
        self.backbone = backbone
        self.head = head.to(backbone.target_device)
        self._tok = GenomicTokenizer()

    @property
    def device(self) -> torch.device:
        return self.backbone.target_device

    def features(self, seqs: Sequence[str]) -> torch.Tensor:
        """(B, L, 2D) float32 features; padded positions are zero."""
        seqs = list(seqs)
        enc_f = self.backbone.encode(seqs)
        enc_r = self.backbone.encode([self._tok.reverse_complement(s) for s in seqs])
        h_f = self.backbone(enc_f["input_ids"], attention_mask=enc_f["attention_mask"]).last_hidden_state.float()
        h_r = self.backbone(enc_r["input_ids"], attention_mask=enc_r["attention_mask"]).last_hidden_state.float()
        # RC position j corresponds to forward position len-1-j; flip each sequence within its own length
        aligned = torch.zeros_like(h_r)
        for i, s in enumerate(seqs):
            n = len(s)
            aligned[i, :n] = h_r[i, :n].flip(0)
        return torch.cat([h_f, aligned], dim=-1)

    @torch.no_grad()
    def predict(self, seqs: Sequence[str]) -> torch.Tensor:
        """(B, L, 3) probabilities on CPU; rows past each sequence's length are zero."""
        self.head.eval()
        seqs = list(seqs)
        probs = torch.softmax(self.head(self.features(seqs)), dim=-1).cpu()
        for i, s in enumerate(seqs):
            probs[i, len(s):] = 0
        return probs

    def save(self, path: str, **meta) -> None:
        torch.save({"state_dict": self.head.state_dict(), "hidden_dim": self.head.hidden_dim,
                    "proj_dim": self.head.net[1].out_features, "backbone": self.backbone.model_name, **meta}, path)

    @classmethod
    def load(cls, path: str, backbone: Optional[SveltoBackbone] = None) -> "SplicePredictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        backbone = backbone or SveltoBackbone(model_name=ckpt["backbone"])
        if backbone.model_name != ckpt["backbone"]:
            raise ValueError(f"Head was trained on {ckpt['backbone']}, not {backbone.model_name}")
        head = SpliceHead(hidden_dim=ckpt["hidden_dim"], proj_dim=ckpt["proj_dim"])
        head.load_state_dict(ckpt["state_dict"])
        return cls(backbone, head)

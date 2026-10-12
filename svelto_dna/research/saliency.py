"""
Locus saliency: which input bases drive the predicted splice probability at one position.

Gradient × input on the backbone's token embeddings, summed over the forward and reverse-complement
passes (both feed the splice head). Uses torch.autograd.grad without retain_graph, so no gradients
accumulate on parameters and the graph is freed when the call returns.
"""

from typing import Literal

import torch
import torch.nn as nn

from svelto_dna.core.tokenizer import GenomicTokenizer
from svelto_dna.model.splice_head import CLASSES, SplicePredictor

_tok = GenomicTokenizer()


def _embedding(module: nn.Module) -> nn.Embedding:
    emb = next((m for m in module.modules() if isinstance(m, nn.Embedding)), None)
    if emb is None:
        raise TypeError("backbone has no token embedding to attribute to")
    return emb


def locus_saliency(predictor: SplicePredictor, seq: str, position: int, target: Literal["donor", "acceptor"] = "donor") -> torch.Tensor:
    """(L,) attribution in [0, 1] for log p(target) at `position`, on CPU."""
    bb = predictor.backbone
    captured: list = []

    def hook(_mod, _inp, out):
        out = out.detach().requires_grad_(True)
        captured.append(out)
        return out

    handle = _embedding(bb.model).register_forward_hook(hook)
    try:
        with torch.enable_grad():
            enc_f, enc_r = bb.encode([seq]), bb.encode([_tok.reverse_complement(seq)])
            dev = bb.target_device
            h_f = bb._forward_impl(enc_f["input_ids"].to(dev), enc_f["attention_mask"].to(dev)).last_hidden_state.float()
            h_r = bb._forward_impl(enc_r["input_ids"].to(dev), enc_r["attention_mask"].to(dev)).last_hidden_state.float()
            feats = torch.cat([h_f, h_r.flip(1)], dim=-1)
            logp = torch.log_softmax(predictor.head(feats)[0, position], dim=-1)[CLASSES.index(target)]
            g_f, g_r = torch.autograd.grad(logp, captured)
        e_f, e_r = captured
        sal = (g_f * e_f).sum(-1).abs()[0] + (g_r * e_r).sum(-1).abs()[0].flip(0)
        sal = sal.detach().float().cpu()
        peak = float(sal.max())
        return sal / peak if peak > 0 else sal
    finally:
        handle.remove()
        captured.clear()

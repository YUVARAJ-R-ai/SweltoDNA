"""Vectorized Splice Disruption & Delta Score (Δ) Calculator.

This module provides high-throughput, vectorized computation of splice donor and
acceptor disruption metrics across genomic sequences using 1D max pooling kernels.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

DeltaComponent = Literal["donor_gain", "donor_loss", "acceptor_gain", "acceptor_loss"]

_COMPONENT_NAMES: Tuple[DeltaComponent, ...] = (
    "donor_gain",
    "donor_loss",
    "acceptor_gain",
    "acceptor_loss",
)


class DeltaResult(dict):
    """Structured container for splice delta scores and disruption metrics.

    Inherits from dict for dictionary compatibility while supporting attribute-style
    access and tensor/NumPy format conversions.
    """

    def __init__(
        self,
        donor_gain: Union[np.ndarray, torch.Tensor],
        donor_loss: Union[np.ndarray, torch.Tensor],
        acceptor_gain: Union[np.ndarray, torch.Tensor],
        acceptor_loss: Union[np.ndarray, torch.Tensor],
        locus_delta: Union[np.ndarray, torch.Tensor],
        peak_delta: float,
        peak_component: DeltaComponent,
        peak_position: int,
        window_size: int,
        raw_differences: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None,
        latency_ms: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        data: Dict[str, Any] = {
            "donor_gain": donor_gain,
            "donor_loss": donor_loss,
            "acceptor_gain": acceptor_gain,
            "acceptor_loss": acceptor_loss,
            "locus_delta": locus_delta,
            "peak_delta": float(peak_delta),
            "peak_component": str(peak_component),
            "peak_position": int(peak_position),
            "window_size": int(window_size),
            "raw_differences": raw_differences or {},
            "latency_ms": latency_ms,
        }
        data.update(kwargs)
        super().__init__(data)

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'DeltaResult' object has no attribute '{item}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a plain Python dictionary."""
        return dict(self)

    def to_numpy(self) -> DeltaResult:
        """Convert any internal PyTorch tensors to NumPy ndarrays."""
        converted = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                converted[k] = v.detach().cpu().numpy()
            elif isinstance(v, dict):
                sub_dict = {}
                for sk, sv in v.items():
                    sub_dict[sk] = sv.detach().cpu().numpy() if isinstance(sv, torch.Tensor) else sv
                converted[k] = sub_dict
            else:
                converted[k] = v
        return DeltaResult(**converted)

    def to_torch(self, device: Optional[Union[str, torch.device]] = None) -> DeltaResult:
        """Convert any internal NumPy arrays to PyTorch tensors on the specified device."""
        converted = {}
        for k, v in self.items():
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v)
                if device is not None:
                    t = t.to(device)
                converted[k] = t
            elif isinstance(v, dict):
                sub_dict = {}
                for sk, sv in v.items():
                    if isinstance(sv, np.ndarray):
                        st = torch.from_numpy(sv)
                        if device is not None:
                            st = st.to(device)
                        sub_dict[sk] = st
                    else:
                        sub_dict[sk] = sv
                converted[k] = sub_dict
            else:
                converted[k] = v
        return DeltaResult(**converted)


def _to_tensor(
    arr: Union[torch.Tensor, np.ndarray, Sequence],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, bool]:
    """Convert input to a PyTorch tensor, tracking if the original input was a torch.Tensor."""
    was_tensor = isinstance(arr, torch.Tensor)
    if was_tensor:
        tensor = arr
        if tensor.dtype != dtype and not tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
    elif isinstance(arr, np.ndarray):
        tensor = torch.from_numpy(arr)
        if tensor.dtype != dtype and not tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)
    else:
        tensor = torch.tensor(arr, dtype=dtype, device=device)
    return tensor, was_tensor


@torch.no_grad()
def compute_delta_scores(
    p_ref: Union[torch.Tensor, np.ndarray],
    p_mut: Union[torch.Tensor, np.ndarray],
    window_size: int = 50,
    variant_pos: Optional[int] = None,
    canonical_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    clamp_non_negative: bool = True,
    return_tensors: Optional[bool] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> DeltaResult:
    """Compute vectorized splice donor and acceptor disruption delta scores (Δ)."""
    t_start = time.perf_counter()

    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, got {window_size}")

    target_device = None
    if device is not None:
        target_device = torch.device(device) if isinstance(device, str) else device
    elif isinstance(p_ref, torch.Tensor):
        target_device = p_ref.device
    elif isinstance(p_mut, torch.Tensor):
        target_device = p_mut.device

    ref_t, ref_was_tensor = _to_tensor(p_ref, device=target_device)
    mut_t, mut_was_tensor = _to_tensor(p_mut, device=target_device)

    is_torch_input = ref_was_tensor or mut_was_tensor
    if return_tensors is None:
        return_tensors = is_torch_input

    if ref_t.shape != mut_t.shape:
        raise ValueError(
            f"Shape mismatch: p_ref shape {ref_t.shape} does not match p_mut shape {mut_t.shape}"
        )

    is_2d = ref_t.dim() == 2
    if is_2d:
        if ref_t.shape[-1] != 3:
            raise ValueError(f"Expected last dimension to be 3 (classes), got shape {ref_t.shape}")
        ref_t = ref_t.unsqueeze(0)
        mut_t = mut_t.unsqueeze(0)
    elif ref_t.dim() == 3:
        if ref_t.shape[-1] != 3:
            raise ValueError(f"Expected last dimension to be 3 (classes), got shape {ref_t.shape}")
    else:
        raise ValueError(f"Expected 2D or 3D probability tensor, got shape {ref_t.shape}")

    batch_size, seq_len, _ = ref_t.shape

    if variant_pos is not None:
        if not (0 <= variant_pos < seq_len):
            raise IndexError(
                f"variant_pos {variant_pos} is out of bounds for sequence of length {seq_len}"
            )

    p_ref_donor = ref_t[:, :, 1]
    p_mut_donor = mut_t[:, :, 1]
    p_ref_acceptor = ref_t[:, :, 2]
    p_mut_acceptor = mut_t[:, :, 2]

    diff_stack = torch.empty((batch_size, 4, seq_len), dtype=ref_t.dtype, device=ref_t.device)

    diff_dg = p_mut_donor - p_ref_donor
    diff_dl = p_ref_donor - p_mut_donor
    diff_ag = p_mut_acceptor - p_ref_acceptor
    diff_al = p_ref_acceptor - p_mut_acceptor

    if clamp_non_negative:
        torch.clamp(diff_dg, min=0.0, out=diff_stack[:, 0, :])
        torch.clamp(diff_dl, min=0.0, out=diff_stack[:, 1, :])
        torch.clamp(diff_ag, min=0.0, out=diff_stack[:, 2, :])
        torch.clamp(diff_al, min=0.0, out=diff_stack[:, 3, :])
    else:
        diff_stack[:, 0, :] = diff_dg
        diff_stack[:, 1, :] = diff_dl
        diff_stack[:, 2, :] = diff_ag
        diff_stack[:, 3, :] = diff_al

    if canonical_mask is not None:
        c_mask_t, _ = _to_tensor(canonical_mask, device=ref_t.device, dtype=torch.bool)
        if c_mask_t.dim() == 1:
            c_mask_t = c_mask_t.unsqueeze(0)

        if c_mask_t.shape[-1] == seq_len and c_mask_t.dim() == 2:
            diff_stack[:, 0, :].masked_fill_(c_mask_t, 0.0)
            diff_stack[:, 2, :].masked_fill_(c_mask_t, 0.0)
        elif c_mask_t.shape[-1] == 3 and c_mask_t.shape[1] == seq_len:
            diff_stack[:, 0, :].masked_fill_(c_mask_t[:, :, 1], 0.0)
            diff_stack[:, 2, :].masked_fill_(c_mask_t[:, :, 2], 0.0)

    raw_diffs: Dict[str, Union[np.ndarray, torch.Tensor]] = {}
    for idx, name in enumerate(_COMPONENT_NAMES):
        raw_slice = diff_stack[:, idx, :]
        if is_2d:
            raw_slice = raw_slice.squeeze(0)
        raw_diffs[name] = raw_slice if return_tensors else raw_slice.detach().cpu().numpy()

    kernel_size = 2 * window_size + 1
    padding = window_size

    pooled = F.max_pool1d(diff_stack, kernel_size=kernel_size, stride=1, padding=padding)

    if variant_pos is not None:
        locus_pooled = pooled[:, :, variant_pos]
        dg_out = locus_pooled[:, 0]
        dl_out = locus_pooled[:, 1]
        ag_out = locus_pooled[:, 2]
        al_out = locus_pooled[:, 3]
        locus_delta = torch.max(locus_pooled, dim=1).values

        start_idx = max(0, variant_pos - window_size)
        end_idx = min(seq_len, variant_pos + window_size + 1)
        slice_diff = diff_stack[:, :, start_idx:end_idx]

        flat_slice = slice_diff.reshape(batch_size, -1)
        max_val, argmax_flat = torch.max(flat_slice, dim=1)
        argmax_idx = argmax_flat[0].item()
        slice_len = end_idx - start_idx
        peak_comp_idx = argmax_idx // slice_len
        peak_pos_offset = argmax_idx % slice_len
        peak_position = start_idx + peak_pos_offset
        peak_component = _COMPONENT_NAMES[peak_comp_idx]
        peak_delta = max_val[0].item()

    else:
        dg_out = pooled[:, 0, :]
        dl_out = pooled[:, 1, :]
        ag_out = pooled[:, 2, :]
        al_out = pooled[:, 3, :]
        locus_delta = torch.max(pooled, dim=1).values

        flat_pooled = diff_stack.reshape(batch_size, -1)
        max_val, argmax_flat = torch.max(flat_pooled, dim=1)
        argmax_idx = argmax_flat[0].item()
        peak_comp_idx = argmax_idx // seq_len
        peak_position = argmax_idx % seq_len
        peak_component = _COMPONENT_NAMES[peak_comp_idx]
        peak_delta = max_val[0].item()

    if is_2d:
        dg_out = dg_out.squeeze(0)
        dl_out = dl_out.squeeze(0)
        ag_out = ag_out.squeeze(0)
        al_out = al_out.squeeze(0)
        locus_delta = locus_delta.squeeze(0)

    if not return_tensors:
        dg_out = dg_out.detach().cpu().numpy()
        dl_out = dl_out.detach().cpu().numpy()
        ag_out = ag_out.detach().cpu().numpy()
        al_out = al_out.detach().cpu().numpy()
        locus_delta = locus_delta.detach().cpu().numpy()

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000.0

    return DeltaResult(
        donor_gain=dg_out,
        donor_loss=dl_out,
        acceptor_gain=ag_out,
        acceptor_loss=al_out,
        locus_delta=locus_delta,
        peak_delta=peak_delta,
        peak_component=peak_component,
        peak_position=peak_position,
        window_size=window_size,
        raw_differences=raw_diffs,
        latency_ms=latency_ms,
    )

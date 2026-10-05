#!/usr/bin/env python3
"""Benchmark runner for Vectorized Splice Disruption & Delta Score (Δ) Calculator.

Measures wall-clock latency, throughput, and memory consumption across sequence
window lengths L in {1024, 2048, 5000, 10000} bp for both continuous sliding window
max-pool profiling and point variant delta extraction.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from svelto_dna.splice.delta import compute_delta_scores


def benchmark_delta_calculator(
    lengths: List[int],
    window_size: int = 50,
    warmup: int = 10,
    repeats: int = 50,
    device_name: str | None = None,
    dtype_str: str = "float32",
) -> Dict[str, Any]:
    """Run benchmark matrix for compute_delta_scores."""
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    dtype = getattr(torch, dtype_str)

    results: Dict[str, Any] = {
        "metadata": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
            "dtype": dtype_str,
            "window_size": window_size,
            "warmup_runs": warmup,
            "repeat_runs": repeats,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "profiles": {},
    }

    for L in lengths:
        # Generate random probability inputs simulating model outputs
        p_ref = torch.softmax(torch.randn(1, L, 3, device=device, dtype=dtype), dim=-1)
        p_mut = torch.softmax(torch.randn(1, L, 3, device=device, dtype=dtype), dim=-1)

        # Warmup
        for _ in range(warmup):
            compute_delta_scores(p_ref, p_mut, window_size=window_size)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        # Timed runs for full-sequence profiling
        full_latencies: List[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            compute_delta_scores(p_ref, p_mut, window_size=window_size)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            full_latencies.append((t1 - t0) * 1000.0)

        # Timed runs for point locus extraction
        variant_idx = L // 2
        point_latencies: List[float] = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            compute_delta_scores(p_ref, p_mut, window_size=window_size, variant_pos=variant_idx)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            point_latencies.append((t1 - t0) * 1000.0)

        full_arr = np.array(full_latencies)
        point_arr = np.array(point_latencies)

        mean_sec = float(np.mean(full_arr)) / 1000.0
        nucleotide_throughput = float(L / mean_sec) if mean_sec > 0 else 0.0

        results["profiles"][str(L)] = {
            "sequence_length": L,
            "full_profile": {
                "mean_ms": float(np.mean(full_arr)),
                "std_ms": float(np.std(full_arr)),
                "median_ms": float(np.median(full_arr)),
                "p95_ms": float(np.percentile(full_arr, 95)),
                "min_ms": float(np.min(full_arr)),
                "max_ms": float(np.max(full_arr)),
            },
            "point_locus": {
                "mean_ms": float(np.mean(point_arr)),
                "std_ms": float(np.std(point_arr)),
                "median_ms": float(np.median(point_arr)),
                "p95_ms": float(np.percentile(point_arr, 95)),
                "min_ms": float(np.min(point_arr)),
                "max_ms": float(np.max(point_arr)),
            },
            "nucleotide_throughput_bp_per_sec": nucleotide_throughput,
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Vectorized Splice Disruption & Delta Score Calculator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="1024,2048,5000,10000",
        help="Comma-separated sequence lengths to benchmark",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Window size W (radius in bp)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of timed benchmark iterations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="delta_benchmark_telemetry.json",
        help="Path to save JSON telemetry output",
    )
    parser.add_argument(
        "--assert-threshold",
        action="store_true",
        default=True,
        help="Assert that L=10,000 bp executes in < 5.0 ms on CPU (< 1.0 ms on GPU)",
    )

    args = parser.parse_args()
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]

    print("=" * 70)
    print("Svelto-DNA Splice Disruption Delta (Δ) Benchmark")
    print("=" * 70)

    telemetry = benchmark_delta_calculator(
        lengths=lengths,
        window_size=args.window_size,
        warmup=args.warmup,
        repeats=args.repeats,
        device_name=args.device,
    )

    meta = telemetry["metadata"]
    print(f"Device:       {meta['device']} ({meta['device_name']})")
    print(f"Window Size:  ±{meta['window_size']} bp")
    print(f"Warmup:       {meta['warmup_runs']} runs")
    print(f"Repeats:      {meta['repeat_runs']} runs")
    print("-" * 70)
    print(f"{'Length (bp)':<12} {'Full Mean (ms)':<16} {'Point Mean (ms)':<16} {'Throughput (bp/s)':<18}")
    print("-" * 70)

    for L_str, p in telemetry["profiles"].items():
        L = int(L_str)
        f_mean = p["full_profile"]["mean_ms"]
        pt_mean = p["point_locus"]["mean_ms"]
        tp = p["nucleotide_throughput_bp_per_sec"]
        print(f"{L:<12} {f_mean:<16.4f} {pt_mean:<16.4f} {tp:<18.0f}")

    print("-" * 70)

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Telemetry saved to {out_path.resolve()}")

    if args.assert_threshold and "10000" in telemetry["profiles"]:
        l10k_mean = telemetry["profiles"]["10000"]["full_profile"]["mean_ms"]
        is_cuda = "cuda" in meta["device"]
        threshold_ms = 1.0 if is_cuda else 5.0
        print(f"L=10000 execution time: {l10k_mean:.4f} ms (Target: < {threshold_ms:.1f} ms)")
        if l10k_mean > threshold_ms:
            print(f"ERROR: Latency {l10k_mean:.4f} ms exceeded threshold {threshold_ms} ms!", file=sys.stderr)
            return 1
        print("✓ Performance threshold passed!")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

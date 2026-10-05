#!/usr/bin/env python3
"""
CLI entry point for GRCh38 Invariant Verification Engine Baseline Profiler.
Measures wall-clock latency, peak VRAM/RSS memory, and throughput across L in {1024, 2048, 5000, 10000}.
Outputs standardized JSON telemetry.
"""

import argparse
import json
import sys
from pathlib import Path
import torch

from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.core.tokenizer import GenomicTokenizer
from svelto_dna.profiler.benchmark import BaselineProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Svelto-DNA Baseline Inference & Verification Profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mock",
        help="Backbone model checkpoint or 'mock' for local architecture-faithful oracle",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="1024,2048,5000,10000",
        help="Comma-separated sequence window lengths L to profile",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup forward passes per window length",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=20,
        help="Number of measured forward passes per window length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('cuda', 'cpu', or None for auto-detect)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float32", "float16", "bfloat16"],
        help="Precision data type (default: float16 if cuda else float32)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="baseline_telemetry.json",
        help="Target file path to save JSON telemetry",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fast sanity check run with 1 warmup and 2 repeats",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]

    num_warmup = 1 if args.dry_run else args.warmup
    num_repeats = 2 if args.dry_run else args.repeats

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype) if args.dtype else None

    print("=================================================================")
    print(" Svelto-DNA: GRCh38 Invariant Verification Engine Profiler")
    print("=================================================================")
    print(f"Model ID:              {args.model}")
    print(f"Window Lengths (L):    {lengths}")
    print(f"Warmup Passes:         {num_warmup}")
    print(f"Measurement Passes:    {num_repeats}")
    print(f"Device Target:         {args.device or 'auto-detect'}")
    print(f"Telemetry Output Path: {args.output_json}")
    print("-----------------------------------------------------------------")

    tokenizer = GenomicTokenizer()
    backbone = SveltoBackbone(
        model_name=args.model,
        device=args.device,
        dtype=dtype,
    )

    print(f"Active Device:         {backbone.target_device}")
    print(f"Active Dtype:          {backbone.target_dtype}")
    print(f"Total Parameters:      {backbone.total_parameters_count:,}")
    print(f"Trainable Parameters:  {backbone.trainable_parameters_count:,}")
    print("-----------------------------------------------------------------")

    # Invariant verification check
    assert backbone.trainable_parameters_count == 0, (
        f"Critical Failure: Backbone has {backbone.trainable_parameters_count} trainable parameters! Must be 0."
    )

    profiler = BaselineProfiler(
        backbone=backbone,
        tokenizer=tokenizer,
        lengths=lengths,
        num_warmup=num_warmup,
        num_repeats=num_repeats,
    )

    print("Executing benchmark runs across context windows...")
    telemetry = profiler.run()

    # Formatted terminal summary table
    print("\nBenchmark Results Summary:")
    print("+--------+------------+------------+------------+------------+---------------+----------------+")
    print("| Length | Mean (ms)  | Median(ms) | P95 (ms)   | Std (ms)   | Peak Mem (MB) | Throughput nt/s|")
    print("+--------+------------+------------+------------+------------+---------------+----------------+")
    for m in telemetry.metrics:
        print(
            f"| {m.seq_length:<6} | {m.mean_latency_ms:<10.2f} | {m.median_latency_ms:<10.2f} | "
            f"{m.p95_latency_ms:<10.2f} | {m.std_latency_ms:<10.2f} | {m.peak_memory_mb:<13.2f} | "
            f"{m.throughput_nt_per_sec:<14.1f} |"
        )
    print("+--------+------------+------------+------------+------------+---------------+----------------+\n")

    # Save JSON telemetry
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(telemetry.to_json())

    print(f"Telemetry successfully written to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

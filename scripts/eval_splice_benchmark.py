#!/usr/bin/env python3
"""
SpliceAI-10k and ClinVar benchmark evaluation harness.
Evaluates foundation backbone inference on canonical autosome test splits,
calculating Top-1/Top-k accuracy, ROC-AUC, and PR-AUC (Average Precision)
under severe class imbalance (>100:1 non-splice vs. splice ratio).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from svelto_dna.core.backbone import SveltoBackbone
from svelto_dna.core.tokenizer import GenomicTokenizer
from svelto_dna.data.benchmark import SpliceMetricsReport, compute_splice_metrics
from svelto_dna.data.spliceai import generate_synthetic_spliceai_dataset
from svelto_dna.speculative.draft_heads import SpeculativeDraftHeads


class SpliceClassificationHead(nn.Module):
    """Linear projection from foundation backbone hidden states to 3-class splice probabilities."""

    def __init__(self, hidden_dim: int = 256, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(hidden_states)
        return torch.softmax(logits, dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark genomic foundation backbone on SpliceAI-10k test split."
    )
    parser.add_argument(
        "--test-parquet",
        type=str,
        default="data/processed/spliceai_test.parquet",
        help="Path to processed test split Parquet file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mock",
        help="Backbone model ID or 'mock' for local verification oracle.",
    )
    parser.add_argument(
        "--draft-heads-checkpoint",
        type=str,
        default=None,
        help="Optional path to trained SpeculativeDraftHeads checkpoint (.pt/.pth) from Issue #3 for downstream draft benchmarking.",
    )
    parser.add_argument(
        "--splice-head",
        type=str,
        default=None,
        help="Trained splice head (scripts/train_splice_head.py); evaluates backbone + head in batches.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Windows per forward pass with --splice-head.",
    )
    parser.add_argument(
        "--output-telemetry",
        type=str,
        default="splice_benchmark_telemetry.json",
        help="Path to export JSON benchmark telemetry.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Maximum number of test sequences to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible evaluation.",
    )
    return parser.parse_args()


def evaluate_trained_head(args: argparse.Namespace, df: pl.DataFrame) -> int:
    """Batched evaluation of a trained SplicePredictor; scores only each window's labelled block."""
    from svelto_dna.model.splice_head import SplicePredictor

    pred = SplicePredictor.load(args.splice_head)
    print(f"Splice head      : {args.splice_head} on {pred.backbone.model_name}")
    rows = list(df.select("window_sequence", "labels", "context").iter_rows())
    ys, ps = [], []
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        probs = pred.predict([r[0] for r in chunk]).numpy()
        for (seq, labels, ctx), p in zip(chunk, probs):
            ctx = int(ctx or 0)
            ys.extend(labels)
            ps.append(p[ctx: ctx + len(labels)])
        if (i // args.batch_size) % 200 == 0:
            print(f"  {i + len(chunk):,}/{len(rows):,} windows", flush=True)
    report = compute_splice_metrics(np.array(ys, dtype=int), np.vstack(ps))
    print("\n" + report.summary() + "\n")
    out = report.to_dict()
    out.update({"model": pred.backbone.model_name, "splice_head": args.splice_head, "windows": len(rows), "split": str(args.test_parquet)})
    Path(args.output_telemetry).write_text(json.dumps(out, indent=2))
    print(f"✓ Telemetry saved to: {args.output_telemetry}")
    return 0


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=== Svelto-DNA Splice Disruption Benchmark Evaluation Suite ===")
    print(f"Model ID         : {args.model_name}")
    print(f"Random seed      : {args.seed}")

    test_path = Path(args.test_parquet)
    if test_path.exists():
        print(f"Loading test split from: {test_path}")
        df = pl.read_parquet(test_path)
    else:
        print(f"Test split not found at {test_path}. Generating synthetic test split (seed={args.seed})...")
        df = generate_synthetic_spliceai_dataset(n_genes=args.max_samples, window_size=1000, seed=args.seed)

    if len(df) > args.max_samples:
        df = df.slice(0, args.max_samples)

    print(f"Evaluating on {len(df)} test sequences...")

    if args.splice_head:
        return evaluate_trained_head(args, df)

    # Initialize tokenizer and backbone
    tokenizer = GenomicTokenizer()
    backbone = SveltoBackbone(
        model_name=args.model_name,
        device="cpu",
    )

    draft_heads_module = None
    head = None

    if args.draft_heads_checkpoint:
        ckpt_path = Path(args.draft_heads_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Draft heads checkpoint not found at: {ckpt_path}")
        print(f"Loading speculative draft heads from checkpoint: {ckpt_path}")
        loaded_ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = loaded_ckpt.get("state_dict", loaded_ckpt) if isinstance(loaded_ckpt, dict) else loaded_ckpt
        num_classes = 3
        num_heads = 3
        found_head_indices = set()
        for k, v in state_dict.items():
            if k.startswith("heads."):
                parts = k.split(".")
                if len(parts) > 1 and parts[1].isdigit():
                    found_head_indices.add(int(parts[1]))
            if k.endswith("fc2.weight") or k.endswith("res_proj.weight"):
                num_classes = v.shape[0]
        if found_head_indices:
            num_heads = max(found_head_indices) + 1

        draft_heads_module = SpeculativeDraftHeads(
            num_heads=num_heads,
            hidden_dim=backbone.hidden_dim,
            num_classes=num_classes,
        )
        # strict load: a mismatched checkpoint must fail, not silently evaluate random weights
        draft_heads_module.load_state_dict(state_dict)
        draft_heads_module.eval()
        print("✓ Successfully loaded SpeculativeDraftHeads from svelto_dna.speculative.draft_heads.")
    else:
        head = SpliceClassificationHead(hidden_dim=backbone.hidden_dim, seed=args.seed)
        head.eval()

    all_y_true = []
    all_y_probs = []

    with torch.no_grad():
        for row in df.iter_rows(named=True):
            seq = row["window_sequence"]
            labels = row["labels"]  # List of int

            encoding = backbone.encode(seq)
            input_ids = encoding["input_ids"]
            attn_mask = encoding["attention_mask"]

            out = backbone(input_ids=input_ids, attention_mask=attn_mask)

            if draft_heads_module is not None:
                # Downstream SpeculativeDraftHeads receives penultimate_hidden_state
                draft_out = draft_heads_module(out.penultimate_hidden_state)
                probs_tensor = draft_out[0] if isinstance(draft_out, tuple) else draft_out
                if probs_tensor.ndim == 4:
                    probs_tensor = probs_tensor[0]  # Select head 0 -> (B, L, C)
                probs = torch.softmax(probs_tensor, dim=-1).squeeze(0).cpu().numpy()
                if probs.shape[-1] > 3:
                    probs = probs[:, :3]
            else:
                hidden = out.last_hidden_state
                probs = head(hidden).squeeze(0).cpu().numpy()  # (L, 3)

            # Real windows carry `context` bp of unlabelled flank on each side; score only the labelled block.
            ctx = int(row.get("context") or 0)
            probs = probs[ctx: ctx + len(labels)]
            all_y_true.extend(labels)
            all_y_probs.append(probs)

    y_true = np.array(all_y_true, dtype=int)
    y_probs = np.vstack(all_y_probs)

    # Compute metrics
    report = compute_splice_metrics(y_true, y_probs)
    print("\n" + report.summary() + "\n")

    # Export telemetry JSON
    out_json = Path(args.output_telemetry)
    report.save_json(out_json)
    print(f"✓ Telemetry saved to: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

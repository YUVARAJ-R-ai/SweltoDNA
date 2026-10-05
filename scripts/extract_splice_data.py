#!/usr/bin/env python3
"""
CLI script to extract, filter, and partition ClinVar splice variants
and SpliceAI-10k benchmark splits into Parquet format with zero chromosome leakage.
"""

import argparse
from pathlib import Path
import sys

from svelto_dna.data.clinvar import (
    ClinVarFilterConfig,
    ClinVarParser,
    generate_synthetic_clinvar_dataset,
)
from svelto_dna.data.leakage import (
    DEFAULT_TEST_CHROMS,
    DEFAULT_TRAIN_CHROMS,
    DEFAULT_VAL_CHROMS,
    partition_by_chromosomes,
    verify_zero_leakage,
)
from svelto_dna.data.spliceai import (
    SpliceAIDatasetConfig,
    SpliceAIParser,
    generate_synthetic_spliceai_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and partition ClinVar and SpliceAI-10k datasets into Parquet."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to store processed Parquet splits.",
    )
    parser.add_argument(
        "--clinvar-input",
        type=str,
        default=None,
        help="Path to ClinVar TSV/VCF file (if omitted or not found, generates synthetic dataset).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Flanking context window size L (e.g. 1000, 2000, 5000, 10000).",
    )
    parser.add_argument(
        "--n-synthetic-samples",
        type=int,
        default=300,
        help="Number of samples to generate in synthetic mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset generation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Svelto-DNA Splice Data Extraction Pipeline ===")
    print(f"Output directory : {out_dir}")
    print(f"Window size (L)  : {args.window_size} bp")

    # 1. ClinVar Extraction
    clinvar_parser = ClinVarParser(ClinVarFilterConfig(max_junction_distance=50, snv_only=True))
    if args.clinvar_input and Path(args.clinvar_input).exists():
        print(f"Parsing ClinVar file from: {args.clinvar_input}")
        if str(args.clinvar_input).endswith(".tsv") or str(args.clinvar_input).endswith(".txt"):
            clinvar_df = clinvar_parser.parse_tsv(args.clinvar_input)
        else:
            clinvar_df = clinvar_parser.parse_vcf(args.clinvar_input)
    else:
        print(f"No local ClinVar file provided. Generating {args.n_synthetic_samples} synthetic ClinVar SNVs (seed={args.seed})...")
        clinvar_df = generate_synthetic_clinvar_dataset(n_samples=args.n_synthetic_samples, seed=args.seed)

    clinvar_parquet = out_dir / "clinvar_splice_variants.parquet"
    clinvar_parser.save_to_parquet(clinvar_df, clinvar_parquet)
    print(f"✓ Saved ClinVar Parquet: {clinvar_parquet} ({len(clinvar_df)} records)")
    print(f"  - Pathogenic: {clinvar_df['is_pathogenic'].sum()} | Benign: {clinvar_df['is_benign'].sum()} | Holdout VUS: {clinvar_df['is_holdout'].sum()}")

    # 2. SpliceAI Ingestion & Partitioning
    print(f"\nGenerating SpliceAI dataset with window_size={args.window_size}...")
    spliceai_df = generate_synthetic_spliceai_dataset(
        n_genes=max(30, args.n_synthetic_samples // 10),
        window_size=args.window_size,
        seed=args.seed,
    )

    # Partition by chromosomes
    train_df, val_df, test_df = partition_by_chromosomes(
        spliceai_df,
        test_chroms=DEFAULT_TEST_CHROMS,
        val_chroms=DEFAULT_VAL_CHROMS,
        train_chroms=DEFAULT_TRAIN_CHROMS,
    )

    # Enforce zero data leakage
    print("Verifying zero chromosome data leakage across splits...")
    leakage_report = verify_zero_leakage(train_df, val_df, test_df, strict=True)
    print(f"✓ {leakage_report.summary()}")

    # Save splits to Parquet
    splice_parser = SpliceAIParser(SpliceAIDatasetConfig(window_size=args.window_size))
    train_path = out_dir / "spliceai_train.parquet"
    val_path = out_dir / "spliceai_val.parquet"
    test_path = out_dir / "spliceai_test.parquet"

    splice_parser.save_to_parquet(train_df, train_path)
    splice_parser.save_to_parquet(val_df, val_path)
    splice_parser.save_to_parquet(test_df, test_path)

    print(f"✓ Saved SpliceAI Train: {train_path} ({len(train_df)} transcripts)")
    print(f"✓ Saved SpliceAI Val  : {val_path} ({len(val_df)} transcripts)")
    print(f"✓ Saved SpliceAI Test : {test_path} ({len(test_df)} transcripts)")
    print("\nDataset extraction and verification complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

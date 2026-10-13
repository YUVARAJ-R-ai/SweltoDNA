#!/usr/bin/env python3
"""
Extract ClinVar splice-proximal SNVs and SpliceAI-style training windows into Parquet, partitioned by
chromosome with zero leakage.

Real mode (needs --gtf and --genome-dir):
  GENCODE GTF → canonical protein-coding transcripts → splice sites;
  GRCh38 per-chromosome FASTA → transcript-oriented windows;
  ClinVar VCF/TSV → SNVs within ±50 bp of a real splice site.
Without them the script generates the synthetic demo data and says so.
"""

import argparse
from pathlib import Path
import sys

import polars as pl

from svelto_dna.data.clinvar import ClinVarFilterConfig, ClinVarParser, generate_synthetic_clinvar_dataset
from svelto_dna.data.leakage import DEFAULT_TEST_CHROMS, DEFAULT_TRAIN_CHROMS, DEFAULT_VAL_CHROMS, partition_by_chromosomes, verify_zero_leakage
from svelto_dna.data.spliceai import SpliceAIDatasetConfig, SpliceAIParser, build_splice_windows, generate_synthetic_spliceai_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--gtf", default=None, help="GENCODE GTF (.gtf or .gtf.gz), e.g. gencode.v46.basic.annotation.gtf.gz")
    p.add_argument("--genome-dir", default=None, help="Directory of Ensembl Homo_sapiens.GRCh38.dna.chromosome.N.fa.gz")
    p.add_argument("--clinvar-input", default=None, help="ClinVar GRCh38 VCF(.gz) or variant_summary TSV")
    p.add_argument("--chroms", default=",".join(str(i) for i in range(1, 23)), help="Comma-separated autosomes to process")
    p.add_argument("--block", type=int, default=1000, help="Labelled block length (bp)")
    p.add_argument("--context", type=int, default=1000, help="Flank on each side of the block (bp)")
    p.add_argument("--background-blocks", type=int, default=1, help="Site-free blocks kept per transcript")
    p.add_argument("--window-size", type=int, default=1000, help="Synthetic mode only")
    p.add_argument("--n-synthetic-samples", type=int, default=300, help="Synthetic mode only")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    real = bool(args.gtf and args.genome_dir)
    chroms = [f"chr{c.strip().removeprefix('chr')}" for c in args.chroms.split(",") if c.strip()]
    print(f"=== Svelto-DNA splice data extraction ({'REAL: GENCODE + GRCh38' if real else 'SYNTHETIC demo data'}) ===")

    sites = None
    if real:
        from svelto_dna.data.annotation import load_canonical_transcripts, splice_site_index
        from svelto_dna.data.genome import Genome
        transcripts = load_canonical_transcripts(args.gtf, chroms=chroms)
        sites = splice_site_index(transcripts)
        print(f"Canonical protein-coding transcripts: {len(transcripts):,} on {len({t.chrom for t in transcripts})} chromosomes")
        spliceai_df = build_splice_windows(Genome(args.genome_dir), transcripts, block=args.block, context=args.context,
                                           background_blocks=args.background_blocks, seed=args.seed)
    else:
        spliceai_df = generate_synthetic_spliceai_dataset(n_genes=max(30, args.n_synthetic_samples // 10), window_size=args.window_size, seed=args.seed)

    # ClinVar
    parser = ClinVarParser(ClinVarFilterConfig(max_junction_distance=50, snv_only=True))
    if args.clinvar_input and Path(args.clinvar_input).exists():
        src = args.clinvar_input
        clinvar_df = parser.parse_tsv(src, sites=sites) if src.endswith((".tsv", ".txt", ".txt.gz")) else parser.parse_vcf(src, sites=sites)
        clinvar_df = clinvar_df.filter(pl.col("chrom").is_in(chroms))
    elif real:
        clinvar_df = None
        print("No --clinvar-input given; skipping ClinVar.")
    else:
        clinvar_df = generate_synthetic_clinvar_dataset(n_samples=args.n_synthetic_samples, seed=args.seed)
    if clinvar_df is not None:
        path = parser.save_to_parquet(clinvar_df, out / "clinvar_splice_variants.parquet")
        print(f"✓ ClinVar splice-proximal SNVs: {len(clinvar_df):,} → {path}")
        print(f"  pathogenic {clinvar_df['is_pathogenic'].sum():,} | benign {clinvar_df['is_benign'].sum():,} | holdout (VUS/conflicting) {clinvar_df['is_holdout'].sum():,}")

    # Splits with zero chromosome and gene leakage
    train_df, val_df, test_df = partition_by_chromosomes(spliceai_df, DEFAULT_TEST_CHROMS, DEFAULT_VAL_CHROMS, DEFAULT_TRAIN_CHROMS)
    print(verify_zero_leakage(train_df, val_df, test_df, strict=True).summary())
    sp = SpliceAIParser(SpliceAIDatasetConfig(window_size=args.window_size))
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = sp.save_to_parquet(df, out / f"spliceai_{name}.parquet")
        n_sites = int(df["donor_count"].sum() + df["acceptor_count"].sum()) if len(df) else 0
        print(f"✓ {name:5s}: {len(df):,} windows, {n_sites:,} splice sites → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

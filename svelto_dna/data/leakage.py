"""
Chromosome-level data leakage prevention and split verification module.
Enforces zero data leakage between training, validation, and evaluation splits
across homologous chromosome groups according to canonical SpliceAI benchmark partitioning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import polars as pl
from svelto_dna.data.clinvar import normalize_chrom

# Canonical SpliceAI autosome benchmark splits
DEFAULT_TEST_CHROMS: Tuple[str, ...] = ("chr1", "chr3", "chr5", "chr7", "chr9")
DEFAULT_VAL_CHROMS: Tuple[str, ...] = ("chr2", "chr4", "chr6", "chr8", "chr10")
DEFAULT_TRAIN_CHROMS: Tuple[str, ...] = tuple(f"chr{i}" for i in range(11, 23))


class DataLeakageError(Exception):
    """Raised when data leakage between dataset splits is detected."""
    pass


@dataclass
class LeakageReport:
    """Report detailing leakage verification across dataset splits."""

    is_leak_free: bool
    train_chromosomes: List[str]
    val_chromosomes: List[str]
    test_chromosomes: List[str]
    overlapping_chromosomes: Dict[str, List[str]] = field(default_factory=dict)
    overlapping_genes: Dict[str, List[str]] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED (Zero Leakage)" if self.is_leak_free else "FAILED (Data Leakage Detected)"
        lines = [
            f"Chromosome Leakage Verification: {status}",
            f"  Train Chromosomes ({len(self.train_chromosomes)}): {', '.join(sorted(self.train_chromosomes))}",
            f"  Val Chromosomes   ({len(self.val_chromosomes)}): {', '.join(sorted(self.val_chromosomes))}",
            f"  Test Chromosomes  ({len(self.test_chromosomes)}): {', '.join(sorted(self.test_chromosomes))}",
        ]
        if self.overlapping_chromosomes:
            for pair, chroms in self.overlapping_chromosomes.items():
                if chroms:
                    lines.append(f"  [ERROR] Chromosome overlap between {pair}: {chroms}")
        if self.overlapping_genes:
            for pair, genes in self.overlapping_genes.items():
                if genes:
                    lines.append(f"  [ERROR] Gene overlap between {pair}: {len(genes)} genes (sample: {genes[:5]})")
        return "\n".join(lines)


def get_unique_chromosomes(df: pl.DataFrame, col: str = "chrom") -> Set[str]:
    """Extracts and normalizes unique chromosome strings from a DataFrame."""
    if col not in df.columns:
        return set()
    raw_chroms = df[col].drop_nulls().unique().to_list()
    return {normalize_chrom(str(c)) for c in raw_chroms}


def get_unique_genes(df: pl.DataFrame, col: str = "gene_id") -> Set[str]:
    """Extracts unique gene symbols from a DataFrame."""
    if col not in df.columns:
        return set()
    return {str(g).strip().upper() for g in df[col].drop_nulls().unique().to_list() if str(g).strip()}


def verify_zero_leakage(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    chrom_col: str = "chrom",
    gene_col: str = "gene_id",
    strict: bool = True,
) -> LeakageReport:
    """
    Verifies that train, validation, and test splits have completely disjoint
    chromosome groups and gene identifiers.

    Args:
        train_df: Training partition Polars DataFrame.
        val_df: Validation partition Polars DataFrame.
        test_df: Test partition Polars DataFrame.
        chrom_col: Column name containing chromosome identifier.
        gene_col: Column name containing gene identifier.
        strict: If True, raises DataLeakageError upon detecting any overlap.

    Returns:
        LeakageReport with detailed overlap findings.
    """
    train_chroms = get_unique_chromosomes(train_df, chrom_col)
    val_chroms = get_unique_chromosomes(val_df, chrom_col)
    test_chroms = get_unique_chromosomes(test_df, chrom_col)

    # Check chromosome pairwise intersections
    chrom_train_val = sorted(list(train_chroms & val_chroms))
    chrom_train_test = sorted(list(train_chroms & test_chroms))
    chrom_val_test = sorted(list(val_chroms & test_chroms))

    overlapping_chroms = {}
    if chrom_train_val:
        overlapping_chroms["train_val"] = chrom_train_val
    if chrom_train_test:
        overlapping_chroms["train_test"] = chrom_train_test
    if chrom_val_test:
        overlapping_chroms["val_test"] = chrom_val_test

    # Check gene pairwise intersections if gene_col is present
    train_genes = get_unique_genes(train_df, gene_col)
    val_genes = get_unique_genes(val_df, gene_col)
    test_genes = get_unique_genes(test_df, gene_col)

    gene_train_val = sorted(list(train_genes & val_genes))
    gene_train_test = sorted(list(train_genes & test_genes))
    gene_val_test = sorted(list(val_genes & test_genes))

    overlapping_genes = {}
    if gene_train_val:
        overlapping_genes["train_val"] = gene_train_val
    if gene_train_test:
        overlapping_genes["train_test"] = gene_train_test
    if gene_val_test:
        overlapping_genes["val_test"] = gene_val_test

    is_leak_free = len(overlapping_chroms) == 0 and len(overlapping_genes) == 0

    report = LeakageReport(
        is_leak_free=is_leak_free,
        train_chromosomes=sorted(list(train_chroms)),
        val_chromosomes=sorted(list(val_chroms)),
        test_chromosomes=sorted(list(test_chroms)),
        overlapping_chromosomes=overlapping_chroms,
        overlapping_genes=overlapping_genes,
    )

    if strict and not is_leak_free:
        raise DataLeakageError(f"Data leakage detected:\n{report.summary()}")

    return report


def partition_by_chromosomes(
    df: pl.DataFrame,
    test_chroms: Sequence[str] = DEFAULT_TEST_CHROMS,
    val_chroms: Sequence[str] = DEFAULT_VAL_CHROMS,
    train_chroms: Sequence[str] = DEFAULT_TRAIN_CHROMS,
    chrom_col: str = "chrom",
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits a DataFrame into (train_df, val_df, test_df) strictly according
    to chromosome assignments.
    """
    # Normalize input partition chromosome lists
    norm_test = {normalize_chrom(c) for c in test_chroms}
    norm_val = {normalize_chrom(c) for c in val_chroms}
    norm_train = {normalize_chrom(c) for c in train_chroms}

    # Ensure partitions themselves are disjoint
    assert norm_train.isdisjoint(norm_val), "Partition config error: train and val chroms overlap"
    assert norm_train.isdisjoint(norm_test), "Partition config error: train and test chroms overlap"
    assert norm_val.isdisjoint(norm_test), "Partition config error: val and test chroms overlap"

    # Add normalized chrom column if needed
    norm_df = df.with_columns(
        pl.col(chrom_col).map_elements(normalize_chrom, return_dtype=pl.String).alias("__norm_chrom")
    )

    train_df = norm_df.filter(pl.col("__norm_chrom").is_in(list(norm_train))).drop("__norm_chrom")
    val_df = norm_df.filter(pl.col("__norm_chrom").is_in(list(norm_val))).drop("__norm_chrom")
    test_df = norm_df.filter(pl.col("__norm_chrom").is_in(list(norm_test))).drop("__norm_chrom")

    return train_df, val_df, test_df

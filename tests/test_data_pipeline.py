"""
Unit tests for ClinVar and SpliceAI data pipelines, strand coordinate resolution,
and zero chromosome leakage verification.
"""

from pathlib import Path
import tempfile
import polars as pl
import pytest

from svelto_dna.core.tokenizer import GenomicTokenizer
from svelto_dna.data.clinvar import (
    ClinVarFilterConfig,
    ClinVarParser,
    generate_synthetic_clinvar_dataset,
    normalize_chrom,
)
from svelto_dna.data.leakage import (
    DEFAULT_TEST_CHROMS,
    DEFAULT_TRAIN_CHROMS,
    DEFAULT_VAL_CHROMS,
    DataLeakageError,
    partition_by_chromosomes,
    verify_zero_leakage,
)
from svelto_dna.data.spliceai import (
    LABEL_ACCEPTOR,
    LABEL_DONOR,
    LABEL_NEITHER,
    SpliceAIDatasetConfig,
    SpliceAIParser,
    StrandCoordinateResolver,
    generate_synthetic_spliceai_dataset,
)


class TestClinVarPipeline:
    """Tests for ClinVar variant extraction, proximity filtering, and VUS partitioning."""

    def test_normalize_chrom(self) -> None:
        assert normalize_chrom("1") == "chr1"
        assert normalize_chrom("chr1") == "chr1"
        assert normalize_chrom("CHR22") == "chr22"
        assert normalize_chrom("X") == "chrX"
        assert normalize_chrom("chrx") == "chrX"

    def test_synthetic_clinvar_generation_and_schema(self) -> None:
        df = generate_synthetic_clinvar_dataset(n_samples=50, seed=42)
        assert len(df) == 50

        expected_cols = {
            "variant_id",
            "chrom",
            "pos",
            "ref",
            "alt",
            "strand",
            "gene_id",
            "junction_type",
            "dist_to_junction",
            "clinical_significance",
            "is_pathogenic",
            "is_benign",
            "is_holdout",
        }
        assert expected_cols.issubset(set(df.columns))

        # Check SNV invariant (length 1 ref and alt)
        assert all(len(r) == 1 for r in df["ref"].to_list())
        assert all(len(a) == 1 for a in df["alt"].to_list())

        # Check proximity invariant (+- 50 bp)
        assert all(abs(d) <= 50 for d in df["dist_to_junction"].to_list())

    def test_vus_holdout_partitioning(self) -> None:
        parser = ClinVarParser()

        # Pathogenic
        p_path, p_ben, p_hold = parser.classify_significance("Pathogenic")
        assert p_path is True and p_ben is False and p_hold is False

        # Benign
        b_path, b_ben, b_hold = parser.classify_significance("Likely benign")
        assert b_path is False and b_ben is True and b_hold is False

        # VUS / Holdout
        v_path, v_ben, v_hold = parser.classify_significance("Uncertain significance")
        assert v_path is False and v_ben is False and v_hold is True

        # Conflicting interpretations -> holdout
        c_path, c_ben, c_hold = parser.classify_significance("Conflicting interpretations of pathogenicity")
        assert c_path is False and c_ben is False and c_hold is True

    def test_filtering_logic_snv_and_distance(self) -> None:
        parser = ClinVarParser(ClinVarFilterConfig(max_junction_distance=50, snv_only=True))

        test_data = pl.DataFrame({
            "variant_id": ["VAR1", "VAR2", "VAR3", "VAR4"],
            "chrom": ["1", "chr2", "3", "4"],
            "pos": [1000, 2000, 3000, 4000],
            "ref": ["A", "ATG", "C", "G"],  # VAR2 is an indel (ref='ATG')
            "alt": ["G", "A", "T", "C"],
            "dist_to_junction": [10, 20, 80, -45],  # VAR3 exceeds +-50 bp
            "clinical_significance": ["Pathogenic", "Benign", "Likely pathogenic", "Uncertain significance"],
        })

        filtered = parser.filter_and_format(test_data)
        var_ids = filtered["variant_id"].to_list()

        assert "VAR1" in var_ids  # Valid SNV within distance
        assert "VAR2" not in var_ids  # Dropped because indel
        assert "VAR3" not in var_ids  # Dropped because distance 80 > 50
        assert "VAR4" in var_ids  # Valid VUS holdout within distance

    def test_parquet_roundtrip(self) -> None:
        df = generate_synthetic_clinvar_dataset(n_samples=20, seed=42)
        parser = ClinVarParser()

        with tempfile.TemporaryDirectory() as tmpdir:
            pq_path = Path(tmpdir) / "test_clinvar.parquet"
            parser.save_to_parquet(df, pq_path)
            assert pq_path.exists()

            loaded = parser.load_from_parquet(pq_path)
            assert len(loaded) == len(df)
            assert loaded.schema == df.schema


class TestStrandCoordinateResolver:
    """Tests for strand polarity and coordinate inversion."""

    def test_positive_strand_resolution(self) -> None:
        tokenizer = GenomicTokenizer()
        resolver = StrandCoordinateResolver(tokenizer)

        # 21-bp reference sequence with GT donor at index 8-9 and AG acceptor at index 14-15
        seq = "AAAAAAAA" + "GT" + "CCCC" + "AG" + "TTTTT"
        locus = 10  # in the middle of sequence
        w_size = 15

        oriented_seq, win_locus, labels = resolver.resolve_window_and_labels(
            sequence=seq,
            locus_idx=locus,
            window_size=w_size,
            strand="+",
            donor_positions=[8],
            acceptor_positions=[14],
        )

        assert len(oriented_seq) == w_size
        assert len(labels) == w_size
        # Left flank = (15 - 1) // 2 = 7. Window starts at 10 - 7 = 3.
        # Donor at 8 -> window index = 8 - 3 = 5.
        # Acceptor at 14 -> window index = 14 - 3 = 11.
        assert labels[5] == LABEL_DONOR
        assert labels[11] == LABEL_ACCEPTOR
        assert oriented_seq[5:7] == "GT"
        assert oriented_seq[11:13] == "AG"

    def test_negative_strand_resolution_biological_polarity(self) -> None:
        tokenizer = GenomicTokenizer()
        resolver = StrandCoordinateResolver(tokenizer)

        # On the negative strand, a transcript reading 5'->3' with a GT donor corresponds
        # to AC (reverse complement) on the reference sequence!
        # Reference: "TTTTT" + "AC" + "GGGG" + "CT" + "AAAAA"
        seq = "TTTTTACGGGGCTAAAAA"
        locus = 9
        w_size = 15

        oriented_seq, win_locus, labels = resolver.resolve_window_and_labels(
            sequence=seq,
            locus_idx=locus,
            window_size=w_size,
            strand="-",
            donor_positions=[5],  # index of 'AC' on reference
            acceptor_positions=[11],  # index of 'CT' on reference
        )

        assert len(oriented_seq) == w_size
        assert len(labels) == w_size

        # In the oriented_seq (which is reverse complemented), the donor motif MUST become 'GT'
        donor_idx = labels.index(LABEL_DONOR)
        assert oriented_seq[donor_idx : donor_idx + 2] == "GT"

        acceptor_idx = labels.index(LABEL_ACCEPTOR)
        assert oriented_seq[acceptor_idx : acceptor_idx + 2] == "AG"


class TestChromosomeLeakagePrevention:
    """Tests for chromosome split disjointness and zero leakage enforcement."""

    def test_valid_disjoint_partitions(self) -> None:
        df = generate_synthetic_spliceai_dataset(n_genes=25, window_size=1000, seed=42)

        train_df, val_df, test_df = partition_by_chromosomes(
            df,
            test_chroms=DEFAULT_TEST_CHROMS,
            val_chroms=DEFAULT_VAL_CHROMS,
            train_chroms=DEFAULT_TRAIN_CHROMS,
        )

        report = verify_zero_leakage(train_df, val_df, test_df, strict=True)
        assert report.is_leak_free is True
        assert len(report.overlapping_chromosomes) == 0

    def test_leakage_detection_raises_error(self) -> None:
        df = generate_synthetic_spliceai_dataset(n_genes=10, window_size=1000, seed=42)

        # Intentionally inject overlapping chromosome 'chr1' in both train and test
        leaky_train = df.filter(pl.col("chrom") == "chr1")
        leaky_test = df.filter(pl.col("chrom") == "chr1")
        val = df.filter(pl.col("chrom") == "chr2")

        with pytest.raises(DataLeakageError):
            verify_zero_leakage(leaky_train, val, leaky_test, strict=True)

"""
SpliceAI-10k dataset ingestion, strand-aware coordinate resolution,
and symmetrical context window extraction for genomic splice benchmarking.
"""

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import polars as pl
from svelto_dna.core.tokenizer import GenomicTokenizer
from svelto_dna.data.clinvar import normalize_chrom

LABEL_NEITHER: int = 0
LABEL_DONOR: int = 1
LABEL_ACCEPTOR: int = 2


@dataclass
class SpliceAIDatasetConfig:
    """Configuration for SpliceAI dataset processing."""

    window_size: int = 1000  # Default context window length (1k - 10k bp)
    flank_size: Optional[int] = None
    pad_char: str = "N"

    def __post_init__(self) -> None:
        if self.flank_size is None:
            self.flank_size = (self.window_size - 1) // 2


class StrandCoordinateResolver:
    """
    Resolves genomic coordinate polarity and strand inversion.
    Ensures extracted sequence windows and splice site labels strictly match
    the biological 5' -> 3' transcript orientation on both positive (+)
    and negative (-) strands.
    """

    def __init__(self, tokenizer: Optional[GenomicTokenizer] = None) -> None:
        self.tokenizer = tokenizer or GenomicTokenizer()

    def resolve_window(
        self,
        sequence: str,
        locus_idx: int,
        window_size: int,
        strand: str = "+",
    ) -> Tuple[str, int]:
        """
        Extracts a symmetrical window and orients it relative to transcript strand.

        Args:
            sequence: Genomic sequence on reference (+) strand.
            locus_idx: 0-indexed position of locus on reference strand.
            window_size: Desired window length.
            strand: Transcript strand ('+' or '-').

        Returns:
            Tuple of (oriented_sequence, locus_idx_in_window)
        """
        # Symmetrically extract window centered at locus_idx
        ref_window = self.tokenizer.extract_context_window(
            sequence=sequence,
            locus_idx=locus_idx,
            window_size=window_size,
        )

        left_flank = (window_size - 1) // 2
        # In the extracted ref_window, the locus is at index `left_flank`
        center_in_window = left_flank

        if strand == "-":
            # For negative strand, compute reverse complement
            oriented_window = self.tokenizer.reverse_complement(ref_window)
            # In reverse-complemented window of length L, position i becomes (L - 1 - i)
            locus_in_window = window_size - 1 - center_in_window
            return oriented_window, locus_in_window

        return ref_window, center_in_window

    def resolve_window_and_labels(
        self,
        sequence: str,
        locus_idx: int,
        window_size: int,
        strand: str,
        donor_positions: Sequence[int],
        acceptor_positions: Sequence[int],
    ) -> Tuple[str, int, List[int]]:
        """
        Extracts window and generates single-nucleotide ground truth label array:
        0 = Neither, 1 = Donor, 2 = Acceptor.
        Correctly accounts for negative-strand coordinate reflection.

        Args:
            sequence: Reference genomic sequence.
            locus_idx: 0-indexed position of the queried variant locus.
            window_size: Total window length L.
            strand: '+' or '-'.
            donor_positions: 0-indexed positions of donor sites on reference sequence.
            acceptor_positions: 0-indexed positions of acceptor sites on reference sequence.

        Returns:
            Tuple of (oriented_sequence, locus_idx_in_window, labels_array)
        """
        left_flank = (window_size - 1) // 2
        window_start = locus_idx - left_flank
        window_end = window_start + window_size
        oriented_window, locus_in_window = self.resolve_window(
            sequence=sequence,
            locus_idx=locus_idx,
            window_size=window_size,
            strand=strand,
        )

        oriented_labels = [LABEL_NEITHER] * window_size

        if strand == "-":
            # On the negative strand, canonical dinucleotide motifs (GT donor, AG acceptor)
            # appear as AC and CT on the reference strand spanning [p, p+1].
            # In biological 5' -> 3' transcript orientation, the 5' base corresponds to p+1 on reference.
            # In the reverse-complemented window of length L, reference position p maps to
            # index L - 1 - (p + 1 - window_start) = L - 2 - (p - window_start).
            for d in donor_positions:
                oriented_d = window_size - 2 - (d - window_start)
                if 0 <= oriented_d < window_size:
                    oriented_labels[oriented_d] = LABEL_DONOR

            for a in acceptor_positions:
                oriented_a = window_size - 2 - (a - window_start)
                if 0 <= oriented_a < window_size:
                    oriented_labels[oriented_a] = LABEL_ACCEPTOR
        else:
            for d in donor_positions:
                if window_start <= d < window_end:
                    oriented_labels[d - window_start] = LABEL_DONOR

            for a in acceptor_positions:
                if window_start <= a < window_end:
                    oriented_labels[a - window_start] = LABEL_ACCEPTOR

        return oriented_window, locus_in_window, oriented_labels


class SpliceAIParser:
    """
    Parses and builds SpliceAI-10k dataset records into Parquet format.
    """

    def __init__(
        self,
        config: Optional[SpliceAIDatasetConfig] = None,
        tokenizer: Optional[GenomicTokenizer] = None,
    ) -> None:
        self.config = config or SpliceAIDatasetConfig()
        self.tokenizer = tokenizer or GenomicTokenizer()
        self.resolver = StrandCoordinateResolver(self.tokenizer)

    def process_records(
        self,
        records: Sequence[Dict[str, Union[str, int, float]]],
    ) -> pl.DataFrame:
        """
        Converts raw transcript records into a standardized Polars DataFrame.
        """
        processed = []
        for r in records:
            chrom = normalize_chrom(str(r["chrom"]))
            strand = str(r.get("strand", "+"))
            seq = str(r["sequence"])
            locus_idx = int(r["locus_idx"])
            donors = [int(x) for x in r.get("donor_positions", [])]  # type: ignore
            acceptors = [int(x) for x in r.get("acceptor_positions", [])]  # type: ignore

            win_seq, win_locus, labels = self.resolver.resolve_window_and_labels(
                sequence=seq,
                locus_idx=locus_idx,
                window_size=self.config.window_size,
                strand=strand,
                donor_positions=donors,
                acceptor_positions=acceptors,
            )

            donor_count = labels.count(LABEL_DONOR)
            acceptor_count = labels.count(LABEL_ACCEPTOR)

            processed.append({
                "gene_id": str(r.get("gene_id", "UNKNOWN")),
                "transcript_id": str(r.get("transcript_id", "UNKNOWN")),
                "chrom": chrom,
                "strand": strand,
                "locus_idx": locus_idx,
                "locus_in_window": win_locus,
                "window_size": self.config.window_size,
                "window_sequence": win_seq,
                "donor_count": donor_count,
                "acceptor_count": acceptor_count,
                # Store label array as list of ints
                "labels": labels,
            })

        return pl.DataFrame(processed)

    def save_to_parquet(self, df: pl.DataFrame, output_path: Union[str, Path]) -> Path:
        """Saves processed SpliceAI DataFrame to Parquet format."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out, compression="zstd")
        return out

    def load_from_parquet(self, input_path: Union[str, Path]) -> pl.DataFrame:
        """Loads SpliceAI Parquet file."""
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        return pl.read_parquet(path)


def generate_synthetic_spliceai_dataset(
    n_genes: int = 25,
    window_size: int = 1000,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generates a deterministic synthetic SpliceAI-10k style dataset
    with realistic exon-intron canonical motifs (GT donor, AG acceptor),
    both positive (+) and negative (-) strand genes, spanning autosomes chr1 - chr22.
    """
    rng = random.Random(seed)
    autosomes = [f"chr{i}" for i in range(1, 23)]
    bases = ["A", "C", "G", "T"]

    records = []
    for g_idx in range(n_genes):
        gene_id = f"GENE_{g_idx+1:04d}"
        tx_id = f"ENST_{g_idx+1:08d}"
        chrom = autosomes[g_idx % len(autosomes)]
        strand = "+" if (g_idx % 2 == 0) else "-"

        # Create a synthetic gene sequence of length >= window_size + 400
        gene_len = window_size + 500
        seq_chars = [rng.choice(bases) for _ in range(gene_len)]

        # Place canonical donor (GT) and acceptor (AG)
        # For positive strand: GT donor, AG acceptor
        # For negative strand: AC (compl. of GT) and CT (compl. of AG) on reference
        donor_idx = gene_len // 3
        acceptor_idx = (2 * gene_len) // 3

        if strand == "+":
            seq_chars[donor_idx : donor_idx + 2] = ["G", "T"]
            seq_chars[acceptor_idx : acceptor_idx + 2] = ["A", "G"]
        else:
            # On reference strand, negative transcript donor is reverse-compl of GT -> AC
            seq_chars[donor_idx : donor_idx + 2] = ["A", "C"]
            seq_chars[acceptor_idx : acceptor_idx + 2] = ["C", "T"]

        full_seq = "".join(seq_chars)

        # Locus centered near splice site
        locus_idx = donor_idx + rng.randint(-30, 30)

        records.append({
            "gene_id": gene_id,
            "transcript_id": tx_id,
            "chrom": chrom,
            "strand": strand,
            "sequence": full_seq,
            "locus_idx": locus_idx,
            "donor_positions": [donor_idx],
            "acceptor_positions": [acceptor_idx],
        })

    config = SpliceAIDatasetConfig(window_size=window_size)
    parser = SpliceAIParser(config=config)
    return parser.process_records(records)

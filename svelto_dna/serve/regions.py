"""
Regions a server session scores. `synthetic_region` is a byte-exact port of the frontend's
web/lib/splice/sequence.ts so the existing UI and the server agree on every base.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

BASES = "ACGT"


@dataclass
class Region:
    seq: List[str]
    exons: List[Tuple[int, int]]
    first_exon_number: int
    feature: int
    coord0: int
    chrom: str
    gene: str
    synthetic: bool
    description: str = ""

    def to_wire(self) -> Dict[str, Any]:
        return {"length": len(self.seq), "chrom": self.chrom, "gene": self.gene, "coord0": self.coord0, "feature": self.feature,
                "exons": [list(e) for e in self.exons], "first_exon_number": self.first_exon_number, "synthetic": self.synthetic,
                "description": self.description, "sequence": "".join(self.seq)}


def synthetic_region() -> Region:
    seed = 1234

    def rnd() -> float:
        nonlocal seed
        seed = (seed * 16807) % 2147483647
        return seed / 2147483647

    seq = [BASES[int(rnd() * 4)] for _ in range(10000)]

    def plant(at: int, s: str) -> None:
        for k, ch in enumerate(s):
            seq[at + k] = ch

    exons = [(620, 760), (1500, 1640), (2400, 2510), (3300, 3460), (4100, 4240), (4870, 5000),
             (5800, 5950), (6700, 6830), (7600, 7790), (8500, 8620), (9300, 9420)]
    for s, e in exons:
        plant(s - 18, "TTTCTTTTCCCTTTTCAG")
        plant(e - 3, "CAGGTAAGT")
    plant(5000 + 17 - 3, "TTGGTGAGC")
    return Region(seq, exons, 5, 5000, 45960000, "chr17", "MAPT", True, "Synthetic 10 kb demo region (matches the web app)")


def genome_region(gtf: str, genome_dir: str, gene: str, exon_number: int, span: int = 10000) -> Region:
    """Real GRCh38 window of `span` bp centred on the donor of exon `exon_number` of the gene's canonical transcript (+ strand genes)."""
    from svelto_dna.data.annotation import load_canonical_transcripts
    from svelto_dna.data.genome import Genome
    tx = next((t for t in load_canonical_transcripts(gtf) if t.gene_name == gene), None)
    if tx is None:
        raise ValueError(f"No canonical protein-coding transcript for {gene}")
    if tx.strand != "+":
        raise ValueError(f"{gene} is on the - strand; only + strand regions are supported by the viewer")
    donor = tx.exons[exon_number - 1][1]
    start = max(0, donor - span // 2)
    seq = list(Genome(genome_dir).fetch(tx.chrom, start, start + span))
    exons = [(max(0, s - start), min(span, e - start)) for s, e in tx.exons if e > start and s < start + span]
    first = next(i for i, (s, e) in enumerate(tx.exons) if e > start) + 1
    return Region(seq, exons, first, donor - start, start + 1, tx.chrom, gene, False,
                  f"GRCh38 {tx.chrom}:{start + 1:,}-{start + span:,}, {gene} {tx.transcript_id}, exon {exon_number} donor centred")

"""
Gene annotation (GENCODE GTF) → canonical transcripts and splice-site coordinates.

Convention (shared with StrandCoordinateResolver): a splice site is stored as the 0-based reference
position of the dinucleotide's first base on the reference + strand.
  + strand: donor "GT" at [p, p+1] (p = exon end), acceptor "AG" at [p, p+1] (p = next exon start - 2)
  - strand: donor appears as "AC" at [p, p+1] (p = exon start - 2), acceptor as "CT" at [p, p+1] (p = exon end)
"""

from bisect import bisect_left
from dataclasses import dataclass
import gzip
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from svelto_dna.data.clinvar import normalize_chrom

_ATTR = re.compile(r'(\S+) "([^"]*)"')


@dataclass(frozen=True)
class Transcript:
    transcript_id: str
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    exons: Tuple[Tuple[int, int], ...]   # 0-based half-open, sorted by genomic position

    @property
    def start(self) -> int:
        return self.exons[0][0]

    @property
    def end(self) -> int:
        return self.exons[-1][1]

    def donors(self) -> List[int]:
        if self.strand == "+":
            return [e for _, e in self.exons[:-1]]
        return [s - 2 for s, _ in self.exons[1:]]

    def acceptors(self) -> List[int]:
        if self.strand == "+":
            return [s - 2 for s, _ in self.exons[1:]]
        return [e for _, e in self.exons[:-1]]


def load_canonical_transcripts(
    gtf_path: Union[str, Path],
    gene_types: Sequence[str] = ("protein_coding",),
    tag: str = "Ensembl_canonical",
    chroms: Optional[Iterable[str]] = None,
) -> List[Transcript]:
    """One canonical transcript per gene (GENCODE `Ensembl_canonical` tag), protein-coding by default."""
    keep_chroms = {normalize_chrom(c) for c in chroms} if chroms else None
    exons: Dict[str, List[Tuple[int, int]]] = {}
    meta: Dict[str, Tuple[str, str, str, str]] = {}
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            chrom = normalize_chrom(parts[0])
            if keep_chroms is not None and chrom not in keep_chroms:
                continue
            attrs: Dict[str, List[str]] = {}
            for k, v in _ATTR.findall(parts[8]):
                attrs.setdefault(k, []).append(v)
            if tag not in attrs.get("tag", []) or attrs.get("gene_type", [""])[0] not in gene_types:
                continue
            tid = attrs["transcript_id"][0]
            exons.setdefault(tid, []).append((int(parts[3]) - 1, int(parts[4])))
            meta[tid] = (attrs["gene_id"][0], attrs.get("gene_name", [""])[0], chrom, parts[6])
    out = []
    for tid, ex in exons.items():
        gene_id, gene_name, chrom, strand = meta[tid]
        out.append(Transcript(tid, gene_id, gene_name, chrom, strand, tuple(sorted(ex))))
    return sorted(out, key=lambda t: (t.chrom, t.start))


@dataclass
class SiteIndex:
    """Splice sites on one chromosome, sorted by position, for nearest-site queries."""
    pos: np.ndarray
    kind: np.ndarray          # 0 = donor, 1 = acceptor
    strand: np.ndarray
    gene: np.ndarray

    def nearest(self, p: int) -> Optional[Tuple[int, str, str, str]]:
        if len(self.pos) == 0:
            return None
        i = bisect_left(self.pos.tolist(), p) if len(self.pos) < 64 else int(np.searchsorted(self.pos, p))
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.pos)), key=lambda j: abs(int(self.pos[j]) - p))
        return int(self.pos[best]), ("donor" if self.kind[best] == 0 else "acceptor"), str(self.strand[best]), str(self.gene[best])


def splice_site_index(transcripts: Iterable[Transcript]) -> Dict[str, SiteIndex]:
    rows: Dict[str, List[Tuple[int, int, str, str]]] = {}
    for t in transcripts:
        for p in t.donors():
            rows.setdefault(t.chrom, []).append((p, 0, t.strand, t.gene_name))
        for p in t.acceptors():
            rows.setdefault(t.chrom, []).append((p, 1, t.strand, t.gene_name))
    out = {}
    for chrom, r in rows.items():
        r.sort()
        out[chrom] = SiteIndex(np.array([x[0] for x in r], dtype=np.int64), np.array([x[1] for x in r], dtype=np.int8),
                               np.array([x[2] for x in r]), np.array([x[3] for x in r]))
    return out

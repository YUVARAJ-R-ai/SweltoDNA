"""
GRCh38 reference access from Ensembl per-chromosome FASTA (Homo_sapiens.GRCh38.dna.chromosome.{N}.fa.gz).
Each chromosome is decompressed once into an uppercase single-line cache file and read via mmap.
"""

import gzip
import mmap
from pathlib import Path
from typing import Dict, Union


class Genome:
    def __init__(self, fasta_dir: Union[str, Path], cache_dir: Union[str, Path, None] = None) -> None:
        self.fasta_dir = Path(fasta_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.fasta_dir / ".seqcache"
        self._maps: Dict[str, mmap.mmap] = {}

    @staticmethod
    def _name(chrom: str) -> str:
        c = str(chrom).strip()
        return c[3:] if c.lower().startswith("chr") else c

    def _map(self, chrom: str) -> mmap.mmap:
        name = self._name(chrom)
        if name in self._maps:
            return self._maps[name]
        cache = self.cache_dir / f"{name}.seq"
        if not cache.exists():
            src = self.fasta_dir / f"Homo_sapiens.GRCh38.dna.chromosome.{name}.fa.gz"
            if not src.exists():
                raise FileNotFoundError(f"No FASTA for chromosome {chrom}: {src}")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".tmp")
            with gzip.open(src, "rt") as f, open(tmp, "wb") as out:
                for line in f:
                    if not line.startswith(">"):
                        out.write(line.strip().upper().encode("ascii"))
            tmp.rename(cache)
        with open(cache, "rb") as fh:
            self._maps[name] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self._maps[name]

    def length(self, chrom: str) -> int:
        return len(self._map(chrom))

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Reference bases for 0-based [start, end); positions outside the chromosome are N."""
        m = self._map(chrom)
        n = len(m)
        left = max(0, -start)
        right = max(0, end - n)
        core = m[max(0, start):min(n, end)].decode("ascii") if end > 0 and start < n else ""
        return "N" * left + core + "N" * right

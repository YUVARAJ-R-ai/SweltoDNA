#!/usr/bin/env python3
"""
Builds tests/fixtures/spliceai_clinvar_ground_truth.json for issue #5.

For a deterministic set of ClinVar *Pathogenic* SNVs at canonical splice dinucleotides (both strands,
donors and acceptors), this script:
  1. runs the official SpliceAI models (5-model ensemble, as published) on the reference and alternate
     11,001 bp windows to get full per-position probability tracks (±50 bp around the variant),
  2. fetches the Broad SpliceAI-lookup published delta scores (DS_*) and positions (DP_*) for the same
     variant (GRCh38, distance 50, unmasked).

The unit test then feeds those tracks to svelto_dna.splice.delta.compute_delta_scores and checks it
reproduces the published scores. The test needs neither TensorFlow nor the network; only this
generator does. It is not part of the package dependencies:

  pip install "tensorflow-cpu==2.15.*" spliceai "setuptools<70"
  python scripts/make_spliceai_fixtures.py --clinvar data/processed_clinvar_preview.parquet \
      --gtf data/raw/gencode.v46.basic.annotation.gtf.gz --genome-dir data/raw/genome
"""

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svelto_dna.data.annotation import load_canonical_transcripts  # noqa: E402
from svelto_dna.data.genome import Genome  # noqa: E402

DIST = 50
CONTEXT = 10000
API = "https://spliceai-38-xwkwwwxdwq-uc.a.run.app/spliceai/?hg=38&variant={v}&distance=50&mask=0&raw=1"
COMP = str.maketrans("ACGTN", "TGCAN")


def one_hot(seq: str) -> np.ndarray:
    m = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq):
        j = "ACGT".find(c)
        if j >= 0:
            m[i, j] = 1
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinvar", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--per-strand", type=int, default=4)
    ap.add_argument("--out", default="tests/fixtures/spliceai_clinvar_ground_truth.json")
    a = ap.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from keras.models import load_model
    from pkg_resources import resource_filename
    models = [load_model(resource_filename("spliceai", f"models/spliceai{i}.h5"), compile=False) for i in range(1, 6)]

    tx = {t.gene_name: t for t in load_canonical_transcripts(a.gtf)}
    genome = Genome(a.genome_dir)
    df = (pl.read_parquet(a.clinvar)
          .filter((pl.col("clinical_significance") == "Pathogenic") & pl.col("dist_to_junction").is_in([0, 1]))
          .unique(subset=["chrom", "pos", "ref", "alt"]).sort(["gene_id", "pos", "alt"]))

    chosen = []
    for strand in ("+", "-"):
        genes = set()
        for r in df.filter(pl.col("strand") == strand).iter_rows(named=True):
            t = tx.get(r["gene_id"])
            p0 = r["pos"] - 1
            # keep clear of transcript ends so SpliceAI's gene-boundary N-padding never applies
            if not t or r["gene_id"] in genes or p0 - t.start < CONTEXT // 2 + DIST or t.end - p0 < CONTEXT // 2 + DIST:
                continue
            genes.add(r["gene_id"])
            chosen.append(r)
            if len(genes) == a.per_strand:
                break

    out = []
    for r in chosen:
        p0, ref, alt, strand = r["pos"] - 1, r["ref"], r["alt"], r["strand"]
        half = (CONTEXT + 2 * DIST + 1) // 2
        seq = genome.fetch(r["chrom"], p0 - half, p0 + half + 1)
        assert seq[half] == ref, (r["variant_id"], seq[half], ref)
        seqs = {"ref": seq, "alt": seq[:half] + alt + seq[half + 1:]}
        tracks = {}
        for k, s in seqs.items():
            if strand == "-":
                s = s.translate(COMP)[::-1]
            x = one_hot(s)[None]
            y = np.mean([m.predict(x, verbose=0) for m in models], axis=0)[0]   # (101, 3): neither, acceptor, donor
            if strand == "-":
                y = y[::-1]
            tracks[k] = {"donor": [round(float(v), 6) for v in y[:, 2]], "acceptor": [round(float(v), 6) for v in y[:, 1]]}
        variant = f"{r['chrom']}-{r['pos']}-{ref}-{alt}"
        with urllib.request.urlopen(API.format(v=variant), timeout=120) as resp:
            scores = json.load(resp)["scores"]
        pub = next((s for s in scores if s.get("t_priority") == "MS"), scores[0])
        out.append({
            "variant": variant, "clinvar_id": r["variant_id"], "gene": r["gene_id"], "strand": strand,
            "junction_type": r["junction_type"], "variant_index": DIST, "tracks": tracks,
            "published": {k: (float(v) if k.startswith("DS_") else int(v)) for k, v in pub.items() if k in ("DS_AG", "DS_AL", "DS_DG", "DS_DL", "DP_AG", "DP_AL", "DP_DG", "DP_DL")},
            "published_transcript": pub.get("t_id"),
        })
        print(variant, r["gene_id"], strand, {k: v for k, v in out[-1]["published"].items() if k.startswith("DS_")})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "source": "Official SpliceAI 5-model ensemble (Illumina, spliceai package) tracks; published scores from the Broad SpliceAI-lookup API (GRCh38, distance 50, unmasked).",
        "convention": "tracks are 101 positions centred on the variant (index 50), reference + strand orientation",
        "variants": out,
    }, indent=1))
    print(f"wrote {len(out)} variants → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Issue #10: ClinVar pathogenic splice-variant case study for the paper.

For variants from the #5 ground-truth fixture (tests/fixtures/spliceai_clinvar_ground_truth.json, PR #17),
compares our model's delta scores with SpliceAI's published ones, plots reference vs mutant probability
tracks, and the saliency of the affected site in the reference sequence. Minus-strand variants are scored
in transcript orientation (as in training); peak offsets are reported in genomic orientation, like
SpliceAI's DP_* positions.

  python scripts/case_study.py --checkpoint checkpoints/splice_head.pt --genome-dir data/raw/genome
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svelto_dna.data.genome import Genome  # noqa: E402
from svelto_dna.model.splice_head import SplicePredictor  # noqa: E402
from svelto_dna.research.saliency import locus_saliency  # noqa: E402
from svelto_dna.splice.delta import compute_delta_scores  # noqa: E402

CTX, W = 1000, 50
COMP = str.maketrans("ACGTN", "TGCAN")
CODES = {"donor_gain": "DG", "donor_loss": "DL", "acceptor_gain": "AG", "acceptor_loss": "AL"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--fixture", default="tests/fixtures/spliceai_clinvar_ground_truth.json")
    ap.add_argument("--genes", default="all", help="Comma-separated gene symbols, or 'all'")
    ap.add_argument("--out-dir", default="docs/paper")
    a = ap.parse_args()
    pred, genome = SplicePredictor.load(a.checkpoint), Genome(a.genome_dir)
    fixture = {v["gene"]: v for v in json.loads(Path(a.fixture).read_text())["variants"]}
    out = Path(a.out_dir); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "data").mkdir(parents=True, exist_ok=True)
    cases = []
    for gene in (fixture if a.genes == "all" else a.genes.split(",")):
        v = fixture[gene]
        chrom, pos, ref, alt = v["variant"].split("-"); p0 = int(pos) - 1
        lo = p0 - CTX - W
        seq = genome.fetch(chrom, lo, p0 + CTX + W + 1)
        assert seq[p0 - lo] == ref
        c = p0 - lo
        minus = v["strand"] == "-"
        if minus:                                   # transcript orientation; the window is symmetric, so c stays the centre
            seq, ref, alt = seq.translate(COMP)[::-1], ref.translate(COMP), alt.translate(COMP)
            c = len(seq) - 1 - c
            assert seq[c] == ref
        mut = seq[:c] + alt + seq[c + 1:]
        pr = pred.predict([seq, mut]).numpy()
        d = compute_delta_scores(pr[0], pr[1], window_size=W, variant_pos=c, return_tensors=False)
        ours = {code: float(np.asarray(d[k]).ravel()[0]) for k, code in CODES.items()}
        peak_code = CODES[d["peak_component"]]
        site = int(d["peak_position"])
        target = "donor" if peak_code.startswith("D") else "acceptor"
        sal = locus_saliency(pred, seq, position=site, target=target).numpy()
        cases.append({"gene": gene, "variant": v["variant"], "clinvar_id": v["clinvar_id"], "strand": v["strand"], "junction_type": v["junction_type"],
                      "ours": ours, "ours_peak": {"component": peak_code, "offset": (c - site) if minus else (site - c)},
                      "spliceai_published": v["published"]})
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
        xs = np.arange(-W - 10, W + 11); idx = c + xs
        k = 1 if target == "donor" else 2
        ax1.plot(xs, pr[0][idx, k], label=f"reference ({target})", color="#0A84FF")
        ax1.plot(xs, pr[1][idx, k], label=f"mutant {ref}>{alt}", color="#FF453A")
        ax1.axvline(0, color="0.6", lw=0.8, ls=":"); ax1.set_ylabel("probability"); ax1.legend(fontsize=8, loc="upper right")
        ax1.set_title(f"{gene} {v['variant']} (ClinVar {v['clinvar_id']}): ours Δ{peak_code} {ours[peak_code]:.2f} vs SpliceAI {v['published']['DS_' + peak_code]:.2f}", fontsize=9)
        ax2.bar(xs, sal[idx], color="#8E8E93", width=0.9); ax2.bar([0], [sal[c]], color="#FF9F0A", width=0.9)
        ax2.set_ylabel("saliency"); ax2.set_xlabel("position relative to variant (bp, transcript orientation)")
        fig.tight_layout(); fig.savefig(out / "figures" / f"case_{gene}.png", dpi=200); plt.close(fig)
        print(gene, v["variant"], "ours", {k2: round(x, 3) for k2, x in ours.items()}, "peak", peak_code, site - c,
              "| SpliceAI", {k2: v["published"][k2] for k2 in ("DS_DL", "DS_AL", "DP_DL", "DP_AL")})
    (out / "data" / "case_study.json").write_text(json.dumps(cases, indent=1))
    # summary: peak component, ours vs SpliceAI published, per variant
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    xs = np.arange(len(cases))
    comp = [cs["ours_peak"]["component"] for cs in cases]
    ax.bar(xs - 0.2, [cs["ours"][k] for cs, k in zip(cases, comp)], 0.4, label="ours", color="#0A84FF")
    ax.bar(xs + 0.2, [cs["spliceai_published"]["DS_" + k] for cs, k in zip(cases, comp)], 0.4, label="SpliceAI (published)", color="#8E8E93")
    ax.set_xticks(xs, [f"{cs['gene']}\n{k} ({cs['strand']})" for cs, k in zip(cases, comp)], fontsize=7)
    ax.set_ylabel("peak Δ"); ax.set_ylim(0, 1.05); ax.axhline(0.2, color="0.6", lw=0.8, ls=":")
    ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(out / "figures" / "case_summary.png", dpi=200); plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())

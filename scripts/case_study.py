#!/usr/bin/env python3
"""
Issue #10: ClinVar pathogenic splice-variant case study for the paper.

For + strand variants from the #5 ground-truth fixture (tests/fixtures/spliceai_clinvar_ground_truth.json,
PR #17), compares our model's delta scores with SpliceAI's published ones, plots reference vs mutant
probability tracks, and the saliency of the affected site in the reference sequence.

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
CODES = {"donor_gain": "DG", "donor_loss": "DL", "acceptor_gain": "AG", "acceptor_loss": "AL"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--fixture", default="tests/fixtures/spliceai_clinvar_ground_truth.json")
    ap.add_argument("--genes", default="ABAT,ABCG8"); ap.add_argument("--out-dir", default="docs/paper")
    a = ap.parse_args()
    pred, genome = SplicePredictor.load(a.checkpoint), Genome(a.genome_dir)
    fixture = {v["gene"]: v for v in json.loads(Path(a.fixture).read_text())["variants"]}
    out = Path(a.out_dir); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "data").mkdir(parents=True, exist_ok=True)
    cases = []
    for gene in a.genes.split(","):
        v = fixture[gene]
        assert v["strand"] == "+", "case study uses + strand variants (model input is reference orientation)"
        chrom, pos, ref, alt = v["variant"].split("-"); p0 = int(pos) - 1
        lo = p0 - CTX - W
        seq = genome.fetch(chrom, lo, p0 + CTX + W + 1)
        assert seq[p0 - lo] == ref
        mut = seq[:p0 - lo] + alt + seq[p0 - lo + 1:]
        pr = pred.predict([seq, mut]).numpy()
        c = p0 - lo
        d = compute_delta_scores(pr[0], pr[1], window_size=W, variant_pos=c, return_tensors=False)
        ours = {code: float(np.asarray(d[k]).ravel()[0]) for k, code in CODES.items()}
        peak_code = CODES[d["peak_component"]]
        site = int(d["peak_position"])
        target = "donor" if peak_code.startswith("D") else "acceptor"
        sal = locus_saliency(pred, seq, position=site, target=target).numpy()
        cases.append({"gene": gene, "variant": v["variant"], "clinvar_id": v["clinvar_id"], "junction_type": v["junction_type"],
                      "ours": ours, "ours_peak": {"component": peak_code, "offset": site - c},
                      "spliceai_published": v["published"]})
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
        xs = np.arange(-W - 10, W + 11); idx = c + xs
        k = 1 if target == "donor" else 2
        ax1.plot(xs, pr[0][idx, k], label=f"reference ({target})", color="#0A84FF")
        ax1.plot(xs, pr[1][idx, k], label=f"mutant {ref}>{alt}", color="#FF453A")
        ax1.axvline(0, color="0.6", lw=0.8, ls=":"); ax1.set_ylabel("probability"); ax1.legend(fontsize=8, loc="upper right")
        ax1.set_title(f"{gene} {v['variant']} (ClinVar {v['clinvar_id']}): ours Δ{peak_code} {ours[peak_code]:.2f} vs SpliceAI {v['published']['DS_' + peak_code]:.2f}", fontsize=9)
        ax2.bar(xs, sal[idx], color="#8E8E93", width=0.9); ax2.bar([0], [sal[c]], color="#FF9F0A", width=0.9)
        ax2.set_ylabel("saliency"); ax2.set_xlabel("position relative to variant (bp)")
        fig.tight_layout(); fig.savefig(out / "figures" / f"case_{gene}.png", dpi=200); plt.close(fig)
        print(gene, v["variant"], "ours", {k2: round(x, 3) for k2, x in ours.items()}, "peak", peak_code, site - c,
              "| SpliceAI", {k2: v["published"][k2] for k2 in ("DS_DL", "DS_AL", "DP_DL", "DP_AL")})
    (out / "data" / "case_study.json").write_text(json.dumps(cases, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

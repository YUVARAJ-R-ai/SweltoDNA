#!/usr/bin/env python3
"""
Issue #4 benchmark: speculative in-silico mutagenesis vs exhaustive, on held-out test chromosomes.

For each sampled window of `--width` positions (3 alternatives each) it times
  vanilla     exhaustive, one forward pass per variant (sequential)
  batched     exhaustive, variants batched (the fair, strong baseline)
  speculative one reference pass + draft + top-K verified in one batch
and reports acceptance α (share of verified candidates that are high-impact) and recall (share of the
exhaustive high-impact variants the speculative run also finds). Half the windows are centred on an
annotated splice site, half are random positions inside the block.

  python scripts/benchmark_speculative.py --checkpoint checkpoints/splice_head.pt \
      --test data/processed/spliceai_test.parquet --windows 40
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svelto_dna.model.splice_head import SplicePredictor  # noqa: E402
from svelto_dna.speculative.verify import SpeculativeISM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--k", default="24,48,96", help="Verification budgets to evaluate")
    ap.add_argument("--threshold", type=float, default=0.2, help="|Δ| defining a high-impact variant (SpliceAI's usual 0.2 cut-off)")
    ap.add_argument("--batch", type=int, default=4, help="Verification batch size (an RTX 5060 Laptop saturates at 2-4 sequences)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="speculative_benchmark_telemetry.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    pred = SplicePredictor.load(a.checkpoint)
    ism = SpeculativeISM(pred, context=1000, delta_window=50, threshold=a.threshold)
    rows = pl.read_parquet(a.test, columns=["window_sequence", "labels", "context", "chrom"]).sample(n=a.windows * 3, seed=a.seed).iter_rows(named=True)
    ks = [int(x) for x in a.k.split(",")]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # warm-up so CUDA initialisation is not billed to the first window
    first = next(rows)
    ism.exhaustive(first["window_sequence"], 1500, 1504, batch_size=4)

    per = []
    for r in rows:
        if len(per) >= a.windows:
            break
        seq, ctx, labels = r["window_sequence"], r["context"], r["labels"]
        sites = [i for i, l in enumerate(labels) if l]
        on_site = len(per) % 2 == 0 and sites
        centre = ctx + (rng.choice(sites) if on_site else rng.randrange(len(labels)))
        s0 = max(ctx, min(centre - a.width // 2, ctx + len(labels) - a.width))
        win = (s0, s0 + a.width)
        vanilla = ism.exhaustive(seq, *win, batch_size=1)
        batched = ism.exhaustive(seq, *win, batch_size=a.batch)
        truth = set(batched.high_impact())
        rec = {"chrom": r["chrom"], "on_site": bool(on_site), "candidates": len(batched.candidates), "high_impact": len(truth),
               "vanilla_ms": vanilla.timings_ms["total"], "batched_ms": batched.timings_ms["total"],
               "max_abs_diff_batched_vs_sequential": float(np.max(np.abs(vanilla.deltas - batched.deltas)))}
        for k in ks:
            sp = ism.speculative(seq, *win, k=k, batch_size=a.batch)
            found = set(sp.high_impact())
            rec[f"k{k}"] = {"ms": sp.timings_ms["total"], "alpha": sp.acceptance_rate,
                            "recall": (len(truth & found) / len(truth)) if truth else None,
                            "flops_saved": 1 - (1 + k) / (1 + len(batched.candidates))}
        per.append(rec)
        print(f"{len(per):3d} {r['chrom']:6s} {'site' if on_site else 'rand'} | high-impact {len(truth):3d} | vanilla {rec['vanilla_ms']:7.1f} ms | batched {rec['batched_ms']:6.1f} ms | "
              + " | ".join(f"K{k} {rec[f'k{k}']['ms']:5.1f} ms α {rec[f'k{k}']['alpha']:.2f} recall {rec[f'k{k}']['recall'] if rec[f'k{k}']['recall'] is not None else float('nan'):.2f}" for k in ks), flush=True)

    med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
    summary = {"vanilla_ms_median": med([p["vanilla_ms"] for p in per]), "batched_ms_median": med([p["batched_ms"] for p in per]),
               "max_abs_diff_batched_vs_sequential": max(p["max_abs_diff_batched_vs_sequential"] for p in per)}
    for k in ks:
        ms = med([p[f"k{k}"]["ms"] for p in per])
        recalls = [p[f"k{k}"]["recall"] for p in per if p[f"k{k}"]["recall"] is not None]
        summary[f"k{k}"] = {
            "ms_median": ms,
            "speedup_vs_sequential": summary["vanilla_ms_median"] / ms,
            "speedup_vs_batched_exhaustive": summary["batched_ms_median"] / ms,
            "alpha_mean": statistics.mean(p[f"k{k}"]["alpha"] for p in per),
            "recall_mean": statistics.mean(recalls) if recalls else None,
            "recall_min": min(recalls) if recalls else None,
            "windows_with_high_impact": len(recalls),
            "flops_saved": per[0][f"k{k}"]["flops_saved"],
        }
    out = {"gpu": gpu, "backbone": pred.backbone.model_name, "dtype": str(pred.backbone.target_dtype), "windows": len(per), "width": a.width,
           "candidates_per_window": per[0]["candidates"] if per else None, "threshold": a.threshold, "seed": a.seed, "summary": summary, "per_window": per}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Issue #10: ablation of speculative ISM over verification budget K and sequence length, on held-out
test chromosomes. Every window is timed three ways (sequential, batched exhaustive, speculative) and
recall/acceptance are measured against the exhaustive result.

Lengths stay within the stored windows' real flank (1 kb per side), so no window is padded with N
(padding pushes the model out of distribution and makes recall meaningless).

  python scripts/run_ablation.py --checkpoint checkpoints/splice_head.pt --test data/processed/spliceai_test.parquet
"""

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svelto_dna.model.splice_head import SplicePredictor  # noqa: E402
from svelto_dna.speculative.verify import SpeculativeISM  # noqa: E402

DELTA_WINDOW = 50


def run_conditions() -> dict:
    """Power state affects batch-1 (launch-overhead-bound) timings a lot on laptops; record it with the results."""
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    ac = [read(f) for f in Path("/sys/class/power_supply").glob("*/online")] if Path("/sys/class/power_supply").exists() else []
    return {"on_ac_power": any(v == "1" for v in ac) if ac else None, "platform_profile": read("/sys/firmware/acpi/platform_profile")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--contexts", default="250,500,900", help="Flank per side; sequence length = width + 100 + 2*context")
    ap.add_argument("--ks", default="12,24,48,96"); ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--width", type=int, default=64); ap.add_argument("--threshold", type=float, default=0.2)
    ap.add_argument("--batch", type=int, default=4, help="Verification batch (an RTX 5060 Laptop saturates at 2-4 sequences)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="docs/paper/data/ablation.json")
    a = ap.parse_args()

    pred = SplicePredictor.load(a.checkpoint)
    ks = [int(x) for x in a.ks.split(",")]
    contexts = [int(x) for x in a.contexts.split(",")]
    need = max(contexts) + DELTA_WINDOW + a.width // 2
    rng = random.Random(a.seed)
    df = pl.read_parquet(a.test, columns=["window_sequence", "labels", "context", "chrom"]).sample(n=a.windows * 6, seed=a.seed)
    picks = []
    for r in df.iter_rows(named=True):
        L, ctx0, n = len(r["window_sequence"]), r["context"], len(r["labels"])
        # centres whose largest-context slice fits inside real sequence
        lo, hi = max(ctx0, need), min(ctx0 + n, L - need)
        if hi <= lo:
            continue
        sites = [ctx0 + i for i, l in enumerate(r["labels"]) if l and lo <= ctx0 + i < hi]
        on_site = len(picks) % 2 == 0 and bool(sites)
        centre = rng.choice(sites) if on_site else rng.randrange(lo, hi)
        picks.append((r, centre, on_site))
        if len(picks) == a.windows:
            break

    results = []
    for ctx in contexts:
        ism = SpeculativeISM(pred, context=ctx, delta_window=DELTA_WINDOW, threshold=a.threshold)
        warm = picks[0][0]["window_sequence"]; ism.exhaustive(warm, len(warm) // 2, len(warm) // 2 + 4, batch_size=4)
        for r, centre, on_site in picks:
            seq = r["window_sequence"]
            win = (centre - a.width // 2, centre + a.width // 2)
            assert win[0] - DELTA_WINDOW - ctx >= 0 and win[1] + DELTA_WINDOW + ctx <= len(seq), "slice must stay inside real sequence"
            van = ism.exhaustive(seq, *win, batch_size=1)
            bat = ism.exhaustive(seq, *win, batch_size=a.batch)
            truth = set(bat.high_impact())
            rec = {"context": ctx, "seq_len": a.width + 2 * DELTA_WINDOW + 2 * ctx, "chrom": r["chrom"], "on_site": on_site, "high_impact": len(truth),
                   "vanilla_ms": van.timings_ms["total"], "batched_ms": bat.timings_ms["total"],
                   "max_abs_diff_batched_vs_sequential": float(np.max(np.abs(van.deltas - bat.deltas))), "k": {}}
            for k in ks:
                sp = ism.speculative(seq, *win, k=k, batch_size=a.batch)
                rec["k"][str(k)] = {"ms": sp.timings_ms["total"], "alpha": sp.acceptance_rate,
                                    "recall": len(truth & set(sp.high_impact())) / len(truth) if truth else None}
            results.append(rec)
            print(f"L={rec['seq_len']:5d} {r['chrom']:5s} {'site' if on_site else 'rand'} hi={len(truth):3d} van {rec['vanilla_ms']:7.1f} bat {rec['batched_ms']:6.1f} "
                  + " ".join(f"K{k}:{rec['k'][str(k)]['ms']:5.1f}ms/r{rec['k'][str(k)]['recall'] if rec['k'][str(k)]['recall'] is not None else float('nan'):.2f}" for k in ks), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                                       "backbone": pred.backbone.model_name, "dtype": str(pred.backbone.target_dtype), "width": a.width,
                                       "candidates": 3 * a.width, "threshold": a.threshold, "ks": ks, "contexts": contexts, "windows": a.windows,
                                       "batch": a.batch, "seed": a.seed, "conditions": run_conditions(), "results": results}, indent=1))
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

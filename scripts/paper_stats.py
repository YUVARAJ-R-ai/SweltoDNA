#!/usr/bin/env python3
"""
Issue #10: derived statistics for the paper, from the committed measurements.

  cost model       speculative speedup predicted from pass counts and the measured batching gain
  draft baselines  expected recall of a uniform-random draft (K/N) and of an oracle draft (min(1, K/H))
  delta validation our Δ calculator on SpliceAI's own tracks vs SpliceAI's published scores (#5 fixture)
  case study       ours vs SpliceAI on the #5 variants, split membership, peak-offset convention check
  batch scaling    (--measure-batch, needs the GPU) per-sequence forward latency against batch size

  python scripts/paper_stats.py                                   # derived numbers only
  python scripts/paper_stats.py --measure-batch --checkpoint checkpoints/splice_head.pt
"""

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
TRAIN = {f"chr{i}" for i in range(11, 23)}
VAL = {"chr2", "chr4", "chr6", "chr8", "chr10"}
TEST = {"chr1", "chr3", "chr5", "chr7", "chr9"}
# Label conventions: we mark the first base of the dinucleotide (G of GT, A of AG); SpliceAI marks the
# exon-side boundary base. In transcript orientation SpliceAI's peak is ours -1 (donor) and +2 (acceptor).
CONVENTION_SHIFT = {"D": -1, "A": 2}


def split_of(chrom: str) -> str:
    return "train" if chrom in TRAIN else "val" if chrom in VAL else "test" if chrom in TEST else "other"


def cost_model(bench: dict) -> dict:
    """T_spec(K) ≈ (1+K) · T_batched/(1+N): verification dominates and runs at the batched per-sequence cost."""
    s, n = bench["summary"], bench["candidates_per_window"]
    gain = s["vanilla_ms_median"] / s["batched_ms_median"]
    out = {"candidates": n, "batching_gain": gain}
    for key, v in s.items():
        if not key.startswith("k"):
            continue
        k = int(key[1:])
        pred = (1 + n) / (1 + k) * gain
        out[key] = {"passes_ratio": (1 + n) / (1 + k), "predicted_speedup": pred, "measured_speedup": v["speedup_vs_sequential"],
                    "rel_error": v["speedup_vs_sequential"] / pred - 1, "flops_saved": 1 - (1 + k) / (1 + n),
                    "measured_vs_batched": v["speedup_vs_batched_exhaustive"]}
    return out


def draft_baselines(abl: dict) -> dict:
    n = abl["candidates"]
    rows = [r for r in abl["results"] if r["high_impact"]]
    out = {"windows": len(rows), "mean_high_impact": statistics.mean(r["high_impact"] for r in rows),
           "mean_high_impact_on_site": statistics.mean(r["high_impact"] for r in rows if r["on_site"]),
           "mean_high_impact_random": statistics.mean(r["high_impact"] for r in rows if not r["on_site"])}
    for k in abl["ks"]:
        rec = statistics.mean(r["k"][str(k)]["recall"] for r in rows)
        oracle = statistics.mean(min(1.0, k / r["high_impact"]) for r in rows)
        out[f"k{k}"] = {"draft_recall": rec, "random_recall": k / n, "oracle_recall": oracle,
                        "gain_over_random": rec / (k / n), "share_of_oracle": rec / oracle,
                        "recall_on_site": statistics.mean(r["k"][str(k)]["recall"] for r in rows if r["on_site"]),
                        "recall_random_window": statistics.mean(r["k"][str(k)]["recall"] for r in rows if not r["on_site"])}
    return out


def case_summary(cases: list) -> dict:
    rows = []
    for c in cases:
        comp = c["ours_peak"]["component"]
        ours, sai = c["ours"][comp], c["spliceai_published"]["DS_" + comp]
        # offsets are genomic; convert the convention shift to genomic orientation for minus-strand variants
        shift = CONVENTION_SHIFT[comp[0]] * (-1 if c["strand"] == "-" else 1)
        rows.append({"gene": c["gene"], "chrom": c["variant"].split("-")[0], "split": split_of(c["variant"].split("-")[0]),
                     "strand": c["strand"], "component": comp, "ours": ours, "spliceai": sai, "abs_diff": abs(ours - sai),
                     "ours_offset": c["ours_peak"]["offset"], "spliceai_offset": c["spliceai_published"]["DP_" + comp],
                     "offset_matches_convention": c["spliceai_published"]["DP_" + comp] == c["ours_peak"]["offset"] + shift,
                     "correct_component": comp[0] == c["junction_type"][0].upper() and comp[1] == "L"})
    return {"variants": rows, "all_correct_component": all(r["correct_component"] for r in rows),
            "all_offsets_match_convention": all(r["offset_matches_convention"] for r in rows),
            "mean_ours": statistics.mean(r["ours"] for r in rows), "mean_spliceai": statistics.mean(r["spliceai"] for r in rows),
            "mean_abs_diff": statistics.mean(r["abs_diff"] for r in rows), "min_ours": min(r["ours"] for r in rows)}


def delta_validation(fixture: dict) -> dict:
    import numpy as np
    from svelto_dna.splice.delta import compute_delta_scores

    def tracks(t):
        d, a = np.asarray(t["donor"], dtype=np.float32), np.asarray(t["acceptor"], dtype=np.float32)
        return np.stack([1 - d - a, d, a], axis=-1)

    names = {"donor_gain": "DG", "donor_loss": "DL", "acceptor_gain": "AG", "acceptor_loss": "AL"}
    rows = []
    for v in fixture["variants"]:
        r = compute_delta_scores(tracks(v["tracks"]["ref"]), tracks(v["tracks"]["alt"]), window_size=50, variant_pos=v["variant_index"], return_tensors=False)
        ours = {code: float(np.asarray(r[k]).reshape(-1)[0]) for k, code in names.items()}
        code = names[r["peak_component"]]
        rows.append({"gene": v["gene"], "variant": v["variant"], "strand": v["strand"], "junction_type": v["junction_type"],
                     "ours": ours, "published": {c: v["published"]["DS_" + c] for c in names.values()},
                     "max_abs_diff": max(abs(ours[c] - v["published"]["DS_" + c]) for c in names.values()),
                     "peak_component": code, "peak_offset": int(r["peak_position"]) - v["variant_index"],
                     "published_peak_offset": v["published"]["DP_" + code]})
    return {"variants": rows, "max_abs_diff": max(r["max_abs_diff"] for r in rows),
            "all_peaks_match": all(r["peak_offset"] == r["published_peak_offset"] for r in rows)}


def batch_scaling(checkpoint: str, test: str, length: int, sizes, repeats: int) -> dict:
    import polars as pl
    import torch
    from svelto_dna.model.splice_head import SplicePredictor
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_ablation import run_conditions as conditions  # same power-state probe as the ablation

    pred = SplicePredictor.load(checkpoint)
    seqs = [s[:length] for s in pl.read_parquet(test, columns=["window_sequence"]).sample(n=max(sizes), seed=3)["window_sequence"]]
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for _ in range(5):
        pred.predict(seqs[:4])
    rows = []
    for b in sizes:
        ts = []
        for _ in range(repeats):
            sync(); t0 = time.perf_counter(); pred.predict(seqs[:b]); sync()
            ts.append((time.perf_counter() - t0) * 1000)
        med = statistics.median(ts)
        rows.append({"batch": b, "median_ms": med, "ms_per_sequence": med / b})
        print(f"batch {b:3d}: {med:7.2f} ms  ({med / b:.2f} ms/sequence)", flush=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"gpu": gpu, "length": length, "repeats": repeats, "rows": rows, "conditions": conditions(),
            "note": "one predict() = forward + reverse-complement backbone passes + head, per sequence"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=str(ROOT / "speculative_benchmark_telemetry.json"))
    ap.add_argument("--ablation", default=str(ROOT / "docs/paper/data/ablation.json"))
    ap.add_argument("--cases", default=str(ROOT / "docs/paper/data/case_study.json"))
    ap.add_argument("--fixture", default=str(ROOT / "tests/fixtures/spliceai_clinvar_ground_truth.json"), help="#5 ground truth (PR #17)")
    ap.add_argument("--out", default=str(ROOT / "docs/paper/data/derived.json"))
    ap.add_argument("--measure-batch", action="store_true")
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints/splice_head.pt"))
    ap.add_argument("--test", default=str(ROOT / "data/processed/spliceai_test.parquet"))
    ap.add_argument("--length", type=int, default=2164)
    ap.add_argument("--batches", default="1,2,4,8,16,32,64")
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()

    out_path = Path(a.out)
    old = json.loads(out_path.read_text()) if out_path.exists() else {}
    out = {"cost_model": cost_model(json.loads(Path(a.bench).read_text())),
           "draft_baselines": draft_baselines(json.loads(Path(a.ablation).read_text())),
           "case_study": case_summary(json.loads(Path(a.cases).read_text()))}
    if Path(a.fixture).exists():
        out["delta_validation"] = delta_validation(json.loads(Path(a.fixture).read_text()))
    elif "delta_validation" in old:
        out["delta_validation"] = old["delta_validation"]
    if a.measure_batch:
        random.seed(0)
        out["batch_scaling"] = batch_scaling(a.checkpoint, a.test, a.length, [int(x) for x in a.batches.split(",")], a.repeats)
    elif "batch_scaling" in old:
        out["batch_scaling"] = old["batch_scaling"]
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "batch_scaling"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

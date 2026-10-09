#!/usr/bin/env python3
"""
Train the splice-site head on frozen backbone features (forward + reverse-complement).

Only the head trains. Validation uses held-out chromosomes (chr2,4,6,8,10); the test chromosomes
(chr1,3,5,7,9) are never touched here. The best checkpoint by validation mean PR-AUC is kept.

  python scripts/train_splice_head.py --train data/processed/spliceai_train.parquet \
      --val data/processed/spliceai_val.parquet --out checkpoints/splice_head.pt
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svelto_dna.core.backbone import SveltoBackbone  # noqa: E402
from svelto_dna.data.benchmark import compute_splice_metrics  # noqa: E402
from svelto_dna.model.splice_head import SpliceHead, SplicePredictor  # noqa: E402


def load(path: str) -> list:
    df = pl.read_parquet(path, columns=["window_sequence", "labels", "context"])
    return list(df.iter_rows())


def evaluate(pred: SplicePredictor, rows: list, batch: int) -> dict:
    ys, ps = [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        probs = pred.predict([r[0] for r in chunk]).numpy()
        for (seq, labels, ctx), p in zip(chunk, probs):
            ys.extend(labels); ps.append(p[ctx: ctx + len(labels)])
    return compute_splice_metrics(np.array(ys), np.vstack(ps)).to_dict()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--model", default="LongSafari/hyenadna-small-32k-seqlen-hf")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--site-weight", type=float, default=20.0, help="Loss weight for donor/acceptor vs neither")
    ap.add_argument("--val-windows", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="checkpoints/splice_head.pt")
    ap.add_argument("--telemetry", default="splice_head_training_telemetry.json")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    train, val = load(a.train), load(a.val)
    val = random.Random(a.seed).sample(val, min(a.val_windows, len(val)))
    print(f"train windows {len(train):,} | val windows (fixed sample) {len(val):,}")

    backbone = SveltoBackbone(model_name=a.model)
    pred = SplicePredictor(backbone, SpliceHead(hidden_dim=backbone.hidden_dim))
    opt = torch.optim.AdamW(pred.head.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=min(0.3, max(0.05, 3 / a.steps)))   # >=3 warm-up steps
    weight = torch.tensor([1.0, a.site_weight, a.site_weight], device=pred.device)
    rng = random.Random(a.seed + 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    history, best, t0 = [], -1.0, time.time()
    for step in range(1, a.steps + 1):
        batch = rng.sample(train, a.batch)
        with torch.no_grad():
            feats = pred.features([r[0] for r in batch])
        ctx, n = batch[0][2], len(batch[0][1])
        logits = pred.head.train()(feats[:, ctx: ctx + n])
        target = torch.tensor([r[1] for r in batch], device=pred.device)
        loss = F.cross_entropy(logits.reshape(-1, 3), target.reshape(-1), weight=weight)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(pred, val, a.batch * 2)
            row = {"step": step, "loss": float(loss.item()), "val_mean_pr_auc": m["mean_pr_auc"], "val_mean_roc_auc": m["mean_roc_auc"],
                   "val_donor_top_1": m["donor_top_1_acc"], "val_acceptor_top_1": m["acceptor_top_1_acc"]}
            history.append(row)
            print(f"  val: PR-AUC {m['mean_pr_auc']:.4f} | ROC-AUC {m['mean_roc_auc']:.4f} | top-1 donor {m['donor_top_1_acc']:.3f} acceptor {m['acceptor_top_1_acc']:.3f}", flush=True)
            if m["mean_pr_auc"] > best:
                best = m["mean_pr_auc"]
                pred.save(a.out, step=step, val_mean_pr_auc=best, train_windows=len(train), seed=a.seed)
    Path(a.telemetry).write_text(json.dumps({
        "model": a.model, "steps": a.steps, "batch": a.batch, "lr": a.lr, "site_weight": a.site_weight, "seed": a.seed,
        "train_windows": len(train), "val_windows": len(val), "minutes": round((time.time() - t0) / 60, 1),
        "best_val_mean_pr_auc": best, "history": history,
    }, indent=2))
    print(f"best val mean PR-AUC {best:.4f} → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

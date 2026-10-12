"""Issue #10 AC2/AC3: the report script turns measured JSON into results_summary.tex and figures."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")


def _ablation(path: Path):
    res = []
    for ctx, L in ((500, 1164), (1000, 2164)):
        for w in range(3):
            res.append({"context": ctx, "seq_len": L, "chrom": "chr1", "on_site": w % 2 == 0, "high_impact": 5,
                        "vanilla_ms": 900.0 + w, "batched_ms": 650.0, "max_abs_diff_batched_vs_sequential": 1e-3,
                        "k": {"12": {"ms": 45.0, "alpha": 0.4, "recall": 0.3 if ctx == 500 else None},
                              "24": {"ms": 90.0, "alpha": 0.3, "recall": 0.6 if ctx == 500 else None}}})
    # context 1000 has no window with a high-impact variant: recall is undefined there
    path.write_text(json.dumps({"gpu": "Test GPU", "backbone": "m", "dtype": "torch.float16", "width": 64, "candidates": 192,
                                "threshold": 0.2, "ks": [12, 24], "contexts": [500, 1000], "windows": 3, "seed": 1, "results": res}))


def test_report_writes_tex_with_measured_numbers_and_figures(tmp_path):
    _ablation(tmp_path / "ablation.json")
    (tmp_path / "eval.json").write_text(json.dumps({"model": "org/model_x", "donor_roc_auc": 0.99891, "acceptor_roc_auc": 0.99842,
        "donor_pr_auc": 0.6295, "acceptor_pr_auc": 0.552, "donor_top_1_acc": 0.6042, "acceptor_top_1_acc": 0.5418,
        "donor_top_k_acc": 0.81, "acceptor_top_k_acc": 0.74, "n_total_loci": 1000, "n_donors": 3, "n_acceptors": 4}))
    (tmp_path / "train.json").write_text(json.dumps({"steps": 10, "minutes": 1.0, "best_val_mean_pr_auc": 0.5}))
    r = subprocess.run([sys.executable, "scripts/research_report.py", "--ablation", str(tmp_path / "ablation.json"), "--eval", str(tmp_path / "eval.json"),
                        "--training", str(tmp_path / "train.json"), "--out-dir", str(tmp_path / "paper")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    tex = (tmp_path / "paper" / "results_summary.tex").read_text()
    assert "0.6295" in tex and "0.9989" in tex                     # copied from the eval JSON, not invented
    assert "1,164" in tex and "2,164" in tex and "K=12" in tex and "K=24" in tex
    assert "recall" in tex and "Do not edit by hand" in tex
    assert "n/a" in tex                                            # undefined recall is shown, not crashed on or invented
    for f in ("speedup_vs_length.png", "recall_alpha_vs_k.png"):
        assert (tmp_path / "paper" / "figures" / f).stat().st_size > 10_000

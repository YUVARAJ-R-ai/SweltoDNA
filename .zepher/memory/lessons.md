# Lessons Learned

- **Eval harness injects ground truth (open, 2026-10-06):** `scripts/eval_splice_benchmark.py` overwrites model probabilities using `labels` before scoring, so `splice_benchmark_telemetry.json` (all 1.0) measures nothing. With the injection removed, the untrained mock head scores donor ROC-AUC 0.35 and PR-AUC 0.0007, which is chance. Still present on `origin/dev`. Never report these metrics until it's removed.
- **ClinVar junction filter is a no-op on real data:** real ClinVar VCF/TSV has no `DIST`/`JUNCTION_TYPE`. The parser defaults distance to 0, so every SNV passes the +-50 bp filter, and VCF `junction_type` is made up from `pos % 2`. A real filter needs exon coordinates from a GTF annotation.
- **No real SpliceAI ingestion exists:** `SpliceAIParser.process_records` only takes in-memory dicts. Every SpliceAI split in `extract_splice_data.py` is synthetic, whatever input you pass.
- **main vs dev drift:** README/ARCHITECTURE docs on `main` describe `speculative/`, `splice/`, `train_draft_heads.py`, and 85+ tests, but that code lives only on `origin/dev` (PR #13). `main` has 34 tests.

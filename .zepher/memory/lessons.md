# Lessons Learned

- **Never let labels reach the prediction path in eval (fixed `ebbcca0`, 2026-10-08):** `scripts/eval_splice_benchmark.py` used to overwrite model probabilities from `labels`, producing all-1.0 telemetry. Guarded by `TestEvalHarnessIntegrity`. Honest baseline for the untrained mock head is chance: donor ROC-AUC 0.35, PR-AUC 0.0007. Perfect scores from an untrained model mean leakage.
- **Splice label 0 is 'Neither', not padding (fixed `3123eb2`):** draft loss/trainer used to default to `ignore_index=0`, which dropped every non-splice position. Default is now -100; padding goes through the attention mask.
- **ClinVar junction filter is a no-op on real data (open):** real ClinVar VCF/TSV has no `DIST`/`JUNCTION_TYPE`. The parser defaults distance to 0, so every SNV passes the +-50 bp filter, and VCF `junction_type` is made up from `pos % 2`. A real filter needs exon coordinates from a GTF annotation.
- **No real SpliceAI ingestion exists (open):** `SpliceAIParser.process_records` only takes in-memory dicts. Every SpliceAI split in `extract_splice_data.py` is synthetic.
- **"Ground truth" tests must use real data:** `tests/test_delta.py` ClinVar tests use hand-placed probabilities; they check arithmetic, not agreement with published SpliceAI/ClinVar scores.
- **Keep README claims tied to measurements:** the 2.0-3.5x speedup is a target until #4 exists. README has a dated Current Status table; update it when status changes.

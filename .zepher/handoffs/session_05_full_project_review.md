# Session Handoff: Full Project Review (main @ bd5cb17)

- **Date:** 2026-10-06
- **Verified:** 34/34 tests pass (CPU torch 2.14.1, py3.11, scratch venv).
- **Critical:** eval harness label injection (see memory/lessons.md). It makes the committed splice telemetry meaningless.
- **High:** ClinVar proximity filter is a no-op on real files; SpliceAI ingestion is synthetic only; real HF backbones get tokens they weren't trained on.
- **Medium:** docs on main describe dev-only code; `MockGenomicBackbone.__init__` calls `torch.manual_seed(42)` (clobbers caller RNG); pos-emb silently skipped when L > max_seq_len; eval silently falls back to mock and swallows checkpoint load errors; `trust_remote_code=True` on arbitrary model ids; ClinVar TSV not filtered by Assembly (GRCh37+38 rows mixed).
- **Next:** remove injection on dev before merging PR #13, regenerate telemetry, then fix the data pipeline realism gaps.

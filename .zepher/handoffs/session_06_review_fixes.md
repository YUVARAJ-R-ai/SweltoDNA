# Session Handoff: Review Fixes (branch `fix/review-findings`)

- **Date:** 2026-10-08
- **Base:** `origin/main` @ 3fcc989 (PR #13 merged). Local `main` rebased onto it; `backup/main-before-rebase` kept.
- **Commits:**
  1. `ebbcca0` fix(eval): removed ground-truth label injection, strict checkpoint load, regenerated `splice_benchmark_telemetry.json` (chance-level, honest).
  2. `3123eb2` fix(speculative): `ignore_index` default 0 -> -100 in loss, trainer, CLI.
  3. `3b3cf20` fix(backbone): mock init restores caller's global RNG state.
  4. docs: README Current Status table, speedup marked as target, real test counts (78), Zepher memory refreshed.
- **Tests:** 78/78 pass (6 new regression tests, each watched failing first).
- **GitHub:** #1, #2, #5 reopened with unmet acceptance criteria. Branch NOT pushed (user's call).
- **Still open:** GTF-based ClinVar junction filter, real SpliceAI-10k loader, HF tokenizer mismatch, #6 payload lacks p_ref/p_mut for #8.
- **Next:** frontend (#7/#8/#9) brainstorming resumes: approach A (virtualized DOM ribbon + canvas tracks), in-browser motif scorer, simulated HUD with SIMULATED badge, real GRCh38 gene region + paste box.

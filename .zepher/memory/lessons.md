# Lessons Learned

- **Never let labels reach the prediction path in eval (fixed `ebbcca0`, 2026-10-08).** Perfect scores from an untrained model mean leakage. Honest untrained baseline: PR-AUC 0.0007; trained head on held-out chr1,3,5,7,9: donor/acceptor PR-AUC 0.630/0.552, ROC-AUC 0.9989/0.9984.
- **Splice label 0 is "Neither", not padding (fixed `3123eb2`).** Draft loss defaults to ignore_index=-100; padding goes through the attention mask.
- **Real checkpoints need their own tokenizer.** HyenaDNA: A=7..N=11, no special tokens (`add_special_tokens=False`, or outputs are one position too long), no `attention_mask` argument. Keep weights fp32 and autocast fp16 (cosine 0.99998).
- **HyenaDNA is causal.** Forward states never see downstream bases; splice features are [forward ; reverse-complement flipped]. FFT round-off leaks ~3e-3 into the forward half; that is noise, not signal.
- **ClinVar changed its wording in 2024:** "Conflicting classifications of pathogenicity" (was "interpretations"). Substring matching on "pathogenic" mislabelled ~87k conflicting variants as pathogenic. Conflicting is always holdout.
- **Junction distance must come from annotation.** Real ClinVar carries no DIST/JUNCTION_TYPE; without a GTF the parser refuses rather than guessing.
- **The root `.gitignore` rule `lib/` swallowed `web/lib/`.** Python packaging patterns are now root-anchored (`/lib/`). Always verify a PR from a clean checkout (`git worktree add` + `npm ci` / fresh venv), never from the working tree.
- **Commit as you go; never leave work only in a scratch/temporary directory.** Issue #10's first build lived uncommitted in a session-scratch worktree and was lost when the session restarted.
- **Batching does not amortize much on an RTX 5060 Laptop.** HyenaDNA-small at ~2 kb saturates the GPU at batch 2–4 (≈2.3 ms/seq/pass); batch ≥8 is ~3.2 ms/seq, and at 4 kb batching was slower than sequential. Measure throughput per batch size before claiming batched speedups.
- **Speculative ISM speed costs recall.** Speedups beyond ~1.2–1.4× come only from verifying fewer variants (K=96: 2.4×, recall 0.83; K=24: 9×, recall 0.31). A first-order gradient draft did not beat the heuristic. Never quote speedup without recall.
- **Don't N-pad windows to test longer lengths.** The model goes out of distribution and predicts no sites, so recall becomes undefined; only lengths within the real flank are meaningful.
- **Fixture strings: build them programmatically.** A hand-counted genomic fixture was off by two; assert motif positions inside the fixture itself.

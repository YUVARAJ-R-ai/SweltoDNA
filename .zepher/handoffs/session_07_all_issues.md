# Session Handoff: All Issues (2026-10-08 → 2026-10-12)

## PRs (all open; merge in stack order, nothing merged by the agent)
| PR | Branch | Closes | Base stack |
| :--- | :--- | :--- | :--- |
| #14 | fix/review-findings | #3 (fix) | main |
| #16 | feat/issue-1-real-hyenadna | #1 | #14 |
| #18 | feat/issue-2-real-data | #2 | #16 |
| #17 | feat/issue-5-clinvar-ground-truth | #5 | #18 |
| #19 | feat/splice-head | — (foundation) | #18 |
| #20 | feat/issue-6-websocket-server | #6 | #19 |
| #21 | feat/issue-4-speculative-ism | refs #4 | #20 |
| #10 PR | feat/issue-10-research | #10 | #21 |
| #15 | feat/frontend-splice-explorer | #7, #8 (refs #9) | origin/main |

## Key measured results
- Real HyenaDNA-small-32k on RTX 5060 Laptop: 10 kb in 15.8 ms, 262 MB.
- Trained splice head, held-out chr1,3,5,7,9: PR-AUC 0.630 / 0.552, ROC-AUC 0.9989 / 0.9984.
- Delta calculator matches published SpliceAI for 8 ClinVar pathogenic variants (max diff 0.000476).
- WebSocket edit round trip 12.5 ms median.
- Speculative ISM: exact batching 1.2×; K=96 2.4× at recall 0.83. The ≥2× target only holds with missed variants.

## Open decisions for the team
- #4: how to state the speedup in the paper (exact 1.2× vs approximate 2.4× at 83% recall).
- #9 closes once #15 and #21 merge (HUD shows measured server telemetry).

## Next
- Remove paralogs from the test set; longer context (train with >1 kb flanks) to raise PR-AUC.
- Try a learned variant-effect draft (trained on ISM labels) to improve recall at small K.

## Paper (2026-10-12)
- IEEE conference paper: `docs/paper/ieee/svelto_dna.tex` + `refs.bib` (46 refs) + compiled PDF, 13 pages. Build: `cd docs/paper/ieee && tectonic svelto_dna.tex`.
- Derived numbers: `scripts/paper_stats.py` -> `docs/paper/data/derived.json` (cost model, random/oracle recall, delta validation; `--fixture data/gt_fixture.json` on this branch since the #5 fixture lives on PR #17; `--measure-batch` re-measures batch scaling).
- Case study covers all 8 #5 variants; ABAT, AAAS, AAGAB are on TRAINING chromosomes (disclosed in the paper). Only ABHD5, ABCA1 are test.
- TODO before submission: author affiliations/e-mails; re-run benchmark, ablation and `paper_stats.py --measure-batch` on AC power (performance profile) and update Tables IX-XI / Fig. 3.

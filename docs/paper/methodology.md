# Svelto-DNA: Methods, Results and Case Study

Draft for the IEEE BIBM / ACM-BCB submission (issue #10). Every number here comes from a file in the repository produced by a script. Regenerate the tables with `scripts/research_report.py` and the case study with `scripts/case_study.py`.

## 1. Data

- **Annotation:** GENCODE v46 basic, one canonical protein-coding transcript per gene (`Ensembl_canonical`): 19,120 transcripts on chr1–22, 366,000 splice sites.
- **Sequence:** Ensembl GRCh38 primary assembly, chromosomes 1–22.
- **Windows:** transcript-oriented 1 kb labelled blocks with 1 kb flanks on both sides (3 kb input). Minus-strand genes are reverse-complemented, and labels mark the first base of the motif (G of GT, A of AG). Every block containing a site is kept, plus one site-free block per transcript, so positives are enriched relative to the genome.
- **Splits:** by chromosome, with zero chromosome and gene overlap. Train chr11–22 (86,811 windows), validation chr2,4,6,8,10 (51,657), test chr1,3,5,7,9 (62,739). Paralogs are not removed from the test set (a known limitation).
- **ClinVar:** GRCh38 VCF, SNVs within ±50 bp of an annotated site (2,118,576). Pathogenic/likely pathogenic 119,787; benign/likely benign 732,647; uncertain or conflicting held out (1,266,142).

## 2. Model

Let $x$ be a DNA window, $\mathrm{rc}(x)$ its reverse complement, and $H(\cdot) \in \mathbb{R}^{L \times D}$ the frozen HyenaDNA-small-32k last hidden state ($D = 256$). HyenaDNA is causal, so a position's forward state only sees upstream sequence. Each position therefore gets both flanks:

$$
h_t = \big[\, H(x)_t \;;\; H(\mathrm{rc}(x))_{L-1-t} \,\big] \in \mathbb{R}^{2D}
$$

A two-hidden-layer MLP head (LayerNorm, GELU) maps $h_t$ to $p_t = \mathrm{softmax}(f(h_t)) \in \Delta^3$ over {neither, donor, acceptor}. Only the head trains, with class-weighted cross-entropy (weight 20 on the two site classes) over the labelled block. Training: 4,000 steps of 16 windows with AdamW and a one-cycle schedule, 14 minutes on an RTX 5060 Laptop GPU. The checkpoint is selected by mean validation PR-AUC.

## 3. Variant effect (Δ)

For a variant at $v$ with reference and mutant tracks $p^{\mathrm{ref}}, p^{\mathrm{mut}}$ and window $W = 50$, following SpliceAI:

$$
\Delta_{\mathrm{DG}} = \max_{|j-v|\le W} \big(p^{\mathrm{mut}}_{j,d} - p^{\mathrm{ref}}_{j,d}\big),\quad
\Delta_{\mathrm{DL}} = \max_{|j-v|\le W} \big(p^{\mathrm{ref}}_{j,d} - p^{\mathrm{mut}}_{j,d}\big),
$$

and likewise $\Delta_{\mathrm{AG}}, \Delta_{\mathrm{AL}}$ for acceptors, with $\Delta = \max(\Delta_{\mathrm{DG}}, \Delta_{\mathrm{DL}}, \Delta_{\mathrm{AG}}, \Delta_{\mathrm{AL}})$. The vectorized implementation (1D max-pooling) was validated against **published** SpliceAI scores for eight ClinVar pathogenic canonical-site variants on both strands. Fed SpliceAI's own probability tracks, it reproduces every published score to within the publication's 3-decimal rounding (max difference 0.000476) and every peak position exactly. It runs in 0.21 ms for a 10 kb window on GPU.

## 4. Speculative in-silico mutagenesis

For a window of $n$ positions, the candidate set is every alternative base, $|C| = N = 3n$. **Vanilla** ISM scores each $c \in C$ with its own forward pass. **Speculative** ISM:

1. one reference pass gives $p^{\mathrm{ref}}$;
2. a draft score ranks candidates without running the model: $d(c) = \max\big(s_{\mathrm{on}}(c),\ s_{\mathrm{near}}(c),\ 0.5\cdot\mathbb{1}[c \text{ creates GT/AG}]\big)$, where $s$ are the reference site probabilities on the dinucleotide and in its neighbourhood (decay $e^{-|j-v|/4}$);
3. the top-$K$ candidates are verified exactly in batched forward passes. The rest are reported as unknown, never as zero.

With high-impact meaning $|\Delta| \ge \tau$ ($\tau = 0.2$), we report the acceptance rate $\alpha = |\{\text{verified}, |\Delta| \ge \tau\}| / K$, the recall against exhaustive ISM, the wall-clock speedup against both sequential and batched exhaustive ISM, and the forward-pass saving $1 - (1+K)/(1+N)$.

Batched verification matches sequential within $6 \times 10^{-8}$ in fp32 (fp16 autocast: up to $4 \times 10^{-3}$).

## 5. Saliency

For a site $t$ and class $k$, the attribution of base $i$ is gradient × input on the token embeddings, summed over the forward and reverse-complement passes and normalized to $[0, 1]$:

$$
a_i = \big| \nabla_{e_i} \log p_{t,k} \cdot e_i \big|_{\mathrm{fwd}} + \big| \nabla_{e'_{L-1-i}} \log p_{t,k} \cdot e'_{L-1-i} \big|_{\mathrm{rc}}
$$

It uses `torch.autograd.grad` without graph retention; the tests check that GPU memory returns to baseline.

## 6. Results

### 6.1 Splice-site prediction (held-out chr1,3,5,7,9; 62.7 M positions, 112,019 sites)

| | Donor | Acceptor |
| :--- | :---: | :---: |
| ROC-AUC | 0.9989 | 0.9984 |
| PR-AUC | 0.630 | 0.552 |
| Top-1 accuracy | 60.4% | 54.2% |
| Top-2k accuracy | 81.1% | 73.8% |

The untrained head scores PR-AUC 0.0007, which is chance. SpliceAI-10k reports top-1 ≈ 0.95 with a 10 kb context and an end-to-end trained model. Ours is a frozen 3.3 M-parameter backbone with 1 kb flanks, so this is a baseline, not a competing model.

### 6.2 Speculative ISM

Table 2 of `results_summary.tex` and the figures `figures/speedup_vs_length.png` and `figures/recall_alpha_vs_k.png` give the K × length ablation (664–2,164 bp, all real sequence). Across all 48 window × length runs (664–2,164 bp of real sequence), mean recall of high-impact variants is 0.28 at K=12, 0.39 at K=24, 0.60 at K=48, 0.87 at K=96 (of 192 candidates). Recall does not depend on hardware. Wall-clock speedups do: this ablation ran on battery in the power-saver profile, which slows the launch-overhead-bound sequential path. That makes exact batching look better (1.4–3.1×) than in the benchmark below (1.2×; its power state was not recorded). Re-run both on AC power, in the performance profile, before quoting times. The headline benchmark (40 held-out 2.2 kb windows, 192 candidates each, PR #21) measured: sequential 1,346 ms, batched exhaustive 1,126 ms (1.2×, exact), speculative 567 ms at K=96 (2.38×, recall 0.83, worst window 0.57) down to 148 ms at K=24 (9.10×, recall 0.31).

### 6.3 Interactive latency

Over the WebSocket server, a 1 kb edit (reference + mutant, forward + reverse-complement passes) takes median 12.5 ms round trip, p95 13.9 ms.

## 7. Negative results and limitations

- **Batching barely helps on this GPU.** HyenaDNA-small at ~2 kb saturates an RTX 5060 Laptop at batch 2–4. In an earlier exploratory run, batching was slower than sequential at 4 kb (0.8×) and equal at 8 kb. Speedups beyond ~1.2–1.4× come only from verifying fewer variants, and therefore cost recall.
- **A first-order gradient draft** ($|\nabla_e S \cdot (e_{\mathrm{alt}} - e_{\mathrm{ref}})|$, with $S$ the splice mass near the window) did not beat the heuristic draft: recall 0.65 vs 0.83 at K=96.
- **N-padding to emulate longer windows is invalid.** The model goes out of distribution and predicts no sites, so recall is undefined. The ablation therefore stays within the windows' real 1 kb flanks.
- **The originally proposed tree-attention verification** does not apply to an attention-free backbone. Medusa-style draft heads (#3) predict future positions' labels, not variant effects, so they are not used as the drafter.
- **The web demo** shows a synthetic region by default; the server can serve real GRCh38 windows.
- Paralogs are not removed from the test set, and the model has only 1 kb flanks.

## 8. Case study: ClinVar pathogenic splice variants

Two ClinVar *Pathogenic* canonical-site SNVs on the + strand from the #5 ground-truth set (`scripts/case_study.py`; figures `figures/case_*.png`). Each is scored by our model and compared with SpliceAI's published scores.

| Variant | ClinVar | Site | Ours Δ (peak) | SpliceAI Δ (published) | Peak offset, ours / SpliceAI |
| :--- | :--- | :--- | :---: | :---: | :---: |
| ABAT `chr16-8746099-G-A` | 635252 | donor | DL 0.937 | DL 0.996 | +0 / -1 |
| ABCG8 `chr2-43852353-G-C` | 1983599 | acceptor | AL 0.840 | AL 0.936 | -1 / +1 |

Both variants are called as loss of the correct site type with high confidence. Our scores are lower than SpliceAI's, consistent with the gap in §6.1. The 1–2 bp peak-offset differences are labelling conventions, not errors: we label the first base of the motif (G of GT, A of AG), while SpliceAI labels the exon-side boundary base. Saliency concentrates on the dinucleotide and its immediate consensus positions (figures).

# Constraints

- `SveltoBackbone`'s real-model path feeds `GenomicTokenizer` ids (PAD0 A1 C2 G3 T4 N5) straight into HF models. HyenaDNA uses a different char vocab and Nucleotide Transformer uses 6-mer tokens, so real checkpoints need their own tokenizer before any non-mock result means anything.
- Leakage split (test chr1,3,5,7,9 / val chr2,4,6,8,10 / train chr11-22) is not the SpliceAI split (SpliceAI trains on everything outside the test chroms, including chrX/Y, and drops paralogs from test). Gene-ID disjointness does not catch paralogs.

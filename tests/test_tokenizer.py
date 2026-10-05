import pytest
import torch
from svelto_dna.core.tokenizer import GenomicTokenizer


class TestGenomicTokenizer:
    def test_vocab_initialization(self):
        tokenizer = GenomicTokenizer()
        assert tokenizer.vocab["<PAD>"] == 0
        assert tokenizer.vocab["A"] == 1
        assert tokenizer.vocab["C"] == 2
        assert tokenizer.vocab["G"] == 3
        assert tokenizer.vocab["T"] == 4
        assert tokenizer.vocab["N"] == 5
        assert tokenizer.vocab_size == 6

    def test_lowercase_normalization(self):
        tokenizer = GenomicTokenizer()
        seq = "acgtacgtn"
        normalized = tokenizer.normalize(seq)
        assert normalized == "ACGTACGTN"

    def test_iupac_mapping_to_n(self):
        tokenizer = GenomicTokenizer(iupac_strategy="map_to_n")
        seq = "ARYSWKMBDHV"
        normalized = tokenizer.normalize(seq)
        # 'A' remains 'A', the rest are IUPAC codes mapped to 'N'
        assert normalized == "ANNNNNNNNNN"

    def test_iupac_mapping_to_first(self):
        tokenizer = GenomicTokenizer(iupac_strategy="first")
        seq = "ARY"
        normalized = tokenizer.normalize(seq)
        assert normalized == "AAC"  # R -> A, Y -> C

    def test_reverse_complement_canonical(self):
        tokenizer = GenomicTokenizer()
        assert tokenizer.reverse_complement("ATCG") == "CGAT"
        assert tokenizer.reverse_complement("AAAA") == "TTTT"
        assert tokenizer.reverse_complement("CCGG") == "CCGG"
        assert tokenizer.reverse_complement("ACGTN") == "NACGT"

    def test_reverse_complement_iupac(self):
        tokenizer = GenomicTokenizer()
        # R complements to Y, Y complements to R.
        # Sequence: "RY" -> complements are "YR", inverted direction -> "RY"
        assert tokenizer.reverse_complement("RY") == "RY"

    def test_context_window_extraction_centered(self):
        tokenizer = GenomicTokenizer()
        # Sequence of length 11: index 5 is 'T'
        seq = "AAAAATGGGGG"
        # Window size 5 around index 5: left_flank = 2, right_flank = 2
        # indices 3, 4, 5, 6, 7 -> "AATGG"
        window = tokenizer.extract_context_window(seq, locus_idx=5, window_size=5)
        assert window == "AATGG"
        assert len(window) == 5

    def test_context_window_extraction_boundary_left(self):
        tokenizer = GenomicTokenizer()
        seq = "ACGT"
        # Locus at 0, window size 6 -> left_flank = 2, right_flank = 3
        # indices -2, -1, 0, 1, 2, 3 -> pad 2 left, 4 actual, 0 right
        window = tokenizer.extract_context_window(seq, locus_idx=0, window_size=6, pad_char="N")
        assert window == "NNACGT"
        assert len(window) == 6

    def test_context_window_extraction_boundary_right(self):
        tokenizer = GenomicTokenizer()
        seq = "ACGT"
        # Locus at 3 (end), window size 6 -> left_flank = 2, right_flank = 3
        # indices 1, 2, 3, 4, 5, 6 -> 3 actual (CGT), 3 right padding
        window = tokenizer.extract_context_window(seq, locus_idx=3, window_size=6, pad_char="N")
        assert window == "CGTNNN"
        assert len(window) == 6

    def test_large_context_window_up_to_10k(self):
        tokenizer = GenomicTokenizer()
        seq = "ACGT" * 3000  # 12,000 bp
        for length in [1024, 2048, 5000, 10000]:
            window = tokenizer.extract_context_window(seq, locus_idx=6000, window_size=length)
            assert len(window) == length

    def test_batch_encode_and_decode(self):
        tokenizer = GenomicTokenizer()
        seqs = ["ACGT", "ACGTACGT"]
        encoded = tokenizer.encode(seqs, padding=True, return_tensors="pt")

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert input_ids.shape == (2, 8)
        assert attention_mask.shape == (2, 8)

        # First sequence is padded by 4 zeros at the end
        assert input_ids[0, 4:].tolist() == [0, 0, 0, 0]
        assert attention_mask[0, 4:].tolist() == [0, 0, 0, 0]

        # Decode without special tokens
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        assert decoded[0] == "ACGT"
        assert decoded[1] == "ACGTACGT"

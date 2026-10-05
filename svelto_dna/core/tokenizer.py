"""
Single-nucleotide resolution genomic tokenizer with IUPAC code support,
reverse complement strand inversion, and symmetrical context windowing.
"""

from typing import Dict, List, Optional, Sequence, Union
import torch

# Standard single-nucleotide vocabulary
DEFAULT_VOCAB: Dict[str, int] = {
    "<PAD>": 0,
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "N": 5,
}

# Standard IUPAC degenerate nucleotide mappings
IUPAC_MAP_TO_N: Dict[str, str] = {
    "R": "N",  # A or G
    "Y": "N",  # C or T
    "S": "N",  # G or C
    "W": "N",  # A or T
    "K": "N",  # G or T
    "M": "N",  # A or C
    "B": "N",  # C or G or T
    "D": "N",  # A or G or T
    "H": "N",  # A or C or T
    "V": "N",  # A or C or G
}

IUPAC_MAP_TO_FIRST: Dict[str, str] = {
    "R": "A",
    "Y": "C",
    "S": "G",
    "W": "A",
    "K": "G",
    "M": "A",
    "B": "C",
    "D": "A",
    "H": "A",
    "V": "A",
}

# Watson-Crick and IUPAC reverse complement pairings
COMPLEMENT_MAP: Dict[str, str] = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",
    "<PAD>": "<PAD>",
    # IUPAC complements
    "R": "Y",
    "Y": "R",
    "S": "S",
    "W": "W",
    "K": "M",
    "M": "K",
    "B": "V",
    "V": "B",
    "D": "H",
    "H": "D",
}


class GenomicTokenizer:
    """
    High-throughput single-nucleotide resolution tokenizer for genomic foundation models.
    Supports IUPAC degenerate resolution, reverse complement inversion, and symmetrical window extraction.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        iupac_strategy: str = "map_to_n",
        pad_token: str = "<PAD>",
        unk_token: str = "N",
    ) -> None:
        """
        Initialize the genomic tokenizer.

        Args:
            vocab: Optional mapping of token string to integer ID.
            iupac_strategy: Strategy for handling IUPAC degenerate codes:
                            'map_to_n' (default) -> maps R, Y, S, etc. to N.
                            'first' -> maps to first canonical base (e.g., R -> A).
                            'keep' -> preserves IUPAC character if in vocab, else unk.
            pad_token: Token used for padding sequences.
            unk_token: Token used for unknown or ambiguous nucleotides.
        """
        self.vocab = dict(vocab or DEFAULT_VOCAB)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]
        self.iupac_strategy = iupac_strategy

        if self.iupac_strategy == "map_to_n":
            self.iupac_map = IUPAC_MAP_TO_N
        elif self.iupac_strategy == "first":
            self.iupac_map = IUPAC_MAP_TO_FIRST
        else:
            self.iupac_map = {}

    @property
    def vocab_size(self) -> int:
        """Returns the number of unique tokens in the vocabulary."""
        return len(self.vocab)

    def normalize(self, sequence: str) -> str:
        """
        Clean, strip whitespaces, and uppercase input genomic sequence.
        Resolves IUPAC degenerate codes according to configured strategy.
        """
        cleaned = sequence.strip().upper()
        # Replace non-base characters or resolve IUPAC
        result = []
        for char in cleaned:
            if char in self.vocab:
                result.append(char)
            elif char in self.iupac_map:
                result.append(self.iupac_map[char])
            else:
                result.append(self.unk_token)
        return "".join(result)

    def reverse_complement(self, sequence: str) -> str:
        """
        Computes the reverse complement of a DNA sequence.
        Handles canonical bases (A, C, G, T), unknown (N), and IUPAC codes.
        """
        cleaned = sequence.strip().upper()
        complemented = [COMPLEMENT_MAP.get(b, self.unk_token) for b in cleaned]
        return "".join(reversed(complemented))

    def extract_context_window(
        self,
        sequence: str,
        locus_idx: int,
        window_size: int,
        pad_char: Optional[str] = None,
    ) -> str:
        """
        Extracts a symmetrical flanking window centered at the locus of interest.

        Args:
            sequence: Input nucleotide sequence.
            locus_idx: 0-indexed position of the locus within sequence.
            window_size: Total target context window length (e.g. 1024, 2048, 5000, 10000).
            pad_char: Character to pad if window exceeds sequence boundaries (default: unk_token).

        Returns:
            Extracted sequence of exact length `window_size`.
        """
        if pad_char is None:
            pad_char = self.unk_token

        norm_seq = self.normalize(sequence)
        seq_len = len(norm_seq)

        if seq_len == 0:
            return pad_char * window_size

        left_flank = (window_size - 1) // 2
        right_flank = window_size - 1 - left_flank

        start_idx = locus_idx - left_flank
        end_idx = locus_idx + right_flank + 1

        left_pad_len = max(0, -start_idx)
        right_pad_len = max(0, end_idx - seq_len)

        actual_start = max(0, start_idx)
        actual_end = min(seq_len, end_idx)

        extracted = norm_seq[actual_start:actual_end]
        padded = (pad_char * left_pad_len) + extracted + (pad_char * right_pad_len)

        assert len(padded) == window_size, f"Extracted length {len(padded)} != target {window_size}"
        return padded

    def tokenize(self, sequence: str) -> List[str]:
        """Splits normalized sequence into single-nucleotide tokens."""
        norm_seq = self.normalize(sequence)
        return list(norm_seq)

    def convert_tokens_to_ids(self, tokens: Sequence[str]) -> List[int]:
        """Maps token strings to discrete integer IDs."""
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        """Maps integer IDs back to token strings."""
        return [self.inv_vocab.get(i, self.unk_token) for i in ids]

    def encode(
        self,
        sequences: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        """
        Encodes sequence(s) into token IDs and attention masks.

        Args:
            sequences: Single DNA string or list of DNA strings.
            max_length: Optional truncate/pad length.
            padding: Whether to pad batch to longest sequence or max_length.
            return_tensors: "pt" for PyTorch tensors, None for raw python lists.

        Returns:
            Dictionary containing 'input_ids' and 'attention_mask'.
        """
        if isinstance(sequences, str):
            seq_list = [sequences]
        else:
            seq_list = list(sequences)

        all_ids = []
        for s in seq_list:
            norm_s = self.normalize(s)
            ids = self.convert_tokens_to_ids(list(norm_s))
            if max_length is not None:
                ids = ids[:max_length]
            all_ids.append(ids)

        # Determine target batch length
        if max_length is not None:
            target_len = max_length
        elif padding and all_ids:
            target_len = max(len(ids) for ids in all_ids)
        else:
            target_len = None

        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []

        for ids in all_ids:
            if target_len is not None and padding:
                pad_len = max(0, target_len - len(ids))
                padded_ids = ids + [self.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
            else:
                padded_ids = ids
                mask = [1] * len(ids)
            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            }

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
        }

    def decode(
        self,
        token_ids: Union[torch.Tensor, Sequence[int], Sequence[Sequence[int]]],
        skip_special_tokens: bool = False,
    ) -> Union[str, List[str]]:
        """
        Decodes token ID tensor or list back into nucleotide string(s).
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if not token_ids:
            return ""

        # Check if single sequence or batch
        if isinstance(token_ids[0], (int,)):
            # Single sequence
            tokens = self.convert_ids_to_tokens(token_ids)  # type: ignore
            if skip_special_tokens:
                tokens = [t for t in tokens if t != self.pad_token]
            return "".join(tokens)
        else:
            # Batch
            results = []
            for seq_ids in token_ids:  # type: ignore
                tokens = self.convert_ids_to_tokens(seq_ids)
                if skip_special_tokens:
                    tokens = [t for t in tokens if t != self.pad_token]
                results.append("".join(tokens))
            return results

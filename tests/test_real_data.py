"""
Issue #2: real data ingestion. Tiny hand-built GTF/FASTA fixtures with known answers.

Coordinate convention (matches StrandCoordinateResolver): a splice site is stored as the reference
position of the dinucleotide's first base on the + strand of the reference.
  + strand donor  "GT" at [p, p+1]   acceptor "AG" at [p, p+1]
  - strand donor  "AC" at [p, p+1]   acceptor "CT" at [p, p+1]   (reverse complements)
"""

import gzip
from pathlib import Path

import polars as pl
import pytest

from svelto_dna.data.annotation import load_canonical_transcripts, splice_site_index
from svelto_dna.data.genome import Genome
from svelto_dna.data.clinvar import ClinVarParser
from svelto_dna.data.spliceai import build_splice_windows
from svelto_dna.core.tokenizer import GenomicTokenizer

# + gene: exon1 [10,15) | intron [15,39) = GT...CAG | exon2 [39,44)
CHR = "N" * 10 + "CCCAG" + "GTAAGT" + "T" * 15 + "CAG" + "GCCCC" + "N" * 20
assert len(CHR) == 64 and CHR[15:17] == "GT" and CHR[37:39] == "AG"


def _gtf_line(chrom, start1, end1, strand, tid, gene, tags=("Ensembl_canonical",), gene_type="protein_coding"):
    attrs = f'gene_id "G{gene}"; transcript_id "{tid}"; gene_type "{gene_type}"; gene_name "{gene}";' + "".join(f' tag "{t}";' for t in tags)
    return f"{chrom}\tTEST\texon\t{start1}\t{end1}\t.\t{strand}\t.\t{attrs}\n"


@pytest.fixture
def fixtures(tmp_path: Path):
    # Build a minus-strand gene whose transcript reads ...exonA | GT...AG | exonB on the - strand.
    # On the reference that intron appears reverse-complemented: exonB_rc  CT ... AC  exonA_rc
    minus = "GGGAAA" + "CT" + "T" * 15 + "AC" + "CCTTT"   # ref [70,100); transcript reads GT A..A AG
    # ref positions: exonB_rc [70,76) "GGGAAA", acceptor "CT" at 76, ..., donor "AC" at 93, exonA_rc [95,100) "CCTTT"
    chrom = CHR + "NNNNNN" + minus + "NNNNNNNNNN"
    assert chrom[76:78] == "CT" and chrom[93:95] == "AC"
    fa = tmp_path / "Homo_sapiens.GRCh38.dna.chromosome.T1.fa.gz"
    with gzip.open(fa, "wt") as f:
        f.write(">T1 dna:chromosome\n" + "\n".join(chrom[i:i + 20] for i in range(0, len(chrom), 20)) + "\n")
    gtf = tmp_path / "test.gtf.gz"
    with gzip.open(gtf, "wt") as f:
        f.write("##comment\n")
        f.write(_gtf_line("chrT1", 11, 15, "+", "TX1", "PLUS"))     # 1-based inclusive → [10,15)
        f.write(_gtf_line("chrT1", 40, 44, "+", "TX1", "PLUS"))     # [39,44)
        f.write(_gtf_line("chrT1", 71, 76, "-", "TX2", "MINUS"))    # [70,76)
        f.write(_gtf_line("chrT1", 96, 100, "-", "TX2", "MINUS"))   # [95,100)
        f.write(_gtf_line("chrT1", 11, 44, "+", "TX3", "NONCANON", tags=()))           # not canonical: ignored
        f.write(_gtf_line("chrT1", 11, 44, "+", "TX4", "LNC", gene_type="lncRNA"))     # not protein coding: ignored
    return {"fa_dir": tmp_path, "gtf": gtf, "chrom": chrom}


def test_loads_only_canonical_protein_coding_transcripts(fixtures):
    tx = load_canonical_transcripts(fixtures["gtf"])
    assert sorted(t.transcript_id for t in tx) == ["TX1", "TX2"]
    plus = next(t for t in tx if t.transcript_id == "TX1")
    assert plus.chrom == "chrT1" and plus.strand == "+" and plus.exons == ((10, 15), (39, 44))


def test_plus_strand_sites(fixtures):
    plus = next(t for t in load_canonical_transcripts(fixtures["gtf"]) if t.strand == "+")
    assert plus.donors() == [15] and plus.acceptors() == [37]
    assert fixtures["chrom"][15:17] == "GT" and fixtures["chrom"][37:39] == "AG"


def test_minus_strand_sites_are_reverse_complement_dinucleotides(fixtures):
    minus = next(t for t in load_canonical_transcripts(fixtures["gtf"]) if t.strand == "-")
    assert minus.donors() == [93] and minus.acceptors() == [76]
    assert fixtures["chrom"][93:95] == "AC" and fixtures["chrom"][76:78] == "CT"


def test_genome_reads_ensembl_chromosome_fasta(fixtures):
    g = Genome(fixtures["fa_dir"])
    assert g.fetch("chrT1", 10, 21) == "CCCAGGTAAGT"
    assert g.fetch("T1", 15, 17) == "GT"
    assert g.length("chrT1") == len(fixtures["chrom"])


def test_genome_pads_with_n_outside_the_chromosome(fixtures):
    g = Genome(fixtures["fa_dir"])
    assert g.fetch("chrT1", -3, 2) == "NNNNN"


def test_clinvar_distance_comes_from_the_annotation_not_from_tags(fixtures, tmp_path):
    vcf = tmp_path / "clinvar.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.1\n"
        "T1\t17\t1\tT\tA\t.\t.\tCLNSIG=Pathogenic;GENEINFO=PLUS:1\n"            # 0-based 16: donor +2 of TX1
        "T1\t38\t2\tA\tG\t.\t.\tCLNSIG=Likely_pathogenic;GENEINFO=PLUS:1\n"     # 0-based 37: acceptor A
        "T1\t94\t3\tC\tT\t.\t.\tCLNSIG=Benign;GENEINFO=MINUS:2\n"               # 0-based 93: minus donor
        "T1\t200\t4\tG\tA\t.\t.\tCLNSIG=Pathogenic;GENEINFO=X:3\n"              # far from any site: dropped
    )
    sites = splice_site_index(load_canonical_transcripts(fixtures["gtf"]))
    df = ClinVarParser().parse_vcf(vcf, sites=sites)
    rows = {r["variant_id"]: r for r in df.iter_rows(named=True)}
    assert set(rows) == {"1", "2", "3"}
    assert rows["1"]["junction_type"] == "donor" and rows["1"]["dist_to_junction"] == 1 and rows["1"]["strand"] == "+"
    assert rows["2"]["junction_type"] == "acceptor" and rows["2"]["dist_to_junction"] == 0
    assert rows["3"]["junction_type"] == "donor" and rows["3"]["strand"] == "-"


def test_clinvar_vcf_without_annotation_is_rejected(tmp_path):
    vcf = tmp_path / "clinvar.vcf"
    vcf.write_text("##fileformat=VCFv4.1\n1\t100\t1\tA\tG\t.\t.\tCLNSIG=Pathogenic\n")
    with pytest.raises(ValueError, match="splice-site annotation"):
        ClinVarParser().parse_vcf(vcf)


def test_clinvar_tsv_keeps_only_grch38(tmp_path):
    tsv = tmp_path / "variant_summary.txt"
    tsv.write_text(
        "VariationID\tChromosome\tPositionVCF\tReferenceAlleleVCF\tAlternateAlleleVCF\tClinicalSignificance\tGeneSymbol\tAssembly\n"
        "1\t1\t100\tA\tG\tPathogenic\tX\tGRCh37\n"
        "2\t1\t200\tA\tG\tPathogenic\tX\tGRCh38\n"
    )
    from svelto_dna.data.annotation import Transcript
    sites = splice_site_index([Transcript("T", "G", "X", "chr1", "+", ((150, 199), (260, 300)))])   # donor at 199
    df = ClinVarParser().parse_tsv(tsv, sites=sites)
    assert df["variant_id"].to_list() == [2]
    assert df["dist_to_junction"].to_list() == [0]


def test_windows_are_transcript_oriented_with_labels_on_the_motif(fixtures):
    tx = load_canonical_transcripts(fixtures["gtf"])
    df = build_splice_windows(Genome(fixtures["fa_dir"]), tx, block=20, context=5)
    tok = GenomicTokenizer()
    for r in df.iter_rows(named=True):
        seq, labels = r["window_sequence"], r["labels"]
        assert len(seq) == 20 + 2 * 5 and len(labels) == 20
        for k, lab in enumerate(labels):
            if lab == 1:
                assert seq[5 + k: 5 + k + 2] == "GT", (r["transcript_id"], k)
            if lab == 2:
                assert seq[5 + k: 5 + k + 2] == "AG", (r["transcript_id"], k)
    assert sum(sum(1 for l in r["labels"] if l) for r in df.iter_rows(named=True)) == 4
    assert set(df["strand"].to_list()) == {"+", "-"}


@pytest.mark.parametrize("clnsig", [
    "Conflicting_classifications_of_pathogenicity",                  # ClinVar wording since 2024
    "Conflicting_interpretations_of_pathogenicity",                  # older wording
    "Conflicting_classifications_of_pathogenicity|risk_factor",
])
def test_conflicting_classifications_are_holdout_not_pathogenic(clnsig):
    assert ClinVarParser().classify_significance(clnsig) == (False, False, True)


@pytest.mark.parametrize("clnsig,expected", [
    ("Pathogenic", (True, False, False)),
    ("Pathogenic/Likely_pathogenic", (True, False, False)),
    ("Likely_benign", (False, True, False)),
    ("Benign/Likely_benign", (False, True, False)),
    ("Uncertain_significance", (False, False, True)),
    ("not_provided", (False, False, True)),
])
def test_clinvar_vcf_significance_terms(clnsig, expected):
    assert ClinVarParser().classify_significance(clnsig) == expected

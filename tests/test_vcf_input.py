"""VCF input handling: ##contig parsing, chromosome resolution, file checks, build guard.

Oracle: hand-written header strings with known contig IDs and lengths.
"""

import gzip

import pytest

from flexsweep.data import (
    check_vcf_file,
    get_contig_from_vcf_filename,
    infer_genome_build,
    parse_contig_headers,
)

PHASE3 = "##contig=<ID=22,assembly=b37,length=51304566>"
NYGC = "##contig=<ID=chr22,length=50818468>"
BODY = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"


def _write_records(path, contig_lines, records):
    header = "##fileformat=VCFv4.2\n" + "".join(c + "\n" for c in contig_lines)
    header += '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    body = "".join(f"{c}\t{p}\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\n" for c, p in records)
    path.write_text(header + BODY + body)
    return str(path)


def test_parse_contig_any_key_order():
    c = parse_contig_headers([PHASE3, NYGC, '##contig=<ID=chrM,length=16569,md5="x,y">'])
    assert c["22"]["length"] == "51304566"
    assert c["22"]["assembly"] == "b37"
    assert c["chr22"]["length"] == "50818468"
    assert c["chrM"]["md5"] == "x,y"


@pytest.mark.parametrize(
    "name,contigs,records,expected",
    [
        # phase3 header (assembly key before length)
        ("ALL.chr22.phase3.vcf", [PHASE3], [("22", 100)], ("22", 51304566)),
        # several contigs, no chromosome in the filename: the first record decides
        ("yri_example.vcf", [NYGC, "##contig=<ID=chr21,length=46709983>"], [("chr21", 100)],
         ("chr21", 46709983)),
        # a number in the filename that is not the chromosome
        ("batch_2.vcf", ["##contig=<ID=chr2,length=242193529>", NYGC], [("chr22", 100)],
         ("chr22", 50818468)),
    ],
)
def test_contig_resolution(tmp_path, name, contigs, records, expected):
    f = _write_records(tmp_path / name, contigs, records)
    assert get_contig_from_vcf_filename(f) == expected


def test_missing_length_falls_back_to_last_position(tmp_path):
    f = _write_records(
        tmp_path / "yri_chr22.vcf", ["##contig=<ID=chr22>"], [("chr22", 100), ("chr22", 98765)]
    )
    with pytest.warns(UserWarning, match="no length="):
        assert get_contig_from_vcf_filename(f) == ("chr22", 98765)


def test_infer_build():
    assert infer_genome_build(parse_contig_headers(["##contig=<ID=chr1,length=248956422>"])) == "GRCh38"
    assert infer_genome_build(parse_contig_headers(["##contig=<ID=1,assembly=b37,length=249250621>"])) == "GRCh37"
    assert infer_genome_build(parse_contig_headers([PHASE3])) == "GRCh37"
    assert infer_genome_build(parse_contig_headers([NYGC])) is None


def test_plain_gzip_rejected(tmp_path):
    f = tmp_path / "x_chr22.vcf.gz"
    with gzip.open(f, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n")
    with pytest.raises(ValueError, match="not bgzip-compressed"):
        check_vcf_file(str(f))


def test_missing_index(tmp_path):
    f = _write_records(tmp_path / "x_chr22.vcf", [NYGC], [("chr22", 100)])
    with pytest.raises(FileNotFoundError, match="tabix -p vcf"):
        check_vcf_file(f)

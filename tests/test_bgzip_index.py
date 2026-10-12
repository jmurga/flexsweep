"""polarize.bgzip_index writes a BGZF VCF plus a tabix index that htslib can query.

Oracle: htslib ``tabix`` (when installed) queried with the index written by
bgzip_index must return exactly the records whose POS lies in the region, enumerated
from the plain-text input.
"""

import gzip
import shutil
import subprocess

import pytest

from flexsweep.data import check_vcf_file
from flexsweep.polarize import bgzip_index

POSITIONS = list(range(100, 2_000_000, 997))  # spans many 16 kb linear-index windows


def _write_plain(path):
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=chr2,length=3000000>\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n")
        for i, p in enumerate(POSITIONS):
            fh.write(f"chr2\t{p}\t.\tA\tC\t.\t.\t.\tGT\t0|1\t{i % 2}|0\n")


def test_bgzip_index_roundtrip(tmp_path):
    plain = tmp_path / "in.vcf"
    _write_plain(plain)
    out = str(tmp_path / "out.vcf.gz")
    tbi = bgzip_index(str(plain), out)
    assert tbi == out + ".tbi"
    check_vcf_file(out)  # BGZF magic + index present
    with gzip.open(out, "rt") as fh, open(plain) as ref:
        got, want = fh.read().splitlines(), ref.read().splitlines()
    # records byte-identical; noodles re-serializes the header (line order may change)
    assert [ln for ln in got if not ln.startswith("#")] == [
        ln for ln in want if not ln.startswith("#")
    ]
    assert {ln for ln in got if ln.startswith("#")} == {ln for ln in want if ln.startswith("#")}


def test_bgzip_index_rejects_plain_output(tmp_path):
    plain = tmp_path / "in.vcf"
    _write_plain(plain)
    with pytest.raises(RuntimeError, match=r"\.gz"):
        bgzip_index(str(plain), str(tmp_path / "out.vcf"))


@pytest.mark.skipif(shutil.which("tabix") is None, reason="htslib tabix not installed")
@pytest.mark.parametrize(
    "start,end", [(1, 50), (100, 100), (15_000, 70_000), (999_000, 1_100_000), (1_990_000, 3_000_000)]
)
def test_tabix_queries_match_text_filter(tmp_path, start, end):
    plain = tmp_path / "in.vcf"
    _write_plain(plain)
    out = str(tmp_path / "out.vcf.gz")
    bgzip_index(str(plain), out)
    res = subprocess.run(
        ["tabix", out, f"chr2:{start}-{end}"], capture_output=True, text=True, check=True
    )
    got = [int(ln.split("\t")[1]) for ln in res.stdout.splitlines()]
    assert got == [p for p in POSITIONS if start <= p <= end]

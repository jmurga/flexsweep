"""Example-data readers and writers (flexsweep example, data step), offline.

Oracle: hand-written MAF blocks, BED intervals and VCF records with known coordinates
and genotypes; the expected kept blocks, rows, intervals, records and AC/AN are
enumerated by hand.
"""

import gzip
import os

import pytest

from flexsweep import example


def _maf(path, sep):
    blocks = [
        # ref 0-based start, size, other species
        ("Homo_sapiens.chr22", 100, 10, ["Pan_troglodytes.CM1", "Mus_musculus.chr1"]),
        ("Homo_sapiens.chr22", 1000, 10, ["Pan_troglodytes.CM1", "Pongo_abelii.X"]),
        ("Homo_sapiens.chr22", 5000, 10, ["Gorilla_gorilla.G1"]),
    ]
    lines = ["##maf version=1 scoring=N/A", ""]
    for ref, start, size, others in blocks:
        lines.append("a")
        lines.append(sep.join(["s", ref, str(start), str(size), "+", "50818468", "A" * size]))
        for o in others:
            lines.append(sep.join(["s", o, "0", str(size), "+", "1000", "C" * size]))
        lines.append("")
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


@pytest.mark.parametrize("sep", ["\t", " "])
def test_region_subset_and_outgroups(tmp_path, sep):
    src = _maf(tmp_path / "in.maf.gz", sep)
    out = str(tmp_path / "out.maf.gz")
    keep = {"Pan_troglodytes", "Gorilla_gorilla", "Pongo_abelii"}
    # Region overlaps the second block only
    n = example._fetch_maf(src, "chr22", 900, 2000, keep, out)
    assert n == 1
    with gzip.open(out, "rt") as fh:
        rows = [ln.split()[1] for ln in fh if ln.startswith("s")]
    assert rows == ["Homo_sapiens.chr22", "Pan_troglodytes.CM1", "Pongo_abelii.X"]


def test_bed_reader_sorts_and_merges(tmp_path):
    bed = tmp_path / "m.bed.gz"
    with gzip.open(bed, "wt") as fh:
        fh.write(
            "chr21\t0\t100\nchr22\t500\t600\nchr22\t10\t20\nchr22\t550\t700\n"
            "chr22\t9000\t9100\n"
        )
    starts, ends = example._read_bed(str(bed), "chr22", 0, 1000)
    assert (starts, ends) == ([10, 500], [20, 700])


_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr22,length=50818468>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc\n"
)


def _records(path):
    with gzip.open(path, "rt") if str(path).endswith(".gz") else open(path) as fh:
        return [ln.rstrip("\n").split("\t") for ln in fh if not ln.startswith("#")]


def test_genotypes_region_snps_polymorphic_and_counts(tmp_path):
    src = tmp_path / "src.vcf.gz"
    with gzip.open(src, "wt") as fh:
        fh.write(_HEADER)
        rows = [
            (5, "A", "G", "0|1\t1|1\t0|0"),    # before region
            (10, "A", "G", "0|1\t1|1\t0|0"),   # kept; in a,b: AC 3, AN 4
            (20, "A", "G", "0|0\t0|0\t1|1"),   # monomorphic in a,b
            (30, "A", "G", "1|1\t1|1\t0|1"),   # fixed ALT in a,b
            (40, "AT", "A", "0|1\t0|0\t0|0"),  # indel
            (50, "C", "T", "0|.\t1|0\t0|0"),   # kept; missing allele: AC 1, AN 3
            (60, "C", "T", "0|1\t1|0\t0|0"),   # after region
        ]
        for pos, ref, alt, gts in rows:
            fh.write(f"chr22\t{pos}\t.\t{ref}\t{alt}\t.\t.\tAF=0.1\tGT\t{gts}\n")
    out = tmp_path / "raw.vcf"
    n = example._fetch_genotypes(str(src), "chr22", 10, 50, ["a", "b"], str(out))
    recs = _records(out)
    assert n == 2
    assert [(r[1], r[7], r[9:]) for r in recs] == [
        ("10", "AC=3;AN=4;AF=0.75", ["0|1", "1|1"]),
        ("50", "AC=1;AN=3;AF=0.333333", ["0|.", "1|0"]),
    ]
    with open(out) as fh:
        assert fh.read().count("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n") == 1
    assert example.vcf_samples(str(src)) == ["a", "b", "c"]


def test_mask_drops_sites_inside_intervals(tmp_path):
    src = tmp_path / "pol.vcf"
    with open(src, "w") as fh:
        fh.write(_HEADER)
        for pos in (10, 11, 20, 21, 100):
            fh.write(f"chr22\t{pos}\t.\tA\tG\t.\t.\t.\tGT\t0|1\t0|0\t0|0\n")
    out = str(tmp_path / "final.vcf.gz")
    # BED [10, 20) covers 1-based POS 11..20; [99, 100) covers POS 100
    n = example._write_final(str(src), out, [10, 99], [20, 100])
    assert n == 2
    assert [r[1] for r in _records(out)] == ["10", "21"]
    assert os.path.exists(out + ".tbi")
    assert not os.path.exists(out + ".tmp.vcf")


class _FakeServer:
    """urlopen stand-in: answers with the given HTTP codes, then serves ``body``."""

    def __init__(self, codes, body=b"data"):
        self.codes, self.body, self.calls = list(codes), body, 0

    def __call__(self, url, timeout=None):
        import email.message
        import io
        import urllib.error

        self.calls += 1
        if self.codes:
            hdrs = email.message.Message()
            hdrs["Retry-After"] = "1"
            raise urllib.error.HTTPError(url, self.codes.pop(0), "err", hdrs, io.BytesIO())
        return io.BytesIO(self.body)


def test_download_retries_on_rate_limit(tmp_path, monkeypatch):
    import hashlib

    fake, slept = _FakeServer([429, 429]), []
    monkeypatch.setattr(example.urllib.request, "urlopen", fake)
    monkeypatch.setattr(example.time, "sleep", slept.append)
    dest = str(tmp_path / "f.gz")
    example._download("https://x/f.gz", dest, hashlib.md5(b"data").hexdigest())
    assert fake.calls == 3 and slept == [1.0, 1.0]
    assert open(dest, "rb").read() == b"data"


def test_download_gives_up_on_client_error(tmp_path, monkeypatch):
    fake = _FakeServer([404])
    monkeypatch.setattr(example.urllib.request, "urlopen", fake)
    monkeypatch.setattr(example.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match=r"\(HTTP 404\) after 1 tries"):
        example._download("https://x/f.gz", str(tmp_path / "f.gz"))
    assert fake.calls == 1

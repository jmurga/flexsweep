"""CLI smoke tests: every documented command must run end to end.

These guard the reviewer-reported failures (cnn/scan crashing, scan writing one stat's
table under another stat's name). They check that commands exit 0 and that outputs have
the expected structure; they make no claim about statistical accuracy.

The end-to-end chain (simulator -> fvs-discoal -> cnn, plus scan on a VCF built from a
simulated replicate) takes a few minutes; skip it with ``-m "not slow"``.
"""

import os

import polars as pl
import pytest
from click.testing import CliRunner

from flexsweep.main import cli

COMMANDS = [
    "simulator",
    "fvs-discoal",
    "fvs-vcf",
    "recombination-bins",
    "cnn",
    "dann",
    "rank",
    "enrichment",
    "enrichment-plot",
    "polarize",
    "split-maf",
    "scan",
    "scan-plot",
    "example",
]


@pytest.mark.parametrize("cmd", COMMANDS)
def test_help(cmd):
    result = CliRunner().invoke(cli, [cmd, "--help"])
    assert result.exit_code == 0, result.output


def _run(args):
    result = CliRunner().invoke(cli, args, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result


def _ms_to_vcf(ms_file: str, vcf_path: str, chrom: str = "chr22") -> None:
    """Write one simulated replicate as a bgzipped, indexed, phased VCF (REF = ancestral)."""
    from flexsweep.fv import parse_ms_numpy
    from flexsweep.polarize import bgzip_index

    hap, _, _, _, positions, _ = parse_ms_numpy(ms_file)
    n_ind = hap.shape[1] // 2
    plain = vcf_path + ".tmp.vcf"
    last = 0
    with open(plain, "w") as out:
        out.write(f"##fileformat=VCFv4.2\n##contig=<ID={chrom},length=1200000>\n")
        out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                  + "\t".join(f"ind{j}" for j in range(n_ind)) + "\n")
        for i, pos in enumerate(positions):
            pos = max(int(pos), last + 1)
            last = pos
            gts = "\t".join(
                f"{int(hap[i, 2 * j])}|{int(hap[i, 2 * j + 1])}" for j in range(n_ind)
            )
            out.write(f"{chrom}\t{pos}\t.\tA\tG\t.\t.\t.\tGT\t{gts}\n")
    bgzip_index(plain, vcf_path)
    os.remove(plain)


@pytest.mark.slow
def test_end_to_end(tmp_path):
    sims = str(tmp_path / "sims")
    _run(
        [
            "simulator",
            "--sample_size", "20",
            "--demes", "yri",
            "--output_folder", sims,
            "--num_simulations", "60",
            "--nthreads", "2",
            "--seed", "1",
        ]
    )
    params = pl.read_csv(os.path.join(sims, "params.txt.gz"))
    assert {"seed1", "seed2"} <= set(params.columns)

    _run(["fvs-discoal", "--simulations_path", sims, "--nthreads", "2"])
    assert os.path.exists(os.path.join(sims, "fvs.parquet"))

    out = str(tmp_path / "cnn")
    _run(
        [
            "cnn",
            "--train_data", os.path.join(sims, "fvs.parquet"),
            "--predict_data", os.path.join(sims, "fvs.parquet"),
            "--output_folder", out,
            "--seed", "1",
        ]
    )
    assert os.path.exists(os.path.join(out, "model.keras"))

    # scan: per-SNP and window stats together, then a window stat alone
    vcf_dir = tmp_path / "vcfs"
    vcf_dir.mkdir()
    _ms_to_vcf(os.path.join(sims, "neutral", "neutral_1.ms.gz"), str(vcf_dir / "sim_chr22.vcf.gz"))
    _run(["scan", "--vcf_path", str(vcf_dir), "--stats", "ihs,nsl,h12", "--out_prefix", str(vcf_dir / "t"), "--nthreads", "2"])
    h12 = pl.read_csv(vcf_dir / "t.h12.txt", separator="\t")
    assert "h12" in h12.columns and "nsl" not in h12.columns
    nsl = pl.read_csv(vcf_dir / "t.nsl.txt", separator="\t")
    assert "nsl" in nsl.columns and "h12" not in nsl.columns

    _run(["scan", "--vcf_path", str(vcf_dir), "--stats", "h12", "--out_prefix", str(vcf_dir / "t2"), "--nthreads", "2"])
    assert "h12_pvalue" in pl.read_csv(vcf_dir / "t2.h12.txt", separator="\t").columns

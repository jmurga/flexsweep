"""Contig names are labels: any name must run, and no name may change a statistic.

Oracle: invariance. The same genotypes under different contig names must give identical
numbers, and the output must carry the name from the VCF. Numeric names (chr22) are the
reference because they worked before non-numeric names did.

Not covered here: full fvs-vcf normalization on real data, one VCF holding several contigs
(unsupported, see docs/data_preparation.rst), and cnn.py or utils.py name handling.
"""

import os

import numpy as np
import polars as pl
import pytest

from flexsweep.data import check_one_vcf_per_contig, contig_ids, parse_region
from flexsweep.fv import genome_reader, run_fs_stats
from flexsweep.polarize import bgzip_index
from flexsweep.scan import scan

NAMES = ["SUPER__1", "SUPER__10", "chrX", "NC_000001.11", "scaffold-1", "HLA:DRB1"]


def _write_vcf(path: str, chrom: str, seed: int = 1, n_sites: int = 400, n_ind: int = 10) -> None:
    """Bgzipped, indexed, phased VCF with reproducible genotypes (REF = ancestral)."""
    rng = np.random.default_rng(seed)
    # Allele frequencies across the range so every statistic has focal SNPs.
    freqs = rng.uniform(0.05, 0.95, n_sites)
    hap = (rng.random((n_sites, 2 * n_ind)) < freqs[:, None]).astype(int)
    positions = np.sort(rng.choice(np.arange(1, 2_000_000), n_sites, replace=False))
    plain = path + ".tmp.vcf"
    with open(plain, "w") as out:
        out.write(f"##fileformat=VCFv4.2\n##contig=<ID={chrom},length=2000000>\n")
        out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        out.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(f"ind{j}" for j in range(n_ind))
            + "\n"
        )
        for i, pos in enumerate(positions):
            gts = "\t".join(f"{hap[i, 2 * j]}|{hap[i, 2 * j + 1]}" for j in range(n_ind))
            out.write(f"{chrom}\t{pos}\t.\tA\tG\t.\t.\t.\tGT\t{gts}\n")
    bgzip_index(plain, path)
    os.remove(plain)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """genome_reader output for the numeric-name reference."""
    path = str(tmp_path_factory.mktemp("ref") / "ref.vcf.gz")
    _write_vcf(path, "chr22")
    return genome_reader(path)


@pytest.mark.parametrize("name", NAMES)
def test_genome_reader_is_numeric_and_name_invariant(tmp_path, reference, name):
    path = str(tmp_path / "a.vcf.gz")
    _write_vcf(path, name)
    hap, rec_map, ac, _, pos, gen = genome_reader(path)
    assert rec_map.dtype != object
    assert np.issubdtype(rec_map.dtype, np.number)
    r_hap, r_rec_map, r_ac, _, r_pos, r_gen = reference
    np.testing.assert_array_equal(hap, r_hap)
    np.testing.assert_array_equal(ac, r_ac)
    np.testing.assert_array_equal(pos, r_pos)
    np.testing.assert_array_equal(gen, r_gen)
    # Only the placeholder column may differ.
    np.testing.assert_array_equal(rec_map[:, 1:], r_rec_map[:, 1:])


@pytest.mark.parametrize("name", NAMES)
def test_fs_stats_compile_and_are_name_invariant(tmp_path, reference, name):
    """The numba call that raised the original TypingError."""
    path = str(tmp_path / "a.vcf.gz")
    _write_vcf(path, name)
    hap, rec_map, ac, *_ = genome_reader(path)
    out = run_fs_stats(hap, ac, rec_map)
    r_hap, r_rec_map, r_ac, *_ = reference
    r_out = run_fs_stats(r_hap, r_ac, r_rec_map)
    assert out[0].height > 0
    for got, expected in zip(out, r_out):
        assert got.equals(expected)


def _scan_values(prefix: str, stat: str = "pi") -> pl.DataFrame:
    return pl.read_csv(f"{prefix}.{stat}.txt", separator="\t", schema_overrides={"chr": pl.String})


@pytest.mark.parametrize("name", ["SUPER__1", "chrX", "scaffold-1", "HLA:DRB1", "22"])
def test_scan_keeps_real_name_and_values(tmp_path, name):
    ref_dir, dir_ = tmp_path / "ref", tmp_path / "other"
    ref_dir.mkdir()
    dir_.mkdir()
    _write_vcf(str(ref_dir / "a.vcf.gz"), "chr22")
    _write_vcf(str(dir_ / "a.vcf.gz"), name)
    for d in (ref_dir, dir_):
        scan(str(d), str(d / "out"), ["pi", "ihs"], nthreads=1)
    for stat in ("pi", "ihs"):
        ref, got = _scan_values(str(ref_dir / "out"), stat), _scan_values(str(dir_ / "out"), stat)
        assert got["chr"].unique().to_list() == [name]
        assert got.drop("chr").equals(ref.drop("chr"))


def test_scan_keeps_contigs_apart_in_natural_order(tmp_path):
    """Non-numeric contigs must not collapse into one, and SUPER__2 sorts before SUPER__10."""
    for i, name in enumerate(["SUPER__10", "SUPER__2", "SUPER__1"]):
        _write_vcf(str(tmp_path / f"f{i}.vcf.gz"), name, seed=i + 1)
    scan(str(tmp_path), str(tmp_path / "out"), ["pi"], nthreads=1)
    labels = _scan_values(str(tmp_path / "out"))["chr"].unique(maintain_order=True).to_list()
    assert labels == ["SUPER__1", "SUPER__2", "SUPER__10"]


def test_scan_rejects_two_vcfs_for_one_contig(tmp_path):
    for f in ("a.vcf.gz", "b.vcf.gz"):
        _write_vcf(str(tmp_path / f), "SUPER__1")
    with pytest.raises(ValueError, match="a.vcf.gz.*b.vcf.gz"):
        scan(str(tmp_path), str(tmp_path / "out"), ["pi"], nthreads=1)


@pytest.mark.parametrize(
    "names, expected",
    [
        (["chr10", "chr2", "chr1"], ["chr1", "chr2", "chr10"]),
        (["SUPER__10", "SUPER__2"], ["SUPER__2", "SUPER__10"]),
        (["chrX", "chr22", "chr2"], ["chr2", "chr22", "chrX"]),
        (["10", "9", "X"], ["9", "10", "X"]),
        (["a.2", "a.10", "b1"], ["a.2", "a.10", "b1"]),
        (["2", "chr1"], ["2", "chr1"]),  # mixed naming must not raise
    ],
)
def test_contig_ids_natural_order(names, expected):
    ids = contig_ids(names + names[:1])  # a repeated name gets one number
    assert sorted(ids, key=ids.get) == expected
    assert sorted(ids.values()) == list(range(1, len(expected) + 1))


@pytest.mark.parametrize(
    "region, expected",
    [
        ("chr1:1-1200000", ("chr1", 1, 1200000)),
        ("SUPER__1:5-10", ("SUPER__1", 5, 10)),
        ("scaffold-1:5-10", ("scaffold-1", 5, 10)),
        ("HLA:DRB1:5-10", ("HLA:DRB1", 5, 10)),
        ("NC_000001.11:5-10", ("NC_000001.11", 5, 10)),
    ],
)
def test_parse_region(region, expected):
    assert parse_region(region) == expected


@pytest.mark.parametrize("region", ["chr1", "chr1:5", ":5-10", "chr1:a-b", "chr1:5-", "chr1-5-10"])
def test_parse_region_rejects_other_formats(region):
    with pytest.raises(ValueError, match="start-end"):
        parse_region(region)


def test_check_one_vcf_per_contig():
    check_one_vcf_per_contig({"a.vcf.gz": "SUPER__1", "b.vcf.gz": "SUPER__11"})
    with pytest.raises(ValueError, match="a.vcf.gz.*b.vcf.gz.*SUPER__1"):
        check_one_vcf_per_contig({"a.vcf.gz": "SUPER__1", "b.vcf.gz": "SUPER__1"})


# -- recombination map and accessibility mask ----------------------------------


@pytest.mark.parametrize("name", ["SUPER__1", "chrX", "scaffold-1"])
def test_recombination_map_and_mask_match_real_name(tmp_path, reference, name):
    path = str(tmp_path / "a.vcf.gz")
    _write_vcf(path, name)
    rec = tmp_path / "map.tsv"
    # A linear map on this contig and a different one on another: the wrong contig or an
    # empty match would give other values.
    rows = ["chr\tstart\tend\tcm_mb\tcm"]
    for c, slope in ((name, 1.0), ("other", 7.0)):
        rows += [f"{c}\t{s}\t{s + 500000}\t{slope}\t{slope * (s + 500000) / 1e6}" for s in range(0, 2000000, 500000)]
    rec.write_text("\n".join(rows) + "\n")
    *_, pos, gen = genome_reader(path, recombination_map=str(rec))
    np.testing.assert_allclose(gen, pos / 1e6, atol=1e-9)

    bed = tmp_path / "mask.bed"
    bed.write_text(f"{name}\t0\t1000000\nother\t1000000\t2000000\n")
    *_, masked_pos, _ = genome_reader(path, accessibility_mask=str(bed))
    np.testing.assert_array_equal(masked_pos, reference[4][reference[4] > 1000000])


# -- cnn ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "regions, chrs",
    [
        (["chr10:1-5", "chr2:1-5", "chr2:6-9"], ["chr10", "chr2", "chr2"]),
        (["SUPER__10:1-5", "SUPER__2:1-5"], ["SUPER__10", "SUPER__2"]),
        (["scaffold-1:1-5", "HLA:DRB1:1-5", "chrX:1-5"], ["scaffold-1", "HLA:DRB1", "chrX"]),
    ],
)
def test_cnn_region_columns(regions, chrs):
    from flexsweep.cnn import _region_columns

    df = pl.DataFrame({"region": regions}).with_columns(*_region_columns(np.array(regions)))
    assert df["chr"].to_list() == chrs
    assert df["start"].dtype == pl.Int64 and df["end"].dtype == pl.Int64
    ids = contig_ids(chrs)
    assert df["nchr"].to_list() == [ids[c] for c in chrs]


@pytest.mark.parametrize("name", ["SUPER__1", "chrX", "scaffold-1", "HLA:DRB1", "22"])
def test_subset_genomic_windows_keeps_any_name(name):
    from flexsweep.cnn import subset_genomic_windows

    def frame(contig):
        return pl.DataFrame(
            {
                "iter": [f"{contig}:1-1200000"],
                **{c: [0.0] for c in ("s", "t", "f_i", "f_t", "mu", "r")},
                "model": ["sweep"],
                "pi_100000_600000": [1.5],
            }
        )

    ref = subset_genomic_windows(frame("chr22"), [600000], ["pi"])
    got = subset_genomic_windows(frame(name), [600000], ["pi"])
    assert ref["iter"][0].startswith("chr22:")
    assert got["iter"].to_list() == [ref["iter"][0].replace("chr22", name)]


# -- plots ----------------------------------------------------------------------


def _windows(names, n=30, seed=0):
    rng = np.random.default_rng(seed)
    return pl.concat(
        pl.DataFrame(
            {
                "chr": [c] * n,
                "start": 100000 * np.arange(n) + 1,
                "end": 100000 * np.arange(n) + 1200000,
                "prob_sweep": rng.uniform(0.01, 0.99, n),
                "cm_mb": rng.uniform(0.1, 2.0, n),
            }
        )
        for c in names
    )


def _xlabels(fig) -> list[str]:
    return [t.get_text() for ax in fig.axes for t in ax.get_xticklabels() if t.get_text()]


def test_manhattan_keeps_every_contig_apart_in_natural_order():
    """scaffold_1 and contig_1 must not merge, and SUPER__30 must not be dropped."""
    from flexsweep.utils import plot_manhattan

    names = ["SUPER__30", "SUPER__2", "scaffold_1", "contig_1", "SUPER__10"]
    fig = plot_manhattan(_windows(names))
    assert _xlabels(fig)[:5] == ["SUPER__2", "SUPER__10", "SUPER__30", "contig_1", "scaffold_1"]


def test_manhattan_human_labels_and_order_are_kept():
    from flexsweep.utils import plot_manhattan

    fig = plot_manhattan(_windows(["chrX", "chr10", "chr2", "chr1"]))
    assert _xlabels(fig)[:4] == ["1", "2", "10", "X"]


def test_manhattan_facets_and_region_use_real_name():
    from flexsweep.utils import plot_manhattan

    df = _windows(["SUPER__10", "SUPER__2"])
    fig = plot_manhattan(df, facet=True)
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titles == ["SUPER__2", "SUPER__10"]
    assert plot_manhattan(df, chrom="SUPER__10", center=1500000, window_bp=500000) is not None
    with pytest.raises(ValueError, match="SUPER__3"):
        plot_manhattan(df, chrom="SUPER__3")


def test_manhattan_recombination_map_matches_real_name(tmp_path):
    from flexsweep.utils import plot_manhattan

    rec = tmp_path / "map.tsv"
    rec.write_text(
        "chr\tstart\tend\tcm_mb\tcm\n"
        + "".join(f"SUPER__2\t{s}\t{s + 1000000}\t1.0\t{(s + 1000000) / 1e6}\n" for s in range(0, 5000000, 1000000))
    )
    df = _windows(["SUPER__2"]).drop("cm_mb")
    fig = plot_manhattan(df, recombination_map=str(rec), chrom="SUPER__2")
    assert any("Recombination" in ax.get_ylabel() for ax in fig.axes)


def test_sweep_density_uses_real_names_in_natural_order():
    from flexsweep.utils import plot_sweep_density

    fig = plot_sweep_density(_windows(["SUPER__10", "SUPER__2", "SUPER__1"]))
    titles = [ax.get_title().split()[0] for ax in fig.axes if ax.get_title()]
    assert titles == ["SUPER__1", "SUPER__2", "SUPER__10"]


def test_plot_diversity_keeps_contigs_in_natural_order(tmp_path):
    from flexsweep.utils import plot_diversity

    for i, name in enumerate(["SUPER__10", "SUPER__2"]):
        _write_vcf(str(tmp_path / f"f{i}.vcf.gz"), name, seed=i + 1)
    df = plot_diversity(str(tmp_path), out=str(tmp_path / "d.png"))
    assert df["contig"].unique(maintain_order=True).to_list() == ["SUPER__2", "SUPER__10"]


def test_interpolate_rates_keeps_real_names(tmp_path):
    from flexsweep.utils import interpolate_rates

    names = ["SUPER__10", "SUPER__2"]
    pred = tmp_path / "pred.csv"
    _windows(names).drop("cm_mb").write_csv(pred)
    rec = tmp_path / "map.tsv"
    rec.write_text(
        "chr\tstart\tend\tcm_mb\tcm\n"
        + "".join(
            f"{c}\t{s}\t{s + 1000000}\t1.0\t{(s + 1000000) / 1e6}\n"
            for c in names
            for s in range(0, 6000000, 1000000)
        )
    )
    out = interpolate_rates(str(pred), str(rec))
    assert out.height == 60 and out["cm_mb"].null_count() == 0
    assert sorted(out["chr"].unique().to_list()) == sorted(names)


def _scan_table(path, names, n=300):
    rng = np.random.default_rng(0)
    pl.concat(
        pl.DataFrame(
            {
                "chr": [c] * n,
                "positions": np.sort(rng.integers(1, 5_000_000, n)),
                "ihs": rng.normal(-1.0, 1.0, n),
            }
        )
        for c in names
    ).write_csv(path, separator="\t")


def test_scan_plot_natural_order_real_name_and_files(tmp_path):
    from flexsweep.utils import plot_scan

    f = tmp_path / "x.ihs.txt"
    _scan_table(f, ["SUPER__10", "SUPER__2", "scaffold-1"])
    fig = plot_scan(str(f), "ihs", out=str(tmp_path / "all.svg"))
    annotated = [t.get_text() for t in fig.axes[-1].texts]
    assert annotated == ["SUPER__2", "SUPER__10", "scaffold-1"]

    plot_scan(str(f), "ihs", split_chr=True, out=str(tmp_path / "s.svg"))
    for name in ("SUPER__2", "SUPER__10", "scaffold-1"):
        assert (tmp_path / f"s.{name}.svg").exists()


def test_scan_plot_region_uses_real_name(tmp_path):
    from flexsweep.utils import plot_scan

    f = tmp_path / "x.ihs.txt"
    _scan_table(f, ["SUPER__10", "SUPER__2"])
    fig = plot_scan(str(f), "ihs", chrom="SUPER__10", center=2_000_000, out=str(tmp_path / "r.svg"))
    assert fig.axes[-1].get_xlabel() == "SUPER__10 position (Mb)"

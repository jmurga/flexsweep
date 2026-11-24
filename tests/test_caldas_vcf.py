"""Caldas (per-window centers) geometry on the VCF path, matched to the simulations.

Oracles:
- The list-centers path, called once per window size: a {window: [centers]} mapping in
  one pass must give the same numbers (exactly for window statistics; the six fs
  statistics at windows larger than the smallest differ by design, because one pass uses
  min(windows)/2 as their neighbourhood, as in the simulations).
- The simulation-path cut_snps on the same SNP table shifted by the locus start.
- Brute-force enumeration of the unique (absolute center, window) pairs (deduplication).
- Hand z-scores of window statistics grouped by (center, window, r_bins), with the
  recombination rate of each locus computed from the map file with numpy.

Not covered: real data (checked on the workstation, see caldas_vcf/run.sh), CMS, scan.
"""

import os
import pickle

import click
import numpy as np
import polars as pl
import pytest
from click.testing import CliRunner

from flexsweep import fv
from flexsweep.main import caldas_window_centers

L = 1_200_000
TESTS = os.path.dirname(__file__)
MS = os.path.join(TESTS, "neutral", "neutral_1.ms.gz")
MS_ALL = [os.path.join(TESTS, d, f"{d}_{i}.ms.gz") for d in ("neutral", "sweep") for i in (1, 2, 3)]
WIN_STATS = ["pi", "theta_w", "tajima_d", "dist_var"]
FS_STATS = ("dind", "high_freq", "low_freq", "s_ratio", "hapdaf_o", "hapdaf_s")
SNP_STATS = FS_STATS + ("ihs", "delta_ihh", "isafe", "nsl")
META = ["iter", "s", "t", "f_i", "f_t", "mu", "r", "model"]
NTHREADS = int(os.environ.get("FS_TEST_THREADS", "4"))
RBINS = [1.0, 2.0, 5.0]  # top bin (5, inf] empty: see GATE.md amendment 1
# ms-like mapping (default center grid with large windows): 6 of 11 pairs reach past the locus
MS_LIKE = {100000: [50000, 600000, 1150000], 200000: [50000, 600000, 1150000],
           500000: [50000, 150000, 600000, 1050000, 1150000]}
# the real ms geometry (fvs-discoal --windows 20000,50000,100000,200000,500000, default centers)
MS_GEOMETRY = {w: list(range(50_000, 1_150_001, 100_000)) for w in (20000, 50000, 100000, 200000, 500000)}
# recombination map blocks (start, end, cM/Mb); covers every locus of the 7.2 Mb fixture
# (loci reach 8.3 Mb because the last ones run past the contig end)
MAP_BLOCKS = [(0, 1, 0.4), (1, 2_400_000, 0.4), (2_400_000, 4_800_000, 1.5), (4_800_000, 9_000_000, 3.0)]


# --------------------------------------------------------------------------- fixtures


def write_multi_vcf(path: str, ms_files: list[str], chrom: str = "chr22") -> int:
    """Concatenate ms replicates (same sample size) into one bgzipped, indexed VCF,
    replicate k shifted by k * 1.2 Mb. Returns the contig length."""
    from flexsweep.fv import parse_ms_numpy
    from flexsweep.polarize import bgzip_index

    reps = [parse_ms_numpy(f) for f in ms_files]
    n_hap = {r[0].shape[1] for r in reps}
    assert len(n_hap) == 1, f"different sample sizes: {n_hap}"
    n_ind = n_hap.pop() // 2
    length = L * len(reps)
    plain = path + ".tmp.vcf"
    last = 0
    with open(plain, "w") as out:
        out.write(f"##fileformat=VCFv4.2\n##contig=<ID={chrom},length={length}>\n")
        out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                  + "\t".join(f"ind{j}" for j in range(n_ind)) + "\n")
        for k, r in enumerate(reps):
            hap, positions = r[0], r[4]
            for i, pos in enumerate(positions):
                pos = max(int(pos) + k * L, last + 1)
                last = pos
                gts = "\t".join(f"{int(hap[i, 2 * j])}|{int(hap[i, 2 * j + 1])}" for j in range(n_ind))
                out.write(f"{chrom}\t{pos}\t.\tA\tG\t.\t.\t.\tGT\t{gts}\n")
    bgzip_index(plain, path)
    os.remove(plain)
    return length


def write_recmap(path: str, chrom: str = "chr22") -> None:
    # deCODE layout: '#' comment lines, then a plain header row (fv reads the map with
    # comment_prefix='#' and has_header=True, so a '#'-only header would eat row 1)
    cm = 0.0
    with open(path, "w") as fh:
        fh.write("# test map: 0.4, 1.5, 3.0 cM/Mb blocks\nchr\tstart\tend\tcm_mb\tcm\n")
        for s, e, rate in MAP_BLOCKS:
            cm += rate * (e - s) / 1e6
            fh.write(f"{chrom}\t{s}\t{e}\t{rate}\t{cm}\n")


def locus_rate(a: int, b: int) -> float:
    """cM/Mb of locus [a, b] from the map blocks (numpy, independent of fv.get_cm)."""
    ends = np.array([e for _, e, _ in MAP_BLOCKS], dtype=float)
    cms = np.cumsum([r * (e - s) / 1e6 for s, e, r in MAP_BLOCKS])
    return (np.interp(b, ends, cms) - np.interp(a, ends, cms)) / ((b - a) / 1e6)


@pytest.fixture(scope="module")
def vcf1(tmp_path_factory):
    from tests.test_cli_smoke import _ms_to_vcf

    d = tmp_path_factory.mktemp("vcf1")
    path = str(d / "sim_chr22.vcf.gz")
    _ms_to_vcf(MS, path)
    return path


def _snp_table(seed=1, n=400, length=L):
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(length, n, replace=False))
    return pl.DataFrame({"positions": pos, "a": rng.normal(size=n), "b": rng.normal(size=n)})


def _shift(df, a):
    return df.with_columns(pl.col("positions") + (a - 1))


# --------------------------------------------------------------------------- cut_snps


def test_cut_snps_dict_fixed_center_equals_sims_call():
    """VCF cut (absolute positions, locus starting at a) == sims cut (relative)."""
    df = _snp_table()
    wc = {10000: [590000, 600000, 610000], 50000: [550000, 600000, 650000]}
    a = 3_000_001
    fixed = (a + (a + L - 1)) // 2
    sims = fv.cut_snps(df, wc, sorted(wc), ["a", "b"], iter_value=7)
    vcf = fv.cut_snps(_shift(df, a), wc, sorted(wc), ["a", "b"], fixed_center=fixed,
                      iter_value=7, sim_mid=L // 2)
    assert vcf.equals(sims)
    assert vcf.select("window", "center").rows() == [
        (w, c) for w in sorted(wc) for c in wc[w]]


def test_cut_snps_dict_fixed_center_controls():
    df = _snp_table()
    wc = {10000: [590000, 600000, 610000]}
    a = 3_000_001
    fixed = (a + (a + L - 1)) // 2
    ok = fv.cut_snps(_shift(df, a), wc, [10000], ["a", "b"], fixed_center=fixed, sim_mid=L // 2)
    off = fv.cut_snps(_shift(df, a), wc, [10000], ["a", "b"], fixed_center=fixed,
                      sim_mid=L // 2 + 4001)
    assert not ok.select("a", "b").equals(off.select("a", "b"))
    with pytest.raises(ValueError, match="sim_mid"):
        fv.cut_snps(_shift(df, a), wc, [10000], ["a", "b"], fixed_center=fixed)


def test_cut_snps_list_midpoint():
    """List path. Symmetric centers (step 100 kb): sim_mid=L/2 changes nothing. Step 70 kb
    (centers 35,000..1,225,000, midpoint 630,000): the old midpoint cut is off by 30 kb
    (documents the bug, still the behaviour without sim_mid); sim_mid=L/2 matches the
    simulation cut."""
    df = _snp_table(n=2000)
    a = 3_000_001
    fixed = (a + (a + L - 1)) // 2
    sym = np.arange(50000, 1150000 + 100000, 100000)
    assert fv.cut_snps(_shift(df, a), sym, [100000], ["a", "b"], fixed_center=fixed).equals(
        fv.cut_snps(_shift(df, a), sym, [100000], ["a", "b"], fixed_center=fixed, sim_mid=L // 2))

    c70 = np.arange(35000, 1165000 + 70000, 70000)
    assert (c70[0] + c70[-1]) // 2 == 630000
    sims = fv.cut_snps(df, c70, [70000], ["a", "b"], iter_value=fixed)
    old = fv.cut_snps(_shift(df, a), c70, [70000], ["a", "b"], fixed_center=fixed, iter_value=fixed)
    new = fv.cut_snps(_shift(df, a), c70, [70000], ["a", "b"], fixed_center=fixed,
                      iter_value=fixed, sim_mid=L // 2)
    assert not old.equals(sims)
    assert new.equals(sims)


# --------------------------------------------------------------------------- window stats


def _flat(vcf, region, **kw):
    return fv.calculate_stats_vcf_flat(vcf, region, stats=WIN_STATS, nthreads=1, **kw)[1]


def _per_window(vcf, region, wc, **kw):
    return pl.concat([_flat(vcf, region, windows=[w], centers=wc[w], **kw) for w in sorted(wc)])


@pytest.mark.parametrize("masked", [False, True])
def test_flat_dict_equals_per_window_list(vcf1, tmp_path, masked):
    kw = {}
    if masked:
        bed = tmp_path / "mask.bed"
        bed.write_text("#mask\nchr22\t570000\t615000\nchr22\t100000\t160000\n")
        kw = dict(accessibility_mask=str(bed), min_accessible=0.6)
    region = ["chr22:1-1200000", "chr22:100001-1300000"]
    wc = {20000: [580000, 600000, 620000], 100000: [500000, 600000, 700000]}
    one = _flat(vcf1, region, windows=sorted(wc), centers=wc, **kw)
    per = _per_window(vcf1, region, wc, **kw)
    key = ["iter", "window", "center"]
    assert one.height == len(region) * 6
    assert one.sort(key).equals(per.select(one.columns).sort(key))
    # rows per locus follow the mapping order (window, then center)
    first = one.filter(pl.col("iter") == "chr22:1-1200000")
    assert first.select("window", "center").rows() == [(w, c) for w in sorted(wc) for c in wc[w]]


def test_flat_dict_control_shift(vcf1):
    region = ["chr22:1-1200000"]
    a = _flat(vcf1, region, windows=[20000], centers={20000: [580000, 600000]})
    b = _flat(vcf1, region, windows=[20000], centers={20000: [580000, 604001]})
    assert not a.select(WIN_STATS).equals(b.select(WIN_STATS))


def _assert_unique_combos(seen, expected):
    if len(seen) != len(set(seen)):
        raise AssertionError(f"{len(seen) - len(set(seen))} window/center pairs computed twice")
    if set(seen) != expected:
        raise AssertionError("computed pairs differ from the brute-force set")


def test_flat_dict_dedup(vcf1, monkeypatch):
    """Overlapping loci share absolute (center, window) pairs; each is computed once."""
    seen = []
    orig = fv.batch_windowed_stats_flat

    def spy(hap_f, ac, pos, combos, *args):
        seen.extend((int(c), int(w)) for c, w, _ in combos)
        return orig(hap_f, ac, pos, combos, *args)

    monkeypatch.setattr(fv, "batch_windowed_stats_flat", spy)
    region = [f"chr22:{s}-{s + L - 1}" for s in (1, 10001, 20001, 30001)]
    wc = {10000: [590000, 595000, 600000, 605000, 610000], 12134: [593933, 600000, 606067]}
    _flat(vcf1, region, windows=sorted(wc), centers=wc)
    expected = {(int(r.split(":")[1].split("-")[0]) + c - 1, w)
                for r in region for w, cs in wc.items() for c in cs}
    assert len(expected) < len(region) * sum(len(c) for c in wc.values())  # overlap exists
    _assert_unique_combos(seen, expected)
    # load balance: pairs are sent center-major (as on the list path), so every batch
    # mixes window sizes instead of one batch holding all the largest windows
    assert seen == sorted(seen)
    # control: a duplicated pair must fail the check
    with pytest.raises(AssertionError, match="twice"):
        _assert_unique_combos(seen + seen[:1], expected)


def test_batch_window_search_int64(monkeypatch):
    """Window boundaries are searched on an int64 copy of the positions made once per
    batch: with the int32 positions of genome_reader and a Python-int query, numpy
    converted the whole array on every call (2 per window; 5.4x slower caldas
    statistics phase on chr1). The statistics still receive the original positions."""
    rng = np.random.default_rng(3)
    pos = np.sort(rng.choice(L, 3000, replace=False)).astype(np.int32)
    hap = rng.integers(0, 2, size=(pos.size, 20)).astype(np.float64)
    ac = np.column_stack([20 - hap.sum(1), hap.sum(1)]).astype(np.int64)
    combos = [(c, w, float(w)) for c in (300_000, 600_000, 900_000) for w in (10_000, 50_000)]
    got_pos = []

    def stat(h, p, a, f, window_size=None, iu=None, ju=None):
        got_pos.append(p.dtype)
        return (float(h.sum()), float(p.min()), float(p.max()), float(p.size))

    ref = [(c, w, (float(hap[(pos >= c - w // 2) & (pos <= c + w // 2)].sum()),
                   float(pos[(pos >= c - w // 2) & (pos <= c + w // 2)].min()),
                   float(pos[(pos >= c - w // 2) & (pos <= c + w // 2)].max()),
                   float(((pos >= c - w // 2) & (pos <= c + w // 2)).sum())))
           for c, w, _ in combos]
    searched = []
    orig = np.searchsorted

    def spy(a, v, side="left", sorter=None):
        searched.append(np.asarray(a).dtype)
        return orig(a, v, side=side, sorter=sorter)

    monkeypatch.setattr(fv.np, "searchsorted", spy)
    out = fv.batch_windowed_stats_flat(hap, ac, pos, combos, stat, ["a", "b", "c", "d"])
    monkeypatch.undo()
    assert [(c, w, tuple(s)) for c, w, s in out] == ref  # oracle: boolean masks
    assert set(got_pos) == {np.dtype(np.int32)}  # statistics see the original positions
    assert searched and set(searched) == {np.dtype(np.int64)}


# --------------------------------------------------------------------------- windows past the locus
# In a simulation nothing exists outside [0, L]: a window reaching past the locus only
# holds in-locus SNPs, with the full window length as denominator. The mapping path
# clips such windows to the locus ([a - 1, a + L - 1], the image of sim positions 0..L)
# so the VCF features are the ones the model was trained on.


def _edge_pairs(wc):
    return [(c, w) for w in sorted(wc) for c in wc[w] if c - w // 2 < 0 or c + w // 2 > L]


@pytest.fixture(scope="module")
def vcf3(tmp_path_factory):
    """Replicates 0-2 end to end: the locus at L + 1 is replicate 1, real SNPs on both flanks."""
    d = tmp_path_factory.mktemp("vcf3")
    path = str(d / "sim_chr22.vcf.gz")
    write_multi_vcf(path, MS_ALL[:3])
    return path


def _sims_windows(ms_file, wc):
    _, wd = fv.calculate_stats_simulations(ms_file, windows=sorted(wc), centers=wc,
                                           locus_length=L, stats=WIN_STATS)
    cols = fv.resolve_stats(WIN_STATS)[0]
    return pl.DataFrame(wd[:, 1:], schema=["center", "window"] + list(cols), orient="row").with_columns(
        pl.col("center").cast(pl.Int64), pl.col("window").cast(pl.Int64))


def test_flat_dict_edge_windows_equal_sims(vcf3):
    """Oracle: the simulation path on the same replicate, every pair, exact."""
    assert len(_edge_pairs(MS_LIKE)) == 6
    region = [f"chr22:{L + 1}-{2 * L}"]
    one = _flat(vcf3, region, windows=sorted(MS_LIKE), centers=MS_LIKE)
    m = one.join(_sims_windows(MS_ALL[1], MS_LIKE), on=["center", "window"], suffix="_sim")
    assert m.height == sum(len(c) for c in MS_LIKE.values())
    for s in WIN_STATS:
        assert np.array_equal(m[s].to_numpy(), m[s + "_sim"].to_numpy(), equal_nan=True), s
    # control: the list path is not clipped (unchanged); its edge pairs take flanking
    # SNPs and differ from the sims, its in-locus pairs are identical
    per = _per_window(vcf3, region, MS_LIKE).join(
        _sims_windows(MS_ALL[1], MS_LIKE), on=["center", "window"], suffix="_sim")
    edge = set(_edge_pairs(MS_LIKE))
    for c, w, row in ((r["center"], r["window"], r) for r in per.iter_rows(named=True)):
        same = all(row[s] == row[s + "_sim"] for s in WIN_STATS)
        assert same != ((c, w) in edge), (c, w)


def test_flat_dict_edge_combos(vcf3, tmp_path, monkeypatch):
    """Edge pairs: one clipped combo per (locus, pair), bounds and masked denominators by
    brute force; in-locus pairs keep the cross-locus deduplication."""
    seen = []
    orig = fv.batch_windowed_stats_flat

    def spy(hap_f, ac, pos, combos, *args):
        seen.extend(combos)
        return orig(hap_f, ac, pos, combos, *args)

    monkeypatch.setattr(fv, "batch_windowed_stats_flat", spy)
    # BED (0-based, half-open): one interval in the left flank, one inside the locus
    bed_iv = [(L - 150_000, L - 50_000), (L + 10_000, L + 30_000)]
    bed = tmp_path / "mask.bed"
    bed.write_text("#mask\n" + "".join(f"chr22\t{s}\t{e}\n" for s, e in bed_iv))
    starts = [L + 1, L + 10_001, L + 20_001]
    region = [f"chr22:{a}-{a + L - 1}" for a in starts]
    _flat(vcf3, region, windows=sorted(MS_LIKE), centers=MS_LIKE,
          accessibility_mask=str(bed), min_accessible=0.0)

    edge = _edge_pairs(MS_LIKE)
    inner = [t for t in seen if len(t) == 3]
    clipped = [t for t in seen if len(t) == 5]
    assert len(inner) + len(clipped) == len(seen)
    expected_inner = {(a + c - 1, w) for a in starts for w, cs in MS_LIKE.items() for c in cs
                      if (c, w) not in edge}
    _assert_unique_combos([(int(t[0]), int(t[1])) for t in inner], expected_inner)
    expected_clip = sorted((a + c - 1, w, max(a + c - 1 - w // 2, a - 1), min(a + c - 1 + w // 2, a + L - 1))
                           for a in starts for c, w in edge)
    assert sorted((int(t[0]), int(t[1]), int(t[3]), int(t[4])) for t in clipped) == expected_clip

    def excluded(lo, hi):  # brute force over 1-based positions [s + 1, e]
        return sum(max(0, min(hi, e) - max(lo, s + 1) + 1) for s, e in bed_iv)

    for _, w, den, lo, hi in clipped:
        assert den == w - excluded(lo, hi)
    # the flank interval lies in some unclipped windows but never counts after clipping
    assert any(excluded(c - w // 2, c + w // 2) > excluded(lo, hi) for c, w, _, lo, hi in clipped)


def test_dict_centers_accepted_for_vcf(tmp_path):
    """A mapping with vcf=True runs (empty folder -> no VCF found), it no longer raises."""
    with pytest.raises(FileNotFoundError):
        fv.summary_statistics(str(tmp_path), vcf=True, centers={10000: [600000]})


# --------------------------------------------------------------------------- matching


def _write_geometry_parquet(path, wc, stats=("dind", "pi", "h2_h1"), drop=None):
    cols = list(META)
    for s in stats:
        cols += [f"{s}_{w}_{c}" for w in sorted(wc) for c in wc[w]]
    if drop is not None:
        cols.remove(drop)
    pl.DataFrame({c: [0.0] for c in cols}).write_parquet(path)
    return cols


def test_geometry_from_parquet(tmp_path):
    from flexsweep.main import check_geometry, geometry_from_parquet

    wc = caldas_window_centers()
    p = str(tmp_path / "fvs_caldas.parquet")
    cols = _write_geometry_parquet(p, wc)
    mapping, columns = geometry_from_parquet(p)
    assert mapping == wc
    assert columns == cols
    check_geometry(mapping, L)
    for bad in (1_000_000, 2_000_000):
        with pytest.raises(click.ClickException):
            check_geometry(mapping, bad)
    # windows past the locus (ms geometry) are accepted: they are clipped like the sims
    check_geometry(MS_LIKE, L)
    with pytest.raises(click.ClickException):
        check_geometry(MS_LIKE, 1_000_000)
    # control: one stat missing one pair
    _write_geometry_parquet(p, wc, drop="pi_10000_565000")
    with pytest.raises(click.ClickException):
        geometry_from_parquet(p)


def test_apply_reference_columns():
    ref = ["iter", "s", "pi_10_5", "pi_10_6", "h1_10_5", "h1_10_6"]
    df = pl.DataFrame({c: [1.0] for c in ["s", "iter", "h1_10_6", "pi_10_6", "h1_10_5", "pi_10_5"]})
    assert fv._apply_reference_columns(df, ref).columns == ref
    with pytest.raises(ValueError, match="missing"):
        fv._apply_reference_columns(df.drop("pi_10_5"), ref)
    with pytest.raises(ValueError, match="extra"):
        fv._apply_reference_columns(df.with_columns(pl.lit(0.0).alias("zns_10_5")), ref)


def test_apply_reference_columns_keeps_rbins_labels():
    """Raw output of an r_bins run carries r_bins_{window}_{center} label columns: they
    are labels, not features. Kept after the reference columns; features still checked."""
    ref = ["iter", "s", "pi_10_5", "pi_10_6"]
    df = pl.DataFrame({c: [1.0] for c in ["pi_10_6", "iter", "s", "pi_10_5"]}).with_columns(
        pl.lit("(1, 2]").alias("r_bins_10_6"), pl.lit("(1, 2]").alias("r_bins_10_5"))
    out = fv._apply_reference_columns(df, ref)
    assert out.columns == ref + ["r_bins_10_6", "r_bins_10_5"]
    with pytest.raises(ValueError, match="missing"):  # control: a feature still missing
        fv._apply_reference_columns(df.drop("pi_10_5"), ref)


def _old_regex_breaks(labels):
    """The HEAD parsing (fv.py _parse_breaks_from_rbins), used as oracle for labels it
    can read."""
    import re
    return [float(re.search(r",\s*([0-9.]+)\]$", s).group(1)) for s in labels]


@pytest.mark.parametrize("values", [[0.2, 0.5, 1.5, 2.5], [0.2, 2.5], [0.5, 1.5]])
def test_parse_breaks_from_rbins(values):
    """Labels without a top bin: same breaks as HEAD. With a top bin '(x, inf]' (HEAD
    crashes): x is the break."""
    df = pl.DataFrame({"r_bins": pl.Series(values).cut(breaks=[1.0, 2.0])})
    labels = df["r_bins"].unique().sort().cast(pl.String).to_list()
    got = fv._parse_breaks_from_rbins(df)
    if any(s.endswith("inf]") for s in labels):
        assert got == sorted({float(x) for s in labels for x in s.strip("(]").split(", ")
                              if x not in ("-inf", "inf")})
    else:
        assert got == _old_regex_breaks(labels)


def _bins_pickle(path, labels):
    n = len(labels) if labels else 1
    snp = pl.DataFrame({"freq_bins": [0.1] * n, "pi": [0.0] * n})
    if labels:
        snp = snp.with_columns(pl.Series("r_bins", labels))
    with open(path, "wb") as fh:
        pickle.dump(fv.binned_stats([snp, None], [snp, None]), fh)


def test_check_rbins_against_bins(tmp_path):
    from flexsweep.main import check_rbins_against_bins

    p = str(tmp_path / "neutral_bins.pickle")
    _bins_pickle(p, None)  # sims without r_bins: anything goes on the VCF side
    check_rbins_against_bins(p, None)
    check_rbins_against_bins(p, [1.0, 2.0])
    labels = pl.Series([0.5, 1.5, 2.5]).cut(breaks=RBINS).cast(pl.String).to_list()
    _bins_pickle(p, labels)
    check_rbins_against_bins(p, RBINS)
    for bad in (None, [1.0], [1.0, 3.0]):
        with pytest.raises(click.ClickException):
            check_rbins_against_bins(p, bad)


def _vcf_dir(tmp_path, vcf):
    d = tmp_path / "vcfs"
    d.mkdir()
    for ext in ("", ".tbi"):
        os.symlink(vcf + ext, d / ("sim_chr22.vcf.gz" + ext))
    return str(d)


def test_cli_fvs_vcf_wiring(vcf1, tmp_path, monkeypatch):
    """--geometry_from passes the mapping and the reference columns; without it the call
    to summary_statistics is the current one, argument for argument."""
    import flexsweep
    from flexsweep.main import cli

    calls = []
    monkeypatch.setattr(flexsweep, "summary_statistics", lambda *a, **k: calls.append((a, k)))
    d = _vcf_dir(tmp_path, vcf1)
    runner = CliRunner()
    res = runner.invoke(cli, ["fvs-vcf", "--vcf_path", d, "--suffix", "x"])
    assert res.exit_code == 0, res.output
    assert sorted(calls[0][1]) == sorted([
        "vcf", "stats", "nthreads", "windows", "step", "step_vcf", "locus_length",
        "recombination_map", "r_bins", "min_rate", "suffix", "func", "save_stats", "tmp_dir",
        "accessibility_mask", "min_accessible", "max_missing"])

    wc = caldas_window_centers()
    p = str(tmp_path / "fvs_caldas.parquet")
    cols = _write_geometry_parquet(p, wc)
    res = runner.invoke(cli, ["fvs-vcf", "--vcf_path", d, "--suffix", "x", "--geometry_from", p])
    assert res.exit_code == 0, res.output
    k = calls[1][1]
    assert k["centers"] == wc and k["windows"] == sorted(wc) and k["reference_columns"] == cols

    res = runner.invoke(cli, ["fvs-vcf", "--vcf_path", d, "--suffix", "x", "--geometry_from", p,
                              "--locus_length", "1000000"])
    assert res.exit_code != 0 and len(calls) == 2

    b = str(tmp_path / "neutral_bins.pickle")
    _bins_pickle(b, pl.Series([0.5, 1.5, 2.5]).cut(breaks=RBINS).cast(pl.String).to_list())
    res = runner.invoke(cli, ["fvs-vcf", "--vcf_path", d, "--suffix", "x", "--bins_from", b,
                              "--r_bins", "1,3"])
    assert res.exit_code != 0 and len(calls) == 2


# --------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def multi(tmp_path_factory):
    d = tmp_path_factory.mktemp("multi")
    vcf = str(d / "sim_chr22.vcf.gz")
    write_multi_vcf(vcf, MS_ALL)
    rmap = str(d / "recmap.tsv")
    write_recmap(rmap)
    return vcf, rmap


def _stat_window(col):
    stat, w, _ = col.rsplit("_", 2)
    return stat, int(w)


def _compare_one_pass(one, per_window, wc, key):
    """One-pass vs per-window outputs: identical except fs stats at windows > min."""
    w_min = min(wc)
    for idx, label in ((0, "normalized"), (1, "raw")):
        per = None
        for w in sorted(wc):
            out = per_window[w][idx]
            feat = [c for c in out.columns if c.endswith(tuple(f"_{w}_{c}" for c in wc[w]))]
            out = out.select(key + feat)
            per = out if per is None else per.join(out, on=key, how="inner")
        # r_bins label columns (raw output of an r_bins run) are text, not features:
        # they must be equal, then are left out of the numeric comparison
        labels = [c for c in per.columns if c.startswith("r_bins_")]
        feat = [c for c in per.columns if c not in key and c not in labels]
        m = one[idx].select(key + feat + labels).join(per, on=key, how="inner", suffix="_ref")
        assert m.height > 0, label
        for c in labels:
            assert m[c].cast(pl.String).equals(m[c + "_ref"].cast(pl.String).alias(c)), c

        def values(cols):
            return (m.select(cols).to_numpy().astype(float),
                    m.select([c + "_ref" for c in cols]).to_numpy().astype(float))

        differs = [c for c in feat if _stat_window(c)[0] in FS_STATS and _stat_window(c)[1] > w_min]
        same = [c for c in feat if c not in differs]
        x, y = values(same)
        assert np.array_equal(np.isfinite(x), np.isfinite(y)), label
        ok = np.isfinite(x)
        np.testing.assert_allclose(x[ok], y[ok], rtol=1e-10, atol=1e-12, err_msg=label)
        x, y = values(differs)
        ok = np.isfinite(x) & np.isfinite(y)
        assert not np.allclose(x[ok], y[ok], rtol=1e-10, atol=1e-12), label


def _caldas_subset():
    wc = caldas_window_centers()
    ws = sorted(wc)
    return {w: wc[w] for w in (ws[0], ws[6], ws[-1])}


@pytest.mark.slow
@pytest.mark.parametrize("rbins", [False, True], ids=["plain", "rbins"])
def test_vcf_caldas_one_pass_equals_per_window(multi, tmp_path, rbins):
    vcf, rmap = multi
    d = _vcf_dir(tmp_path, vcf)
    kw = dict(vcf=True, step_vcf=int(1e5), nthreads=NTHREADS, locus_length=L)
    if rbins:
        kw.update(recombination_map=rmap, r_bins=RBINS, min_rate=0.0)
    wc = _caldas_subset()
    one = fv.summary_statistics(d, windows=sorted(wc), centers=wc, suffix="one", **kw)
    per = {w: fv.summary_statistics(d, windows=[w], centers=wc[w], suffix=f"w{w}", **kw)
           for w in sorted(wc)}
    _compare_one_pass(one, per, wc, ["iter"])

    if rbins:
        # hand z-score: window stat grouped by (center, window, r_bins) over all loci
        norm, raw = one
        edges = np.asarray(RBINS)
        loci = raw.select("iter").to_series().to_list()
        bounds = [tuple(map(int, s.split(":")[1].split("-"))) for s in loci]
        rbin = np.searchsorted(edges, [locus_rate(a, b) for a, b in bounds], side="left")
        assert len(set(rbin.tolist())) == 3  # every bin populated
        n_checked = 0
        for col in ("pi_10000_600000", f"tajima_d_{sorted(wc)[1]}_600000", "h12_150000_600000"):
            x = raw[col].cast(pl.Float64).to_numpy()
            z = np.full_like(x, np.nan)
            for b in np.unique(rbin):
                sel = (rbin == b) & np.isfinite(x)
                z[sel] = (x[sel] - x[sel].mean()) / x[sel].std(ddof=1)
            hand = pl.DataFrame({"iter": loci, "hand": z})
            m = norm.select("iter", col).join(hand, on="iter", how="inner").filter(
                pl.col("hand").is_finite())
            np.testing.assert_allclose(m[col].to_numpy(), m["hand"].to_numpy(), rtol=1e-9, atol=1e-12)
            n_checked += m.height
        assert n_checked > 0


@pytest.mark.slow
def test_vcf_rbins_top_bin(multi, tmp_path):
    """Breaks [1, 2] put the 3.0 cM/Mb loci in the top bin '(2, inf]' (HEAD crashes).
    Their SNP statistics must be normalized (finite raw -> non-zero z), not left as
    nulls filled with 0 after a failed bin join."""
    vcf, rmap = multi
    d = _vcf_dir(tmp_path, vcf)
    norm, raw = fv.summary_statistics(
        d, vcf=True, step_vcf=int(1e5), nthreads=NTHREADS, locus_length=L, windows=[100000],
        step=int(1e5), recombination_map=rmap, r_bins=[1.0, 2.0], min_rate=0.0, suffix="top")
    rate = {s: locus_rate(*map(int, s.split(":")[1].split("-"))) for s in norm["iter"].to_list()}
    top = [s for s, r in rate.items() if r > 2.0]
    assert len(top) > 5
    cols = [c for c in norm.columns if c.split("_")[0] in ("ihs", "nsl", "dind")]
    n = norm.filter(pl.col("iter").is_in(top)).select(["iter"] + cols)
    r = raw.filter(pl.col("iter").is_in(top)).select(["iter"] + cols)
    m = n.join(r, on="iter", suffix="_raw")
    z = m.select(cols).to_numpy().astype(float)
    x = m.select([c + "_raw" for c in cols]).to_numpy().astype(float)
    ok = np.isfinite(x)
    assert ok.sum() > 50
    assert (z[ok] != 0).mean() > 0.9


@pytest.mark.slow
def test_sims_caldas_rbins_one_pass_equals_per_window(tmp_path):
    from flexsweep.simulate_discoal import Simulator

    sims = str(tmp_path / "sims")
    sim = Simulator(sample_size=20, demes="constant", output_folder=sims,
                    num_simulations=30, nthreads=NTHREADS, seed=3)
    sim.create_params()
    sim.simulate()
    wc = _caldas_subset()
    kw = dict(nthreads=NTHREADS, r_bins=[0.5, 1.0, 2.0])
    one = fv.summary_statistics(sims, windows=sorted(wc), centers=wc, suffix="one", **kw)
    per = {w: fv.summary_statistics(sims, windows=[w], centers=wc[w], suffix=f"w{w}", **kw)
           for w in sorted(wc)}
    _compare_one_pass(one, per, wc, ["iter", "model"])


@pytest.mark.slow
def test_vcf_caldas_matches_sims_columns(multi, tmp_path):
    """Full 15 x 15 caldas: VCF columns == simulation columns, same order."""
    from flexsweep.main import check_geometry, geometry_from_parquet
    from flexsweep.simulate_discoal import Simulator

    sims = str(tmp_path / "sims")
    sim = Simulator(sample_size=20, demes="constant", output_folder=sims,
                    num_simulations=20, nthreads=NTHREADS, seed=5)
    sim.create_params()
    sim.simulate()
    wc = caldas_window_centers()
    fv.summary_statistics(sims, windows=sorted(wc), centers=wc, nthreads=NTHREADS, suffix="caldas")
    mapping, cols = geometry_from_parquet(os.path.join(sims, "fvs_caldas.parquet"))
    assert mapping == wc
    check_geometry(mapping, L)

    vcf, _ = multi
    d = _vcf_dir(tmp_path, vcf)
    kw = dict(vcf=True, step_vcf=int(1e5), nthreads=NTHREADS, locus_length=L,
              windows=sorted(mapping), centers=mapping, reference_columns=cols)
    norm, raw = fv.summary_statistics(d, suffix="caldas", **kw)
    assert norm.columns == cols and raw.columns == cols
    assert list(pl.read_parquet_schema(os.path.join(d, "fvs_caldas.parquet"))) == cols
    assert norm.height > 0
    # control: a different statistic set must be refused
    with pytest.raises(ValueError, match="missing"):
        fv.summary_statistics(d, suffix="bad", stats=["pi", "tajima_d"], **kw)


@pytest.mark.slow
def test_vcf_ms_geometry_mimics_sims(multi, tmp_path, monkeypatch):
    """Real ms geometry end to end (60 pairs, 6 reach past the locus).
    Window stats of the 6 replicate loci == simulation path on the same replicate, all
    pairs, exact. SNP stats: every per-locus SNP table handed to cut_snps lies inside
    [a - 1, a + L - 1] (nthreads=1 runs joblib in-process, so the spy sees every cut)."""
    vcf, _ = multi
    d = _vcf_dir(tmp_path, vcf)
    edge = _edge_pairs(MS_GEOMETRY)
    assert len(edge) == 6
    cuts = []
    orig = fv.cut_snps

    def spy(df, centers, windows, stats_names, fixed_center=None, iter_value=1, sim_mid=None):
        if df.height:
            cuts.append((int(fixed_center), int(df["positions"].min()), int(df["positions"].max())))
        return orig(df, centers, windows, stats_names, fixed_center=fixed_center,
                    iter_value=iter_value, sim_mid=sim_mid)

    monkeypatch.setattr(fv, "cut_snps", spy)
    _, raw = fv.summary_statistics(d, vcf=True, step_vcf=int(1e5), nthreads=1, locus_length=L,
                                   windows=sorted(MS_GEOMETRY), centers=MS_GEOMETRY, suffix="ms")
    win_cols = fv.resolve_stats(None)[0]
    n_cmp = 0
    for k, ms in enumerate(MS_ALL):
        row = raw.filter(pl.col("iter") == f"chr22:{k * L + 1}-{(k + 1) * L}")
        assert row.height == 1, k
        _, wd = fv.calculate_stats_simulations(ms, windows=sorted(MS_GEOMETRY), centers=MS_GEOMETRY,
                                               locus_length=L)
        for r in wd:
            c, w = int(r[1]), int(r[2])
            for j, s in enumerate(win_cols):
                v = row[f"{s}_{w}_{c}"].cast(pl.Float64)[0]
                v = np.nan if v is None else v
                assert (np.isnan(v) and np.isnan(r[3 + j])) or v == r[3 + j], (k, s, w, c, v, r[3 + j])
                n_cmp += 1
    assert n_cmp == len(MS_ALL) * 60 * len(win_cols)

    assert cuts
    for fc, lo, hi in cuts:
        a = fc - L // 2 + 1
        assert lo >= a - 1 and hi <= a + L - 1, (fc, lo, hi)
    # the check has teeth: SNPs exist on both sides of every interior replicate locus
    import gzip
    with gzip.open(vcf, "rt") as fh:
        pos = np.array([int(x.split("\t", 2)[1]) for x in fh if not x.startswith("#")])
    a = 2 * L + 1
    assert ((pos < a - 1) & (pos >= a - 250_000)).any() and ((pos > a + L - 1) & (pos <= a + L + 250_000)).any()

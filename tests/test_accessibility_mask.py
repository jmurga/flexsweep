"""Accessibility mask: GIAB-style BED of excluded regions read once with the VCF.

Oracles:
- Brute force: a per-base boolean array built from the BED gives the excluded-base
  counts and the excluded positions.
- Hand computation: with a mask, per-bp pi / theta_w equal the absolute value over
  the kept SNPs divided by the window's accessible bases; statistics without a bp
  denominator equal their value on the same SNP set.
- Frozen values: without a mask the window statistics equal those of main at
  a2838f2 (same function, same input).

Not covered: multi-chromosome BEDs beyond name matching, real GIAB data (checked
separately on the YRI chr22 example).
"""

import os

import allel
import numpy as np
import pytest

from flexsweep.fv import (
    calculate_stats_vcf_flat,
    clean_genotypes,
    excluded_bp,
    is_excluded,
    load_accessibility_mask,
    tajima_d,
)
from tests.test_cli_smoke import _ms_to_vcf

MS = os.path.join(os.path.dirname(__file__), "neutral", "neutral_1.ms.gz")
STATS = ["pi", "theta_w", "tajima_d", "dist_var"]
CENTERS = dict(center=[500000, 700000], windows=[100000], step=1e5)

# main at a2838f2, calculate_stats_vcf_flat(neutral_1 VCF, no mask):
# (center, window, pi, theta_w, tajima_d, dist_var)
MAIN_NO_MASK = [
    (500000, 100000, 0.0006814948320413437, 0.0005730921499813946, 0.600572510221275, 1.234863523140511e-07),
    (600000, 100000, 0.0006113027562446167, 0.00045544860013184136, 1.082761795579057, 1.0583550935111782e-07),
    (700000, 100000, 0.000465475021533161, 0.0003865430923628174, 0.644195867621435, 6.563878980630093e-08),
]


def _write_bed(path, rows):
    with open(path, "w") as fh:
        fh.write("#GIAB-style header line\n")
        for chrom, s, e in rows:
            fh.write(f"{chrom}\t{s}\t{e}\n")


@pytest.fixture(scope="module")
def vcf(tmp_path_factory):
    d = tmp_path_factory.mktemp("vcf")
    path = str(d / "sim_chr22.vcf.gz")
    _ms_to_vcf(MS, path)
    return path


@pytest.mark.parametrize("name", ["chr22", "22"])
def test_excluded_bp_matches_brute_force(tmp_path, name):
    rng = np.random.default_rng(1)
    L = 20_000
    starts = np.sort(rng.integers(0, L - 500, 60))
    rows = [("chr22", int(s), int(s + rng.integers(1, 400))) for s in starts]
    rows += [("chr22", 100, 200), ("chr22", 200, 260), ("chr1", 0, L)]  # adjacent + other chrom
    bed = tmp_path / "mask.bed"
    _write_bed(bed, rows)
    mask = load_accessibility_mask(str(bed), name)

    excluded = np.zeros(L + 2, dtype=bool)  # index = 1-based position
    for c, s, e in rows:
        if c == "chr22":
            excluded[s + 1 : e + 1] = True

    a = rng.integers(1, L, 500)
    b = np.minimum(a + rng.integers(0, 3000, 500), L)
    brute = np.array([excluded[x : y + 1].sum() for x, y in zip(a, b)])
    assert np.array_equal(excluded_bp(mask, a, b), brute)
    pos = np.arange(1, L + 1)
    assert np.array_equal(is_excluded(mask, pos), excluded[1 : L + 1])


def test_clean_genotypes_drops_missing_masked_and_monomorphic(tmp_path):
    g = allel.GenotypeArray(
        np.array(
            [
                [[0, 1], [0, 0]],  # pos 10: keep
                [[0, -1], [1, 1]],  # pos 20: half-missing call
                [[-1, -1], [0, 1]],  # pos 30: missing call
                [[0, 0], [0, 0]],  # pos 40: monomorphic
                [[0, 1], [1, 1]],  # pos 50: inside mask (49, 55]
                [[1, 0], [0, 0]],  # pos 60: keep
            ],
            dtype="i1",
        )
    )
    pos = np.array([10, 20, 30, 40, 50, 60])
    bed = tmp_path / "mask.bed"
    _write_bed(bed, [("chr1", 49, 55)])
    with pytest.warns(UserWarning, match="missing calls"):
        keep, ac = clean_genotypes(g, pos, "chr1", mask=load_accessibility_mask(str(bed), "1"))
    assert keep.tolist() == [True, False, False, False, False, True]
    assert ac.shape == (6, 2)


def test_no_mask_matches_main(vcf):
    _, win = calculate_stats_vcf_flat(vcf, ["chr22:1-1200000"], stats=STATS, nthreads=1, **CENTERS)
    got = win.sort("center").select("center", "window", *STATS).rows()
    for g, m in zip(got, MAIN_NO_MASK):
        assert g[:2] == m[:2]
        assert np.allclose(g[2:], m[2:], rtol=1e-12, atol=0)


def test_mask_divides_by_accessible_bases(vcf, tmp_path):
    # exclude 30 kb inside the 600 kb window and 10 kb overlapping its edge
    bed = tmp_path / "mask.bed"
    _write_bed(bed, [("chr22", 580_000, 610_000), ("chr22", 645_000, 655_000)])
    _, win = calculate_stats_vcf_flat(
        vcf, ["chr22:1-1200000"], stats=STATS, nthreads=1,
        accessibility_mask=str(bed), **CENTERS,
    )
    row = win.filter(win["center"] == 600000).select(STATS).row(0, named=True)

    raw = allel.read_vcf(vcf)
    pos = raw["variants/POS"]
    ac = allel.GenotypeArray(raw["calldata/GT"]).count_alleles()
    a, b = 600000 - 50000, 600000 + 50000
    sel = (pos >= a) & (pos <= b) & ~(((pos > 580_000) & (pos <= 610_000)) | ((pos > 645_000) & (pos <= 655_000)))
    sel &= ac.is_biallelic_01()
    n = ac[sel][0].sum()
    k = ac[sel][:, 1]
    acc = 100_000 - (30_000 + 5_000)  # 645,000-650,000 lies inside the window
    pi_abs = np.sum(2.0 * k * (n - k) / (n * (n - 1.0)))
    theta_abs = sel.sum() / np.sum(1.0 / np.arange(1, n))
    assert row["pi"] == pytest.approx(pi_abs / acc, rel=1e-12)
    assert row["theta_w"] == pytest.approx(theta_abs / acc, rel=1e-12)
    assert row["tajima_d"] == pytest.approx(tajima_d(np.asarray(ac[sel])), rel=1e-12)

    # windows without masked bases keep their values
    _, win0 = calculate_stats_vcf_flat(vcf, ["chr22:1-1200000"], stats=STATS, nthreads=1, **CENTERS)
    r5 = win.filter(win["center"] == 500000).select(STATS).row(0)
    r5_0 = win0.filter(win0["center"] == 500000).select(STATS).row(0)
    assert r5 == r5_0


def test_scan_pi_per_bp(vcf, tmp_path):
    from flexsweep.scan import scan

    d = os.path.dirname(vcf)
    bed = tmp_path / "mask.bed"
    _write_bed(bed, [("chr22", 580_000, 610_000)])
    kw = dict(nthreads=1, w_size_bp=100_000, step_bp=50_000)
    p0 = scan(d, str(tmp_path / "nomask"), ["pi", "neutrality"], **kw)
    p1 = scan(d, str(tmp_path / "mask"), ["pi", "neutrality"], accessibility_mask=str(bed), **kw)

    raw = allel.read_vcf(vcf)
    pos = raw["variants/POS"]
    ac = allel.GenotypeArray(raw["calldata/GT"]).count_alleles()
    n = ac[0].sum()
    masked = (pos > 580_000) & (pos <= 610_000)

    def pi_abs(c, drop):
        s = (pos >= c - 50_000) & (pos <= c + 50_000)
        if drop:
            s &= ~masked
        k = ac[s][:, 1]
        return np.sum(2.0 * k * (n - k) / (n * (n - 1.0)))

    def excluded(c):  # masked bases in [c - 50 kb, c + 50 kb]
        return max(0, min(c + 50_000, 610_000) - max(c - 50_000 - 1, 580_000))

    centers = p0["pi"]["positions"].to_list()
    assert centers == p1["pi"]["positions"].to_list()
    assert any(excluded(c) > 0 for c in centers) and any(excluded(c) == 0 for c in centers)
    for c, v0, v1, nv in zip(
        centers, p0["pi"]["pi"], p1["pi"]["pi"], p1["neutrality"]["pi"]
    ):
        assert v0 == pytest.approx(pi_abs(c, False) / 100_000, rel=1e-12)
        assert v1 == pytest.approx(pi_abs(c, True) / (100_000 - excluded(c)), rel=1e-12)
        assert nv == pytest.approx(v1, rel=1e-12)


def test_min_accessible_fvs(vcf, tmp_path):
    # 45 kb masked inside the 600 kb window: 55% accessible
    bed = tmp_path / "mask.bed"
    _write_bed(bed, [("chr22", 570_000, 615_000)])
    run = lambda m: calculate_stats_vcf_flat(
        vcf, ["chr22:1-1200000"], stats=STATS, nthreads=1,
        accessibility_mask=str(bed), min_accessible=m, **CENTERS,
    )[1]
    strict, loose = run(0.6), run(0.5)
    r6 = strict.filter(strict["center"] == 600000).select(STATS).row(0)
    r5 = loose.filter(loose["center"] == 600000).select(STATS).row(0)
    assert all(np.isnan(v) for v in r6)  # below 60%: every window statistic missing
    assert all(np.isfinite(v) for v in r5)  # 55% >= 50%: computed
    # windows away from the mask are untouched by the cutoff
    for c in (500000, 700000):
        assert strict.filter(strict["center"] == c).select(STATS).row(0) == \
            loose.filter(loose["center"] == c).select(STATS).row(0)


def test_min_accessible_scan(vcf, tmp_path):
    from flexsweep.scan import scan

    d = os.path.dirname(vcf)
    bed = tmp_path / "mask.bed"
    _write_bed(bed, [("chr22", 560_000, 620_000)])  # 60 kb
    kw = dict(nthreads=1, w_size_bp=100_000, step_bp=10_000)
    p0 = scan(d, str(tmp_path / "nomask"), ["pi", "tajima_d"], **kw)
    p1 = scan(d, str(tmp_path / "mask"), ["pi", "tajima_d"], accessibility_mask=str(bed), **kw)

    def frac(c):  # accessible fraction of [c - 50 kb, c + 50 kb]
        ex = max(0, min(c + 50_000, 620_000) - max(c - 50_000 - 1, 560_000))
        return (100_000 - ex) / 100_000

    for stat in ("pi", "tajima_d"):
        all_c = p0[stat]["positions"].to_list()
        kept = p1[stat]["positions"].to_list()
        assert kept == [c for c in all_c if frac(c) >= 0.6]
        assert len(kept) < len(all_c)

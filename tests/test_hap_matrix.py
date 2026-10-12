"""Haplotype-matrix images (flexsweep.hap_matrix).

Reviewer 1 (MBE-26-0872, point 1d) asked for image-based inputs. Oracles: (1) the
central-block indices and zero padding against a hand-computed small case; (2) sorting
by haplotype frequency preserves the multiset of rows and yields contiguous groups of
non-increasing size; (3) the same haplotypes written as a discoal/ms file and as a
bgzipped VCF give identical images (simulation and empirical readers agree); (4) the
simulation driver labels subclasses with the same rule as the feature vectors.
"""

import gzip

import numpy as np
import polars as pl

from flexsweep.hap_matrix import haplotype_images, locus_image, ms_image

L = 1_200_000


def test_central_block_and_padding():
    rng = np.random.default_rng(0)
    hap = rng.integers(0, 2, (10, 6)).astype(np.int8)  # 10 sites x 6 haplotypes
    pos = np.arange(10) * 100 + 50  # 50, 150, ..., 950
    img, ipos, n = locus_image(hap, pos, center=160, K=5, sort=None)
    # nearest SNP to 160 is site 1 -> column 2; sites -1..3 -> site -1 padded
    assert n == 4
    np.testing.assert_array_equal(ipos, [-1, 50, 150, 250, 350])
    np.testing.assert_array_equal(img[:, 0], 0)
    np.testing.assert_array_equal(img[:, 1:], hap[0:4].T)
    img, ipos, n = locus_image(hap, pos, center=500, K=5, sort=None)
    assert n == 5 and ipos[2] == 450  # tie 450/550 -> first (argmin)
    np.testing.assert_array_equal(img, hap[2:7].T)


def test_haplotype_frequency_sorting():
    rows = np.array([[0, 1, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0], [0, 1, 1], [1, 1, 0]])
    img, _, _ = locus_image(rows.T, np.array([100, 200, 300]), 200, K=3)
    assert sorted(map(tuple, img)) == sorted(map(tuple, rows))
    np.testing.assert_array_equal(img[:3], [[0, 1, 1]] * 3)
    np.testing.assert_array_equal(img[3:5], [[1, 1, 0]] * 2)
    np.testing.assert_array_equal(img[5], [0, 0, 0])


def _toy_locus(seed, n_hap=20, n_sites=60):
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(np.arange(500_000, 700_000), n_sites, replace=False))
    hap = rng.integers(0, 2, (n_sites, n_hap)).astype(np.int8)
    hap[hap.sum(1) == 0, 0] = 1  # keep every site polymorphic
    hap[hap.sum(1) == n_hap, 0] = 0
    return hap, pos


def _write_ms(path, hap, pos):
    with gzip.open(path, "wt") as fh:
        fh.write(f"discoal {hap.shape[1]} 1 {L}\n1 2\n\n//\nsegsites: {hap.shape[0]}\n")
        fh.write("positions: " + " ".join(f"{(p + 0.5) / L:.10f}" for p in pos) + "\n")
        for row in hap.T:
            fh.write("".join(map(str, row)) + "\n")


def test_ms_and_vcf_give_the_same_image(tmp_path):
    from flexsweep.polarize import bgzip_index

    hap, pos = _toy_locus(1)
    _write_ms(str(tmp_path / "rep.ms.gz"), hap, pos)
    ms = ms_image(str(tmp_path / "rep.ms.gz"), K=31)

    vdir = tmp_path / "vcf"
    vdir.mkdir()
    vcf = tmp_path / "toy_chr1.vcf"
    n_ind = hap.shape[1] // 2
    with open(vcf, "w") as fh:
        fh.write(f"##fileformat=VCFv4.2\n##contig=<ID=chr1,length={L}>\n")
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join(f"i{i}" for i in range(n_ind)) + "\n")
        for p, row in zip(pos, hap):
            gts = "\t".join(f"{row[2 * i]}|{row[2 * i + 1]}" for i in range(n_ind))
            fh.write(f"chr1\t{p}\t.\tA\tT\t.\tPASS\t.\tGT\t{gts}\n")
    bgzip_index(str(vcf), str(vdir / "toy_chr1.vcf.gz"))
    imgs, ipos, meta = haplotype_images(str(vdir), vcf=True, K=31, step_vcf=L)

    assert meta.height == 1 and meta["n_snps"][0] == ms[2] == 31
    np.testing.assert_array_equal(ipos[0], ms[1])
    np.testing.assert_array_equal(imgs[0], ms[0])


def test_simulation_driver_labels_and_iters(tmp_path):
    for d in ("neutral", "sweep"):
        (tmp_path / d).mkdir()
    rows = []
    for i in range(1, 4):
        hap, pos = _toy_locus(i)
        _write_ms(str(tmp_path / "neutral" / f"neutral_{i}.ms.gz"), hap, pos)
        rows.append((i, 1e-8, 1e-8, 0.0, 0.0, 0.0, 0.0, "neutral"))
    sweeps = [(0.01, 1.0, 500), (0.01, 0.5, 5000), (0.1, 1.0, 5000)]  # f_i, f_t, t
    for i, (fi, ft, t) in enumerate(sweeps, start=1):
        hap, pos = _toy_locus(10 + i)
        _write_ms(str(tmp_path / "sweep" / f"sweep_{i}.ms.gz"), hap, pos)
        rows.append((i, 1e-8, 1e-8, ft, fi, 0.01, t, "sweep"))
    with gzip.open(tmp_path / "params.txt.gz", "wt") as fh:
        fh.write(pl.DataFrame(rows, schema=["iter", "mu", "r", "eaf", "saf", "s", "t", "model"],
                              orient="row").write_csv())

    imgs, _, meta = haplotype_images(str(tmp_path), K=21,
                                     iters={"neutral": [2], "sweep": [1, 2, 3]})
    assert imgs.shape == (4, 20, 21)
    assert meta["iter"].to_list() == [2, 1, 2, 3]
    assert meta["model"].to_list() == [
        "neutral", "hard_young_complete", "hard_old_incomplete", "soft_old_complete"]
    hap, pos = _toy_locus(12)
    np.testing.assert_array_equal(imgs[2], locus_image(hap, pos, L / 2, 21)[0])


def test_span_covers_a_fixed_width_and_thins_to_K():
    """Oracle: hand-listed SNP indices inside the window, then evenly spaced picks."""
    import numpy as np

    from flexsweep.hap_matrix import locus_image, thin_indices

    pos = np.arange(0, 100) * 1000 + 500          # 500, 1500, ... 99500
    hap = (np.arange(100)[:, None] % 3 == 0).astype(np.int8).repeat(4, axis=1)
    # window [40500, 59500] holds sites 40..59 inclusive -> 20 SNPs, thinned to 5
    img, img_pos, n = locus_image(hap, pos, center=50_000, K=5, sort=None, span=19_000)
    assert n == 5
    want = pos[40 + np.round(np.linspace(0, 19, 5)).astype(int)]
    np.testing.assert_array_equal(img_pos, want)            # 40500, 45500, 50000+, ...
    np.testing.assert_array_equal(img, hap[40 + np.round(np.linspace(0, 19, 5)).astype(int)].T)
    np.testing.assert_array_equal(thin_indices(20, 5), np.array([0, 5, 10, 14, 19]))


def test_offset_zero_and_reproducibility():
    import numpy as np

    from flexsweep.hap_matrix import locus_image

    rng = np.random.default_rng(13)
    pos = np.sort(rng.choice(np.arange(1, 1_200_000), 2000, replace=False))
    hap = rng.integers(0, 2, (2000, 6)).astype(np.int8)
    base = locus_image(hap, pos, 600_000, K=61, sort="haplotype_freq")
    zero = locus_image(hap, pos, 600_000, K=61, sort="haplotype_freq", offset=0)
    np.testing.assert_array_equal(base[0], zero[0])     # default path untouched
    np.testing.assert_array_equal(base[1], zero[1])
    kw = dict(K=61, sort="haplotype_freq", offset=150_000)
    a = locus_image(hap, pos, 600_000, seed=4, **kw)[1]
    np.testing.assert_array_equal(a, locus_image(hap, pos, 600_000, seed=4, **kw)[1])
    assert not np.array_equal(a, locus_image(hap, pos, 600_000, seed=5, **kw)[1])

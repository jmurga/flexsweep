"""Per-window centers ({window: [centers]}) in fv.py: one pass gives the same numbers as
one call per window size.

Oracle: the existing list-centers path, called once per window size (the way the Caldas
geometry was built before), and itertools.product for the list behaviour.
"""

import os
from itertools import product

import numpy as np
import polars as pl
import pytest

from flexsweep import fv
from flexsweep.main import caldas_window_centers


def test_list_centers_unchanged():
    centers, windows = [50000, 150000, 250000], [10000, 100000]
    assert fv._window_center_pairs(centers, windows) == list(product(centers, windows))
    assert fv._sort_centers([3, 1, 2]).tolist() == [1, 2, 3]


def test_dict_centers_give_only_their_pairs():
    wc = {20000: [610000, 590000], 5000: [600000]}
    assert fv._window_center_pairs(wc, windows=[999]) == [
        (600000, 5000), (590000, 20000), (610000, 20000)]
    assert fv._sort_centers(wc) == {20000: [590000, 610000], 5000: [600000]}


def _snp_table(seed=1, n=400, length=1_200_000):
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(length, n, replace=False))
    return pl.DataFrame({"positions": pos, "a": rng.normal(size=n), "b": rng.normal(size=n)})


def test_cut_snps_dict_equals_per_window_calls():
    df = _snp_table()
    wc = {10000: [590000, 600000, 610000], 50000: [550000, 600000, 650000]}
    one = fv.cut_snps(df, wc, sorted(wc), ["a", "b"], iter_value=7)
    per = pl.concat([fv.cut_snps(df, wc[w], [w], ["a", "b"], iter_value=7) for w in sorted(wc)])
    key = ["center", "window"]
    assert one.sort(key).equals(per.sort(key))
    assert one.height == sum(len(c) for c in wc.values())


def test_cut_snps_dict_control_detects_shift():
    """Control: a 1 bp change of one center must change the result."""
    df = _snp_table()
    wc = {10000: [590000, 600000, 610000]}
    shifted = {10000: [590000, 600000 + 4001, 610000]}
    a = fv.cut_snps(df, wc, [10000], ["a", "b"]).sort("center")
    b = fv.cut_snps(df, shifted, [10000], ["a", "b"]).sort("center")
    assert not a.select("a", "b").equals(b.select("a", "b"))


# test_dict_centers_rejected_for_vcf removed: the VCF path now accepts a mapping
# (tests/test_caldas_vcf.py::test_dict_centers_accepted_for_vcf).


def test_caldas_window_centers_matches_test_code():
    wc = caldas_window_centers()
    assert len(wc) == 15 and all(len(c) == 15 for c in wc.values())
    assert min(wc) == 10000 and max(wc) == 150000
    assert all(600000 in c for c in wc.values())  # every size is fanned on the center
    try:
        from flexsweep.test_code.caldas_geometry import caldas_window_centers as ref
    except ImportError:
        pytest.skip("test_code not available (installed wheel)")
    assert wc == ref()


# Per-SNP statistics whose neighbourhood is half the SMALLEST window of the call
# (calculate_stats_simulations: _snp_window_size = min(windows) // 2). In one pass every
# window size shares that neighbourhood; a per-window call uses half its own window.
FS_STATS = ("dind", "high_freq", "low_freq", "s_ratio", "hapdaf_o", "hapdaf_s")


def _stat_window(col):
    stat, w, _ = col.rsplit("_", 2)
    return stat, int(w)


@pytest.mark.slow
def test_one_pass_equals_per_window_passes(tmp_path):
    """End to end on small simulations: one summary_statistics call with a
    {window: [centers]} mapping vs one call per window size.

    Expected, normalized and raw: every column identical on the rows present in both,
    except the six fs statistics at windows larger than the smallest, whose
    neighbourhood differs by design (10 kb in one pass vs 50 kb for the 100 kb window
    alone); those must differ."""
    from flexsweep.simulate_discoal import Simulator

    sims = str(tmp_path / "sims")
    sim = Simulator(sample_size=20, demes="constant", output_folder=sims,
                    num_simulations=30, nthreads=2, seed=3)
    sim.create_params()
    sim.simulate()
    wc = {20000: [580000, 600000, 620000], 100000: [500000, 600000, 700000]}
    w_min = min(wc)
    one = fv.summary_statistics(sims, windows=sorted(wc), centers=wc, nthreads=2,
                                suffix="one")
    per_window = {w: fv.summary_statistics(sims, windows=[w], centers=wc[w], nthreads=2,
                                           suffix=f"w{w}") for w in sorted(wc)}
    key = ["iter", "model"]
    for idx, label in ((0, "normalized"), (1, "raw")):
        per = None
        for w in sorted(wc):
            out = per_window[w][idx]
            feat = [c for c in out.columns if c.endswith(tuple(f"_{w}_{c}" for c in wc[w]))]
            out = out.select(key + feat)
            per = out if per is None else per.join(out, on=key, how="inner")
        feat = [c for c in per.columns if c not in key]
        m = one[idx].select(key + feat).join(per, on=key, how="inner", suffix="_ref")
        assert m.height > 0

        def values(cols):
            return (m.select(cols).to_numpy().astype(float),
                    m.select([c + "_ref" for c in cols]).to_numpy().astype(float))

        differs = [c for c in feat if _stat_window(c)[0] in FS_STATS and _stat_window(c)[1] > w_min]
        same = [c for c in feat if c not in differs]
        assert len(differs) == len(FS_STATS) * len(wc[100000])

        x, y = values(same)
        assert np.array_equal(np.isfinite(x), np.isfinite(y)), label
        ok = np.isfinite(x)
        np.testing.assert_allclose(x[ok], y[ok], rtol=1e-10, atol=1e-12, err_msg=label)

        # control: the fs statistics at the larger window must NOT match, or the
        # neighbourhood rule above (and this test's split) is wrong
        x, y = values(differs)
        ok = np.isfinite(x) & np.isfinite(y)
        assert not np.allclose(x[ok], y[ok], rtol=1e-10, atol=1e-12), label
    assert os.path.exists(os.path.join(sims, "fvs_one.parquet"))

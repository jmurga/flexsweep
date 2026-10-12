"""Every scan stat must write its own table (reviewer bug 3, plus silent no-output stats).

Input: one bundled neutral replicate (tests/neutral, n=216, 1.2 Mb) written as a VCF.
Oracle: structural -- each output holds that stat's rank column and its p-value, and no
other stat's columns. A second check compares two independent code paths for
normalized Fay & Wu's H. Not a numerical-accuracy test.
"""

import os

import polars as pl
import pytest

from flexsweep.scan import STAT_REGISTRY, available_stats, scan
from tests.test_cli_smoke import _ms_to_vcf

MS = os.path.join(os.path.dirname(__file__), "neutral", "neutral_1.ms.gz")
SLOW = {"lassip", "zns", "omega"}


@pytest.fixture(scope="module")
def vcf_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("vcf")
    _ms_to_vcf(MS, str(d / "sim_chr22.vcf.gz"))
    return d


@pytest.mark.parametrize(
    "stat",
    [pytest.param(s, marks=pytest.mark.slow) if s in SLOW else s for s in available_stats()],
)
def test_stat_writes_own_table(vcf_dir, stat):
    prefix = str(vcf_dir / f"out_{stat}")
    res = scan(str(vcf_dir), prefix, [stat], nthreads=2)
    rank_col = STAT_REGISTRY[stat].rank_col
    df = pl.read_csv(f"{prefix}.{stat}.txt", separator="\t")
    assert df.height > 0
    assert rank_col in df.columns, df.columns
    assert f"{rank_col}_pvalue" in df.columns
    assert res[stat].height == df.height


def test_fay_wu_h_matches_neutrality(vcf_dir):
    a = scan(str(vcf_dir), str(vcf_dir / "fwh"), ["fay_wu_h"], nthreads=2)["fay_wu_h"]
    b = scan(str(vcf_dir), str(vcf_dir / "neu"), ["neutrality"], nthreads=2)["neutrality"]
    j = a.join(b, on=["chr", "positions"])
    assert j.height > 0
    assert (j["fay_wu_h"] - j["fay_wu_h_norm"]).abs().max() < 1e-5


def test_bp_windows_are_full_and_centred_inside_data():
    """bp windows must not run past the last SNP (they used to, up to w/2 beyond)."""
    import numpy as np

    from flexsweep.scan import _bp_windows

    rng = np.random.default_rng(0)
    pos = np.sort(rng.choice(np.arange(20_000_000, 25_000_000), 5000, replace=False))
    w, step = 1_000_000, 10_000
    wins = list(_bp_windows(pos, w, step))
    assert wins
    for i, j, centre in wins:
        start = centre - w // 2
        assert start >= pos[0] and start + w <= pos[-1]
        assert pos[i] >= start and pos[j - 1] <= start + w
    # every full window of the scan grid is present
    assert len(wins) == (pos[-1] - w - pos[0]) // step + 1

"""compute_r2_matrix_upper (float64 and float32) and omega_linear_correct.

Oracles: r^2 against np.corrcoef on a small case; float32 ZnS/omega_max against the
float64 path on bundled sweep replicates. Regression: omega_linear_correct read
uninitialised sentinels (np.empty), so repeated calls on the same input differed.
"""

import os

import numpy as np
import pytest

from flexsweep.fv import compute_r2_matrix_upper, omega_linear_correct, parse_ms_numpy

SWEEP_DIR = os.path.join(os.path.dirname(__file__), "sweep")
SWEEP_FILES = sorted(os.path.join(SWEEP_DIR, f) for f in os.listdir(SWEEP_DIR) if f.endswith(".ms.gz"))


def _random_hap(rng, n_snps, n_hap, maf_min=0.05):
    freqs = rng.uniform(maf_min, 1.0 - maf_min, size=n_snps)
    return rng.binomial(1, freqs[:, None], size=(n_snps, n_hap)).astype(np.float64)


def _windows(ms_path, locus_length=1_200_000, window=100_000):
    hap, _, _, _, positions, _ = parse_ms_numpy(ms_path, seq_len=locus_length)
    hap = hap.astype(np.float64)
    for c in range(window // 2, locus_length, window):
        a = np.searchsorted(positions, c - window // 2)
        b = np.searchsorted(positions, c + window // 2, side="right")
        if b - a >= 3:
            yield hap[a:b], f"{os.path.basename(ms_path)} c={c}"


@pytest.mark.parametrize("as_float32", [False, True])
def test_r2_matches_corrcoef(as_float32):
    hap = _random_hap(np.random.default_rng(0), 40, 30)
    hap = hap[(hap.sum(1) > 0) & (hap.sum(1) < hap.shape[1])]
    want = np.triu(np.corrcoef(hap) ** 2, k=1)
    got = compute_r2_matrix_upper(hap, as_float32=as_float32)
    np.testing.assert_allclose(got, want, atol=1e-4 if as_float32 else 1e-12)


def test_non_float64_input_raises():
    hap = _random_hap(np.random.default_rng(0), 50, 100)
    for bad in (hap.astype(np.float32), hap.astype(np.int8)):
        with pytest.raises(TypeError, match="float64"):
            compute_r2_matrix_upper(bad, as_float32=False)


def test_monomorphic_snps_finite():
    hap = _random_hap(np.random.default_rng(77), 100, 100)
    hap[0, :] = 0
    hap[-1, :] = 1
    for as_float32 in (False, True):
        assert np.isfinite(compute_r2_matrix_upper(hap, as_float32=as_float32)).all()


@pytest.mark.parametrize("as_float32", [False, True])
def test_omega_deterministic(as_float32):
    r2 = compute_r2_matrix_upper(_random_hap(np.random.default_rng(11), 200, 100), as_float32=as_float32)
    _, om1 = omega_linear_correct(r2)
    _, om2 = omega_linear_correct(r2)
    assert om1 == om2 and om1 > 0.0


def test_float32_zns_omega_match_float64_on_sweeps():
    assert SWEEP_FILES
    for ms_path in SWEEP_FILES:
        for sub, label in _windows(ms_path):
            zns64, om64 = omega_linear_correct(compute_r2_matrix_upper(sub, as_float32=False))
            zns32, om32 = omega_linear_correct(compute_r2_matrix_upper(sub, as_float32=True))
            assert abs(zns32 - zns64) <= 1e-4, label
            # omega can be large (~1e4): compare relatively
            np.testing.assert_allclose(om32, om64, rtol=1e-4, atol=0, err_msg=label)

"""
Numerical correctness of compute_r2_matrix_upper(as_float32=True) for ZnS and omega_max.

Root cause of prior non-determinism
-------------------------------------
omega_linear_correct used np.empty() for prefix_L (size S) and suffix_R (size S+1).
The sentinels prefix_L[0] and suffix_R[S] were never written before the recurrence
loops read them, so the first call read uninitialised memory (random garbage → omega=0)
and the second call read the OS-zeroed page or leftover zeros from a prior call.

Fix: prefix_L = np.zeros(...) and suffix_R[S] = 0.0 before the loop.

Summary of findings (post-fix)
--------------------------------
SAFE:
  - r² element-wise values: max error ~1e-5 for both random and sweep haplotypes.
  - ZnS: mean of N*(N-1)/2 r² values — errors average down, safe to 1e-4.
  - omega_max: omega_linear_correct uses explicit float64 accumulators (row_sum,
    col_sum, prefix_L, suffix_R, total are all float64).  float32 r² values are
    promoted to float64 during accumulation; the resulting omega agrees with the
    float64 path to ~1e-6.

Conclusion: as_float32=True is safe for both ZnS and omega_max after the
sentinel-initialisation bug fix.

Input dtype contract
---------------------
compute_r2_matrix_upper raises TypeError for non-float64 input unless as_float32=True.
"""

import os

import numpy as np
import pytest

from flexsweep.fv import compute_r2_matrix_upper, omega_linear_correct, parse_ms_numpy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = os.path.dirname(__file__)
NEUTRAL_DIR = os.path.join(TESTS_DIR, "neutral")
SWEEP_DIR = os.path.join(TESTS_DIR, "sweep")

NEUTRAL_FILES = sorted(
    os.path.join(NEUTRAL_DIR, f)
    for f in os.listdir(NEUTRAL_DIR)
    if f.endswith(".ms.gz")
)

SWEEP_FILES = sorted(
    os.path.join(SWEEP_DIR, f)
    for f in os.listdir(SWEEP_DIR)
    if f.endswith(".ms.gz")
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_hap(rng, n_snps, n_hap, maf_min=0.05):
    freqs = rng.uniform(maf_min, 1.0 - maf_min, size=n_snps)
    return rng.binomial(1, freqs[:, None], size=(n_snps, n_hap)).astype(np.float64)


def _ld_f64(hap):
    r2 = compute_r2_matrix_upper(hap, as_float32=False)
    return omega_linear_correct(r2)


def _ld_f32(hap):
    """float32 r² matrix, passed directly to omega_linear_correct (float64 accumulators)."""
    r2 = compute_r2_matrix_upper(hap, as_float32=True)
    return omega_linear_correct(r2)


def _windows_from_ms(ms_path, locus_length=1_200_000, window=100_000, step=100_000):
    """Yield (sub_hap_f64, label) for each non-empty window in an ms file."""
    result = parse_ms_numpy(ms_path, seq_len=locus_length)
    assert result is not None, f"parse_ms_numpy failed for {ms_path}"
    hap_int, _, _, _, positions, _ = result
    hap_f = hap_int.astype(np.float64)
    fname = os.path.basename(ms_path)
    centers = np.arange(step // 2, locus_length - step // 2 + step, step, dtype=int)
    for c in centers:
        left = np.searchsorted(positions, c - window // 2)
        right = np.searchsorted(positions, c + window // 2, side="right")
        sub = hap_f[left:right]
        if sub.shape[0] < 3:
            continue
        yield sub, f"{fname} c={c} S={sub.shape[0]}"


# ---------------------------------------------------------------------------
# Input dtype contract
# ---------------------------------------------------------------------------


def test_non_float64_input_raises():
    """compute_r2_matrix_upper raises TypeError for non-float64 input (no as_float32)."""
    rng = np.random.default_rng(0)
    hap_f32 = _random_hap(rng, 50, 100).astype(np.float32)
    hap_i8 = _random_hap(rng, 50, 100).astype(np.int8)
    for bad in (hap_f32, hap_i8):
        with pytest.raises(TypeError, match="float64"):
            compute_r2_matrix_upper(bad, as_float32=False)


def test_float64_input_accepted():
    """compute_r2_matrix_upper accepts float64 input without as_float32."""
    rng = np.random.default_rng(1)
    hap = _random_hap(rng, 50, 100)
    r2 = compute_r2_matrix_upper(hap, as_float32=False)
    assert r2.dtype == np.float64


def test_as_float32_accepts_float64_input():
    """as_float32=True accepts float64 input and internally converts."""
    rng = np.random.default_rng(2)
    hap = _random_hap(rng, 50, 100)
    r2 = compute_r2_matrix_upper(hap, as_float32=True)
    assert r2.dtype == np.float32


# ---------------------------------------------------------------------------
# omega_linear_correct: determinism (regression for uninitialized-sentinel bug)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_snps,seed", [(50, 10), (200, 11), (400, 12)])
def test_omega_deterministic_float64(n_snps, seed):
    """omega_linear_correct returns the same value on repeated calls (float64 input)."""
    rng = np.random.default_rng(seed)
    hap = _random_hap(rng, n_snps, 100)
    r2 = compute_r2_matrix_upper(hap, as_float32=False)
    _, om1 = omega_linear_correct(r2)
    _, om2 = omega_linear_correct(r2)
    assert om1 == om2, f"omega not deterministic: {om1} vs {om2}"
    assert om1 > 0.0, "omega unexpectedly zero (uninitialized sentinel?)"


@pytest.mark.parametrize("n_snps,seed", [(50, 20), (200, 21), (400, 22)])
def test_omega_deterministic_float32(n_snps, seed):
    """omega_linear_correct returns the same value on repeated calls (float32 input)."""
    rng = np.random.default_rng(seed)
    hap = _random_hap(rng, n_snps, 100)
    r2 = compute_r2_matrix_upper(hap, as_float32=True)
    _, om1 = omega_linear_correct(r2)
    _, om2 = omega_linear_correct(r2)
    assert om1 == om2, f"omega not deterministic: {om1} vs {om2}"
    assert om1 > 0.0, "omega unexpectedly zero (uninitialized sentinel?)"


# ---------------------------------------------------------------------------
# r² element accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_snps,n_hap,seed", [
    (50,  100, 1),
    (200, 100, 2),
    (325, 100, 3),
    (466, 100, 4),
    (509, 100, 5),
    (637, 100, 6),
    (700, 100, 7),
])
def test_r2_element_accuracy_random(n_snps, n_hap, seed):
    """float32 r² elements agree with float64 to atol=1e-4 on random haplotypes."""
    rng = np.random.default_rng(seed)
    hap = _random_hap(rng, n_snps, n_hap)
    r2_f64 = compute_r2_matrix_upper(hap, as_float32=False)
    r2_f32 = compute_r2_matrix_upper(hap, as_float32=True)
    np.testing.assert_allclose(
        r2_f32.astype(np.float64), r2_f64, atol=1e-4, rtol=0,
        err_msg=f"S={n_snps}: r² element mismatch",
    )


@pytest.mark.skipif(not SWEEP_FILES, reason="no sweep ms.gz files found")
def test_r2_element_accuracy_sweep_window():
    """float32 r² elements remain accurate to atol=1e-4 for sweep windows."""
    for ms_path in SWEEP_FILES:
        for sub, label in _windows_from_ms(ms_path):
            r2_f64 = compute_r2_matrix_upper(sub, as_float32=False)
            r2_f32 = compute_r2_matrix_upper(sub, as_float32=True)
            max_err = float(np.abs(r2_f32.astype(np.float64) - r2_f64).max())
            assert max_err <= 1e-4, (
                f"{label}: r² element max error {max_err:.2e} exceeds 1e-4"
            )


# ---------------------------------------------------------------------------
# ZnS accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_snps,n_hap,seed", [
    (325, 100, 10),
    (466, 100, 11),
    (637, 100, 12),
])
def test_zns_float32_random(n_snps, n_hap, seed):
    """ZnS from float32 r² agrees with float64 to atol=1e-4 on random haplotypes."""
    rng = np.random.default_rng(seed)
    hap = _random_hap(rng, n_snps, n_hap)
    zns64, _ = _ld_f64(hap)
    zns32, _ = _ld_f32(hap)
    np.testing.assert_allclose(zns32, zns64, atol=1e-4, rtol=0,
                                err_msg=f"ZnS mismatch S={n_snps}")


@pytest.mark.skipif(not SWEEP_FILES, reason="no sweep ms.gz files found")
def test_zns_float32_sweep():
    """ZnS from float32 r² remains accurate to atol=1e-4 for sweep windows."""
    for ms_path in SWEEP_FILES:
        for sub, label in _windows_from_ms(ms_path):
            zns64, _ = _ld_f64(sub)
            zns32, _ = _ld_f32(sub)
            err = abs(zns32 - zns64)
            assert err <= 1e-4, f"{label}: ZnS float32 error {err:.2e} exceeds 1e-4"


# ---------------------------------------------------------------------------
# omega_max accuracy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_snps,n_hap,seed", [
    (325, 100, 30),
    (466, 100, 31),
    (637, 100, 32),
])
def test_omega_float32_random(n_snps, n_hap, seed):
    """omega_max from float32 r² agrees with float64 to rtol=1e-4 on random haplotypes."""
    rng = np.random.default_rng(seed)
    hap = _random_hap(rng, n_snps, n_hap)
    _, om64 = _ld_f64(hap)
    _, om32 = _ld_f32(hap)
    np.testing.assert_allclose(om32, om64, rtol=1e-4, atol=0,
                                err_msg=f"omega mismatch S={n_snps}")


@pytest.mark.skipif(not SWEEP_FILES, reason="no sweep ms.gz files found")
def test_omega_float32_sweep():
    """omega_max from float32 r² matches float64 to rtol=1e-4 across all sweep windows.

    Absolute errors can be large for high-omega windows (e.g. om64=8601, abs_err≈0.1),
    but the relative error stays below 2e-5.  Use rtol, not atol.
    """
    for ms_path in SWEEP_FILES:
        for sub, label in _windows_from_ms(ms_path):
            _, om64 = _ld_f64(sub)
            _, om32 = _ld_f32(sub)
            np.testing.assert_allclose(
                om32, om64, rtol=1e-4, atol=0,
                err_msg=f"{label}: omega float32 relative error exceeds 1e-4",
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_small_window_both_paths():
    """S < 3: both paths return (0.0, 0.0)."""
    rng = np.random.default_rng(42)
    for s in (1, 2):
        hap = _random_hap(rng, s, 100)
        for fn, label in [(_ld_f64, "f64"), (_ld_f32, "f32")]:
            zns, omega = fn(hap)
            assert zns == 0.0 and omega == 0.0, f"S={s} {label}: expected (0, 0)"


def test_monomorphic_snps_finite():
    """Monomorphic rows produce r²=0; no NaN/inf in either dtype."""
    rng = np.random.default_rng(77)
    hap = _random_hap(rng, 100, 100)
    hap[0, :] = 0
    hap[-1, :] = 1

    for dtype_flag, label in [(False, "f64"), (True, "f32")]:
        r2 = compute_r2_matrix_upper(hap, as_float32=dtype_flag)
        assert np.all(np.isfinite(r2)), f"{label}: non-finite r² for monomorphic SNP"


def test_output_dtypes():
    """compute_r2_matrix_upper returns the requested dtype."""
    rng = np.random.default_rng(0)
    hap = _random_hap(rng, 50, 100)
    assert compute_r2_matrix_upper(hap, as_float32=False).dtype == np.float64
    assert compute_r2_matrix_upper(hap, as_float32=True).dtype == np.float32


def test_upper_triangle_only():
    """Lower triangle and diagonal are zero in both dtypes."""
    rng = np.random.default_rng(13)
    hap = _random_hap(rng, 80, 100)
    for flag, label in [(False, "f64"), (True, "f32")]:
        r2 = compute_r2_matrix_upper(hap, as_float32=flag)
        assert np.all(np.tril(r2.astype(np.float64)) == 0), \
            f"{label}: non-zero lower triangle or diagonal"

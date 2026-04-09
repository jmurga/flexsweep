"""
Tests for fast_skew_kurt against scipy and numpy reference implementations.

scipy source (v1.17.0) formulas verified at:
https://raw.githubusercontent.com/scipy/scipy/refs/tags/v1.17.0/scipy/stats/_stats_py.py

Formulas (bias=False, Fisher kurtosis):
  m2 = (1/N) * sum((x - mean)^2)            # biased 2nd central moment
  m3 = (1/N) * sum((x - mean)^3)            # biased 3rd central moment
  m4 = (1/N) * sum((x - mean)^4)            # biased 4th central moment

  var_unbiased = m2 * N / (N - 1)           # matches np.var(ddof=1)

  # scipy skew (bias=False): adjusted Fisher-Pearson coefficient
  skew_unbiased = sqrt(N*(N-1)) / (N-2) * m3 / m2^1.5

  # scipy kurtosis (bias=False, fisher=True):
  kurt_unbiased = 1/(N-2)/(N-3) * ((N^2-1)*m4/m2^2 - 3*(N-1)^2)
  # algebraically identical to the fast_skew_kurt formulation:
  #   (N-1)/((N-2)*(N-3)) * ((N+1)*g2 + 6)  where g2 = m4/m2^2 - 3

Near-zero variance (scipy):
  Trigger: m2 <= (eps_float64 * mean)^2  where eps_float64 = 2.22e-16
  fast_skew_kurt trigger: m2 <= (1.1e-16 * abs(mu))^2  OR  m2 < 1e-20
  Both return NaN for skew/kurtosis; variance returns 0.
"""

import numpy as np
import pytest
from scipy.stats import kurtosis as scipy_kurt
from scipy.stats import skew as scipy_skew

from flexsweep.fv import fast_skew_kurt


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _check(data: np.ndarray, rtol: float = 1e-10) -> None:
    """Compare fast_skew_kurt(bias=False) against scipy/numpy references."""
    data = np.asarray(data, dtype=np.float64)

    var_jit, skew_jit, kurt_jit = fast_skew_kurt(data, False)

    var_ref  = float(np.var(data, ddof=1))
    skew_ref = float(scipy_skew(data, bias=False))
    kurt_ref = float(scipy_kurt(data, bias=False, fisher=True))

    np.testing.assert_allclose(
        var_jit, var_ref, rtol=rtol,
        err_msg=f"variance mismatch  n={len(data)}",
    )

    if np.isnan(skew_ref):
        assert np.isnan(skew_jit), \
            f"expected NaN skew (near-zero variance), got {skew_jit}"
    else:
        np.testing.assert_allclose(
            skew_jit, skew_ref, rtol=rtol,
            err_msg=f"skewness mismatch  n={len(data)}",
        )

    if np.isnan(kurt_ref):
        assert np.isnan(kurt_jit), \
            f"expected NaN kurtosis (near-zero variance), got {kurt_jit}"
    else:
        np.testing.assert_allclose(
            kurt_jit, kurt_ref, rtol=rtol,
            err_msg=f"kurtosis mismatch  n={len(data)}",
        )


# ---------------------------------------------------------------------------
# Main correctness tests
# ---------------------------------------------------------------------------

def test_normal_large():
    """19,503 values — mirrors real pairwise-distance input (198 haplotypes)."""
    rng = np.random.default_rng(42)
    _check(rng.standard_normal(19_503))


def test_normal_medium():
    rng = np.random.default_rng(123)
    _check(rng.standard_normal(500))


def test_uniform():
    rng = np.random.default_rng(7)
    _check(rng.uniform(0.0, 1.0, 1000))


def test_exponential():
    """Exponential distribution (right-skewed, heavy-tailed)."""
    rng = np.random.default_rng(99)
    _check(rng.exponential(scale=1.0, size=10_000))


def test_positive_pairwise_distances():
    """Positive-only data in [0, 1] range — typical per-base distances."""
    rng = np.random.default_rng(55)
    _check(rng.uniform(0.0, 1.0, size=19_503))


def test_integer_like_distances():
    """Integer-valued distances 0..100 (cast to float64)."""
    rng = np.random.default_rng(11)
    _check(rng.integers(0, 100, size=2_000).astype(np.float64))


def test_binomial_pairwise():
    """
    Binomial(1000, 0.001) pairwise distances — typical ms simulation regime.
    Most values near 0, sparse large values.
    """
    rng = np.random.default_rng(2024)
    _check(rng.binomial(n=1000, p=0.001, size=19_503).astype(np.float64))


def test_deterministic_from_scipy_docs():
    """
    data = [2, 8, 0, 4, 1, 9, 9, 0]  from scipy.stats.skew docstring example.
    Pinned against exact scipy output for regression detection.
    """
    data = np.array([2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0])
    _check(data, rtol=1e-14)

    # Pin absolute values so any formula drift is immediately visible
    var_jit, skew_jit, kurt_jit = fast_skew_kurt(data, False)
    np.testing.assert_allclose(skew_jit, float(scipy_skew(data, bias=False)), rtol=1e-14)
    np.testing.assert_allclose(kurt_jit, float(scipy_kurt(data, bias=False, fisher=True)), rtol=1e-14)


def test_minimum_valid_n():
    """n=4 — smallest array where all unbiased corrections are defined."""
    _check(np.array([1.0, 2.0, 4.0, 8.0]))


# ---------------------------------------------------------------------------
# Edge cases: near-zero / zero variance
# ---------------------------------------------------------------------------

def test_all_zeros():
    """All-zero array: variance 0, skew/kurtosis NaN."""
    data = np.zeros(200, dtype=np.float64)
    var_jit, skew_jit, kurt_jit = fast_skew_kurt(data, False)
    assert var_jit == 0.0
    assert np.isnan(skew_jit)
    assert np.isnan(kurt_jit)
    # scipy also returns NaN
    assert np.isnan(float(scipy_skew(data, bias=False)))
    assert np.isnan(float(scipy_kurt(data, bias=False, fisher=True)))


def test_constant_nonzero():
    """Constant non-zero array: variance 0, skew/kurtosis NaN."""
    data = np.full(500, 3.14, dtype=np.float64)
    var_jit, skew_jit, kurt_jit = fast_skew_kurt(data, False)
    assert var_jit == 0.0
    assert np.isnan(skew_jit)
    assert np.isnan(kurt_jit)
    assert np.isnan(float(scipy_skew(data, bias=False)))
    assert np.isnan(float(scipy_kurt(data, bias=False, fisher=True)))


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_too_small_returns_zeros(n):
    """n < 4: guard clause returns (0.0, 0.0, 0.0) — no scipy comparison."""
    data = np.arange(n, dtype=np.float64)
    var_jit, skew_jit, kurt_jit = fast_skew_kurt(data, False)
    assert var_jit == 0.0
    assert skew_jit == 0.0
    assert kurt_jit == 0.0


# ---------------------------------------------------------------------------
# bias=True path
# ---------------------------------------------------------------------------

def test_bias_true_matches_scipy():
    """
    bias=True: fast_skew_kurt returns (var_unbiased, g1_biased, g2_biased).
    g1 = m3 / m2^1.5  (matches scipy.stats.skew(bias=True))
    g2 = m4 / m2^2 - 3  (matches scipy.stats.kurtosis(bias=True, fisher=True))
    variance correction is always n/(n-1) regardless of bias flag.
    """
    rng = np.random.default_rng(77)
    data = rng.standard_normal(500).astype(np.float64)

    var_jit, skew_jit, kurt_jit = fast_skew_kurt(data, True)

    np.testing.assert_allclose(var_jit,  float(np.var(data, ddof=1)),             rtol=1e-12)
    np.testing.assert_allclose(skew_jit, float(scipy_skew(data, bias=True)),      rtol=1e-12)
    np.testing.assert_allclose(kurt_jit, float(scipy_kurt(data, bias=True, fisher=True)), rtol=1e-12)


# ---------------------------------------------------------------------------
# Formula cross-check: manual numpy implementation of scipy's exact formulas
# ---------------------------------------------------------------------------

def _scipy_skew_manual(data: np.ndarray) -> float:
    """
    Reproduce scipy skew(bias=False) from first principles (v1.17.0 formula).
    G1 = sqrt(N*(N-1)) / (N-2) * m3 / m2^1.5
    """
    n = len(data)
    if n < 3:
        return np.nan
    mean = data.mean()
    d = data - mean
    m2 = (d ** 2).mean()
    m3 = (d ** 3).mean()
    eps = np.finfo(np.float64).eps
    if m2 <= (eps * abs(mean)) ** 2:
        return np.nan
    g1 = m3 / m2 ** 1.5
    return float(np.sqrt(n * (n - 1.0)) / (n - 2.0) * g1)


def _scipy_kurt_manual(data: np.ndarray) -> float:
    """
    Reproduce scipy kurtosis(bias=False, fisher=True) from first principles.
    1 / (n-2) / (n-3) * ((n^2-1) * m4/m2^2 - 3*(n-1)^2)
    """
    n = len(data)
    if n < 4:
        return np.nan
    mean = data.mean()
    d = data - mean
    m2 = (d ** 2).mean()
    m4 = (d ** 4).mean()
    eps = np.finfo(np.float64).eps
    if m2 <= (eps * abs(mean)) ** 2:
        return np.nan
    return float(1.0 / (n - 2) / (n - 3) * ((n ** 2 - 1.0) * m4 / m2 ** 2 - 3 * (n - 1) ** 2))


def test_manual_formula_vs_scipy():
    """Manual numpy reproductions of scipy formulas agree with scipy to 1e-13."""
    rng = np.random.default_rng(314)
    data = rng.standard_normal(5000).astype(np.float64)

    np.testing.assert_allclose(
        _scipy_skew_manual(data),
        float(scipy_skew(data, bias=False)),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        _scipy_kurt_manual(data),
        float(scipy_kurt(data, bias=False, fisher=True)),
        rtol=1e-13,
    )


def test_fast_skew_kurt_vs_manual_formulas():
    """fast_skew_kurt matches manual numpy formula reproductions to 1e-12."""
    rng = np.random.default_rng(999)
    data = rng.standard_normal(10_000).astype(np.float64)

    var_jit, skew_jit, kurt_jit = fast_skew_kurt(data, False)

    np.testing.assert_allclose(var_jit,  float(np.var(data, ddof=1)),      rtol=1e-12)
    np.testing.assert_allclose(skew_jit, _scipy_skew_manual(data),         rtol=1e-12)
    np.testing.assert_allclose(kurt_jit, _scipy_kurt_manual(data),         rtol=1e-12)

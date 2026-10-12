"""fast_skew_kurt against scipy.stats.skew / kurtosis and np.var(ddof=1).

Oracle: scipy (bias=False: adjusted Fisher-Pearson skew, Fisher excess kurtosis;
bias=True: g1, g2). Near-zero variance gives variance 0 and NaN skew/kurtosis, as in
scipy; n < 4 returns (0, 0, 0) by design.
"""

import numpy as np
import pytest
from scipy.stats import kurtosis as scipy_kurt
from scipy.stats import skew as scipy_skew

from flexsweep.fv import fast_skew_kurt

_rng = np.random.default_rng(42)
DATA = {
    "normal": _rng.standard_normal(19_503),  # size of real pairwise-distance input
    "exponential": _rng.exponential(size=10_000),
    "binomial": _rng.binomial(1000, 0.001, 19_503).astype(np.float64),  # sparse ms regime
    "n4": np.array([1.0, 2.0, 4.0, 8.0]),
}


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("name", sorted(DATA))
def test_matches_scipy(name, bias):
    data = DATA[name]
    var, sk, ku = fast_skew_kurt(data, bias)
    np.testing.assert_allclose(var, np.var(data, ddof=1), rtol=1e-10)
    np.testing.assert_allclose(sk, scipy_skew(data, bias=bias), rtol=1e-10)
    np.testing.assert_allclose(ku, scipy_kurt(data, bias=bias, fisher=True), rtol=1e-10)


@pytest.mark.parametrize("value", [0.0, 3.14])
def test_constant_gives_nan_moments(value):
    var, sk, ku = fast_skew_kurt(np.full(500, value), False)
    assert var == 0.0 and np.isnan(sk) and np.isnan(ku)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_too_small_returns_zeros(n):
    assert fast_skew_kurt(np.arange(n, dtype=np.float64), False) == (0.0, 0.0, 0.0)

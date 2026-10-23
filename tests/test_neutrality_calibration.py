"""Standardized SFS neutrality tests must have the null variance they claim.

Oracles, strongest first:

1. Analytic: Fu (1995) sigma_ij summed over all frequency classes equals a2
   (Var(S) = a1 theta + a2 theta^2).
2. Brute force (slow): Kingman coalescent with no recombination at a KNOWN theta.
   The simulated variance of each numerator must match alpha*theta + beta*theta^2,
   and the singleton-free S_A moments must match g1*theta and g2*theta^2.
3. Calibration: under the standard neutral model every standardized statistic
   has mean ~0 and sd ~1.

Not covered: recombination, demography, sample sizes outside 20-100, or sites with
different sample sizes (all functions assume a constant n).
"""

import numpy as np
import pytest

from flexsweep.fv import (
    _achaz_y_coeffs,
    achaz_t,
    achaz_y,
    achaz_y_star,
    fay_wu_h_norm,
    fuli_d,
    fuli_d_star,
    fuli_f,
    fuli_f_star,
    neutrality_stats,
    sigma,
    tajima_d,
    zeng_e,
)


def _a1_a2(n):
    k = np.arange(1, n, dtype=np.float64)
    return np.sum(1.0 / k), np.sum(1.0 / (k * k))


def _ac_from_sfs(xi, n):
    """Allele-count array (S, 2) with xi[i] sites at derived count i."""
    dac = np.repeat(np.arange(len(xi)), xi.astype(np.int64))
    return np.column_stack([n - dac, dac]).astype(np.int64)


@pytest.mark.parametrize("n", [10, 20, 50, 100])
def test_fu_sigma_sums_to_a2(n):
    k = np.arange(1, n, dtype=np.int64)
    ii, jj = np.meshgrid(k, k, indexing="ij")
    total = sigma(n, np.column_stack([ii.ravel(), jj.ravel()])).sum()
    assert total == pytest.approx(_a1_a2(n)[1], rel=1e-10)


# --------------------------------------------------------------------------------
# Calibration and brute-force variance under the standard neutral model

SEED = 20260812


def _kingman_sfs(n, theta, rng):
    """One neutral Kingman genealogy (no recombination) -> unfolded SFS xi[0..n]."""
    sizes = [1] * n  # leaves subtended by each active lineage
    lengths = np.zeros(n)  # total branch length subtending i leaves
    k = n
    while k > 1:
        t = rng.exponential(2.0 / (k * (k - 1)))
        for s in sizes:
            lengths[s] += t
        a, b = sorted(rng.choice(k, size=2, replace=False))
        merged = sizes[a] + sizes[b]
        sizes = [s for j, s in enumerate(sizes) if j not in (a, b)] + [merged]
        k -= 1
    xi = np.zeros(n + 1)
    xi[1:n] = rng.poisson(theta / 2.0 * lengths[1:n])
    return xi


def _neutral_sfs(n, s_target, reps):
    rng = np.random.default_rng(SEED + n + s_target)
    a1, _ = _a1_a2(n)
    return [_kingman_sfs(n, s_target / a1, rng) for _ in range(reps)]


STATS = {
    "tajima_d": tajima_d,
    "fuli_d": fuli_d,
    "fuli_d_star": fuli_d_star,
    "fuli_f": fuli_f,
    "fuli_f_star": fuli_f_star,
    "zeng_e": zeng_e,
    "fay_wu_h_norm": lambda ac: fay_wu_h_norm(ac, None)[2],
    "neutrality_stats_h_norm": lambda ac: neutrality_stats(
        ac, np.arange(1, ac.shape[0] + 1, dtype=np.int64)
    )[3],
    "achaz_y": achaz_y,
    "achaz_y_star": achaz_y_star,
    "achaz_t": achaz_t,
}


@pytest.fixture(scope="module", params=[(20, 30), (74, 100), (74, 400), (100, 1500)],
                ids=lambda c: f"n{c[0]}_S{c[1]}")
def neutral_acs(request):
    n, s_target = request.param
    return [_ac_from_sfs(xi, n) for xi in _neutral_sfs(n, s_target, 200)]


@pytest.mark.parametrize("name", list(STATS))
def test_neutral_calibration(neutral_acs, name):
    vals = np.array([float(STATS[name](ac)) for ac in neutral_acs])
    finite = np.isfinite(vals)
    v = vals[finite]
    assert 1 - finite.mean() < 0.02, f"{name}: {100 * (1 - finite.mean()):.1f}% NaN"
    assert abs(v.mean()) < 0.35, f"{name}: mean {v.mean():.2f} (expected ~0)"
    assert 0.6 < v.std() < 1.3, f"{name}: sd {v.std():.2f} (expected ~1)"
    assert np.mean(np.abs(v) > 4) < 0.025, f"{name}: exploding normalization"


def _zeng_h_var(n, theta):
    """Var(pi - theta_L), Zeng et al. (2006) Eq. 12."""
    _, a2 = _a1_a2(n)
    bn1 = a2 + 1.0 / n**2
    v1 = (n - 2.0) / (6.0 * (n - 1.0))
    v2 = (18 * n**2 * (3 * n + 2) * bn1 - (88 * n**3 + 9 * n**2 - 13 * n + 6)) / (
        9.0 * n * (n - 1.0) ** 2
    )
    return v1 * theta + v2 * theta**2


@pytest.mark.slow
@pytest.mark.parametrize("n,theta", [(20, 10.0), (100, 30.0)])
def test_null_moments_brute_force(n, theta):
    rng = np.random.default_rng(SEED + n)
    sfs = np.array([_kingman_sfs(n, theta, rng)[1:n] for _ in range(4000)])  # xi_1..xi_{n-1}
    i = np.arange(1, n, dtype=np.float64)

    h_num = sfs @ (2.0 * i * (n - i) / (n * (n - 1.0)) - i / (n - 1.0))
    ratio = np.var(h_num) / _zeng_h_var(n, theta)
    assert ratio == pytest.approx(1.0, abs=0.1), f"H': Var ratio {ratio:.3f}"

    for folded in (False, True):
        A, c, alpha, beta, g1, g2 = _achaz_y_coeffs(n, folded)
        ratio = np.var(sfs @ c) / (alpha * theta + beta * theta**2)
        assert ratio == pytest.approx(1.0, abs=0.1), f"folded={folded}: Var ratio {ratio:.3f}"
        S_A = sfs[:, A].sum(axis=1)
        assert S_A.mean() / (g1 * theta) == pytest.approx(1.0, abs=0.05)
        assert np.mean(S_A * (S_A - 1.0)) / (g2 * theta**2) == pytest.approx(1.0, abs=0.1)

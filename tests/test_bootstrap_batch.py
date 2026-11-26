"""Bootstrap sampler (enrichment.batch_control_chain).

Oracles: the exact probability of every first block, enumerated on a small pool from the
published rule (uniform gene 1; genes 2 and 3 uniform in the top 10% of the factor furthest
off; accepted when every running mean is inside its limits); an independent numpy replay of
the running-mean rule and of the reuse cap over whole chains. Agreement of whole control
sets with the C program is checked outside the package (fs_enrichment/bench, gates.md).
"""

import numpy as np
import pytest
from scipy import stats

from flexsweep import enrichment as E

from .test_enrichment_fast import _matching_inputs

ALL_POOL = (10_000_000, 150_000, 200_000, 200_000, 20_000_000)  # direct proposals
SCAN = (0, 10**9, 10**9, 0, 1)  # every pair is screened against all last genes


def _problem(seed=3, n_case=2, n_pool=40, nf=4, off=1.5):
    r = np.random.default_rng(seed)
    V = r.normal(10, 0.5, (n_case, nf))
    P = r.normal(10, 2.0, (n_pool, nf)) + np.array([off, -off, 0.3, -off])[:nf]
    return V, P


def _limits(V, P, tol):
    avg, pavg = V.mean(0), P.mean(0)
    lo, hi = (1 - tol) * avg, (1 + tol) * avg
    hi = np.where(avg > pavg, np.inf, hi)
    lo = np.where(avg > pavg, lo, -np.inf)
    return avg, pavg, lo, hi


def _first_block_law(V, P, tol):
    """P(i, j, g) of the first accepted block: 100 fake case means, 3 genes."""
    avg, pavg, lo, hi = _limits(V, P, tol)
    n, nf = P.shape
    top = max(1, n // 10)
    s = np.where(avg > pavg, 1.0, -1.0)
    Y, best = s * P, (s * P).max(0)
    lists = np.argsort(-Y, axis=0, kind="stable")[:top]
    need = s * np.where(s > 0, (1 - tol) * avg, (1 + tol) * avg) * 103 - s * avg * 100

    def furthest(done, left):
        z = np.where(best > 0, (need - done) / (left * np.where(best > 0, best, 1)), -np.inf)
        return int(np.argmax(z)) if (best > 0).any() else 0

    law = {}
    for i in range(n):
        for j in lists[:, furthest(Y[i], 2)]:
            for g in lists[:, furthest(Y[i] + Y[j], 1)]:
                m = (100 * avg + P[i] + P[j] + P[g]) / 103
                if np.all((m >= lo) & (m <= hi)):
                    law[i, j, g] = law.get((i, j, g), 0) + 1
    total = sum(law.values())
    return {k: v / total for k, v in law.items()}


@pytest.mark.parametrize("limits", [ALL_POOL, SCAN], ids=["direct", "scan"])
def test_first_block_follows_the_enumerated_law(limits):
    V, P = _problem()
    law = _first_block_law(V, P, 0.005)
    assert len(law) >= 20
    tables = E._match_tables(V, P, 0.005)
    N = 4000
    counts = dict.fromkeys(law, 0)
    for s in np.random.SeedSequence(8).spawn(N):
        chain = E.batch_control_chain(
            *tables, 2, 10**9, np.random.Generator(np.random.PCG64(s)), 1, 10**7, limits
        )[0]
        counts[tuple(int(x) for x in chain[:3])] += 1  # KeyError: a block that cannot occur
    obs = np.array([counts[k] for k in law])
    exp = N * np.array([law[k] for k in law])
    keep = exp >= 5
    obs = np.append(obs[keep], obs[~keep].sum())
    exp = np.append(exp[keep], exp[~keep].sum())
    ok = exp > 0
    assert stats.chisquare(obs[ok], exp[ok]).pvalue > 1e-3


def test_scan_weights_pairs_by_their_number_of_fits():
    """A pool whose pairs have 0 to 60 fitting last genes, far more than the initial M = 16:
    the first gene of the first block follows its enumerated law."""
    r = np.random.default_rng(4)
    V = np.full((2, 1), 10.0)
    P = np.concatenate([r.normal(5, 3, 500), r.normal(11, 2, 100)])[:, None]
    tol, top = 0.002, 60
    need = 0.998 * 10 * 103 - 1000  # what the three genes must sum to
    best = np.sort(P[:, 0])[::-1][:top]
    fits = (P[:, 0][:, None, None] + best[None, :, None] + best[None, None, :] >= need).sum((1, 2))
    per_pair = (P[:, 0][:, None, None] + best[None, :, None] + best[None, None, :] >= need).sum(2)
    assert per_pair.max() == 60 and (per_pair[per_pair > 0] < 16).mean() > 0.1
    tables = E._match_tables(V, P, tol)
    N, obs = 2500, np.zeros(600)
    for s in np.random.SeedSequence(21).spawn(N):
        chain = E.batch_control_chain(
            *tables, 2, 10**9, np.random.Generator(np.random.PCG64(s)), 1, 3000, SCAN
        )[0]  # gives up soon after the first block
        obs[chain[0]] += 1
    order = np.argsort(fits)  # 10 groups of genes by how often they can start a block
    groups = np.array_split(order[fits[order] > 0], 10)
    o = np.array([obs[g].sum() for g in groups])
    e = N * np.array([fits[g].sum() for g in groups]) / fits.sum()
    assert o.sum() == N
    assert stats.chisquare(o, e * o.sum() / e.sum()).pvalue > 1e-3


def _replay(chain, V, P, tol, max_rep, size=3):
    """Number of blocks that break the running-mean rule or the reuse cap."""
    avg, _, lo, hi = _limits(V, P, tol)
    eps = 1e-9 * np.abs(avg)
    cur, fake, n, used, bad = avg.copy(), 100, 0, np.zeros(len(P), int), 0
    for b in range(0, len(chain), size):
        g = chain[b : b + size]
        new = (fake * avg + cur * n + P[g].sum(0)) / (n + size + fake)
        bad += bool(np.any(new < lo - eps) or np.any(new > hi + eps))
        bad += bool(used[g].max() > max_rep * (n + 1) / len(V))
        np.add.at(used, g, 1)
        cur, fake, n = new, max(fake - 1, 0), n + size
    return bad


@pytest.mark.parametrize("limits", [ALL_POOL, (2000, 50, 120, 500, 4000), SCAN],
                         ids=["direct", "all stages", "scan"])
def test_chain_obeys_rule_and_cap(limits):
    V, P = _problem(5, n_case=40, n_pool=300, nf=4, off=1.0)
    max_rep = 1.0  # the cap binds: the most used genes end within 3 of it
    tables = E._match_tables(V, P, 0.02)
    chain, ends, used, stuck = E.batch_control_chain(
        *tables, 40, max_rep, np.random.default_rng(2), 3, 10**7, limits
    )
    assert stuck == -1 and len(chain) % 3 == 0
    assert np.array_equal(used, np.bincount(chain, minlength=300))
    assert all(e >= 100 + 11 * 40 * (r + 1) for r, e in enumerate(ends)) and ends[-1] == len(chain)
    assert _replay(chain, V, P, 0.02, max_rep) == 0
    assert used.max() >= max_rep * len(chain) / 40 - 3
    swapped = chain.copy()
    swapped[300:330] = np.argsort(P[:, 1])[:30]  # control: blocks that break the rule
    assert _replay(swapped, V, P, 0.02, max_rep) > 0


def test_level_bitsets_never_lose_a_gene():
    V, P = _problem(7, n_case=20, n_pool=150, nf=4)
    P[:, 2] = np.round(P[:, 2])
    D, _, order, top_bits, bits, d_min, inv_step, *_ = E._match_tables(V, P, 0.05)
    unpack = lambda w: np.unpackbits(w.view(np.uint8), bitorder="little")[:150].astype(bool)
    q = np.random.default_rng(1).uniform(D.min(0) - 1, D.max(0) + 1, (200, D.shape[1]))
    for need in np.vstack([q, D[:50]]):  # random needs, and needs equal to a gene's value
        for c in range(D.shape[1]):
            lev = int((need[c] - d_min[c]) * inv_step[c])
            if lev > E._N_LEVELS:
                assert not (D[:, c] >= need[c]).any()
            elif lev > 0:
                got = unpack(bits[c, min(lev, E._N_LEVELS - 1)])
                assert not ((D[:, c] >= need[c]) & ~got).any()
    for f in range(4):
        assert set(np.flatnonzero(unpack(top_bits[f]))) == set(order[f])


def test_top_lists_break_ties_in_pool_order():
    """As the C: a binary factor lists the first pool genes among the ties."""
    V = np.full((5, 2), 1.0)
    P = np.zeros((100, 2))
    P[10:60, 0] = 1.0
    P[:, 1] = np.arange(100) / 200
    order = E._match_tables(V, P, 0.05)[2]
    assert order[0].tolist() == list(range(10, 20))


def test_gives_up_instead_of_loosening():
    V, P = _problem(5, n_case=40, n_pool=300, nf=4)
    tables = E._match_tables(V, P - 100.0, 0.01)
    chain, ends, used, stuck = E.batch_control_chain(
        *tables, 40, 25, np.random.default_rng(2), 1, 20_000
    )
    assert stuck == 0 and len(chain) == 0 and used.sum() == 0


def test_larger_blocks_when_three_do_not_fit():
    """Four kinds of genes, each +3 in its factor and -1 in the others (the rest of the pool
    is low): one of each kind balances, three genes always miss a factor."""
    V = np.full((4, 4), 10.0)
    P = np.full((40, 4), 1.0)
    P[:16] = 9.0
    for k in range(4):
        P[4 * k : 4 * k + 4, k] = 13.0
    tables = E._match_tables(V, P, 1e-6)
    chain, _, _, stuck = E.batch_control_chain(
        *tables, 4, 10**9, np.random.default_rng(1), 1, 50_000
    )
    assert stuck == -1 and len(chain) % 4 == 0
    assert _replay(chain, V, P, 1e-6, 10**9, size=4) == 0


def test_run_bootstrap_batch(tmp_path):
    kw = _matching_inputs(tmp_path)
    a, b = E.run_bootstrap(**kw)[1], E.run_bootstrap(**kw)[1]
    assert a.height == 20 and a.equals(b)
    assert not a.equals(E.run_bootstrap(**{**kw, "seed": 6})[1])
    assert not set(a.drop("sample_id").to_numpy().ravel()) & set(kw["case_genes"])
    two = E.run_bootstrap(**{**kw, "nthreads": 2})[1]
    assert two.height == 20


def test_one_set_of_matching_arguments():
    import inspect

    for f in (E.run_bootstrap, E.run_enrichment):
        names = set(inspect.signature(f).parameters)
        assert {"n_control_sets", "skip_factors", "tolerance", "max_rep"} <= names
        assert not names & {"sampler", "bounds", "offset_factor_idx", "offset_value",
                            "skip_factor_idx", "runs_number", "n_batches"}


def test_chain_tasks_are_pickled_by_reference(tmp_path, monkeypatch):
    """Workers must import the chain runner: a nested function drags the compiled kernels
    along by value, without their numba cache, and every worker compiles them again."""
    import pickle

    kw = _matching_inputs(tmp_path)
    sizes = []

    class Serial:
        def __init__(self, **kwargs):
            pass

        def __call__(self, tasks):
            out = []
            for f, args, kwargs in tasks:
                sizes.append(len(pickle.dumps(f)))
                out.append(f(*args, **kwargs))
            return out

    monkeypatch.setattr(E, "Parallel", Serial)
    E.run_bootstrap(**kw)
    assert sizes and max(sizes) < 200

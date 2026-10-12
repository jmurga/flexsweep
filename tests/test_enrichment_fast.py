"""run_enrichment pipeline and bootstrap sampler.

Counting/shuffling/FDR semantics are checked against Perl in test_enrichment_parity.py.
Here: run_enrichment is reproducible with a seed and a reused bootstrap; distances against
brute force; the sampler law against its analytic distribution; the bootstrap stops (not
hangs) on an impossible pool and restarts a stuck chain reproducibly.
"""

import numpy as np
import pytest

from flexsweep import enrichment as E

POPS = ["p1", "p2", "p3", "p4"]
GROUPS = ["ga", "ga", "gb", "gc"]
THR = [2, 5, 10, 20, 40]
N_GENES = 80


def _inputs(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i:03d}" for i in range(N_GENES)]
    ranks = np.stack([rng.permutation(N_GENES) + 1 for _ in POPS], axis=1)
    rank_file = tmp_path / "all_ranks_x"
    with open(rank_file, "w") as fh:
        for g, r in zip(genes, ranks):
            fh.write(g + " " + " ".join(str(v) for v in r) + "\n")
    vip = genes[:12]
    pool = genes[12:-3]
    sets = [list(rng.choice(pool, len(vip))) for _ in range(60)]  # with duplicates
    return str(rank_file), vip, sets


def test_count_sweeps_fast_rejects_unknown_genes(tmp_path):
    rank_file, vip, sets = _inputs(tmp_path)
    bad = [s[:-1] + ["NOT_A_GENE"] for s in sets]
    with pytest.raises(ValueError, match="NOT_A_GENE"):
        E.count_sweeps_fast(E._read_rank_table(rank_file), vip, bad, THR, POPS, GROUPS)


@pytest.mark.parametrize("nthreads", [1, 32, 200])
def test_chain_layout(nthreads):
    layout = E._chain_layout(100, nthreads)  # 1000 control sets = 100 batches of 10
    assert sum(layout) == 100
    assert len(layout) == min(nthreads, 100)  # one chain per thread
    assert max(layout) - min(layout) <= 1


def _pipeline_inputs(tmp_path):
    rng = np.random.default_rng(3)
    n = 200
    genes = [f"ENSG{i:05d}" for i in range(n)]
    with open(tmp_path / "factors.txt", "w") as fh:
        for g in genes:
            vals = rng.normal(10.0, 0.5, 32)
            fh.write(g + " " + " ".join(f"{v:.4f}" for v in vals) + "\n")
    with open(tmp_path / "genes.bed", "w") as fh:
        for i, g in enumerate(genes):
            fh.write(f"{1 + i // 100}\t{i * 200_000}\t{i * 200_000 + 5000}\t{g}\t+\n")
    with open(tmp_path / "all_ranks_t", "w") as fh:
        ranks = np.stack([rng.permutation(n) + 1 for _ in POPS], axis=1)
        for g, r in zip(genes, ranks):
            fh.write(g + " " + " ".join(map(str, r)) + "\n")
    (tmp_path / "cases.txt").write_text("\n".join(genes[::10]) + "\n")
    return dict(
        sweep_files=[str(tmp_path / "all_ranks_t")],
        gene_set=str(tmp_path / "cases.txt"),
        factors_file=str(tmp_path / "factors.txt"),
        annotation_file=str(tmp_path / "genes.bed"),
        populations=POPS,
        groups=GROUPS,
        thresholds=[100, 50, 20, 10, 5],
        pop_interest="p1",
        n_control_sets=200,
        min_distance=0,
        n_shuffles=16,
        nthreads=2,
    )


def test_run_enrichment_seed_and_reuse(tmp_path):
    kw = _pipeline_inputs(tmp_path)
    out = tmp_path / "run1"
    a = E.run_enrichment(**kw, seed=11, output_folder=str(out))
    b = E.run_enrichment(**kw, seed=11)
    for x, y in zip(a, b):
        assert x.equals(y)
    case, control, curves, fdr = a
    assert control.height == 2 * 10 * 10
    assert set(curves["scope"]) == set(POPS) | {"ga", "gb", "gc", "All"}
    assert {"p_value", "p_value_perm", "n_replicates"} <= set(fdr.columns)
    used = (out / "used_genes.txt").read_text().split()
    assert set(case["gene_id"]) <= set(used)
    assert set(np.unique(control.drop("sample_id").to_numpy())) <= set(used)

    c = E.run_enrichment(**kw, seed=11, reuse_bootstrap=str(out))
    assert c[2].equals(curves) and c[3].equals(fdr)

    (tmp_path / "cases.txt").write_text("ENSG00001\n")
    with pytest.raises(ValueError, match="not case genes"):
        E.run_enrichment(**kw, seed=11, reuse_bootstrap=str(out))


def test_compute_distances_brute_force(tmp_path):
    """Oracle: brute-force min |centre - case centre| on 1-based integer centres,
    1e11 on chromosomes without case genes (the Perl distance_file convention)."""
    rng = np.random.default_rng(9)
    rows = []
    for i in range(120):
        c = ["1", "2", "3", "CHR_PATCH"][i % 4]
        s0 = int(rng.integers(0, 5_000_000)); rows.append((c, s0, s0 + int(rng.integers(1, 9000)), f"G{i:03d}"))
    with open(tmp_path / "g.bed", "w") as fh:
        for c, s0, e, g in rows:
            fh.write(f"{c}\t{s0}\t{e}\t{g}\t+\n")
    case = [g for c, _, _, g in rows if c in ("1", "2")][::5] + ["NOT_IN_BED"]
    got = dict(E.compute_distances(case, str(tmp_path / "g.bed")).iter_rows())
    centre = {g: (s0 + 1 + e) // 2 for c, s0, e, g in rows}
    chrom = {g: c for c, _, _, g in rows}
    for c, s0, e, g in rows:
        same = [centre[v] for v in case if v in chrom and chrom[v] == c]
        expected = min(abs(centre[g] - x) for x in same) if same else 100_000_000_000
        assert got[g] == expected, g
    assert len(got) == len(rows)


def _law_pool():
    """
    8 control genes, 2 matched factors (case mean 1) + 1 unconstrained factor.

    Factor 0: gene 0 = 1e6, the rest 1. Factor 1: neutral genes 1-4 = 1, high genes 5-6
    = 501, low gene 7 = -499 (the last index, so an off-by-one upper bound is caught).
    A pair passes the +-5% check for every running state iff its factor-1 sum is 2 (the
    new mean is then a convex combination of the case mean and the in-bounds running
    mean) and fails for every state otherwise (|sum - 2| >= 1000 while the running-mean
    denominator stays < 300). The valid ordered pairs are fixed: distinct neutral pairs
    (12) plus high/low in either order (4).
    """
    f1 = np.array([1, 1, 1, 1, 1, 501, 501, -499], dtype=np.float64)
    f0 = np.ones(8)
    f0[0] = 1e6
    f2 = np.linspace(-1e3, 1e3, 8)  # unconstrained factor: must not reject anything
    nonvip = np.ascontiguousarray(np.stack([f0, f1, f2], axis=1))
    vip = np.ones((2, 3))
    valid = [(a, b) for a in range(1, 5) for b in range(1, 5) if a != b]
    valid += [(5, 7), (6, 7), (7, 5), (7, 6)]
    return vip, nonvip, valid


def test_sampler_law_uniform_over_valid_pairs():
    """Analytic oracle: with a state-independent acceptance region and no repetition
    limit, every accepted ordered pair is uniform over the valid ordered pairs."""
    from scipy import stats

    vip, nonvip, valid = _law_pool()
    counts = dict.fromkeys(valid, 0)
    for s in np.random.SeedSequence(12345).spawn(400):
        sets, used, stuck_at = E.iterative_control_set_fast(
            vip, nonvip, 2, 0.05, 10**9, np.random.Generator(np.random.PCG64(s)), 2, 1
        )
        assert stuck_at == -1
        assert sets.shape == (10, 2)  # vip_number = 2: each set is one accepted pair
        for a, b in sets:
            assert (a, b) in counts, f"invalid pair {(a, b)} accepted"
            counts[(a, b)] += 1
        assert used[0] == 0
    obs = np.array(list(counts.values()))
    assert obs.sum() == 4000
    assert stats.chisquare(obs).pvalue > 1e-3


def test_bootstrap_stops_when_no_pair_can_match(tmp_path):
    """A pool that can never match (tolerance 0) stops with an error, not a hang."""
    vip, nonvip, _ = _law_pool()
    sets, used, stuck_at = E.iterative_control_set_fast(
        vip, nonvip + 0.5, 2, 0.0, 10**9, np.random.default_rng(1), 2, 1, 10_000
    )
    assert stuck_at == 0 and sets.shape == (0, 2) and used.sum() == 0

    kw = _pipeline_inputs(tmp_path)
    case, ctrl, _ = E.load_case_control(
        kw["gene_set"], [f"ENSG{i:05d}" for i in range(200)], set()
    )
    with pytest.raises(RuntimeError, match="lower --min_distance or raise --tolerance"):
        E.run_bootstrap_nomatchomega(
            case_genes=case, factors_file=kw["factors_file"],
            annotation_file=kw["annotation_file"], runs_number=2, n_batches=1,
            tolerance=0.0, min_dist=0, flip=False, max_rep=25, seed=1, nthreads=1,
            control_genes=ctrl, max_tries=10_000,
        )


def test_bootstrap_restarts_stuck_chain(tmp_path, monkeypatch, capsys):
    """A chain whose first attempt is stuck restarts on a seed-derived stream, warns,
    and the run stays reproducible."""
    kw = _pipeline_inputs(tmp_path)
    case, ctrl, _ = E.load_case_control(
        kw["gene_set"], [f"ENSG{i:05d}" for i in range(200)], set()
    )
    args = dict(
        case_genes=case, factors_file=kw["factors_file"],
        annotation_file=kw["annotation_file"], runs_number=2, n_batches=1,
        tolerance=0.05, min_dist=0, flip=False, max_rep=25, seed=3, nthreads=1,
        control_genes=ctrl,
    )
    clean = E.run_bootstrap_nomatchomega(**args)[1]

    real = E.iterative_control_set_fast
    calls = []

    def first_attempt_stuck(*a):
        calls.append(1)
        if len(calls) == 1:  # chain 0, attempt 0
            return np.empty((0, a[2]), dtype=np.int64), np.zeros(a[1].shape[0], np.int32), 7
        return real(*a)

    monkeypatch.setattr(E, "iterative_control_set_fast", first_attempt_stuck)
    a = E.run_bootstrap_nomatchomega(**args)[1]
    assert "chain 0 got stuck 1 time(s) (after 7 control genes)" in capsys.readouterr().out
    calls.clear()
    b = E.run_bootstrap_nomatchomega(**args)[1]
    assert a.equals(b) and a.height == clean.height
    assert not a.equals(clean)  # nthreads=1 -> one chain, rerun on the restart stream

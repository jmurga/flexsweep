"""run_enrichment pipeline and bootstrap sampler.

Counting/shuffling/FDR semantics are checked against Perl in test_enrichment_parity.py.
Here: run_enrichment is reproducible with a seed and a reused bootstrap; distances against
brute force; the bootstrap stops (not
hangs) on an impossible pool and restarts a stuck chain reproducibly.
"""

import re

import numpy as np
import polars as pl
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


def test_bootstrap_stops_when_nothing_can_match(tmp_path):
    """A pool that cannot match (tolerance 0) stops with an error, not a hang."""
    kw = _pipeline_inputs(tmp_path)
    case, ctrl, _ = E.load_case_control(
        kw["gene_set"], [f"ENSG{i:05d}" for i in range(200)], set()
    )
    with pytest.raises(RuntimeError, match="lower --min_distance or raise --tolerance"):
        E.run_bootstrap(
            case_genes=case, factors_file=kw["factors_file"],
            annotation_file=kw["annotation_file"], n_control_sets=20,
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
        annotation_file=kw["annotation_file"], n_control_sets=20,
        tolerance=0.05, min_dist=0, flip=False, max_rep=25, seed=3, nthreads=1,
        control_genes=ctrl,
    )
    clean = E.run_bootstrap(**args)[1]

    real = E.batch_control_chain
    calls = []

    def first_attempt_stuck(*a):
        calls.append(1)
        if len(calls) == 1:  # chain 0, attempt 0
            n_pool = a[0].shape[0]
            return np.empty(0, np.int32), np.zeros(1, np.int64), np.zeros(n_pool, np.int32), 7
        return real(*a)

    monkeypatch.setattr(E, "batch_control_chain", first_attempt_stuck)
    a = E.run_bootstrap(**args)[1]
    assert "chain 0 got stuck 1 time(s) (after 7 control genes)" in capsys.readouterr().out
    calls.clear()
    b = E.run_bootstrap(**args)[1]
    assert a.equals(b) and a.height == clean.height
    assert not a.equals(clean)  # nthreads=1 -> one chain, rerun on the restart stream


# --- skipped factors, shift of non-positive case means


def _matching_inputs(tmp_path):
    """200 genes x 32 factors; factor 9 is noisy, so matching it changes the sets, and
    factor 28 has a negative case mean (-1), which is shifted."""
    rng = np.random.default_rng(17)
    genes = [f"ENSG{i:05d}" for i in range(200)]
    X = rng.normal(10.0, 0.5, (200, 32))
    X[:, 27] = rng.normal(-1.0, 0.05, 200)
    X[:, 8] = rng.normal(0.0, 5.0, 200)
    with open(tmp_path / "factors.txt", "w") as fh:
        for g, row in zip(genes, X):
            fh.write(g + " " + " ".join(f"{v:.4f}" for v in row) + "\n")
    with open(tmp_path / "genes.bed", "w") as fh:
        for i, g in enumerate(genes):
            fh.write(f"{1 + i // 100}\t{i * 200_000}\t{i * 200_000 + 5000}\t{g}\t+\n")
    return dict(
        case_genes=genes[::10], factors_file=str(tmp_path / "factors.txt"),
        annotation_file=str(tmp_path / "genes.bed"), n_control_sets=20,
        tolerance=0.05, min_dist=0, flip=False, max_rep=25, seed=5, nthreads=1,
        max_restarts=0,
    )


def test_skipped_factors_change_the_sets(tmp_path, capsys):
    kw = _matching_inputs(tmp_path)
    out = [E.run_bootstrap(**kw, skip_factors=k)[1] for k in (None, 9, [9], [9, 28])]
    assert out[1].equals(out[2]) and not out[0].equals(out[1]) and not out[1].equals(out[3])
    log = capsys.readouterr().out
    assert log.count("Factor 28: case mean") == 3  # skipped factors are not shifted


@pytest.mark.parametrize(
    "skip, expected",
    [(None, []), ([], []), (9, [8]), ([9], [8]), ([4, 9], [3, 8]), (np.int64(1), [0]), (28, [27])],
)
def test_skip_mask(skip, expected):
    assert np.flatnonzero(E._skip_mask(skip, 28)).tolist() == expected


@pytest.mark.parametrize("bad", [0, -1, 29, [1, 29], list(range(1, 29))])
def test_skip_mask_rejects(bad):
    with pytest.raises(ValueError, match="skip_factors"):
        E._skip_mask(bad, 28)


def test_shift_nonpositive_by_hand():
    """Case means -3, 0, 2, -3 (skipped) and -4 (constant). Factor 0 over cases + pool is
    -4, -2, 0, 2: population sd sqrt(5). Factor 1 is -1, 1, 1, -1: sd 1."""
    vip = np.array([[-4, -1, 2, -3, -4], [-2, 1, 2, -3, -4]], dtype=float)
    pool = np.array([[0, 1, 5, 1, -4], [2, -1, 6, 2, -4]], dtype=float)
    skip = np.array([False, False, False, True, False])
    v0, p0 = vip.copy(), pool.copy()
    v, p, shifts = E._shift_nonpositive(vip, pool, skip)
    assert [s[0] for s in shifts] == [0, 1, 4]
    np.testing.assert_allclose(v.mean(axis=0), [5**0.5, 1.0, 2.0, -3.0, 1.0])
    # cases and pool move together, untouched columns stay, inputs are not modified
    c = np.array([5**0.5 + 3, 1.0, 0.0, 0.0, 5.0])
    np.testing.assert_allclose(v, v0 + c)
    np.testing.assert_allclose(p, p0 + c)
    assert np.array_equal(vip, v0) and np.array_equal(pool, p0)


@pytest.mark.parametrize("arg, expected", [(None, None), ("9", [9]), ("9,28", [9, 28])])
def test_cli_skip_factors(monkeypatch, tmp_path, arg, expected):
    from click.testing import CliRunner

    from flexsweep.main import cli

    seen = {}

    def fake(**kw):
        seen.update(kw)
        raise RuntimeError("stop here")

    monkeypatch.setattr(E, "run_enrichment", fake)
    args = ["enrichment", "--sweep_files", "r", "--gene_set", "g", "--factors", "f",
            "--annotation", "a", "--populations", "p", "--groups", "g", "--thresholds", "5"]
    CliRunner().invoke(cli, args + (["--skip_factors", arg] if arg else []))
    assert seen["skip_factors"] == expected


@pytest.mark.parametrize("bad", ["0", "x", "9,,3", "-2"])
def test_cli_skip_factors_rejects(bad):
    from click.testing import CliRunner

    from flexsweep.main import cli

    args = ["enrichment", "--sweep_files", "r", "--gene_set", "g", "--factors", "f",
            "--annotation", "a", "--populations", "p", "--groups", "g", "--thresholds", "5",
            f"--skip_factors={bad}"]
    res = CliRunner().invoke(cli, args)
    assert res.exit_code == 2 and "factor numbers from 1" in res.output


def test_run_enrichment_passes_matching_options(tmp_path, monkeypatch):
    kw = _pipeline_inputs(tmp_path)
    seen = {}

    def fake(**k):
        seen.update(k)
        raise RuntimeError("stop here")

    monkeypatch.setattr(E, "run_bootstrap", fake)
    with pytest.raises(RuntimeError, match="stop here"):
        E.run_enrichment(**kw, seed=1, skip_factors=[9], tolerance=0.07, max_rep=11)
    assert (seen["skip_factors"], seen["tolerance"], seen["max_rep"]) == ([9], 0.07, 11)


def test_defaults_are_those_of_the_c_parameter_file():
    """exdef_input_parameters_file_hassan.txt: Max_rep 25, Min_distance 200000,
    Tolerance_range 0.05, Flip no, and 100 control sets (the file has 10) in both functions."""
    import inspect

    d = {k: p.default for k, p in inspect.signature(E.run_bootstrap).parameters.items()}
    assert (d["tolerance"], d["max_rep"], d["min_dist"], d["flip"]) == (0.05, 25, 200_000, False)
    assert d["n_control_sets"] == 100
    assert d["skip_factors"] is None
    e = {k: p.default for k, p in inspect.signature(E.run_enrichment).parameters.items()}
    for a, b in (("tolerance", "tolerance"), ("max_rep", "max_rep"), ("flip", "flip"),
                 ("min_distance", "min_dist"), ("skip_factors", "skip_factors")):
        assert e[a] == d[b], a

    from flexsweep.main import cli

    c = {o.name: o.default for o in cli.commands["enrichment"].params}
    assert (c["tolerance"], c["max_rep"], c["min_distance"], c["flip"]) == (0.05, 25, 200_000, False)
    assert c["skip_factors"] is None and c["cluster_distance"] == 500_000
    assert c["shuffling_segs"] == 2 and c["count_sweeps"] is False
    assert c["n_shuffles"] == e["n_shuffles"] == 10_000 and c["pop_interest"] == "All"
    assert e["n_control_sets"] == 100


def test_fdr_scope_names():
    pops, groups = ["YRI", "CEU", "gbr"], ["AFR", "EUR", "EUR"]
    f = lambda x: E._fdr_scopes(x, pops, groups)
    assert f("All") == [("All", 5)]
    assert f("all,yri") == f(["ALL", "YRI"]) == [("All", 5), ("YRI", 0)]
    assert f(" EUR , GBR ") == [("EUR", 4), ("gbr", 2)]
    assert f("YRI,yri,All") == [("YRI", 0), ("All", 5)]
    # exact labels win over case; a label of a population and of a group is ambiguous
    assert E._fdr_scopes("afr", ["afr"], ["AFR"]) == [("afr", 0)]
    with pytest.raises(ValueError, match="fits 2 scopes"):
        E._fdr_scopes("Afr", ["afr"], ["AFR"])
    with pytest.raises(ValueError, match="fits 2 scopes"):
        E._fdr_scopes("YRI", ["YRI"], ["YRI"])
    with pytest.raises(ValueError, match="unknown scope 'XXX'.*YRI, CEU, gbr, AFR, EUR, All"):
        f("All,XXX")
    for bad in ("", "All,", ",", []):
        with pytest.raises(ValueError, match="empty scope"):
            f(bad)


def test_fdr_scope_rows_follow_add_scopes():
    pops, groups = ["a", "b", "c", "d"], ["g2", "g1", "g2", "g3"]
    vip = np.arange(8).reshape(4, 2)
    _, _, names = E._add_scopes(vip, vip[:, :, None], pops, groups)
    assert E._scope_names(pops, groups) == names


def test_run_enrichment_fdr_scopes_and_log(tmp_path, capsys):
    kw = {**_pipeline_inputs(tmp_path), "pop_interest": "all,p1,GA"}
    out = tmp_path / "run"
    _, _, curves, fdr = E.run_enrichment(**kw, seed=11, output_folder=str(out))
    log = capsys.readouterr().out
    assert fdr["scope"].to_list() == ["All", "p1", "ga"]
    assert pl.read_csv(out / "fdr.txt")["scope"].to_list() == ["All", "p1", "ga"]
    assert curves["n_sets"].unique().to_list() == [200]
    lines = log.splitlines()
    assert all(re.match(r"\[\d\d:\d\d:\d\d\] ", ln) for ln in lines)
    for r in fdr.iter_rows(named=True):
        want = f"FDR t {r['scope']}: p={r['p_value']:.4f}"
        assert sum(want in ln for ln in lines) == 1, want
    for part in ("Seed: 11", "FDR: All, p1, ga from 16 shuffles", "Matching: 200 control sets",
                 f"Output: {out}", "Time: bootstrap"):
        assert part in log, part
    one = E.run_enrichment(**{**kw, "pop_interest": "All"}, seed=11)
    assert one[3].equals(fdr.filter(pl.col("scope") == "All")) and one[2].equals(curves)


def test_unknown_fdr_scope_stops_before_the_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "run_bootstrap", lambda **k: pytest.fail("bootstrap started"))
    with pytest.raises(ValueError, match="unknown scope 'YRI'"):
        E.run_enrichment(**{**_pipeline_inputs(tmp_path), "pop_interest": "All,YRI"}, seed=1)


def test_cli_rejects_unknown_fdr_scope():
    from click.testing import CliRunner

    from flexsweep.main import cli

    args = ["enrichment", "--sweep_files", "r", "--gene_set", "g", "--factors", "f",
            "--annotation", "a", "--populations", "p", "--groups", "g", "--thresholds", "5",
            "--pop_interest", "All,q"]
    res = CliRunner().invoke(cli, args)
    assert res.exit_code == 2 and "unknown scope 'q'" in res.output


def test_fdr_progress_lines(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(E, "_PROGRESS_SECONDS", 0)
    E.run_enrichment(**{**_pipeline_inputs(tmp_path), "nthreads": 1}, seed=11)
    log = capsys.readouterr().out
    done = [int(x) for x in re.findall(r"FDR: (\d+) of 16 shuffles, about", log)]
    assert done and done == sorted(set(done)) and done[-1] < 16


def test_duration_text():
    assert [E._duration(x) for x in (0.4, 59.4, 59.6, 200, 3900)] == [
        "0 s", "59 s", "1 min 00 s", "3 min 20 s", "1 h 05 min"
    ]


@pytest.mark.parametrize("n", [0, -10, 5, 15])
def test_bootstrap_rejects_set_numbers_not_multiple_of_10(tmp_path, n):
    with pytest.raises(ValueError, match="multiple of 10"):
        E.run_bootstrap(**{**_matching_inputs(tmp_path), "n_control_sets": n})
